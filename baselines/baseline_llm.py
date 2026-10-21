"""
Baseline C: an LLM as the entity extractor, scored exactly like the other baselines.

Same test sentences (data/processed/ner_test.parquet), same gold handling
(resolve_overlaps + spans_to_bio), same seqeval entity-level metric. The only
difference is where the predicted spans come from.

How it works
  1. Sentences are sent in batches (default 10 per request) so the 156 test
     sentences cost 16 requests, inside a 20-request free daily quota.
  2. The LLM returns entity TEXT and LABEL only. LLMs are unreliable at
     character offsets, so this script finds each span in the sentence itself
     (left to right, case-insensitive, tolerant of repeated whitespace).
     Spans it cannot find are counted as "not found" and dropped.
  3. Every response is cached on disk. If the quota runs out, rerun the same
     command later and it resumes from the cache. Scoring only runs once every
     batch is done.

Modes
  zero : label definitions only
  few  : label definitions + K example sentences with gold labels, taken from
         the TRAIN part of ner_train (never dev, never test)

Protocol (avoid test peeking): iterate on the prompt with --split dev, then run
--split test once per mode and report that.

Setup (the same values the Streamlit app reads from secrets):
  set LLM_API_KEY=...
  set LLM_MODEL=<model name, e.g. the Flash-Lite model the RAG page uses>
  set LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/   (default)

Run from the repository root:
  python baselines/baseline_llm.py --mode few --split dev
  python baselines/baseline_llm.py --mode zero --split test
  python baselines/baseline_llm.py --mode few --split test
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "annotation"))

import pandas as pd
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from bio_utils import spans_to_bio, resolve_overlaps

LABELS = [
    "EQUIPMENT", "FAILURE_MODE", "CAUSE_FACTOR", "ACTION_TAKEN",
    "CONSEQUENCE", "QUANTITY", "LOCATION", "DATE_TIME", "MATERIAL_SPEC",
    "INSPECTION_FINDING", "PARTY_ROLE", "REGULATORY_REF",
]

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DATA = root_dir / "data" / "processed"
CACHE_DIR = DATA / "llm_cache"

# Written from annotation/ANNOTATION_GUIDELINES.md and the schema, the same
# rules the human annotator worked from. Do not tune this on the test set.
SYSTEM_PROMPT = """You are an information extraction system for U.S. pipeline incident reports (PHMSA).
Tag named entity spans in each sentence. Use ONLY these 12 labels:

EQUIPMENT: a physical pipeline-related component (valve, service line, pipe, regulator, tank).
FAILURE_MODE: what broke and how; the physical mechanism or defect (leak, external corrosion, crack, rupture). The category "excavation damage" is a FAILURE_MODE.
CAUSE_FACTOR: the root cause, why it happened (frost heave, improper installation, inadequate cathodic protection). Do not tag when the text says the cause is unknown or still under investigation. An expected condition that was absent counts ("not properly marked").
ACTION_TAKEN: a deliberate response after the incident (shut down, isolated, repaired, replaced, excavated, notified).
CONSEQUENCE: the resulting outcome (fire, explosion, injury, fatality, evacuation, release). Never the triggering event. Never tag negated mentions ("no injuries" -> tag nothing).
QUANTITY: a number with its unit as ONE span ("2.5 barrels", "9,303 MCF", "4 INCH", "$50,000"). A bare number with no unit is not a QUANTITY.
LOCATION: where it happened (station names, addresses, roads, facility areas).
DATE_TIME: dates and times.
MATERIAL_SPEC: what something is made of (steel, plastic, polyethylene, cast iron) and the substances involved (natural gas, crude oil).
INSPECTION_FINDING: what an inspection or examination found.
PARTY_ROLE: a party described by its role or action (third-party contractor, operator technician, excavator), not a bare company name.
REGULATORY_REF: a citation of a rule, regulation, procedure or code (49 CFR 192.605).

Rules:
- "Fire" in "fire department", "fire marshal", "fire chief" is not a CONSEQUENCE.
- Spans must not overlap. If two would nest, keep only the narrower, more specific one.
- Copy each span EXACTLY as it appears in the sentence. Do not paraphrase, correct spelling or change case.
- List entities in the order they appear. If a span appears twice, list it twice.
- A sentence may have no entities.

Return JSON only, in this shape:
{"results": [{"id": <sentence id>, "entities": [{"text": "<exact span>", "label": "<LABEL>"}]}]}
Return one result for every sentence id you were given."""


# ---------------------------------------------------------------- data

def load_splits():
    """Exactly the split used by convert_to_spacy.py and export_for_colab.py."""
    train_df = pd.read_parquet(DATA / "ner_train.parquet")
    test_df = pd.read_parquet(DATA / "ner_test.parquet")
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_dev = int(len(train_df) * 0.15)
    dev_df, fit_df = train_df.iloc[:n_dev], train_df.iloc[n_dev:]
    return fit_df.reset_index(drop=True), dev_df.reset_index(drop=True), test_df.reset_index(drop=True)


def gold_entities(row):
    """Gold spans after the schema's overlap rule, with their text."""
    text = row["sentence_text"]
    ents = resolve_overlaps([dict(e) for e in row["entities"]])
    return [{"text": text[e["start"]:e["end"]], "label": e["label"]} for e in ents]


def pick_few_shot(fit_df, k, seed=42):
    """Greedy pick of K train sentences that together cover as many labels as possible.
    Prefers sentences with 2 to 8 entities so examples are informative but short."""
    pool = fit_df.sample(frac=1, random_state=seed)
    pool = pool[pool["entities"].apply(lambda e: 2 <= len(e) <= 8)]
    chosen, covered = [], set()
    candidates = list(pool.itertuples(index=False))
    while len(chosen) < k and candidates:
        def gain(r):
            return len({e["label"] for e in r.entities} - covered)
        best = max(candidates, key=gain)
        candidates.remove(best)
        chosen.append(best)
        covered |= {e["label"] for e in best.entities}
    return [row._asdict() for row in chosen], covered


def build_messages(batch, examples):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if examples:
        ex_in = [{"id": i, "sentence": ex["sentence_text"]} for i, ex in enumerate(examples)]
        ex_out = {"results": [{"id": i, "entities": gold_entities(ex)} for i, ex in enumerate(examples)]}
        msgs.append({"role": "user", "content": "Sentences:\n" + json.dumps(ex_in, ensure_ascii=False)})
        msgs.append({"role": "assistant", "content": json.dumps(ex_out, ensure_ascii=False)})
    payload = [{"id": i, "sentence": s} for i, s in enumerate(batch)]
    msgs.append({"role": "user", "content": "Sentences:\n" + json.dumps(payload, ensure_ascii=False)})
    return msgs


# ---------------------------------------------------------------- LLM call

def parse_json(raw):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def call_llm(client, model, messages, retries=2):
    """Returns (parsed_json, usage_dict, seconds). Tries JSON mode, falls back to plain."""
    from openai import RateLimitError, AuthenticationError
    last_err = None
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages, temperature=0,
                    response_format={"type": "json_object"})
            except (RateLimitError, AuthenticationError):
                raise
            except Exception:
                resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
            secs = time.time() - t0
            content = resp.choices[0].message.content or ""
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None) if resp.usage else None,
                "completion_tokens": getattr(resp.usage, "completion_tokens", None) if resp.usage else None,
            }
            return parse_json(content), usage, secs
        except (RateLimitError, AuthenticationError):
            raise
        except Exception as e:  # bad JSON or a transient error: retry the same batch
            last_err = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"batch failed after {retries + 1} attempts: {last_err}")


# ---------------------------------------------------------------- spans

def locate(sentence, entities):
    """Map LLM (text, label) pairs to character offsets in the sentence.
    Returns (spans, stats). Repeated mentions match successive occurrences."""
    spans, cursor = [], {}
    stats = {"returned": 0, "found": 0, "not_found": 0, "bad_label": 0}
    for ent in entities or []:
        stats["returned"] += 1
        text = str(ent.get("text", "")).strip()
        label = str(ent.get("label", "")).strip().upper()
        if label not in LABELS:
            stats["bad_label"] += 1
            continue
        words = text.split()
        if not words:
            stats["not_found"] += 1
            continue
        pattern = re.compile(r"\s+".join(re.escape(w) for w in words), re.I)
        key = " ".join(words).upper()
        m = pattern.search(sentence, cursor.get(key, 0)) or pattern.search(sentence)
        if not m:
            stats["not_found"] += 1
            continue
        cursor[key] = m.end()
        spans.append({"start": m.start(), "end": m.end(), "label": label,
                      "text": sentence[m.start():m.end()]})
        stats["found"] += 1
    return spans, stats


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["zero", "few"], default="few")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--k", type=int, default=8, help="few-shot examples")
    ap.add_argument("--batch", type=int, default=10, help="sentences per request")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--sleep", type=float, default=4.0, help="seconds between requests")
    args = ap.parse_args()

    if not args.model:
        sys.exit("Set LLM_MODEL (or pass --model).")
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        sys.exit("Set LLM_API_KEY.")

    from openai import OpenAI, RateLimitError
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    fit_df, dev_df, test_df = load_splits()
    eval_df = dev_df if args.split == "dev" else test_df
    examples = []
    if args.mode == "few":
        examples, covered = pick_few_shot(fit_df, args.k)
        print(f"few-shot: {len(examples)} train examples covering {len(covered)}/12 labels; "
              f"missing: {sorted(set(LABELS) - covered) or 'none'}")

    sentences = eval_df["sentence_text"].tolist()
    batches = [sentences[i:i + args.batch] for i in range(0, len(sentences), args.batch)]

    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    run_name = f"{safe_model}_{args.mode}_{args.split}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{run_name}.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            cache[rec["key"]] = rec

    print(f"{run_name}: {len(sentences)} sentences, {len(batches)} requests, "
          f"{sum(1 for b in batches if _key(b, examples, args.model) in cache)} already cached")

    for bi, batch in enumerate(batches):
        key = _key(batch, examples, args.model)
        if key in cache:
            continue
        try:
            parsed, usage, secs = call_llm(client, args.model, build_messages(batch, examples))
        except RateLimitError:
            done = sum(1 for b in batches if _key(b, examples, args.model) in cache)
            sys.exit(f"\nQuota or rate limit hit after {done}/{len(batches)} batches. "
                     f"Progress is cached in {cache_path}; rerun the same command later to resume.")
        by_id = {}
        for r in parsed.get("results", []):
            try:
                by_id[int(r.get("id"))] = r.get("entities", [])
            except (TypeError, ValueError):
                continue
        rec = {"key": key, "batch": bi, "sentences": batch,
               "entities": [by_id.get(i) for i in range(len(batch))],
               "usage": usage, "seconds": round(secs, 2)}
        cache[key] = rec
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  batch {bi + 1}/{len(batches)}: {secs:.1f}s, "
              f"{usage.get('prompt_tokens')} in / {usage.get('completion_tokens')} out tokens")
        time.sleep(args.sleep)

    # ---------- score ----------
    totals = {"returned": 0, "found": 0, "not_found": 0, "bad_label": 0, "missing_sentences": 0}
    preds_out = []
    pred_by_sentence = []
    for batch in batches:
        rec = cache[_key(batch, examples, args.model)]
        for sent, ents in zip(rec["sentences"], rec["entities"]):
            if ents is None:  # the LLM skipped this id: score as no entities
                totals["missing_sentences"] += 1
                ents = []
            spans, st = locate(sent, ents)
            for k_ in ("returned", "found", "not_found", "bad_label"):
                totals[k_] += st[k_]
            pred_by_sentence.append(spans)

    y_true, y_pred = [], []
    per_commodity = {}
    for (_, row), spans in zip(eval_df.iterrows(), pred_by_sentence):
        _, gold_tags = spans_to_bio(row["sentence_text"], [dict(e) for e in row["entities"]])
        _, pred_tags = spans_to_bio(row["sentence_text"], spans)
        y_true.append(gold_tags)
        y_pred.append(pred_tags)
        c = per_commodity.setdefault(row["commodity_type"], ([], []))
        c[0].append(gold_tags)
        c[1].append(pred_tags)
        preds_out.append({"commodity_type": row["commodity_type"],
                          "sentence_text": row["sentence_text"],
                          "gold": gold_entities(row), "pred": spans})

    for commodity, (t, p) in per_commodity.items():
        print(f"\n{'=' * 60}\n{commodity.upper()}\n{'=' * 60}")
        print(classification_report(t, p, digits=3, zero_division=0))

    print(f"\n{'=' * 60}\nOVERALL ({args.split}, {args.mode}-shot, {args.model})\n{'=' * 60}")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))
    p, r, f = (precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred))
    print(f"micro: precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}")

    recs = [cache[_key(b, examples, args.model)] for b in batches]
    secs = [x["seconds"] for x in recs]
    tin = [x["usage"]["prompt_tokens"] for x in recs if x["usage"].get("prompt_tokens") is not None]
    tout = [x["usage"]["completion_tokens"] for x in recs if x["usage"].get("completion_tokens") is not None]
    print(f"\nspans returned {totals['returned']}, located {totals['found']}, "
          f"not found in sentence {totals['not_found']}, invalid label {totals['bad_label']}, "
          f"sentences the model skipped {totals['missing_sentences']}")
    print(f"requests {len(recs)}, total {sum(secs):.0f}s, "
          f"{sum(secs) / len(sentences):.2f}s per sentence (batched {args.batch})")
    if tin:
        print(f"tokens: {sum(tin)} in, {sum(tout)} out "
              f"({sum(tin) / len(sentences):.0f} in, {sum(tout) / len(sentences):.0f} out per sentence)")

    out_path = DATA / f"llm_preds_{run_name}.json"
    out_path.write_text(json.dumps({
        "model": args.model, "mode": args.mode, "split": args.split, "k": len(examples),
        "micro": {"precision": p, "recall": r, "f1": f}, "span_stats": totals,
        "sentences": preds_out}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\npredictions for error analysis: {out_path}")


def _key(batch, examples, model):
    h = hashlib.sha1()
    h.update(model.encode())
    h.update(SYSTEM_PROMPT.encode())
    h.update(json.dumps([e["sentence_text"] for e in examples]).encode())
    h.update(json.dumps(batch).encode())
    return h.hexdigest()[:16]


if __name__ == "__main__":
    main()
