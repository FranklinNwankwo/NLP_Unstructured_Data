"""
Evaluate the RAG layer against the gold question set in eval/questions.jsonl.

    python -m eval.run_eval --index-dir rag_index                  # full pipeline (2 LLM calls/question)
    python -m eval.run_eval --index-dir rag_index --filters-only   # filter parsing only (1 call/question)
    python -m eval.run_eval                                        # index from the HF dataset repo

How ground truth works
- Gold filter values are short tokens ("CORROSION", "GAS TRANSMISSION"). They are resolved
  to the exact values in your data first. A token that matches nothing, or more than one
  value, stops the run with the list of available values, so you fix the token once.
- Expected counts are never typed by hand: they come from applying the gold filters to the
  full incidents table, so they stay correct after every index rebuild.

What gets scored
- filter accuracy per field, and all fields exact
- count_correct: the engine's matching-incident count equals the gold count
- source_precision: share of retrieved reports that satisfy the gold filters
- citations_valid: every [id] cited in the answer is one of the retrieved reports
- count_in_answer: for "how many" questions, the gold count appears in the answer text

LLM settings come from environment variables or .streamlit/secrets.toml
(LLM_API_KEY, LLM_MODEL, LLM_BASE_URL).
"""
import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import RateLimitError

from rag.engine import Filters, RAGEngine

HERE = Path(__file__).parent
LIST_FIELDS = ("state", "system_type", "cause")
INT_FIELDS = ("year_min", "year_max")


# ---------------- setup ----------------
def load_llm_config() -> dict:
    cfg = {}
    secrets = Path(".streamlit") / "secrets.toml"
    if secrets.exists():
        import tomllib
        cfg = tomllib.loads(secrets.read_text(encoding="utf-8"))
    for k in ("LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "LLM_ANSWER_MODEL"):
        cfg[k] = os.environ.get(k, cfg.get(k))
    if not cfg["LLM_API_KEY"] or not cfg["LLM_MODEL"]:
        raise SystemExit("Set LLM_API_KEY and LLM_MODEL (env vars or .streamlit/secrets.toml).")
    return cfg


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def resolve_token(token: str, field: str, allowed: list[str]) -> str:
    t = str(token).strip().upper()
    exact = [a for a in allowed if a.upper() == t]
    if exact:
        return exact[0]
    if field == "state":
        raise ValueError(f"state {token!r} not in data")
    hits = [a for a in allowed if t in a.upper()]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(
        f"{field} token {token!r} matched {hits or 'nothing'}. Available: {allowed}"
    )


def build_gold(questions: list[dict], engine: RAGEngine) -> dict[str, Filters]:
    gold, errors = {}, []
    for q in questions:
        g = q["gold"]
        resolved = {}
        for field in LIST_FIELDS:
            vals = []
            for tok in g.get(field) or []:
                try:
                    vals.append(resolve_token(tok, field, engine.allowed[field]))
                except ValueError as e:
                    errors.append(f"{q['id']}: {e}")
            resolved[field] = sorted(set(vals))
        gold[q["id"]] = Filters(
            **resolved, year_min=g.get("year_min"), year_max=g.get("year_max"),
            search_text=q["question"],
        )
    if errors:
        raise SystemExit("Fix these gold tokens in eval/questions.jsonl:\n  " + "\n  ".join(errors))
    return gold


# ---------------- scoring helpers ----------------
def field_scores(pred: Filters, gold: Filters) -> dict:
    s = {f: sorted(getattr(pred, f)) == sorted(getattr(gold, f)) for f in LIST_FIELDS}
    s.update({f: getattr(pred, f) == getattr(gold, f) for f in INT_FIELDS})
    s["filters_exact"] = all(s.values())
    return s


def cited_ids(answer: str) -> set[str]:
    ids = set()
    for group in re.findall(r"\[([^\[\]]+)\]", answer or ""):
        for part in group.split(","):
            part = part.strip()
            if part:
                ids.add(part)
    return ids


def count_in_text(n: int, text: str) -> bool:
    variants = {str(n), f"{n:,}"}
    return any(re.search(rf"(?<![\d,]){re.escape(v)}(?![\d,])", text or "") for v in variants)


class DailyQuotaExceeded(Exception):
    pass


def quota_detail(err: Exception) -> str:
    """Pull the useful part ('Quota exceeded for metric: ..., limit: N, model: ...') out of a 429."""
    text = str(err)
    m = re.search(r"Quota exceeded for metric:[^\\\n'\"]*", text)
    return m.group(0) if m else text[:600]


def with_retries(fn, attempts: int = 4):
    for i in range(attempts):
        try:
            return fn()
        except RateLimitError as e:
            text = str(e)
            if "PerDay" in text or "per day" in text.lower():
                raise DailyQuotaExceeded(quota_detail(e)) from e
            if i == attempts - 1:
                raise
            m = re.search(r"retry(?:Delay)?\D{0,10}(\d+(?:\.\d+)?)\s*s", text, re.I)
            wait = float(m.group(1)) + 2 if m else 20.0 * (i + 1)
            print(f"rate limited, waiting {wait:.0f}s ...", end=" ", flush=True)
            time.sleep(wait)


def fmt_filters(f: Filters) -> str:
    return f.describe()


# ---------------- main loop ----------------
def run(args):
    cfg = load_llm_config()
    engine = RAGEngine(
        api_key=cfg["LLM_API_KEY"], model=cfg["LLM_MODEL"],
        base_url=cfg.get("LLM_BASE_URL"), index_dir=args.index_dir,
        answer_model=cfg.get("LLM_ANSWER_MODEL"),
    )
    questions = load_questions(Path(args.questions))
    gold = build_gold(questions, engine)
    if args.ids:
        wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        questions = [q for q in questions if q["id"] in wanted]
        if not questions:
            raise SystemExit(f"No questions match --ids {args.ids}")

    rows = []
    for i, q in enumerate(questions, 1):
        g = gold[q["id"]]
        gold_ids = set(engine.apply_filters(g)["report_id"])
        row = {"id": q["id"], "question": q["question"], "tests": q.get("tests", ""),
               "gold_filters": fmt_filters(g), "gold_count": len(gold_ids)}
        print(f"[{i}/{len(questions)}] {q['id']} ...", end=" ", flush=True)

        try:
            if args.filters_only:
                pred = with_retries(lambda: engine.parse_filters(q["question"]))
                pred_count = len(engine.apply_filters(pred))
                answer, sources = "", pd.DataFrame(columns=["report_id"])
            else:
                res = with_retries(lambda: engine.ask(q["question"]))
                pred, answer, sources = res.filters, res.answer, res.sources
                pred_count = res.stats["matching_incidents"]
        except DailyQuotaExceeded as e:
            done = [r["id"] for r in rows if "error" not in r]
            left = [x["id"] for x in questions[i - 1:]]
            print(f"\n\nSTOPPED: daily quota used up.\n  {e}\n"
                  f"Rerun the rest after the quota resets:\n"
                  f"  python -m eval.run_eval --index-dir {args.index_dir or '<dir>'}"
                  f"{' --filters-only' if args.filters_only else ''} --ids {','.join(left)}\n")
            if not done:
                return
            break
        except Exception as e:
            row.update({"error": quota_detail(e) if isinstance(e, RateLimitError) else str(e)[:600]})
            rows.append(row)
            print("ERROR")
            time.sleep(args.sleep)
            continue

        row["pred_filters"] = fmt_filters(pred)
        row.update(field_scores(pred, g))
        row["pred_count"] = pred_count
        row["count_correct"] = pred_count == len(gold_ids)

        if not args.filters_only:
            src_ids = set(sources["report_id"].astype(str)) if len(sources) else set()
            cites = cited_ids(answer)
            row["n_sources"] = len(src_ids)
            row["source_precision"] = (len(src_ids & gold_ids) / len(src_ids)) if src_ids else None
            row["has_citation"] = bool(cites) if src_ids else None
            row["citations_valid"] = cites <= src_ids if cites else None
            row["uncited_ids"] = ", ".join(sorted(cites - src_ids))
            row["count_in_answer"] = (
                count_in_text(len(gold_ids), answer) if q.get("check_count_in_answer") else None
            )
            row["answer"] = answer

        rows.append(row)
        print("ok" if row["filters_exact"] else "filters differ")
        time.sleep(args.sleep)

    write_outputs(pd.DataFrame(rows), args)


# ---------------- reporting ----------------
def rate(df: pd.DataFrame, col: str):
    if col not in df:
        return None
    s = df[col].dropna()
    if not len(s):
        return None
    return float(s.astype(float).mean()), int(len(s))


def write_outputs(df: pd.DataFrame, args):
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"results_{stamp}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    metrics = [
        ("Filters fully correct", "filters_exact"),
        ("State correct", "state"),
        ("System type correct", "system_type"),
        ("Cause correct", "cause"),
        ("Year min correct", "year_min"),
        ("Year max correct", "year_max"),
        ("Matching count correct", "count_correct"),
        ("Retrieved reports satisfy gold filters", "source_precision"),
        ("Answer has citations", "has_citation"),
        ("All citations are retrieved reports", "citations_valid"),
        ("Gold count stated in answer", "count_in_answer"),
    ]
    lines = [
        f"# RAG evaluation ({'filters only' if args.filters_only else 'full pipeline'})",
        "",
        f"Run: {stamp} · questions: {len(df)} · errors: {int(df['error'].notna().sum()) if 'error' in df else 0}",
        "",
        "| Metric | Score | n |",
        "|---|---|---|",
    ]
    for label, col in metrics:
        r = rate(df, col)
        if r:
            lines.append(f"| {label} | {r[0]:.0%} | {r[1]} |")

    zero = df[df["gold_count"] == 0]
    if len(zero):
        lines += ["", "**Questions with 0 gold matches** (consider swapping them for "
                  "questions your data covers): " + ", ".join(zero["id"])]

    if "filters_exact" in df:
        bad = df[df["filters_exact"] == False]  # noqa: E712
        if len(bad):
            lines += ["", "## Filter mismatches", "", "| id | tests | gold | predicted |", "|---|---|---|---|"]
            for _, r in bad.iterrows():
                lines.append(f"| {r['id']} | {r['tests']} | {r['gold_filters']} | {r['pred_filters']} |")

    if "error" in df and df["error"].notna().any():
        lines += ["", "## Errors", ""]
        for _, r in df[df["error"].notna()].iterrows():
            lines.append(f"- {r['id']}: {r['error']}")

    report = "\n".join(lines) + "\n"
    report_path = out_dir / f"report_{stamp}.md"
    report_path.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Saved {csv_path} and {report_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(HERE / "questions.jsonl"))
    ap.add_argument("--index-dir", default=None, help="Local rag_index folder; omit to download from HF")
    ap.add_argument("--filters-only", action="store_true", help="Score filter parsing only (1 LLM call/question)")
    ap.add_argument("--sleep", type=float, default=4.0, help="Seconds between questions (free-tier rate limits)")
    ap.add_argument("--ids", default=None, help="Comma-separated question ids to run, e.g. q03,q04")
    run(ap.parse_args())
