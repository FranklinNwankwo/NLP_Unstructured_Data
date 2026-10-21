"""
Why does CAUSE_FACTOR score 0.000? A finer look than strict span F1.

Strict seqeval F1 only counts a span when both edges and the label are exact,
so 0.000 cannot tell these cases apart:
  A. right region, wrong edges  -> the model tags CAUSE_FACTOR on part of the gold span
  B. right region, wrong label  -> the model tags the words, but as another type
  C. nothing                    -> the model tags the words as O
and it cannot tell "the model never predicts CAUSE_FACTOR" from "it predicts it
in the wrong places".

For every gold CAUSE_FACTOR span this script reports which case applies, how
much probability the model gave CAUSE_FACTOR on those words, and where
CAUSE_FACTOR ranked among the 12 types. It also prints token-level precision,
recall and F1 for CAUSE_FACTOR (B- and I- merged, one point per word), and how
often the model predicts CAUSE_FACTOR anywhere.

Run from the repository root, with the model in ner_model/phmsa_ner_model:
    python ner_model/cause_factor_diagnosis.py                 # test set (14 spans)
    python ner_model/cause_factor_diagnosis.py --split train   # did it even fit the training spans?

--split train uses the same 760 training sentences as the Colab export (seed 42).
If the model scores near zero there too, it never learned the type at all
(underfitting), which points at the data rather than at generalization.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "annotation"))
sys.path.insert(0, str(ROOT / "ner_model"))

import pandas as pd
import torch

from bio_utils import resolve_overlaps
from predict import NERPredictor, _word_tokenizer

TARGET = "CAUSE_FACTOR"


def load_split(name):
    if name == "test":
        return pd.read_parquet(ROOT / "data/processed/ner_test.parquet").reset_index(drop=True)
    df = pd.read_parquet(ROOT / "data/processed/ner_train.parquet")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)   # same as export_for_colab.py
    n_dev = int(len(df) * 0.15)
    return (df.iloc[:n_dev] if name == "dev" else df.iloc[n_dev:]).reset_index(drop=True)


@torch.no_grad()
def word_level(predictor, sentences, batch_size=32):
    """Per sentence: (doc, predicted type per word, {type: prob} per word). Types have B-/I- merged."""
    id2label = predictor.id2label
    types = sorted({l.split("-", 1)[1] for l in id2label.values() if l != "O"})
    out = []
    for i in range(0, len(sentences), batch_size):
        docs = [_word_tokenizer(s) for s in sentences[i:i + batch_size]]
        enc = predictor.tokenizer([[t.text for t in d] for d in docs], is_split_into_words=True,
                                  truncation=True, max_length=256, padding=True, return_tensors="pt")
        inputs = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        probs = predictor.model(**inputs).logits.softmax(dim=-1)
        for b, doc in enumerate(docs):
            pred_type = ["O"] * len(doc)
            type_probs = [None] * len(doc)
            seen = set()
            for pos, wid in enumerate(enc.word_ids(batch_index=b)):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                p = probs[b, pos]
                tp = {t: 0.0 for t in types}
                tp["O"] = 0.0
                for idx, lab in id2label.items():
                    key = "O" if lab == "O" else lab.split("-", 1)[1]
                    tp[key] += float(p[idx])
                type_probs[wid] = tp
                lab = id2label[int(p.argmax())]
                pred_type[wid] = "O" if lab == "O" else lab.split("-", 1)[1]
            out.append((doc, pred_type, type_probs))
    return out


def gold_types(doc, entities):
    """Gold type per word (B-/I- merged), after the schema's overlap rule."""
    g = ["O"] * len(doc)
    for e in resolve_overlaps([dict(x) for x in entities]):
        for i, tok in enumerate(doc):
            if tok.idx < e["end"] and tok.idx + len(tok.text) > e["start"]:
                g[i] = e["label"]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "dev", "train"], default="test")
    args = ap.parse_args()

    df = load_split(args.split)
    predictor = NERPredictor()
    results = word_level(predictor, df["sentence_text"].tolist())

    tp = fp = fn = 0
    pred_cf_words = 0
    cases = Counter()
    other_labels = Counter()
    ranks = Counter()
    span_rows = []

    for (_, row), (doc, pred, probs) in zip(df.iterrows(), results):
        gold = gold_types(doc, row["entities"])
        words = [i for i, t in enumerate(doc) if not t.is_space and probs[i] is not None]
        for i in words:
            g, p = gold[i] == TARGET, pred[i] == TARGET
            tp += g and p
            fp += p and not g
            fn += g and not p
            pred_cf_words += p

        for e in resolve_overlaps([dict(x) for x in row["entities"]]):
            if e["label"] != TARGET:
                continue
            idx = [i for i in words if doc[i].idx < e["end"] and doc[i].idx + len(doc[i].text) > e["start"]]
            if not idx:
                continue
            span_pred = [pred[i] for i in idx]
            if TARGET in span_pred:
                case = "A. right region, wrong edges"
            elif any(p != "O" for p in span_pred):
                case = "B. tagged as another type"
                other_labels[Counter(p for p in span_pred if p != "O").most_common(1)[0][0]] += 1
            else:
                case = "C. tagged as nothing (O)"
            cases[case] += 1
            mean_p = sum(probs[i][TARGET] for i in idx) / len(idx)
            # rank of CAUSE_FACTOR among the 12 entity types (O excluded), averaged over the span
            avg = {t: sum(probs[i][t] for i in idx) / len(idx) for t in probs[idx[0]] if t != "O"}
            rank = sorted(avg, key=avg.get, reverse=True).index(TARGET) + 1
            ranks[rank] += 1
            pred_spans = [f"{doc[i].text}/{pred[i]}" for i in idx]
            span_rows.append((e["text"] if "text" in e else row["sentence_text"][e["start"]:e["end"]],
                              case[0], mean_p, rank, " ".join(pred_spans), row["sentence_text"]))

    n_spans = sum(cases.values())
    print(f"\n{TARGET} diagnosis on {args.split} ({len(df)} sentences, {n_spans} gold spans)\n")
    print("Per gold span:")
    for case in sorted(cases):
        extra = ""
        if case.startswith("B") and other_labels:
            extra = "   (" + ", ".join(f"{k} {v}" for k, v in other_labels.most_common()) + ")"
        print(f"  {case:<32}{cases[case]:>3}{extra}")
    print(f"\nRank of {TARGET} among the 12 types on the gold words (1 = model's top entity type):")
    print("  " + ", ".join(f"rank {r}: {ranks[r]}" for r in sorted(ranks)))

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"\nToken-level {TARGET}: precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}  "
          f"(gold words {tp + fn}, predicted words {pred_cf_words})")
    if pred_cf_words == 0:
        print(f"The model never predicted {TARGET} on any word in this split.")

    print("\nEach gold span (case, mean P(CAUSE_FACTOR) on its words, rank, model tags):")
    for text, case, mp, rank, tags, sent in span_rows:
        print(f"\n  [{case}] \"{text}\"   P={mp:.3f}  rank {rank}")
        print(f"      model: {tags}")
        print(f"      {sent[:160]}{'...' if len(sent) > 160 else ''}")


if __name__ == "__main__":
    main()
