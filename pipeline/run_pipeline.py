"""
Runs the full pipeline on real narratives and prints a readable summary.

    python pipeline\run_pipeline.py                  one narrative per commodity from the severity TEST split
    python pipeline\run_pipeline.py --text "..."     your own narrative
    python pipeline\run_pipeline.py --file note.txt  a narrative from a text file

The full JSON records are written to pipeline/sample_output.json.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

from incident_extractor import IncidentExtractor


def show(rec, truth=None):
    print("=" * 78)
    print((rec["commodity_type"] or "narrative").upper())
    print("=" * 78)
    text = rec["narrative"]
    print(text[:700] + (" ..." if len(text) > 700 else ""))

    sev = rec["severity"]
    if sev:
        probs = ", ".join(f"{k} {v:.0%}" for k, v in sorted(sev["probabilities"].items(), key=lambda kv: -kv[1]))
        line = f"\nSeverity: {sev['label']}   ({probs})"
        if truth:
            line += f"\nPHMSA flag-derived label: {truth}"
        print(line)
    else:
        print(f"\nSeverity: not predicted ({rec.get('note')})")

    by = {}
    for e in rec["entities"]:
        texts = by.setdefault(e["label"], [])
        if e["text"] not in texts:
            texts.append(e["text"])
    print("\nEntities:")
    for label, texts in sorted(by.items()):
        more = f" (+{len(texts) - 8} more)" if len(texts) > 8 else ""
        print(f"  {label}: {texts[:8]}{more}")

    print("\nRelations:")
    if not rec["relations"]:
        print("  (none)")
    for r in rec["relations"]:
        tail = r["tail"]["text"] + (f" ({r['tail']['unit_type']})" if "unit_type" in r["tail"] else "")
        flag = "  (low confidence)" if r["confidence"] == "low" else ""
        print(f"  s{r['sentence_index']}: {r['head']['text']} --{r['type']}--> {tail}{flag}")

    if rec["dropped_negated"]:
        print("\nDropped as negated: " + ", ".join(d["text"] for d in rec["dropped_negated"]))
    print()


def pick_demo_narratives(seed, per_commodity):
    data = json.load(open(ROOT / "data/processed/severity_colab_export.json", encoding="utf-8"))
    labels = data["labels"]
    by = {}
    for r in data["test"]:
        if 60 <= len(r["text"].split()) <= 250:       # readable length
            by.setdefault(r["commodity_type"], []).append(r)
    rng = random.Random(seed)
    picked = []
    for c in sorted(by):
        for r in rng.sample(by[c], min(per_commodity, len(by[c]))):
            picked.append((r["text"], c, labels[r["label"]]))
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--file")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per_commodity", type=int, default=1)
    args = ap.parse_args()

    if args.file:
        items = [(Path(args.file).read_text(encoding="utf-8", errors="replace"), None, None)]
    elif args.text:
        items = [(args.text, None, None)]
    else:
        items = pick_demo_narratives(args.seed, args.per_commodity)

    print("Loading models...")
    extractor = IncidentExtractor.load_default()
    t0 = time.time()
    records = extractor.extract_many([i[0] for i in items], [i[1] for i in items])
    print(f"Processed {len(items)} narrative(s) in {time.time() - t0:.1f}s\n")

    for rec, (_, _, truth) in zip(records, items):
        show(rec, truth)

    out = ROOT / "pipeline/sample_output.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Full records written to {out}")


if __name__ == "__main__":
    main()