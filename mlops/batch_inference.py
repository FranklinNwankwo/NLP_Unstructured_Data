"""
Batch inference + drift report. Pulls PHMSA's current flat files, keeps reports the models have
never seen (not in the training snapshot, mlops/seen_reports.txt), runs the full pipeline on them,
and compares the output with the reference profile (mlops/reference_profile.json).

    python mlops/batch_inference.py                          # download PHMSA files, pinned models
    python mlops/batch_inference.py --source-dir data/raw    # use flat files already on disk (.txt or .zip)
    python mlops/batch_inference.py --model-dir local        # the repo's own model folders, no download
    python mlops/batch_inference.py --all --max 500          # ignore the seen list; newest 500 reports
    python mlops/batch_inference.py --fail-on-alert          # exit 1 if any drift check alerts (for CI)

What it watches (PSI warn 0.10 / alert 0.20):
  - input:   narrative length distribution, commodity mix
  - NER:     share of narratives with zero entities, entities per narrative, entity-type mix
  - classes: predicted severity mix and cause mix
  - quality: agreement of predicted severity with PHMSA's fatality/injury/fire/explosion flags, and of
             predicted cause with PHMSA's coded cause. These are the only ground truth that arrives
             with new reports, so they are the closest thing to live accuracy. Alert on a drop of
             5 points or more against the reference.

Writes mlops/runs/<YYYY-MM-DD>/: predictions.jsonl.gz (one record per report), drift_report.md,
drift_report.json, and disagreements.csv (confident predictions that contradict PHMSA's coded cause,
for review: either model errors or coding errors worth reading).
"""

import argparse
import csv
import datetime as dt
import gzip
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MIN_WORDS, REFERENCE_PATH, ROOT, SEEN_REPORTS, clean_cause, compare,  # noqa: E402
                    get_extractor, load_manifest, load_phmsa, profile, report_key, report_markdown,
                    severity_from_flags)


def select_reports(df, seen, use_all=False, max_n=0):
    """Reports with a usable narrative, not in the training snapshot (unless use_all), newest first."""
    import pandas as pd
    text = df["NARRATIVE"].fillna("").astype(str).str.strip()
    df = df.assign(_text=text)[text.str.split().str.len() >= MIN_WORDS].copy()
    df["_key"] = [report_key(c, n) for c, n in zip(df["commodity_type"], df["REPORT_NUMBER"])]
    df = df.drop_duplicates("_key", keep="last")
    total = len(df)
    if not use_all:
        df = df[~df["_key"].isin(seen)]
    for col in ("REPORT_RECEIVED_DATE", "LOCAL_DATETIME"):
        if col in df.columns:
            df["_date"] = pd.to_datetime(df[col], errors="coerce")
            df = df.sort_values("_date", ascending=False, na_position="last")
            break
    if max_n:
        df = df.head(max_n)
    return df, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", help="folder with the three PHMSA flat files (.txt or .zip); default: download")
    ap.add_argument("--model-dir")
    ap.add_argument("--all", action="store_true", help="score every report, not only ones missing from the snapshot")
    ap.add_argument("--max", type=int, default=0, help="score at most N reports (newest first)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out-dir", default=str(ROOT / "mlops/runs"))
    ap.add_argument("--reference", default=str(REFERENCE_PATH))
    ap.add_argument("--fail-on-alert", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.out_dir) / dt.date.today().isoformat()
    run_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = load_phmsa(args.source_dir, download_to=None if args.source_dir else run_dir / "raw")
    seen = set(SEEN_REPORTS.read_text(encoding="utf-8").split()) if SEEN_REPORTS.exists() else set()
    if not seen and not args.all:
        print("WARNING: mlops/seen_reports.txt missing, so every report counts as new. Run mlops/freeze_eval_sets.py.")
    batch, total = select_reports(df, seen, args.all, args.max)
    print(f"PHMSA files: {len(df):,} rows, {total:,} reports with a narrative; "
          f"{len(batch):,} selected ({'all' if args.all else 'not in the training snapshot'})  ({time.time() - t0:.0f}s)")

    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    flags = [severity_from_flags(r) for r in batch.to_dict("records")]
    coded = [clean_cause(c) for c in batch.get("CAUSE", [None] * len(batch))]

    records = []
    if len(batch):
        extractor, source = get_extractor(args.model_dir, load_manifest())
        texts, comms = batch["_text"].tolist(), batch["commodity_type"].tolist()
        t0 = time.time()
        for i in range(0, len(texts), args.batch):
            records += extractor.extract_many(texts[i:i + args.batch], comms[i:i + args.batch])
            print(f"  {min(i + args.batch, len(texts)):,} / {len(texts):,}  ({time.time() - t0:.0f}s)", end="\r")
        print()
    else:
        source = "(no models loaded: empty batch)"

    with gzip.open(run_dir / "predictions.jsonl.gz", "wt", encoding="utf-8") as f:
        for key, rec, flag, cause in zip(batch["_key"], records, flags, coded):
            f.write(json.dumps({"report": key, "phmsa_severity_flags": flag, "phmsa_cause": cause, **rec},
                               ensure_ascii=False) + "\n")

    if not records:
        md = "# PHMSA drift report\n\nNo new reports since the training snapshot. Nothing to score.\n"
        (run_dir / "drift_report.md").write_text(md, encoding="utf-8")
        (run_dir / "drift_report.json").write_text(json.dumps({"n_narratives": 0, "alerts": []}) + "\n")
        _summary(md)
        print(md)
        return 0

    current = profile(records, flags, coded)
    rows, alerts = compare(reference, current)

    # confident cause predictions that contradict PHMSA's code: worth a human look
    dis = []
    for key, rec, cause in zip(batch["_key"], records, coded):
        if rec.get("cause") and cause and rec["cause"]["label"] != cause:
            conf = max(rec["cause"]["probabilities"].values())
            if conf >= 0.8:
                dis.append((conf, key, cause, rec["cause"]["label"], rec["narrative"][:300].replace("\n", " ")))
    dis.sort(reverse=True)
    with open(run_dir / "disagreements.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["report", "phmsa_cause", "predicted_cause", "confidence", "narrative_start"])
        for conf, key, cause, pred, text in dis:
            w.writerow([key, cause, pred, f"{conf:.3f}", text])

    comm = Counter(batch["commodity_type"])
    extra = [f"Models: `{source}`  ",
             f"Reference: {reference.get('source', 'reference_profile.json')}  ",
             "Batch by commodity: " + ", ".join(f"{k} {v}" for k, v in sorted(comm.items())) + "  ",
             f"Confident (>= 0.8) cause predictions that contradict PHMSA's coded cause: {len(dis)} "
             "(see disagreements.csv)"]
    if "_date" in batch.columns and batch["_date"].notna().any():
        extra.insert(2, f"Report dates: {batch['_date'].min():%Y-%m-%d} to {batch['_date'].max():%Y-%m-%d}  ")
    md = report_markdown("PHMSA drift report " + dt.date.today().isoformat(), reference, current, rows, alerts, extra)
    (run_dir / "drift_report.md").write_text(md, encoding="utf-8")
    (run_dir / "drift_report.json").write_text(json.dumps(
        {"date": dt.date.today().isoformat(), "models": source, "current": current, "checks": rows,
         "alerts": alerts, "disagreements": len(dis)}, indent=2) + "\n", encoding="utf-8")
    _summary(md)
    print(md)
    print(f"wrote {run_dir}")
    return 1 if (alerts and args.fail_on_alert) else 0


def _summary(md):
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(md)


if __name__ == "__main__":
    sys.exit(main())
