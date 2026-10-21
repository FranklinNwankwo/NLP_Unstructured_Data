"""
Builds the cause-classification dataset for Colab: predict WHY an incident happened
from its narrative.

Labels come from PHMSA's structured apparent-cause field, so every report has one
without any hand annotation:
  --level cause   (default) CAUSE: 8 categories (corrosion, excavation damage, ...)
  --level detail  CAUSE_DETAILS: the sub-cause under each category (for example
                  natural force damage -> frost heave). Sub-causes with fewer than
                  --min_train training examples are merged into "<CAUSE> - OTHER".

Gas distribution words two categories differently; they are mapped onto the other
forms' wording with the same aliases the RAG layer uses (rag/config.py).

Split: IDENTICAL to the severity split (severity_model/export_for_colab.py): same
parquet, same 5-word cut-off, same strata (commodity x severity), same report-number
groups, same seed. So a narrative in the severity test set is in the cause test set
too, and the demo's example narratives stay unseen by both models. If
severity_colab_export.json exists, the script checks this narrative by narrative.
Rows without a cause label are dropped AFTER splitting, so they cannot shift the split.

Run from the repository root:
    python cause_model/export_for_colab.py
    python cause_model/export_for_colab.py --level detail
"""

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", message="The least populated class")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag.config import CAUSE_ALIASES  # noqa: E402

MIN_WORDS = 5
SEED = 42

ap = argparse.ArgumentParser()
ap.add_argument("--level", choices=["cause", "detail"], default="cause")
ap.add_argument("--min_train", type=int, default=20,
                help="detail level only: sub-causes with fewer training examples become '<CAUSE> - OTHER'")
args = ap.parse_args()


def clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = " ".join(str(v).split()).upper()
    return None if s in {"", "NAN", "NONE", "<NA>"} else s


df = pd.read_parquet(ROOT / "data/processed/phmsa_combined_with_severity.parquet").reset_index(drop=True)

# ---------- 1. the severity split, reproduced exactly ----------
text = df["NARRATIVE"].fillna("").astype(str).str.strip()
keep = text.str.split().str.len() >= MIN_WORDS
df, text = df[keep].reset_index(drop=True), text[keep].reset_index(drop=True)
print(f"Narratives with at least {MIN_WORDS} words: {len(df):,}")

rn = df["REPORT_NUMBER"]
missing = rn.isna() | rn.astype(str).str.strip().isin({"", "nan", "<NA>", "None"})
groups = np.where(missing, "row" + df.index.astype(str), df["commodity_type"] + "|" + rn.astype(str))
strata = (df["commodity_type"] + "|" + df["severity_weak_label"]).to_numpy()


def hold_out(idx, n_splits):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rest, held = next(sgkf.split(idx, strata[idx], groups[idx]))
    return idx[rest], idx[held]


trainval, test_idx = hold_out(np.arange(len(df)), 5)
train_idx, dev_idx = hold_out(trainval, 8)
splits = {"train": train_idx, "dev": dev_idx, "test": test_idx}

sev_path = ROOT / "data/processed/severity_colab_export.json"
if sev_path.exists():
    sev = json.load(open(sev_path, encoding="utf-8"))
    for name, idx in splits.items():
        same = sorted(text[i] for i in idx) == sorted(r["text"] for r in sev[name])
        print(f"  {name}: same narratives as the severity split: {'yes' if same else 'NO'}")
        if not same:
            sys.exit("Split differs from the severity export. Rebuild that export first, then rerun.")
else:
    print("  (severity_colab_export.json not found, so the split could not be cross-checked)")

# ---------- 2. labels ----------
cause = df["CAUSE"].map(clean).map(lambda c: CAUSE_ALIASES.get(c, c) if isinstance(c, str) else None)
if args.level == "cause":
    label = cause
else:
    detail = df["CAUSE_DETAILS"].map(clean)
    label = pd.Series([f"{c} - {d}" if isinstance(c, str) and isinstance(d, str) else None
                       for c, d in zip(cause, detail)], dtype=object)
    train_counts = Counter(label[i] for i in train_idx if isinstance(label[i], str))
    rare = {l for l, n in train_counts.items() if n < args.min_train}
    label = label.map(lambda l: (l.split(" - ")[0] + " - OTHER") if isinstance(l, str) and l in rare else l)
    print(f"\nSub-causes: {len(train_counts)} distinct in train; {len(rare)} with fewer than "
          f"{args.min_train} training examples merged into '<CAUSE> - OTHER'")

no_label = ~label.map(lambda l: isinstance(l, str))
print(f"\nRows without a {args.level} label (dropped after splitting): {int(no_label.sum())}")
if no_label.any():
    print(df.loc[no_label, "commodity_type"].value_counts().to_string())

LABELS = sorted(label.dropna().unique())
lab2id = {l: i for i, l in enumerate(LABELS)}

export = {"labels": LABELS, "level": args.level}
for name, idx in splits.items():
    export[name] = [{"text": text[i], "label": lab2id[label[i]], "commodity_type": df.loc[i, "commodity_type"],
                     "severity": df.loc[i, "severity_weak_label"]}
                    for i in idx if not no_label[i]]

out = ROOT / "data/processed" / ("cause_colab_export.json" if args.level == "cause"
                                 else "cause_detail_colab_export.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(export, f)

print(f"\nWrote {out.name}: {len(LABELS)} labels; " + ", ".join(f"{n} {len(export[n]):,}" for n in splits))
tab = pd.DataFrame([(n, LABELS[r["label"]]) for n in splits for r in export[n]], columns=["split", "label"])
print("\nExamples per label x split:")
print(pd.crosstab(tab["label"], tab["split"])[["train", "dev", "test"]].to_string())
test_tab = pd.DataFrame([(r["commodity_type"], LABELS[r["label"]]) for r in export["test"]],
                        columns=["commodity", "label"])
print("\nTest examples per commodity x label:")
print(pd.crosstab(test_tab["label"], test_tab["commodity"]).to_string())