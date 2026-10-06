"""
Builds the Phase 6 severity dataset for Colab.

Label = severity_weak_label from Phase 1 (decision A: unchanged). It comes from
PHMSA's FATALITY / INJURY / IGNITE / EXPLODE flags, not from the text, so it
says nothing about spill size or environmental damage.

Split 70/10/20 (train/dev/test), stratified by commodity x severity, with every
row of the same PHMSA report kept in the same split so near-duplicate
supplemental narratives can't leak from train into test.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# rare classes (e.g. 10 liquid "critical" reports) trigger a harmless sklearn warning
warnings.filterwarnings("ignore", message="The least populated class")

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["minor", "moderate", "severe", "critical"]
MIN_WORDS = 5          # narratives shorter than this carry no usable signal
SEED = 42

df = pd.read_parquet(ROOT / "data/processed/phmsa_combined_with_severity.parquet")
df = df.reset_index(drop=True)

text = df["NARRATIVE"].fillna("").astype(str).str.strip()
keep = text.str.split().str.len() >= MIN_WORDS
print(f"Rows: {len(df):,}. Dropped {int((~keep).sum())} with fewer than {MIN_WORDS} words:")
print(df.loc[~keep, "commodity_type"].value_counts().to_string())
df, text = df[keep].reset_index(drop=True), text[keep].reset_index(drop=True)

# one group per PHMSA report, so supplemental versions stay together
if "REPORT_NUMBER" in df.columns:
    rn = df["REPORT_NUMBER"]
    missing = rn.isna() | rn.astype(str).str.strip().isin({"", "nan", "<NA>", "None"})
    groups = np.where(missing, "row" + df.index.astype(str),
                      df["commodity_type"] + "|" + rn.astype(str))
    shared = int(pd.Series(groups).duplicated(keep=False).sum())
    print(f"\nRows that share a REPORT_NUMBER with another row: {shared}")
else:
    groups = "row" + df.index.astype(str)
    print("\nWARNING: no REPORT_NUMBER column, splitting row by row")

y = df["severity_weak_label"].map({l: i for i, l in enumerate(LABELS)})
assert y.notna().all(), "unexpected severity label"
y = y.astype(int).to_numpy()
strata = (df["commodity_type"] + "|" + df["severity_weak_label"]).to_numpy()


def hold_out(idx, n_splits):
    """Hold out ~1/n_splits of idx, stratified, group-aware."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rest, held = next(sgkf.split(idx, strata[idx], groups[idx]))
    return idx[rest], idx[held]


all_idx = np.arange(len(df))
trainval, test_idx = hold_out(all_idx, 5)      # 20% test
train_idx, dev_idx = hold_out(trainval, 8)     # 10% of total as dev
splits = {"train": train_idx, "dev": dev_idx, "test": test_idx}

for a in splits:
    for b in splits:
        if a < b:
            assert not (set(groups[splits[a]]) & set(groups[splits[b]])), f"group leak {a}/{b}"

export = {"labels": LABELS}
for name, idx in splits.items():
    export[name] = [{"text": text[i], "label": int(y[i]), "commodity_type": df.loc[i, "commodity_type"]}
                    for i in idx]

out = ROOT / "data/processed/severity_colab_export.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(export, f)

print(f"\nWrote {out.name}: " + ", ".join(f"{n} {len(v):,}" for n, v in splits.items()))
tab = pd.DataFrame({
    "split": np.concatenate([[n] * len(v) for n, v in splits.items()]),
    "commodity": np.concatenate([df["commodity_type"].to_numpy()[v] for v in splits.values()]),
    "severity": np.concatenate([df["severity_weak_label"].to_numpy()[v] for v in splits.values()]),
})
print("\nExamples per split x severity:")
print(pd.crosstab(tab["split"], tab["severity"])[LABELS].to_string())
print("\nTest examples per commodity x severity:")
print(pd.crosstab(tab[tab.split == "test"]["commodity"], tab[tab.split == "test"]["severity"])
      .reindex(columns=LABELS, fill_value=0).to_string())