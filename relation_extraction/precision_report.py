"""Scores the filled-in precision sheet: Y/N per relation type, with 95% Wilson intervals."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sheet = pd.read_csv(ROOT / "relation_extraction/relation_precision_sheet.csv",
                    encoding="utf-8-sig", keep_default_na=False)
sheet["correct"] = sheet["correct"].astype(str).str.strip().str.upper()
sheet["error_type"] = sheet["error_type"].astype(str).str.strip().str.upper()

unmarked = sheet[~sheet["correct"].isin({"Y", "N"})]
if len(unmarked):
    print(f"{len(unmarked)} rows are not marked Y or N yet. Fill them in and save as CSV UTF-8.")
    raise SystemExit(1)

sheet["ok"] = sheet["correct"] == "Y"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (c - m) / d, (c + m) / d


print(f"{'RELATION':<16}{'N':>4}{'CORRECT':>9}{'PRECISION':>11}   95% interval   errors (NER / RULE)")
for name, g in sheet.groupby("relation"):
    k, n = int(g["ok"].sum()), len(g)
    lo, hi = wilson(k, n)
    bad = g[~g["ok"]]
    print(f"{name:<16}{n:>4}{k:>9}{k / n:>11.0%}   {lo:>4.0%} - {hi:<4.0%}      "
          f"{int((bad['error_type'] == 'NER').sum())} / {int((bad['error_type'] == 'RULE').sum())}")

k, n = int(sheet["ok"].sum()), len(sheet)
lo, hi = wilson(k, n)
print(f"\nOverall: {k}/{n} = {k / n:.0%}  (95% interval {lo:.0%} - {hi:.0%})")
unlabeled = sheet[(~sheet["ok"]) & (~sheet["error_type"].isin({"NER", "RULE"}))]
if len(unlabeled):
    print(f"Note: {len(unlabeled)} incorrect rows have no error_type (NER or RULE).")