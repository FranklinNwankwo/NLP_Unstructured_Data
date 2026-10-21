"""
Blind human re-judgment of the 90 sampled relations in relation_precision_sheet.csv.

The original sheet's `correct` / `error_type` columns were drafted by an AI assistant.
This script lets a human judge the same 90 rows WITHOUT seeing those answers, then
reports human precision and how often the human and the AI agreed.

Step 1: make the blind sheet (shuffled, AI answers removed):
    python relation_extraction/human_review.py make

Step 2: open relation_extraction/relation_human_review.csv in Excel, fill in
        `correct` (Y or N) and, for every N, `error_type` (NER or RULE).
        Save it as "CSV UTF-8 (Comma delimited)", keeping the same name.

Step 3: score it:
    python relation_extraction/human_review.py score

The original AI-judged sheet is never modified.

Rubric (same as the original check):
  correct = Y  the sentence supports the relation between those two spans, read as
               the `means` column says. A slightly truncated or over-long span still
               counts as Y when the idea is right.
  correct = N  otherwise. Then pick ONE error_type:
     NER   a span itself is wrong: not an entity of that type, or a negated mention
           that should not exist ("no injuries")
     RULE  both spans are fine, but the sentence does not link them that way
           (boilerplate, reversed cause and effect, an investigative action read as
           a fix, the wrong nearby span picked)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AI_SHEET = ROOT / "relation_extraction" / "relation_precision_sheet.csv"
BLIND_SHEET = ROOT / "relation_extraction" / "relation_human_review.csv"
DISAGREE_OUT = ROOT / "relation_extraction" / "relation_review_disagreements.csv"

MEANS = {
    "CAUSED_BY": "HEAD happened because of TAIL",
    "RESULTED_IN": "HEAD led to TAIL",
    "REMEDIATED_BY": "HEAD was fixed, stopped or contained by TAIL",
    "LOCATED_AT": "HEAD is located at TAIL",
    "HAS_QUANTITY": "TAIL is a measurement of HEAD",
    "MADE_OF": "HEAD is made of TAIL",
}


def load_ai():
    df = pd.read_csv(AI_SHEET, encoding="utf-8-sig", keep_default_na=False)
    df["correct"] = df["correct"].astype(str).str.strip().str.upper()
    df["error_type"] = df["error_type"].astype(str).str.strip().str.upper()
    df["row_id"] = range(1, len(df) + 1)
    return df


def make():
    if BLIND_SHEET.exists():
        sys.exit(f"{BLIND_SHEET.name} already exists. Delete it first if you really want "
                 f"a fresh blank sheet (your judgments in it would be lost).")
    ai = load_ai()
    blind = ai.sample(frac=1, random_state=7).reset_index(drop=True)
    blind["means"] = blind["relation"].map(MEANS)
    blind["correct"] = ""
    blind["error_type"] = ""
    blind["notes"] = ""
    cols = ["row_id", "relation", "means", "head", "head_label", "tail", "tail_label",
            "sentence", "correct", "error_type", "notes"]
    blind[cols].to_csv(BLIND_SHEET, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(blind)} rows to {BLIND_SHEET}")
    print("Rows are shuffled and the AI's answers are removed. Fill in `correct` (Y/N) and,")
    print("for each N, `error_type` (NER/RULE). Save as CSV UTF-8, then run: ... human_review.py score")


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (c - m) / d, (c + m) / d


def cohen_kappa(a, b):
    a, b = list(a), list(b)
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def score():
    if not BLIND_SHEET.exists():
        sys.exit("No review sheet yet. Run: python relation_extraction/human_review.py make")
    human = pd.read_csv(BLIND_SHEET, encoding="utf-8-sig", keep_default_na=False)
    human["correct"] = human["correct"].astype(str).str.strip().str.upper()
    human["error_type"] = human["error_type"].astype(str).str.strip().str.upper()

    bad = human[~human["correct"].isin({"Y", "N"})]
    if len(bad):
        print(f"{len(bad)} rows are not marked Y or N yet (row_id: "
              f"{', '.join(map(str, bad['row_id'].head(15)))}{' ...' if len(bad) > 15 else ''}).")
        sys.exit(1)
    missing_type = human[(human["correct"] == "N") & (~human["error_type"].isin({"NER", "RULE"}))]
    if len(missing_type):
        print(f"Note: {len(missing_type)} N rows have no error_type (NER or RULE): row_id "
              f"{', '.join(map(str, missing_type['row_id']))}")

    ai = load_ai()[["row_id", "correct", "error_type"]].rename(
        columns={"correct": "ai_correct", "error_type": "ai_error_type"})
    m = human.merge(ai, on="row_id", how="inner")
    if len(m) != len(ai):
        sys.exit(f"Matched {len(m)} of {len(ai)} rows; do not edit or delete the row_id column.")
    m["ok"] = m["correct"] == "Y"
    m["ai_ok"] = m["ai_correct"] == "Y"

    print("\nHUMAN-JUDGED PRECISION")
    print(f"{'RELATION':<16}{'N':>4}{'CORRECT':>9}{'PRECISION':>11}   95% interval   "
          f"errors (NER / RULE)   AI precision")
    for name, g in m.groupby("relation"):
        k, n = int(g["ok"].sum()), len(g)
        lo, hi = wilson(k, n)
        wrong = g[~g["ok"]]
        print(f"{name:<16}{n:>4}{k:>9}{k / n:>11.0%}   {lo:>4.0%} - {hi:<4.0%}      "
              f"{int((wrong['error_type'] == 'NER').sum()):>2} / {int((wrong['error_type'] == 'RULE').sum()):<2}"
              f"              {int(g['ai_ok'].sum())}/{n}")
    k, n = int(m["ok"].sum()), len(m)
    lo, hi = wilson(k, n)
    wrong = m[~m["ok"]]
    print(f"\nOverall (human): {k}/{n} = {k / n:.0%}  (95% interval {lo:.0%} - {hi:.0%}); "
          f"errors NER {int((wrong['error_type'] == 'NER').sum())} / "
          f"RULE {int((wrong['error_type'] == 'RULE').sum())}")
    print(f"Overall (AI):    {int(m['ai_ok'].sum())}/{n} = {m['ai_ok'].mean():.0%}")

    agree = (m["correct"] == m["ai_correct"])
    kappa = cohen_kappa(m["correct"], m["ai_correct"])
    print(f"\nHUMAN vs AI AGREEMENT on Y/N: {int(agree.sum())}/{n} = {agree.mean():.0%}, "
          f"Cohen's kappa {kappa:.2f}")
    both_n = m[(m["correct"] == "N") & (m["ai_correct"] == "N")]
    if len(both_n):
        same_type = (both_n["error_type"] == both_n["ai_error_type"]).sum()
        print(f"Error type agreement where both said N: {same_type}/{len(both_n)}")

    dis = m[~agree].copy()
    if len(dis):
        dis = dis[["row_id", "relation", "head", "tail", "sentence", "correct", "error_type",
                   "ai_correct", "ai_error_type", "notes"]].rename(
            columns={"correct": "human_correct", "error_type": "human_error_type"})
        dis.to_csv(DISAGREE_OUT, index=False, encoding="utf-8-sig")
        print(f"\n{len(dis)} disagreements written to {DISAGREE_OUT}")
        for _, r in dis.iterrows():
            print(f"  #{r['row_id']:<3} {r['relation']:<14} {r['head']} -> {r['tail']}: "
                  f"human {r['human_correct']}, AI {r['ai_correct']}")
    else:
        print("\nNo disagreements.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["make", "score"])
    args = ap.parse_args()
    make() if args.step == "make" else score()