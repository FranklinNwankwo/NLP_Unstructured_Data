import re
import pandas as pd
from collections import Counter

r = pd.read_parquet("data/processed/weak_labeled_narratives.parquet")
neg = re.compile(r"\b(no|not|without|never|none)\b(\s+[\w'-]+){0,4}\s*$", re.I)

for label in ("EQUIPMENT", "FAILURE_MODE"):
    total, negated = Counter(), Counter()
    for _, row in r.iterrows():
        text = row["narrative"]
        for e in row["entities"]:
            if e["label"] != label:
                continue
            term = e["text"].upper()
            total[term] += 1
            if neg.search(text[max(0, e["start"] - 30):e["start"]]):
                negated[term] += 1

    all_total, all_neg = sum(total.values()), sum(negated.values())
    print(f"\n{'='*60}\n{label}: {all_total:,} spans, {all_neg:,} negated overall ({all_neg/all_total:.1%})\n{'='*60}")
    print(f"{'TERM':<20}{'SPANS':>8}{'NEGATED':>9}{'SHARE':>8}")
    flagged = 0
    for term, n in total.most_common(30):
        share = negated[term] / n
        marker = "  <-- check" if share >= 0.20 and n >= 10 else ""
        if marker:
            flagged += 1
        print(f"{term:<20}{n:>8}{negated[term]:>9}{share:>8.0%}{marker}")
    print(f"\n{flagged} term(s) at or above 20% negated (min 10 spans)")