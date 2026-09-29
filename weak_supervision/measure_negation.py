import re
import pandas as pd
from collections import Counter

r = pd.read_parquet("data/processed/weak_labeled_narratives.parquet")
neg = re.compile(r"\b(NO|NOT|WITHOUT|NEVER|NONE)\b(\s+[\w'-]+){0,2}\s*$", re.I)

total, negated = Counter(), Counter()
for _, row in r.iterrows():
    text = row["narrative"]
    for e in row["entities"]:
        if e["label"] != "CONSEQUENCE":
            continue
        term = e["text"].upper()
        total[term] += 1
        if neg.search(text[max(0, e["start"] - 25):e["start"]]):
            negated[term] += 1

print(f"{'TERM':<14}{'SPANS':>8}{'NEGATED':>9}{'SHARE':>8}")
for term, n in total.most_common(20):
    print(f"{term:<14}{n:>8}{negated[term]:>9}{negated[term] / n:>8.0%}")

all_total, all_neg = sum(total.values()), sum(negated.values())
print(f"\nOverall: {all_neg:,} of {all_total:,} CONSEQUENCE spans negated ({all_neg / all_total:.1%})")