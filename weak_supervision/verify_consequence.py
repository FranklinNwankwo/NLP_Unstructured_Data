import re
import pandas as pd
from collections import Counter

r = pd.read_parquet("data/processed/weak_labeled_narratives.parquet")
rows = [(row["narrative"], e) for _, row in r.iterrows() for e in row["entities"]]

for label in ("CONSEQUENCE", "ACTION_TAKEN"):
    c = Counter(e["text"].upper() for _, e in rows if e["label"] == label)
    print(f"\n{label}: {sum(c.values()):,} spans, top 15:")
    for text, n in c.most_common(15):
        print(f"  {n:>6}  {text}")

# Any FIRE consequence still followed by a fire-service word?
svc = re.compile(r"^[\s\-/]*(DEPT|DEPARTMENT|FIGHTER|MARSHAL|ALARM|CREW|TRUCK|CHIEF|POLICE|EMS)", re.I)
leaks = [n[max(0, e["start"] - 20):e["end"] + 30]
         for n, e in rows
         if e["label"] == "CONSEQUENCE" and e["text"].upper() == "FIRE"
         and svc.match(n[e["end"]:e["end"] + 15])]
print(f"\nFire-service leaks remaining: {len(leaks)} {leaks[:5]}")