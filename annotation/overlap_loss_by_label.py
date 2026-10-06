import sys
sys.path.insert(0, "annotation")
import pandas as pd
from collections import Counter
from bio_utils import resolve_overlaps

df = pd.read_parquet("data/processed/annotated_sentences.parquet")
before, after = Counter(), Counter()
for _, row in df.iterrows():
    for e in row["entities"]:
        before[e["label"]] += 1
    for e in resolve_overlaps(row["entities"]):
        after[e["label"]] += 1

print(f"{'LABEL':<20}{'BEFORE':>8}{'AFTER':>8}{'DROPPED':>9}{'SHARE':>8}")
for label, n in before.most_common():
    d = n - after[label]
    print(f"{label:<20}{n:>8}{after[label]:>8}{d:>9}{d / n:>8.0%}")