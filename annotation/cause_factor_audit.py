import sys
sys.path.insert(0, "annotation")
import pandas as pd
from bio_utils import resolve_overlaps

df = pd.read_parquet("data/processed/annotated_sentences.parquet")

rows = []
for _, row in df.iterrows():
    for e in resolve_overlaps(row["entities"]):
        rows.append({"label": e["label"], "tokens": len(e["text"].split()),
                     "text": e["text"].strip(), "sentence": row["sentence_text"]})
spans = pd.DataFrame(rows)

print("Span length in words, by label:")
print(spans.groupby("label")["tokens"].agg(["count", "mean", "median", "max"]).round(1)
      .sort_values("mean", ascending=False))

cf = spans[spans["label"] == "CAUSE_FACTOR"].sample(40, random_state=1)
print("\n40 random CAUSE_FACTOR spans:")
for _, r in cf.iterrows():
    print(f"  [{r['tokens']:>2}w] {r['text']}")