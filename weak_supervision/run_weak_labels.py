"""
Runs the assembled pipeline (build_pipeline.py) over the full combined
PHMSA parquet, producing weak-labeled entity spans per narrative.
"""

import pandas as pd
from build_pipeline import build_pipeline

nlp = build_pipeline()

df = pd.read_parquet("data/processed/phmsa_combined_raw.parquet")
print(f"\nProcessing {len(df):,} narratives across "
      f"{df['commodity_type'].nunique()} commodity types...")

results = []
# nlp.pipe() batches processing — much faster than looping doc = nlp(text)
for idx, doc in enumerate(nlp.pipe(df["NARRATIVE"].fillna("").astype(str), batch_size=100)):
    row = df.iloc[idx]
    entities = [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
            "unit_type": ent._.unit_type if ent.label_ == "QUANTITY" else None,
        }
        for ent in doc.ents
    ]
    results.append({
        "report_index": idx,
        "commodity_type": row["commodity_type"],
        "narrative": row["NARRATIVE"],
        "entities": entities,
        "num_entities": len(entities),
    })
    if (idx + 1) % 1000 == 0:
        print(f"  ...processed {idx + 1:,}")

results_df = pd.DataFrame(results)
results_df.to_parquet("data/processed/weak_labeled_narratives.parquet", index=False)

print(f"\nDone. Saved to data/processed/weak_labeled_narratives.parquet")
print(f"\nEntity count summary by commodity:")
print(results_df.groupby("commodity_type")["num_entities"].agg(["mean", "median", "min", "max"]).round(1))

zero_entity = (results_df["num_entities"] == 0).sum()
print(f"\nNarratives with ZERO entities tagged: {zero_entity:,} "
      f"({zero_entity/len(results_df)*100:.1f}%)")