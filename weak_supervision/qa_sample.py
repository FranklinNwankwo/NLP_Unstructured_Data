"""
Stratified QA sample: pulls a random set of narratives per commodity type
and prints the full tagged output (all entity types together) alongside
the narrative, for manual review. Same discipline as Phase 0's Step 11
manual read, now applied to the pipeline's output rather than raw text.
"""

import pandas as pd

results = pd.read_parquet("data/processed/weak_labeled_narratives.parquet")

N_PER_COMMODITY = 10

for commodity in results["commodity_type"].unique():
    sample = results[results["commodity_type"] == commodity].sample(
        N_PER_COMMODITY, random_state=42
    )
    print(f"\n{'#'*70}\n# {commodity.upper()}\n{'#'*70}")

    for _, row in sample.iterrows():
        print(f"\n{'-'*70}")
        print(row["narrative"][:500])
        print(f"\nTagged ({row['num_entities']} entities):")
        if len(row["entities"]) == 0:
            print("  (none)")
        for e in row["entities"]:
            extra = f" [{e['unit_type']}]" if e["unit_type"] else ""
            print(f"  [{e['label']}]{extra} '{e['text']}'")