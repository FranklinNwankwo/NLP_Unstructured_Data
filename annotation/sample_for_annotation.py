"""
Stratified annotation sample: ~350 sentences per commodity, allocated across
severity so critical/severe incidents (rare in the raw data) get proportionally
more annotation attention than their natural frequency would give them.
"""

import pandas as pd

df = pd.read_parquet("data/processed/sentences_weak_labeled.parquet")

TARGET_PER_COMMODITY = 350
SEVERITY_SHARE = {"critical": 0.20, "severe": 0.20, "moderate": 0.20, "minor": 0.35}
ZERO_ENTITY_SHARE = 0.05  # explicit negative examples, not a byproduct
RANDOM_STATE = 42

samples = []
for commodity in df["commodity_type"].unique():
    sub = df[df["commodity_type"] == commodity]
    picked = []

    zero_ent_quota = int(TARGET_PER_COMMODITY * ZERO_ENTITY_SHARE)
    zero_pool = sub[sub["num_entities"] == 0]
    n = min(zero_ent_quota, len(zero_pool))
    if n > 0:
        picked.append(zero_pool.sample(n, random_state=RANDOM_STATE))

    non_zero = sub[sub["num_entities"] > 0]
    for severity, share in SEVERITY_SHARE.items():
        quota = int(TARGET_PER_COMMODITY * share)
        pool = non_zero[non_zero["severity_weak_label"] == severity]
        n = min(quota, len(pool))
        if n > 0:
            picked.append(pool.sample(n, random_state=RANDOM_STATE))

    combined = pd.concat(picked) if picked else pd.DataFrame(columns=sub.columns)
    shortfall = TARGET_PER_COMMODITY - len(combined)
    if shortfall > 0:
        remaining = non_zero[~non_zero.index.isin(combined.index)]
        top_up = remaining.sample(min(shortfall, len(remaining)), random_state=RANDOM_STATE)
        combined = pd.concat([combined, top_up])

    samples.append(combined)

annotation_sample = pd.concat(samples).reset_index(drop=True)

print("Sampled counts by commodity x severity:")
print(annotation_sample.groupby(["commodity_type", "severity_weak_label"]).size())
print(f"\nTotal sampled: {len(annotation_sample):,}")

annotation_sample.to_parquet("data/processed/annotation_sample.parquet", index=False)