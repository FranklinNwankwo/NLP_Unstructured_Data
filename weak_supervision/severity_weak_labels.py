"""
Weak-labels severity directly from PHMSA's own structured fields — no NER
involved. This is the "nearly free ground truth" source flagged in the
original dev plan (Phase 6 preview), built here in Phase 1 since it doesn't
depend on the entity pipeline at all and can be validated independently.

Severity classes, from least to most severe:
  near_miss    - no release-related indicators fired at all
  minor        - release occurred, no fire/explosion/injury/fatality
  moderate     - fire OR explosion, but no injury/fatality
  severe       - injury occurred (with or without fire/explosion)
  critical     - fatality occurred
"""

import pandas as pd

df = pd.read_parquet("data/processed/phmsa_combined_raw.parquet")


def classify_severity(row):
    # FATALITY_IND / INJURY_IND / IGNITE_IND / EXPLODE_IND confirmed present
    # across all three commodities (column list, all three field_notes.md
    # sections — these are core PHMSA fields common to every commodity type)
    fatality = str(row.get("FATALITY_IND", "")).strip().upper() == "YES"
    injury = str(row.get("INJURY_IND", "")).strip().upper() == "YES"
    ignite = str(row.get("IGNITE_IND", "")).strip().upper() == "YES"
    explode = str(row.get("EXPLODE_IND", "")).strip().upper() == "YES"

    if fatality:
        return "critical"
    if injury:
        return "severe"
    if ignite or explode:
        return "moderate"
    return "minor"  # release occurred (every row here IS an incident report)


df["severity_weak_label"] = df.apply(classify_severity, axis=1)

print("Severity distribution, overall:")
print(df["severity_weak_label"].value_counts())

print("\nSeverity distribution, per commodity:")
print(pd.crosstab(df["commodity_type"], df["severity_weak_label"]))

# Sanity check: confirm the source columns actually have real values, not
# all-null (which would silently produce an all-"minor" label column)
print("\nSource column value check (should NOT be all one value):")
for col in ["FATALITY_IND", "INJURY_IND", "IGNITE_IND", "EXPLODE_IND"]:
    if col in df.columns:
        print(f"  {col}: {df[col].value_counts(dropna=False).to_dict()}")
    else:
        print(f"  {col}: COLUMN NOT FOUND")

df.to_parquet("data/processed/phmsa_combined_with_severity.parquet", index=False)
print("\nSaved to data/processed/phmsa_combined_with_severity.parquet")