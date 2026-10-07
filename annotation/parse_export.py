"""
Parses the Label Studio JSON export into one row per annotated sentence,
with entities as (text, label, start, end) tuples using character offsets
relative to the sentence text.
"""

import json
import pandas as pd

with open("annotation/label_studio_export.json", encoding="utf-8") as f:
    raw = json.load(f)

rows = []
skipped_unannotated = 0

for task in raw:
    data = task["data"]
    annotations = task.get("annotations", [])
    if not annotations:
        skipped_unannotated += 1
        continue

    # Using the first completed annotation (there is exactly one,
    # since this was a single-annotator pass)
    result = annotations[0]["result"]
    entities = []
    for r in result:
        if r.get("type") != "labels":
            continue
        v = r["value"]
        entities.append({
            "text": v["text"],
            "label": v["labels"][0],
            "start": v["start"],
            "end": v["end"],
        })

    rows.append({
        "report_index": data["report_index"],
        "commodity_type": data["commodity_type"],
        "severity_weak_label": data["severity_weak_label"],
        "sentence_text": data["text"],
        "entities": entities,
        "num_entities": len(entities),
    })

df = pd.DataFrame(rows)
df.to_parquet("data/processed/annotated_sentences.parquet", index=False)

print(f"Parsed {len(df):,} annotated sentences ({skipped_unannotated} tasks had no annotation, skipped)")
print(f"\nBy commodity:")
print(df["commodity_type"].value_counts())
print(f"\nEntity count stats:")
print(df["num_entities"].describe())

label_counts = pd.Series(
    [e["label"] for ents in df["entities"] for e in ents]
).value_counts()
print(f"\nLabel distribution across all annotated spans:")
print(label_counts)