import json
import pandas as pd

df = pd.read_parquet("data/processed/annotation_sample.parquet")

tasks = []
for _, row in df.iterrows():
    results = [
        {
            "from_name": "label", "to_name": "text", "type": "labels",
            "value": {"start": e["start"], "end": e["end"],
                      "text": e["text"], "labels": [e["label"]]},
        }
        for e in row["entities"]
    ]
    tasks.append({
        "data": {
            "text": row["sentence_text"],
            "commodity_type": row["commodity_type"],
            "severity_weak_label": row["severity_weak_label"],
            "report_index": int(row["report_index"]),
        },
        "predictions": [{"model_version": "weak_supervision_v1", "result": results}],
    })

with open("annotation/label_studio_import.json", "w", encoding="utf-8") as f:
    json.dump(tasks, f, indent=2)

print(f"Wrote {len(tasks):,} tasks to annotation/label_studio_import.json")