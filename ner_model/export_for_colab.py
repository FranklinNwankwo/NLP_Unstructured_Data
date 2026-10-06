"""
Exports train/dev/test splits as one JSON for Colab, using the same
resolve_overlaps + spans_to_bio logic already verified against both
baselines, so Phase 4 trains on exactly the same gold labels they were
scored against.
"""

import sys
sys.path.insert(0, "annotation")

import json
import pandas as pd
from bio_utils import spans_to_bio, resolve_overlaps

LABELS = [
    "EQUIPMENT", "FAILURE_MODE", "CAUSE_FACTOR", "ACTION_TAKEN",
    "CONSEQUENCE", "QUANTITY", "LOCATION", "DATE_TIME", "MATERIAL_SPEC",
    "INSPECTION_FINDING", "PARTY_ROLE", "REGULATORY_REF",
]
TAG_LIST = ["O"] + [f"{p}-{l}" for l in LABELS for p in ("B", "I")]
TAG2ID = {t: i for i, t in enumerate(TAG_LIST)}

# train/dev split must match convert_to_spacy.py's split exactly, so the
# spaCy baseline and the transformer train on identical data
train_df = pd.read_parquet("data/processed/ner_train.parquet")
test_df = pd.read_parquet("data/processed/ner_test.parquet")

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
n_dev = int(len(train_df) * 0.15)
dev_df, fit_df = train_df.iloc[:n_dev], train_df.iloc[n_dev:]


def to_records(df):
    records = []
    for _, row in df.iterrows():
        entities = resolve_overlaps(row["entities"])
        tokens, tags = spans_to_bio(row["sentence_text"], entities)
        records.append({
            "tokens": tokens,
            "ner_tags": [TAG2ID[t] for t in tags],
            "commodity_type": row["commodity_type"],
        })
    return records


export = {
    "label_list": TAG_LIST,
    "train": to_records(fit_df),
    "dev": to_records(dev_df),
    "test": to_records(test_df),
}

with open("data/processed/ner_colab_export.json", "w", encoding="utf-8") as f:
    json.dump(export, f)

print(f"train: {len(export['train'])}  dev: {len(export['dev'])}  test: {len(export['test'])}")
print(f"label count: {len(TAG_LIST)} (O + {len(LABELS)} types x B/I)")