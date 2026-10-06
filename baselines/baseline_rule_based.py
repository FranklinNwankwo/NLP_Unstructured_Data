import sys
from pathlib import Path

# Add project root and folders to sys.path relative to this file's location
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "weak_supervision"))
sys.path.insert(0, str(root_dir / "annotation"))

import pandas as pd
from seqeval.metrics import classification_report
from build_pipeline import build_pipeline
from bio_utils import spans_to_bio, resolve_overlaps

test_df = pd.read_parquet("data/processed/ner_test.parquet")
nlp = build_pipeline()

y_true, y_pred = [], []
for commodity in test_df["commodity_type"].unique():
    print(f"\n{'='*60}\n{commodity.upper()}\n{'='*60}")
    sub = test_df[test_df["commodity_type"] == commodity]

    c_true, c_pred = [], []
    for _, row in sub.iterrows():
        gold_entities = resolve_overlaps(row["entities"])
        gold_tokens, gold_tags = spans_to_bio(row["sentence_text"], row["entities"])

        doc = nlp(row["sentence_text"])
        pred_entities = [
            {"start": e.start_char, "end": e.end_char, "label": e.label_}
            for e in doc.ents
        ]
        _, pred_tags = spans_to_bio(row["sentence_text"], pred_entities)

        c_true.append(gold_tags)
        c_pred.append(pred_tags)

    print(classification_report(c_true, c_pred, digits=3))
    y_true.extend(c_true)
    y_pred.extend(c_pred)

print(f"\n{'='*60}\nOVERALL (all commodities)\n{'='*60}")
print(classification_report(y_true, y_pred, digits=3))