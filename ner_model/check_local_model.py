"""Re-scores the downloaded model locally on the held-out test set. Should match Colab's Cell 6."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "annotation"))
sys.path.insert(0, str(ROOT / "ner_model"))

import pandas as pd
from seqeval.metrics import classification_report, f1_score

from bio_utils import spans_to_bio, resolve_overlaps
from predict import NERPredictor

test_df = pd.read_parquet(ROOT / "data/processed/ner_test.parquet")
predictor = NERPredictor()
preds = predictor.predict(test_df["sentence_text"].tolist())

y_true, y_pred = [], []
for (_, row), ents in zip(test_df.iterrows(), preds):
    _, gold = spans_to_bio(row["sentence_text"], resolve_overlaps(row["entities"]))
    _, pred = spans_to_bio(row["sentence_text"], ents)
    y_true.append(gold)
    y_pred.append(pred)

print(classification_report(y_true, y_pred, digits=3))
print(f"overall micro-F1: {f1_score(y_true, y_pred):.3f}   (Colab reported 0.608)")