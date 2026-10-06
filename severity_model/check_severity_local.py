"""Re-scores the downloaded severity model locally on the held-out test split. Should match Colab's Cell 5."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "severity_model"))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from severity_predict import SeverityPredictor

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=0, help="score only the first N test narratives (smoke test)")
args = ap.parse_args()

data = json.load(open(ROOT / "data/processed/severity_colab_export.json", encoding="utf-8"))
LABELS = data["labels"]
ALL = list(range(len(LABELS)))
test = data["test"][:args.n] if args.n else data["test"]

predictor = SeverityPredictor()
t0 = time.time()
preds = predictor.predict([r["text"] for r in test])
print(f"scored {len(test):,} narratives in {time.time() - t0:.0f}s")

y_true = np.array([r["label"] for r in test])
y_pred = np.array([LABELS.index(p["label"]) for p in preds])
present = sorted(set(y_true))

print(classification_report(y_true, y_pred, labels=ALL, target_names=LABELS, digits=3, zero_division=0))
print(f"Overall macro-F1 {f1_score(y_true, y_pred, labels=present, average='macro', zero_division=0):.3f}"
      f"   accuracy {np.mean(y_true == y_pred):.3f}")
print("\nConfusion matrix (rows = true, columns = predicted):")
print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=ALL), index=LABELS, columns=LABELS).to_string())
if not args.n:
    print("\nColab reported: macro-F1 0.777, accuracy 0.956, correct per class 1616 / 174 / 31 / 16")