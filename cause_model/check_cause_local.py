"""
Re-scores the downloaded cause model locally on the held-out test split. Should match
the test cell of notebooks/Cause Classifier.ipynb. Unzip the model from Colab into
cause_model/phmsa_cause_model first.

    python cause_model/check_cause_local.py
    python cause_model/check_cause_local.py --n 50      # quick smoke test
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cause_model"))

import numpy as np
from sklearn.metrics import classification_report, f1_score

from cause_predict import CausePredictor

ap = argparse.ArgumentParser()
ap.add_argument("--level", choices=["cause", "detail"], default="cause")
ap.add_argument("--n", type=int, default=0, help="score only the first N test narratives")
args = ap.parse_args()

path = ROOT / "data/processed" / ("cause_colab_export.json" if args.level == "cause"
                                  else "cause_detail_colab_export.json")
data = json.load(open(path, encoding="utf-8"))
LABELS = data["labels"]
ALL = list(range(len(LABELS)))
test = data["test"][:args.n] if args.n else data["test"]

predictor = CausePredictor()
model_labels = [predictor.id2label[i] for i in range(len(predictor.id2label))]
if model_labels != LABELS:
    sys.exit("The model's labels do not match the export's labels: wrong model folder or wrong --level?")

t0 = time.time()
preds = predictor.predict([r["text"] for r in test])
print(f"scored {len(test):,} narratives in {time.time() - t0:.0f}s")

y_true = np.array([r["label"] for r in test])
y_pred = np.array([LABELS.index(p["label"]) for p in preds])
present = sorted(set(y_true))
print(classification_report(y_true, y_pred, labels=ALL, target_names=LABELS, digits=3, zero_division=0))
print(f"Overall macro-F1 {f1_score(y_true, y_pred, labels=present, average='macro', zero_division=0):.3f}"
      f"   accuracy {np.mean(y_true == y_pred):.3f}")