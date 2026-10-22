"""
Re-scores the downloaded cause model locally on the held-out test split. Should match
the test cell of notebooks/Cause Classifier.ipynb. Unzip the model from Colab into
cause_model/phmsa_cause_model (or cause_model/phmsa_cause_detail_model for sub-causes) first.

    python cause_model/check_cause_local.py
    python cause_model/check_cause_local.py --level detail   # sub-causes, plus a head-to-head with
                                                             # the 8-category model at parent level
    python cause_model/check_cause_local.py --n 50           # quick smoke test
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
ap.add_argument("--model-dir", help="default: cause_model/phmsa_cause_model, or phmsa_cause_detail_model for --level detail")
args = ap.parse_args()

path = ROOT / "data/processed" / ("cause_colab_export.json" if args.level == "cause"
                                  else "cause_detail_colab_export.json")
data = json.load(open(path, encoding="utf-8"))
LABELS = data["labels"]
ALL = list(range(len(LABELS)))
test = data["test"][:args.n] if args.n else data["test"]

model_dir = Path(args.model_dir) if args.model_dir else ROOT / "cause_model" / (
    "phmsa_cause_model" if args.level == "cause" else "phmsa_cause_detail_model")
if not (model_dir / "config.json").exists():
    sys.exit(f"No model in {model_dir}")
predictor = CausePredictor(model_dir)
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


def macro(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=sorted(set(y_true)), average="macro", zero_division=0)


if args.level == "detail":
    # Sub-causes rolled up to their parent category, against the 8-category model on the SAME narratives.
    parent = lambda label: label.split(" - ")[0]
    pt = np.array([parent(LABELS[i]) for i in y_true])
    pp = np.array([parent(p["label"]) for p in preds])
    print(f"\nRolled up to the 8 parent causes: macro-F1 {macro(pt, pp):.3f}   accuracy {np.mean(pt == pp):.3f}")
    print(f"Sub-cause wrong but parent right: {np.mean((y_true != y_pred) & (pt == pp)):.1%} of test narratives")
    base_dir = ROOT / "cause_model/phmsa_cause_model"
    if (base_dir / "config.json").exists():
        t0 = time.time()
        base = [p["label"] for p in CausePredictor(base_dir).predict([r["text"] for r in test])]
        bp = np.array(base)
        print(f"8-category model on the same {len(test):,} narratives: macro-F1 {macro(pt, bp):.3f}   "
              f"accuracy {np.mean(pt == bp):.3f}   ({time.time() - t0:.0f}s)")
        both = np.mean((pp == pt) & (bp == pt))
        only_detail, only_base = np.mean((pp == pt) & (bp != pt)), np.mean((pp != pt) & (bp == pt))
        print(f"Both right {both:.1%}; only the sub-cause model right {only_detail:.1%}; "
              f"only the 8-category model right {only_base:.1%}")
    else:
        print("(cause_model/phmsa_cause_model not found: no head-to-head with the 8-category model)")
