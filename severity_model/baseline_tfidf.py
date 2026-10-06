"""
Phase 6 baselines: always-"minor" and TF-IDF + logistic regression.
C is picked on dev; the test set is scored once.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
data = json.load(open(ROOT / "data/processed/severity_colab_export.json", encoding="utf-8"))
LABELS = data["labels"]
ALL = list(range(len(LABELS)))


def unpack(split):
    rows = data[split]
    return ([r["text"] for r in rows], np.array([r["label"] for r in rows]),
            np.array([r["commodity_type"] for r in rows]))


Xtr_t, ytr, _ = unpack("train")
Xdv_t, ydv, _ = unpack("dev")
Xte_t, yte, cte = unpack("test")


def macro_present(y_true, y_pred):
    """Macro-F1 over classes that actually occur in y_true."""
    present = sorted(set(y_true))
    return f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)


always_minor = np.zeros_like(yte)
print(f"Always-minor baseline: accuracy {np.mean(always_minor == yte):.3f}, "
      f"macro-F1 {macro_present(yte, always_minor):.3f}  (why accuracy alone is useless here)\n")

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
Xtr = vec.fit_transform(Xtr_t)
Xdv, Xte = vec.transform(Xdv_t), vec.transform(Xte_t)

best = None
for C in (0.3, 1, 3, 10, 30):
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xtr, ytr)
    f = macro_present(ydv, clf.predict(Xdv))
    print(f"C={C:<5} dev macro-F1 {f:.3f}")
    if best is None or f > best[0]:
        best = (f, C, clf)

_, C, clf = best
pred = clf.predict(Xte)
print(f"\nBest C = {C}. TEST (scored once):\n")
print(classification_report(yte, pred, labels=ALL, target_names=LABELS, digits=3, zero_division=0))
print(f"Overall macro-F1 {macro_present(yte, pred):.3f}   accuracy {np.mean(pred == yte):.3f}")
print("\nConfusion matrix (rows = true, columns = predicted):")
print(pd.DataFrame(confusion_matrix(yte, pred, labels=ALL), index=LABELS, columns=LABELS).to_string())

print("\nPer commodity (macro-F1 over classes present; small classes are very noisy):")
for c in sorted(set(cte)):
    m = cte == c
    support = {LABELS[k]: int((yte[m] == k).sum()) for k in ALL}
    print(f"  {c:<28} n={int(m.sum()):<5} macro-F1 {macro_present(yte[m], pred[m]):.3f}   support {support}")