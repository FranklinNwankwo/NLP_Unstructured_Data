"""
Model regression test: re-scores the three deployed models (at the revision pinned in
mlops/model_manifest.json) on the frozen test sets and fails if any score drops more than
`tolerance` below the score recorded when the model was trained.

    python mlops/regression_test.py                       # download the pinned models from Hugging Face
    python mlops/regression_test.py --model-dir local     # the repo's own model folders (ner_model/phmsa_ner_model, ...)
    python mlops/regression_test.py --model-dir PATH      # use local folders PATH/ner, PATH/severity, PATH/cause
    python mlops/regression_test.py --limit 200           # smoke test on the first 200 narratives (no pass/fail)
    python mlops/regression_test.py --only ner severity

Checks, in order:
  1. the frozen eval files match the hashes in the manifest (the test set has not been edited)
  2. NER micro-F1 (seqeval, same BIO conversion as training), severity and cause macro-F1
     over classes present plus accuracy
  3. each score >= expected - tolerance
Writes mlops/results/regression_latest.json and, under GitHub Actions, a table in the job summary.
Exit code 1 on any failure.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DOCS_TEST, NER_TEST, ROOT, load_manifest, model_paths, read_jsonl,  # noqa: E402
                    sha256)


def macro_f1_present(y_true, y_pred):
    """Macro-F1 over the classes present in y_true (as in training), plus accuracy."""
    present = sorted(set(y_true))
    f1s = []
    for c in present:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
    return (sum(f1s) / len(f1s) if f1s else 0.0), acc


ECE_TOLERANCE = 0.01


def check(results, name, metric, got, expected, tol, higher_is_better=True):
    got = float(got)
    ok = bool(got >= expected - tol) if higher_is_better else bool(got <= expected + tol)
    results.append({"model": name, "metric": metric, "expected": expected, "got": round(got, 4),
                    "delta": round(got - expected, 4), "pass": ok})
    return ok


def expected_calibration_error(preds, gold):
    """Top-label ECE (15 bins) from predictor output, same definition as mlops/calibrate.py."""
    import numpy as np
    from calibrate import reliability
    labels = list(preds[0]["probabilities"])
    probs = np.array([[p["probabilities"][l] for l in labels] for p in preds])
    y = np.array([labels.index(g) for g in gold])
    return reliability(probs, y)[0]


def score_ner(predictor, rows):
    sys.path.insert(0, str(ROOT / "annotation"))
    from bio_utils import resolve_overlaps, spans_to_bio
    from seqeval.metrics import f1_score
    preds = predictor.predict([r["text"] for r in rows])
    y_true = [spans_to_bio(r["text"], resolve_overlaps(r["entities"]))[1] for r in rows]
    y_pred = [spans_to_bio(r["text"], p)[1] for r, p in zip(rows, preds)]
    return f1_score(y_true, y_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", help='"local" (the repo\'s model folders) or a folder with ner/, severity/, cause/ (skips the download)')
    ap.add_argument("--manifest", default=str(ROOT / "mlops/model_manifest.json"))
    ap.add_argument("--only", nargs="+", choices=["ner", "severity", "cause"])
    ap.add_argument("--limit", type=int, default=0, help="smoke test on the first N rows; scores are reported, not judged")
    ap.add_argument("--skip-hash", action="store_true", help="do not verify eval-set hashes (for synthetic tests)")
    args = ap.parse_args()

    m = load_manifest(args.manifest)
    which = args.only or ["ner", "severity", "cause"]
    tol = m.get("tolerance", 0.005)
    failures, results = [], []

    # 1. the test sets are the ones the expected scores were measured on
    for key, path in (("ner_test", NER_TEST), ("docs_test", DOCS_TEST)):
        if not path.exists():
            sys.exit(f"{path.relative_to(ROOT)} missing. Run: python mlops/freeze_eval_sets.py")
        want = m.get("eval_sets", {}).get(key, {}).get("sha256")
        if args.skip_hash:
            continue
        if not want:
            sys.exit(f"No hash for {key} in the manifest. Run: python mlops/freeze_eval_sets.py")
        if sha256(path) != want:
            failures.append(f"{path.name} has changed since it was frozen (sha256 mismatch)")

    # 2. models at the pinned revision
    t0 = time.time()
    paths = model_paths(args.model_dir, m)
    print(f"models: {args.model_dir or m['model_repo'] + '@' + m['revision'][:12]}  ({time.time() - t0:.0f}s)")
    from predict import NERPredictor
    from severity_predict import SeverityPredictor
    from cause_predict import CausePredictor

    ner_rows = read_jsonl(NER_TEST)
    docs = read_jsonl(DOCS_TEST)
    if args.limit:
        ner_rows, docs = ner_rows[:args.limit], docs[:args.limit]

    if "ner" in which:
        t0 = time.time()
        f1 = score_ner(NERPredictor(paths["ner"]), ner_rows)
        if not check(results, "ner", "micro_f1", f1, m["models"]["ner"]["expected"]["micro_f1"], tol):
            failures.append(f"NER micro-F1 {f1:.3f}")
        print(f"ner       micro-F1 {f1:.3f}   ({len(ner_rows)} sentences, {time.time() - t0:.0f}s)")

    for name, cls, field in (("severity", SeverityPredictor, "severity"), ("cause", CausePredictor, "cause")):
        if name not in which:
            continue
        rows = [d for d in docs if d.get(field)]
        t0 = time.time()
        preds = cls(paths[name]).predict([d["text"] for d in rows])
        mf1, acc = macro_f1_present([d[field] for d in rows], [p["label"] for p in preds])
        exp = m["models"][name]["expected"]
        ok1 = check(results, name, "macro_f1", mf1, exp["macro_f1"], tol)
        ok2 = check(results, name, "accuracy", acc, exp["accuracy"], tol)
        if not (ok1 and ok2):
            failures.append(f"{name} macro-F1 {mf1:.3f} / accuracy {acc:.3f}")
        ece_note = ""
        if "ece" in exp:                     # set by mlops/calibrate.py; lower is better
            ece = expected_calibration_error(preds, [d[field] for d in rows])
            if not check(results, name, "ece", ece, exp["ece"], ECE_TOLERANCE, higher_is_better=False):
                failures.append(f"{name} ECE {ece:.3f}")
            ece_note = f"   ECE {ece:.3f}"
        print(f"{name:<9} macro-F1 {mf1:.3f}   accuracy {acc:.3f}{ece_note}   "
              f"({len(rows)} narratives, {time.time() - t0:.0f}s)")

    # 3. report
    smoke = bool(args.limit)
    out = {"revision": m["revision"], "model_dir": args.model_dir, "limit": args.limit, "tolerance": tol,
           "results": results, "failures": [] if smoke else failures}
    res_dir = ROOT / "mlops/results"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / ("regression_smoke.json" if smoke else "regression_latest.json")).write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")

    lines = [f"### Model regression test ({'smoke, not judged' if smoke else 'full'})", "",
             f"`{m['model_repo']}@{m['revision'][:12]}`, tolerance {tol} (ECE: +{ECE_TOLERANCE})", "",
             "| Model | Metric | Expected | Got | Δ | |", "|---|---|---|---|---|---|"]
    for r in results:
        mark = "✅" if r["pass"] else ("⚠️" if smoke else "❌")
        lines.append(f"| {r['model']} | {r['metric']} | {r['expected']:.3f} | {r['got']:.3f} | {r['delta']:+.3f} | {mark} |")
    if failures and not smoke:
        lines += ["", "**Failed:** " + "; ".join(failures)]
    table = "\n".join(lines) + "\n"
    print("\n" + table)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(table)

    if smoke:
        print("smoke run: scores reported, not judged")
        return 0
    if failures:
        print("REGRESSION: " + "; ".join(failures))
        return 1
    print("all models at or above their recorded scores")
    return 0


if __name__ == "__main__":
    sys.exit(main())
