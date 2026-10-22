"""
Temperature scaling for the severity and cause classifiers: are the probabilities they show
trustworthy, and if not, one number per model that makes them so.

    python mlops/calibrate.py                          # both models, the repo's local model folders
    python mlops/calibrate.py --task severity
    python mlops/calibrate.py --model-dir PATH         # PATH/severity, PATH/cause
    python mlops/calibrate.py --dry-run                # measure only, write nothing into the model folders

Method (Guo et al., 2017): fit a single temperature T on the DEV split by minimising negative
log-likelihood of softmax(logits / T); report on the TEST split before and after. T > 1 means the
model was overconfident. Argmax never changes, so accuracy and macro-F1 are untouched.

Metrics on test:
  ECE    expected calibration error, 15 equal-width confidence bins (top-label): the average gap
         between "the model said 90%" and "it was right 90% of the time"
  NLL    negative log-likelihood of the true class
  Brier  mean squared error of the full probability vector

Writes:
  <model folder>/calibration.json       {"temperature": T, ...}: the predictors apply it automatically
  eval/calibration/<task>.json          before/after metrics and the reliability-table bins
  eval/calibration/reliability_<task>.png   reliability diagram (if matplotlib is installed)
  mlops/model_manifest.json             expected test ECE per model, checked by the regression test
Logits are cached in data/processed/calib_logits_<task>.npz, so reruns take seconds.

Afterwards: python demo/upload_models.py, python mlops/pin_models.py, python demo/build_space.py.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from common import LOCAL_MODEL_DIRS, ROOT, load_manifest, save_manifest  # noqa: E402

EXPORTS = {"severity": "severity_colab_export.json", "cause": "cause_colab_export.json"}
N_BINS = 15


# ---------------------------------------------------------------- metrics (numpy only, unit-tested)

def softmax(logits, t=1.0):
    z = logits / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits, y, t=1.0):
    z = logits / t
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


def reliability(probs, y, n_bins=N_BINS):
    """Top-label reliability table and ECE."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        n = int(m.sum())
        if n:
            acc, avg = float(correct[m].mean()), float(conf[m].mean())
            ece += n / len(y) * abs(acc - avg)
            bins.append({"lo": round(lo, 3), "hi": round(hi, 3), "n": n, "accuracy": round(acc, 4),
                         "confidence": round(avg, 4)})
    return ece, bins


def brier(probs, y):
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def fit_temperature(logits, y, lo=-3.0, hi=3.0, iters=80):
    """Golden-section search on log T (NLL is smooth and unimodal in T for one scalar)."""
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(logits, y, math.exp(c)), nll(logits, y, math.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(logits, y, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(logits, y, math.exp(d))
    return math.exp((a + b) / 2)


def summary(logits, y, t):
    p = softmax(logits, t)
    ece, bins = reliability(p, y)
    return {"temperature": round(t, 4), "ece": round(ece, 4), "nll": round(nll(logits, y, t), 4),
            "brier": round(brier(p, y), 4), "accuracy": round(float((p.argmax(1) == y).mean()), 4),
            "mean_confidence": round(float(p.max(1).mean()), 4), "bins": bins}


# ---------------------------------------------------------------- running the models

def get_logits(task, model_dir, data, n, refresh):
    cache = ROOT / "data/processed" / f"calib_logits_{task}.npz"
    if cache.exists() and not refresh and not n:
        z = np.load(cache)
        if len(z["dev"]) == len(data["dev"]) and len(z["test"]) == len(data["test"]):
            print(f"  logits from cache ({cache.name})")
            return z["dev"], z["test"]
    from severity_predict import SeverityPredictor
    predictor = SeverityPredictor(model_dir)
    labels = [predictor.id2label[i] for i in range(len(predictor.id2label))]
    if labels != data["labels"]:
        sys.exit(f"{task}: the model's labels {labels} do not match the export's {data['labels']}")
    out = {}
    for split in ("dev", "test"):
        rows = data[split][:n] if n else data[split]
        t0 = time.time()
        out[split] = predictor.logits([r["text"] for r in rows]).numpy()
        print(f"  {split}: {len(rows):,} narratives in {time.time() - t0:.0f}s")
    if not n:
        np.savez_compressed(cache, dev=out["dev"], test=out["test"])
    return out["dev"], out["test"]


def plot(task, before, after, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed: no reliability diagram)")
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--", label="perfect calibration")
    for res, name, color in ((before, "before", "#c0392b"), (after, "after", "#1f6f8b")):
        xs = [b["confidence"] for b in res["bins"]]
        ys = [b["accuracy"] for b in res["bins"]]
        ax.plot(xs, ys, marker="o", color=color, label=f"{name} (T={res['temperature']:.2f}, ECE {res['ece']:.3f})")
    ax.set_xlabel("confidence (top class)")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{task} classifier: reliability on test")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", nargs="+", choices=list(EXPORTS), default=list(EXPORTS))
    ap.add_argument("--model-dir", help="folder with severity/ and cause/ (default: the repo's local model folders)")
    ap.add_argument("--n", type=int, default=0, help="smoke test on the first N dev and test rows (writes nothing)")
    ap.add_argument("--refresh", action="store_true", help="recompute logits even if cached")
    ap.add_argument("--dry-run", action="store_true", help="measure only; do not write calibration.json or the manifest")
    args = ap.parse_args()
    write = not (args.dry_run or args.n)

    out_dir = ROOT / "eval/calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    rows = []
    for task in args.task:
        model_dir = Path(args.model_dir) / task if args.model_dir else LOCAL_MODEL_DIRS[task]
        if not (model_dir / "config.json").exists():
            sys.exit(f"No model in {model_dir}")
        export = ROOT / "data/processed" / EXPORTS[task]
        if not export.exists():
            sys.exit(f"{export.relative_to(ROOT)} missing (run {task}_model/export_for_colab.py)")
        data = json.load(open(export, encoding="utf-8"))
        print(f"{task}: {model_dir}")
        dev_logits, test_logits = get_logits(task, model_dir, data, args.n, args.refresh)
        y_dev = np.array([r["label"] for r in (data["dev"][:args.n] if args.n else data["dev"])])
        y_test = np.array([r["label"] for r in (data["test"][:args.n] if args.n else data["test"])])

        t = fit_temperature(dev_logits, y_dev)
        if not 0.06 < t < 19:
            print(f"  WARNING: T={t:.3f} is at the edge of the search range: the model or the labels look wrong")
        before, after = summary(test_logits, y_test, 1.0), summary(test_logits, y_test, t)
        dev_before, dev_after = summary(dev_logits, y_dev, 1.0), summary(dev_logits, y_dev, t)
        rows.append((task, before, after))
        result = {"task": task, "fitted_on": "dev", "n_dev": int(len(y_dev)), "n_test": int(len(y_test)),
                  "temperature": round(t, 4), "dev_nll_before": dev_before["nll"], "dev_nll_after": dev_after["nll"],
                  "test_before": before, "test_after": after}
        if write:
            (out_dir / f"{task}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            plot(task, before, after, out_dir / f"reliability_{task}.png")
            (model_dir / "calibration.json").write_text(json.dumps(
                {"temperature": round(t, 4), "method": "temperature scaling, fitted on dev (NLL)",
                 "test_ece_before": before["ece"], "test_ece_after": after["ece"]}, indent=2) + "\n", encoding="utf-8")
            manifest["models"][task]["expected"]["ece"] = after["ece"]
            manifest["models"][task]["temperature"] = round(t, 4)

    print("\nTest set, before -> after temperature scaling (accuracy cannot change):\n")
    print(f"{'model':<9} {'T':>6}  {'ECE':>15}  {'NLL':>15}  {'Brier':>15}  {'mean conf':>15}  {'accuracy':>8}")
    for task, b, a in rows:
        print(f"{task:<9} {a['temperature']:>6.3f}  {b['ece']:.4f} -> {a['ece']:.4f}  {b['nll']:.4f} -> {a['nll']:.4f}  "
              f"{b['brier']:.4f} -> {a['brier']:.4f}  {b['mean_confidence']:.4f} -> {a['mean_confidence']:.4f}  "
              f"{a['accuracy']:>8.3f}")
        worst = max(b["bins"], key=lambda x: x["n"] * abs(x["accuracy"] - x["confidence"]))
        print(f"          before: narratives given {worst['lo']:.2f}-{worst['hi']:.2f} confidence (n={worst['n']}) "
              f"were right {worst['accuracy']:.1%} of the time (mean confidence {worst['confidence']:.1%})")
    if write:
        save_manifest(manifest)
        print(f"\nwrote calibration.json into each model folder, eval/calibration/, and expected ECE into the manifest")
        print("next: python demo/upload_models.py, python mlops/pin_models.py, python demo/build_space.py")
    else:
        print("\n(dry run / smoke test: nothing written)")


if __name__ == "__main__":
    main()
