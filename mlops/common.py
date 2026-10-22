"""
Shared pieces for the MLOps layer: paths, the pinned model manifest, loading the three models,
reading PHMSA's flat files, and the monitoring statistics used by the drift report.

Nothing here imports torch at module level, so the offline unit tests run without it.
"""

import hashlib
import io
import json
import math
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "mlops" / "model_manifest.json"
FROZEN_DIR = ROOT / "eval" / "frozen"
NER_TEST = FROZEN_DIR / "ner_test.jsonl"
DOCS_TEST = FROZEN_DIR / "docs_test.jsonl"
SEEN_REPORTS = ROOT / "mlops" / "seen_reports.txt"
REFERENCE_PATH = ROOT / "mlops" / "reference_profile.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _sub in ("pipeline", "relation_extraction", "ner_model", "severity_model", "cause_model", "annotation"):
    if str(ROOT / _sub) not in sys.path:
        sys.path.insert(0, str(ROOT / _sub))

# PHMSA's official "January 2010 to present" flat files, one ZIP per report form.
PHMSA_FILES = {
    "gas_distribution": "https://www.phmsa.dot.gov/sites/phmsa.dot.gov/files/data_statistics/pipeline/"
                        "incident_gas_distribution_jan2010_present.zip",
    "gas_transmission_gathering": "https://www.phmsa.dot.gov/sites/phmsa.dot.gov/files/data_statistics/pipeline/"
                                  "incident_gas_transmission_gathering_jan2010_present.zip",
    "hazardous_liquid": "https://www.phmsa.dot.gov/sites/phmsa.dot.gov/files/data_statistics/pipeline/"
                        "accident_hazardous_liquid_jan2010_present.zip",
}
FILE_STEMS = {
    "gas_distribution": "incident_gas_distribution_jan2010_present",
    "gas_transmission_gathering": "incident_gas_transmission_gathering_jan2010_present",
    "hazardous_liquid": "accident_hazardous_liquid_jan2010_present",
}
MIN_WORDS = 5
SEVERITY_ORDER = ["minor", "moderate", "severe", "critical"]


# ---------------------------------------------------------------- manifest and hashes

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path=MANIFEST_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_manifest(manifest, path=MANIFEST_PATH):
    Path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:     # LF on Windows too: the hash must match CI
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- models

MODELS_DIR = ROOT / ".models"          # pinned downloads live here (git-ignored), one folder per revision
LOCAL_MODEL_DIRS = {"ner": ROOT / "ner_model/phmsa_ner_model",
                    "severity": ROOT / "severity_model/phmsa_severity_model",
                    "cause": ROOT / "cause_model/phmsa_cause_model"}


def download_models(manifest, dest=None):
    """Download the three model folders at the manifest's pinned revision into .models/<revision>/.

    Uses a plain folder (local_dir), not the Hugging Face cache, because the cache needs symlinks,
    which Windows refuses without Developer Mode or admin rights (WinError 1314). Skips the download
    when the folder is already complete."""
    rev = manifest["revision"]
    root = Path(dest) if dest else MODELS_DIR / rev[:12]
    subs = [m["subfolder"] for m in manifest["models"].values()]
    if rev != "main" and all((root / s / "config.json").exists() for s in subs):
        return root
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=manifest["model_repo"], revision=rev, local_dir=str(root),
                      allow_patterns=[f"{s}/*" for s in subs])
    return root


def model_paths(model_dir, manifest):
    """{name: folder} for the three models.
    model_dir None    -> the pinned revision (downloaded once into .models/)
    model_dir "local" -> the repo's own folders (ner_model/phmsa_ner_model, ...), no download
    any other path    -> <path>/ner, <path>/severity, <path>/cause"""
    if model_dir == "local":
        missing = [str(p.relative_to(ROOT)) for p in LOCAL_MODEL_DIRS.values() if not (p / "config.json").exists()]
        if missing:
            raise FileNotFoundError("No model in: " + ", ".join(missing))
        return dict(LOCAL_MODEL_DIRS)
    root = Path(model_dir) if model_dir else download_models(manifest)
    return {k: root / v["subfolder"] for k, v in manifest["models"].items()}


def load_predictors(paths):
    from predict import NERPredictor
    from severity_predict import SeverityPredictor
    from cause_predict import CausePredictor
    return NERPredictor(paths["ner"]), SeverityPredictor(paths["severity"]), CausePredictor(paths["cause"])


# ---------------------------------------------------------------- PHMSA flat files

def report_key(commodity, report_number):
    return f"{commodity}|{str(report_number).strip()}"


def severity_from_flags(row):
    """The same rule as weak_supervision/severity_weak_labels.py."""
    yes = lambda c: str(row.get(c, "")).strip().upper() == "YES"
    if yes("FATALITY_IND"):
        return "critical"
    if yes("INJURY_IND"):
        return "severe"
    if yes("IGNITE_IND") or yes("EXPLODE_IND"):
        return "moderate"
    return "minor"


def clean_cause(value):
    """Upper-case, collapse spaces, and map gas distribution's wording onto the other forms' (as the cause export does)."""
    from rag.config import CAUSE_ALIASES
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = " ".join(str(value).split()).upper()
    if s in {"", "NAN", "NONE", "<NA>"}:
        return None
    return CAUSE_ALIASES.get(s, s)


def read_flat_file(raw_bytes, commodity):
    """One PHMSA flat file (tab-delimited, cp1252) -> DataFrame with a commodity_type column."""
    import pandas as pd
    df = pd.read_csv(io.BytesIO(raw_bytes), sep="\t", encoding="cp1252", dtype=str, low_memory=False)
    df["commodity_type"] = commodity
    return df


def load_phmsa(source_dir=None, download_to=None):
    """All three forms as one DataFrame. Reads <source_dir>/<stem>.txt or .zip; otherwise downloads the ZIPs."""
    import pandas as pd
    frames = []
    for commodity, stem in FILE_STEMS.items():
        raw = None
        if source_dir:
            d = Path(source_dir)
            for cand in (d / f"{stem}.txt", d / commodity / f"{stem}.txt", d / f"{stem}.zip"):
                if cand.exists():
                    raw = cand.read_bytes()
                    break
            if raw is None:
                raise FileNotFoundError(f"No {stem}.txt or .zip under {d}")
        else:
            import requests
            r = requests.get(PHMSA_FILES[commodity], timeout=300,
                             headers={"User-Agent": "phmsa-incident-extraction monitoring job"})
            r.raise_for_status()
            raw = r.content
            if download_to:
                Path(download_to).mkdir(parents=True, exist_ok=True)
                (Path(download_to) / f"{stem}.zip").write_bytes(raw)
        if raw[:2] == b"PK":                                   # a ZIP: take the .txt inside
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".txt")]
                if not names:
                    raise ValueError(f"{stem}.zip has no .txt file inside")
                pick = next((n for n in names if stem in n), names[0])
                raw = z.read(pick)
        frames.append(read_flat_file(raw, commodity))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- monitoring statistics

LENGTH_BINS = [0, 25, 50, 100, 200, 400, 800, float("inf")]


def distribution(values, categories):
    c = Counter(values)
    n = sum(c[k] for k in categories) or 1
    return {k: c[k] / n for k in categories}


def length_distribution(word_counts):
    labels = [f"{LENGTH_BINS[i]}-{LENGTH_BINS[i + 1]}" for i in range(len(LENGTH_BINS) - 1)]
    binned = []
    for w in word_counts:
        for i in range(len(LENGTH_BINS) - 1):
            if LENGTH_BINS[i] <= w < LENGTH_BINS[i + 1]:
                binned.append(labels[i])
                break
    return distribution(binned, labels)


def psi(expected, actual, eps=1e-4):
    """Population stability index between two distributions over the same categories."""
    total = 0.0
    for k in set(expected) | set(actual):
        e = max(expected.get(k, 0.0), eps)
        a = max(actual.get(k, 0.0), eps)
        total += (a - e) * math.log(a / e)
    return total


def profile(records, flag_severity=None, coded_cause=None):
    """Summary statistics of pipeline output. records: IncidentExtractor records, in order.
    flag_severity / coded_cause: optional lists of the labels PHMSA's own fields give, for agreement rates."""
    n = len(records)
    words = [len((r.get("narrative") or "").split()) for r in records]
    ents = [len(r["entities"]) for r in records]
    rels = [len(r["relations"]) for r in records]
    labels = [e["label"] for r in records for e in r["entities"]]
    label_set = sorted(set(labels))
    sev = [r["severity"]["label"] for r in records if r.get("severity")]
    cau = [r["cause"]["label"] for r in records if r.get("cause")]
    out = {
        "n_narratives": n,
        "words_mean": sum(words) / n if n else 0.0,
        "length_dist": length_distribution(words),
        "entities_per_narrative": sum(ents) / n if n else 0.0,
        "zero_entity_share": sum(1 for x in ents if x == 0) / n if n else 0.0,
        "relations_per_narrative": sum(rels) / n if n else 0.0,
        "entity_label_dist": distribution(labels, label_set),
        "severity_dist": distribution(sev, SEVERITY_ORDER),
        "severity_confidence_mean": (sum(max(r["severity"]["probabilities"].values()) for r in records if r.get("severity"))
                                     / max(len(sev), 1)),
    }
    if cau:
        cats = sorted(set(cau) | set(c for c in (coded_cause or []) if c))
        out["cause_dist"] = distribution(cau, cats)
        out["cause_confidence_mean"] = (sum(max(r["cause"]["probabilities"].values()) for r in records if r.get("cause"))
                                        / len(cau))
    if flag_severity is not None:
        pairs = [(r["severity"]["label"], f) for r, f in zip(records, flag_severity) if r.get("severity") and f]
        out["severity_agreement"] = sum(p == f for p, f in pairs) / len(pairs) if pairs else None
    if coded_cause is not None:
        pairs = [(r["cause"]["label"], c) for r, c in zip(records, coded_cause) if r.get("cause") and c]
        out["cause_agreement"] = sum(p == c for p, c in pairs) / len(pairs) if pairs else None
    return out


# Thresholds: PSI 0.1 to 0.2 is a moderate shift, above 0.2 a large one (the usual rule of thumb).
PSI_WARN, PSI_ALERT = 0.10, 0.20
AGREEMENT_DROP = 0.05
MIN_BATCH = 100


def compare(reference, current):
    """Checks a batch profile against the reference profile. Returns (rows, alerts)."""
    rows, alerts = [], []
    small = current["n_narratives"] < MIN_BATCH

    def add(name, ref, cur, status, note=""):
        rows.append({"check": name, "reference": ref, "current": cur, "status": status, "note": note})
        if status == "ALERT":
            alerts.append(name)

    for key, label in (("length_dist", "Narrative length (PSI)"), ("severity_dist", "Severity mix (PSI)"),
                       ("cause_dist", "Cause mix (PSI)"), ("entity_label_dist", "Entity type mix (PSI)")):
        if key not in reference or key not in current:
            continue
        v = psi(reference[key], current[key])
        status = "ALERT" if v >= PSI_ALERT else "WARN" if v >= PSI_WARN else "OK"
        if small and status == "ALERT":
            status = "WARN"
        add(label, "0", round(v, 3), status, "small batch: treat as indicative" if small else "")

    ref_z, cur_z = reference["zero_entity_share"], current["zero_entity_share"]
    status = "ALERT" if cur_z > max(2 * ref_z, ref_z + 0.05) else "OK"
    add("Narratives with zero entities", f"{ref_z:.1%}", f"{cur_z:.1%}", "WARN" if small and status == "ALERT" else status)

    for key, label in (("severity_agreement", "Severity agrees with PHMSA flags"),
                       ("cause_agreement", "Cause agrees with PHMSA's coded cause")):
        ref_a, cur_a = reference.get(key), current.get(key)
        if ref_a is None or cur_a is None:
            continue
        status = "ALERT" if cur_a < ref_a - AGREEMENT_DROP else "OK"
        add(label, f"{ref_a:.1%}", f"{cur_a:.1%}", "WARN" if small and status == "ALERT" else status)

    for key, label in (("entities_per_narrative", "Entities per narrative"),
                       ("severity_confidence_mean", "Mean severity confidence"),
                       ("cause_confidence_mean", "Mean cause confidence")):
        if key in reference and key in current:
            add(label, round(reference[key], 3), round(current[key], 3), "INFO")
    return rows, alerts


def report_markdown(title, reference, current, rows, alerts, extra_lines=()):
    status = "ALERT" if alerts else ("WARN" if any(r["status"] == "WARN" for r in rows) else "OK")
    lines = [f"# {title}", "", f"**Status: {status}** · {current['n_narratives']} narratives in this batch, "
             f"{reference['n_narratives']} in the reference.", ""]
    lines += list(extra_lines)
    lines += ["", "| Check | Reference | This batch | Status | Note |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['check']} | {r['reference']} | {r['current']} | {r['status']} | {r['note']} |")
    lines += ["", "PSI: under 0.10 stable, 0.10 to 0.20 moderate shift, over 0.20 large shift. "
              f"Batches under {MIN_BATCH} narratives never raise an ALERT, only WARN."]
    if alerts:
        lines += ["", "**Alerts:** " + ", ".join(alerts)]
    return "\n".join(lines) + "\n"


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def get_extractor(model_dir=None, manifest=None):
    """IncidentExtractor on the pinned HF revision, the repo's local model folders ("local"),
    or a folder holding ner/, severity/, cause/."""
    manifest = manifest or load_manifest()
    paths = model_paths(model_dir, manifest)
    from incident_extractor import IncidentExtractor
    ner, sev, cause = load_predictors(paths)
    return IncidentExtractor(ner, sev, cause), (str(model_dir) if model_dir else
                                               f"{manifest['model_repo']}@{manifest['revision']}")
