"""Offline tests for the MLOps layer: drift statistics, PHMSA file parsing, report selection,
regression scoring and manifest consistency. No models, no torch, no network.

    python mlops/test_mlops.py
"""

import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402

import common as c  # noqa: E402
from batch_inference import select_reports  # noqa: E402
from regression_test import macro_f1_present  # noqa: E402

failures = 0


def check(name, ok, detail=""):
    global failures
    failures += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n        {detail}" if not ok else ""))


def rec(words, ents=(), sev="minor", cause="CORROSION FAILURE", conf=0.9):
    return {"narrative": " ".join(["W"] * words), "entities": [{"label": l} for l in ents], "relations": [],
            "severity": {"label": sev, "probabilities": {sev: conf}},
            "cause": {"label": cause, "probabilities": {cause: conf}}}


# ---- PSI
d = {"a": 0.5, "b": 0.5}
check("PSI of identical distributions is 0", abs(c.psi(d, d)) < 1e-12)
check("PSI is positive and grows with the shift",
      0 < c.psi(d, {"a": 0.6, "b": 0.4}) < c.psi(d, {"a": 0.9, "b": 0.1}))
check("PSI handles a category missing on one side", c.psi({"a": 1.0}, {"a": 0.5, "b": 0.5}) > c.PSI_ALERT)

# ---- length bins
ld = c.length_distribution([3, 30, 30, 150, 5000])
check("length bins sum to 1 and land in the right bins",
      abs(sum(ld.values()) - 1) < 1e-9 and ld["25-50"] == 0.4 and ld["800-inf"] == 0.2, ld)

# ---- labels from PHMSA fields
check("fatality flag -> critical", c.severity_from_flags({"FATALITY_IND": "YES", "INJURY_IND": "YES"}) == "critical")
check("injury flag -> severe", c.severity_from_flags({"INJURY_IND": "yes "}) == "severe")
check("fire or explosion -> moderate", c.severity_from_flags({"EXPLODE_IND": "YES"}) == "moderate")
check("no flags (or missing columns) -> minor", c.severity_from_flags({}) == "minor")
check("cause alias maps gas distribution wording",
      c.clean_cause("pipe, weld,  or joint failure") == "MATERIAL FAILURE OF PIPE OR WELD")
check("blank / NaN cause -> None", c.clean_cause(" ") is None and c.clean_cause(float("nan")) is None)

# ---- flat files: tab-delimited cp1252, plain .txt and .zip
tmp = Path(__file__).resolve().parent / "_test_tmp"
tmp.mkdir(exist_ok=True)
txt = "REPORT_NUMBER\tNARRATIVE\tCAUSE\n20240001\tCAF\xc9 GAS LEAK AT THE METER SET\tCORROSION FAILURE\n".encode("cp1252")
for commodity, stem in c.FILE_STEMS.items():
    if commodity == "hazardous_liquid":
        b = io.BytesIO()
        with zipfile.ZipFile(b, "w") as z:
            z.writestr(f"{stem}.txt", txt)
        (tmp / f"{stem}.zip").write_bytes(b.getvalue())
    else:
        (tmp / f"{stem}.txt").write_bytes(txt)
df = c.load_phmsa(tmp)
check("loads all three forms (txt and zip) with commodity_type",
      len(df) == 3 and sorted(df["commodity_type"]) == sorted(c.FILE_STEMS), df.shape)
check("cp1252 characters decode", df["NARRATIVE"].iloc[0].startswith("CAFÉ"), df["NARRATIVE"].iloc[0])
for p in tmp.iterdir():
    p.unlink()
tmp.rmdir()

# ---- selecting new reports
raw = pd.DataFrame({
    "commodity_type": ["gas_distribution"] * 5,
    "REPORT_NUMBER": ["1", "2", "3", "4", "4"],
    "NARRATIVE": ["A LONG ENOUGH NARRATIVE HERE"] * 3 + ["TOO SHORT", "ANOTHER LONG ENOUGH NARRATIVE TEXT"],
    "REPORT_RECEIVED_DATE": ["1/1/2025", "3/1/2025", "2/1/2025", "4/1/2025", "5/1/2025"],
})
sel, total = select_reports(raw, {"gas_distribution|1"})
check("skips seen reports, keeps the last copy of a duplicate, sorts newest first",
      list(sel["_key"]) == ["gas_distribution|4", "gas_distribution|2", "gas_distribution|3"] and total == 4,
      list(sel["_key"]))
sel, _ = select_reports(raw, {"gas_distribution|1"}, use_all=True, max_n=2)
check("--all and --max", len(sel) == 2 and sel["_key"].iloc[0] == "gas_distribution|4")

# ---- profile and comparison
ref_recs = [rec(40, ["EQUIPMENT", "CONSEQUENCE"]) for _ in range(150)] + [rec(40, [], sev="severe") for _ in range(50)]
ref = c.profile(ref_recs, ["minor"] * 150 + ["severe"] * 50, ["CORROSION FAILURE"] * 200)
check("profile: zero-entity share and agreement", ref["zero_entity_share"] == 0.25
      and ref["severity_agreement"] == 1.0 and ref["cause_agreement"] == 1.0, ref)
check("profile: severity mix", ref["severity_dist"]["minor"] == 0.75 and ref["severity_dist"]["critical"] == 0)

same = c.profile(ref_recs, ["minor"] * 150 + ["severe"] * 50, ["CORROSION FAILURE"] * 200)
rows, alerts = c.compare(ref, same)
check("identical batch raises no alert", alerts == [] and all(r["status"] in ("OK", "INFO") for r in rows), alerts)

drift_recs = [rec(600, []) for _ in range(200)]                      # long narratives, no entities
drift = c.profile(drift_recs, ["severe"] * 200, ["EXCAVATION DAMAGE"] * 200)
rows, alerts = c.compare(ref, drift)
check("shifted batch alerts on length, zero-entity share and both agreements",
      {"Narrative length (PSI)", "Narratives with zero entities", "Severity agrees with PHMSA flags",
       "Cause agrees with PHMSA's coded cause"} <= set(alerts), alerts)

small = c.profile(drift_recs[:40], ["severe"] * 40, ["EXCAVATION DAMAGE"] * 40)
rows, alerts = c.compare(ref, small)
check("batches under MIN_BATCH only WARN", alerts == [] and any(r["status"] == "WARN" for r in rows), alerts)

md = c.report_markdown("t", ref, drift, *c.compare(ref, drift))
check("markdown report shows ALERT status and a table", "**Status: ALERT**" in md and "| Check |" in md)

no_cause = [dict(r, cause=None) for r in ref_recs]
p = c.profile(no_cause)
check("profile works without the cause model", "cause_dist" not in p and p["n_narratives"] == 200)

# ---- regression scoring
f1, acc = macro_f1_present(["a", "a", "b", "b"], ["a", "b", "b", "b"])
check("macro-F1 over present classes", abs(f1 - (2 / 3 + 0.8) / 2) < 1e-9 and acc == 0.75, (f1, acc))
f1, _ = macro_f1_present(["a", "a"], ["a", "c"])
check("classes only predicted (not present) don't count", abs(f1 - 2 / 3) < 1e-9, f1)

# ---- calibration (temperature scaling)
import numpy as np  # noqa: E402

from calibrate import brier, fit_temperature, nll, reliability, softmax  # noqa: E402

rng = np.random.default_rng(0)
true_logits = rng.normal(0, 2, size=(4000, 4))
y = np.array([rng.choice(4, p=row) for row in softmax(true_logits)])
over = true_logits * 2.5                                      # an overconfident model: logits 2.5x too large
t = fit_temperature(over, y)
check("temperature fit recovers the overconfidence factor", abs(t - 2.5) < 0.25, t)
check("calibrated NLL is lower than raw NLL", nll(over, y, t) < nll(over, y, 1.0))
ece_raw, _ = reliability(softmax(over), y)
ece_cal, bins = reliability(softmax(over, t), y)
check("ECE drops after scaling", ece_cal < ece_raw / 2, (ece_raw, ece_cal))
check("reliability bins cover every row", sum(b["n"] for b in bins) == len(y))
check("temperature never changes the argmax", (softmax(over).argmax(1) == softmax(over, t).argmax(1)).all())
perfect = np.eye(3)[[0, 1, 2, 0]]
check("Brier is 0 for perfect one-hot predictions", brier(perfect, np.array([0, 1, 2, 0])) == 0)
check("ECE is 0 for perfect one-hot predictions", reliability(perfect, np.array([0, 1, 2, 0]))[0] == 0)

# ---- manifest consistency (the files CI and the app rely on)
m = c.load_manifest()
check("manifest has repo, revision, three models with expected scores",
      m["model_repo"] and m["revision"] and set(m["models"]) == {"ner", "severity", "cause"}
      and all(v["expected"] for v in m["models"].values()))
pinned = m["revision"] != "main"
check("revision is a 40-character commit once pinned", not pinned or len(m["revision"]) == 40, m["revision"])
rev_file = c.ROOT / "demo/model_revision.txt"
if rev_file.exists():
    check("demo/model_revision.txt matches the manifest", rev_file.read_text().strip() == m["revision"])
for key, path in (("ner_test", c.NER_TEST), ("docs_test", c.DOCS_TEST)):
    want = m.get("eval_sets", {}).get(key, {}).get("sha256")
    if want and path.exists():
        check(f"{path.name} matches its frozen hash", c.sha256(path) == want)
if c.REFERENCE_PATH.exists():
    r = json.loads(c.REFERENCE_PATH.read_text())
    check("reference profile has the fields the drift report compares",
          {"length_dist", "severity_dist", "zero_entity_share", "n_narratives"} <= set(r))

print(f"\n{'All MLOps tests passed' if not failures else f'{failures} MLOps test(s) FAILED'}")
sys.exit(1 if failures else 0)
