"""
Freezes the held-out test sets into small committed files so CI can re-score the models
without the gitignored data/processed/ folder.

Writes:
  eval/frozen/ner_test.jsonl    the 156 hand-annotated test sentences (text, commodity, gold spans)
  eval/frozen/docs_test.jsonl   the severity/cause test narratives (text, commodity, severity, cause)
  mlops/seen_reports.txt        every report key (commodity|REPORT_NUMBER) in the training snapshot,
                                so the batch job can tell new PHMSA reports from ones the models saw
and records their sha256 hashes, sizes and the source-data hash in mlops/model_manifest.json.

Run from the repository root, after the three Colab exports exist:
    python mlops/freeze_eval_sets.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import json  # noqa: E402

import pandas as pd  # noqa: E402

from common import (DOCS_TEST, MANIFEST_PATH, NER_TEST, ROOT, SEEN_REPORTS, load_manifest,  # noqa: E402
                    report_key, save_manifest, sha256, write_jsonl)

PROC = ROOT / "data/processed"
need = ["ner_test.parquet", "phmsa_combined_with_severity.parquet",
        "severity_colab_export.json", "cause_colab_export.json"]
missing = [n for n in need if not (PROC / n).exists()]
if missing:
    sys.exit("Missing in data/processed/: " + ", ".join(missing))

# ---- NER: gold spans exactly as stored (resolve_overlaps is applied at scoring time, as in training)
ner = pd.read_parquet(PROC / "ner_test.parquet")
ner_rows = []
for _, r in ner.iterrows():
    ents = [{"start": int(e["start"]), "end": int(e["end"]), "label": str(e["label"]),
             "text": str(e.get("text", r["sentence_text"][int(e["start"]):int(e["end"])]))}
            for e in r["entities"]]
    ner_rows.append({"text": r["sentence_text"], "commodity_type": r.get("commodity_type"), "entities": ents})
write_jsonl(ner_rows, NER_TEST)
print(f"wrote {NER_TEST.relative_to(ROOT)}: {len(ner_rows)} sentences")

# ---- documents: severity test split, with the cause label joined on (same split, same order;
#      the cause export dropped rows without a coded cause, so walk both lists together)
sev = json.load(open(PROC / "severity_colab_export.json", encoding="utf-8"))
cau = json.load(open(PROC / "cause_colab_export.json", encoding="utf-8"))
ctest, j, doc_rows = cau["test"], 0, []
for r in sev["test"]:
    cause = None
    if j < len(ctest) and ctest[j]["text"] == r["text"]:
        cause = cau["labels"][ctest[j]["label"]]
        j += 1
    doc_rows.append({"text": r["text"], "commodity_type": r.get("commodity_type"),
                     "severity": sev["labels"][r["label"]], "cause": cause})
if j != len(ctest):
    sys.exit(f"Could only match {j} of {len(ctest)} cause test rows to the severity test split. "
             "Rebuild the cause export (python cause_model/export_for_colab.py) and rerun.")
write_jsonl(doc_rows, DOCS_TEST)
print(f"wrote {DOCS_TEST.relative_to(ROOT)}: {len(doc_rows)} narratives "
      f"({sum(1 for d in doc_rows if d['cause'])} with a coded cause)")

# ---- every report in the training snapshot
df = pd.read_parquet(PROC / "phmsa_combined_with_severity.parquet", columns=["commodity_type", "REPORT_NUMBER"])
keys = sorted({report_key(c, n) for c, n in zip(df["commodity_type"], df["REPORT_NUMBER"])
               if str(n).strip() not in {"", "nan", "None", "<NA>"}})
SEEN_REPORTS.write_text("\n".join(keys) + "\n", encoding="utf-8")
print(f"wrote {SEEN_REPORTS.relative_to(ROOT)}: {len(keys):,} report keys")

# ---- manifest
m = load_manifest()
m["eval_sets"] = {
    "ner_test": {"path": str(NER_TEST.relative_to(ROOT)).replace("\\", "/"), "rows": len(ner_rows),
                 "sha256": sha256(NER_TEST)},
    "docs_test": {"path": str(DOCS_TEST.relative_to(ROOT)).replace("\\", "/"), "rows": len(doc_rows),
                  "rows_with_cause": sum(1 for d in doc_rows if d["cause"]), "sha256": sha256(DOCS_TEST)},
}
m["training_data"] = {
    "phmsa_combined_with_severity.parquet": sha256(PROC / "phmsa_combined_with_severity.parquet"),
    "ner_train.parquet": sha256(PROC / "ner_train.parquet") if (PROC / "ner_train.parquet").exists() else None,
    "severity_split_sizes": {k: len(sev[k]) for k in ("train", "dev", "test")},
    "cause_split_sizes": {k: len(cau[k]) for k in ("train", "dev", "test")},
    "reports_in_snapshot": len(keys),
}
save_manifest(m)
print(f"updated {MANIFEST_PATH.relative_to(ROOT)}")
