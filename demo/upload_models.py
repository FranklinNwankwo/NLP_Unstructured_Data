"""
Uploads the three fine-tuned models to ONE Hugging Face model repo (public), as the
subfolders ner/, severity/ and cause/, plus a model card. Safe to rerun
after adding a model. Writes demo/model_repo.txt so
the Space knows where to download from.

Needs a Hugging Face account and a token with write access
(huggingface.co > Settings > Access Tokens > New token > type "Write").
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi, login

try:
    from huggingface_hub import get_token
except ImportError:                                   # older huggingface_hub
    from huggingface_hub import HfFolder
    get_token = HfFolder.get_token

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "ner": ROOT / "ner_model/phmsa_ner_model",
    "severity": ROOT / "severity_model/phmsa_severity_model",
    "cause": ROOT / "cause_model/phmsa_cause_model",
}
REPO_NAME = "phmsa-incident-models"

MODEL_CARD = """---
language: en
library_name: transformers
base_model: distilbert/distilbert-base-uncased
tags:
  - token-classification
  - text-classification
  - pipeline-safety
  - phmsa
---

# PHMSA incident models

Three DistilBERT models fine-tuned on public PHMSA pipeline incident reports (gas distribution,
gas transmission and gathering, hazardous liquid; 2010 to present). Used by the Space
`phmsa-incident-extraction`.

| Folder | Task | Held-out result |
|---|---|---|
| `ner/` | token classification, 12 entity types | micro-F1 0.608 on 840 spans (1,050 sentences labelled by one annotator) |
| `severity/` | narrative classification: minor, moderate, severe, critical | macro-F1 0.777, accuracy 0.956 on 1,921 narratives ; calibration error 0.023 |
| `cause/` | narrative classification: PHMSA's 8 apparent-cause categories | macro-F1 0.748, accuracy 0.818 on 1,921 narratives ; calibration error 0.050 |

## Calibration

`severity/` and `cause/` each contain `calibration.json` with a temperature T fitted on the dev split
(T = 1.656 and 1.927). Divide the logits by T before the softmax for calibrated probabilities; labels do not
change. Uncalibrated, the cause model was overconfident (average confidence 0.95 vs accuracy 0.82).

## Limits

- Severity labels come from PHMSA's structured flags (fatality, injury requiring inpatient
  hospitalization, ignition, explosion), not from the narrative text. About 40% of critical narratives
  never mention a death (rough keyword estimate) and the model detects none of those.
  Severity says nothing about spill size or environmental damage.
- Cause labels are PHMSA's apparent cause as chosen by the operator on the report form, not an independent
  root-cause investigation. The weakest class is the catch-all "other incident cause" (F1 0.38), and
  "incorrect operation" is often confused with "equipment failure".
- The entity model does not extract root-cause spans (`CAUSE_FACTOR` scores 0) and is weak on locations and
  regulatory references. Rare classes per commodity are too small to evaluate.
- Tested only on PHMSA narratives. Not for operational, safety or regulatory decisions.

## Loading

Download the repo with `huggingface_hub.snapshot_download("REPO_ID")`, then load the two subfolders with
`AutoModelForTokenClassification.from_pretrained(<path>/ner)` and
`AutoModelForSequenceClassification.from_pretrained(<path>/severity)` (or `<path>/cause`).
"""

missing = [n for n, p in MODELS.items() if not (p / "config.json").exists()]
if missing:
    sys.exit(f"Model folder not found for: {', '.join(missing)}. Check the unzipped folders.")

if not get_token():
    login()                                           # asks you to paste the token
api = HfApi()
user = api.whoami()["name"]
repo_id = f"{user}/{REPO_NAME}"
api.create_repo(repo_id, repo_type="model", exist_ok=True)     # public
print(f"Uploading to https://huggingface.co/{repo_id}")

for name, path in MODELS.items():
    print(f"  {name}/ ...")
    api.upload_folder(folder_path=str(path), path_in_repo=name, repo_id=repo_id, repo_type="model",
                      ignore_patterns=["training_args.bin", "*.zip", "__pycache__/*"],
                      commit_message=f"Add {name} model")
api.upload_file(path_or_fileobj=MODEL_CARD.replace("REPO_ID", repo_id).encode("utf-8"),
                path_in_repo="README.md", repo_id=repo_id, repo_type="model",
                commit_message="Add model card")

(ROOT / "demo/model_repo.txt").write_text(repo_id, encoding="utf-8")
print(f"\nDone. Wrote demo/model_repo.txt ({repo_id})")