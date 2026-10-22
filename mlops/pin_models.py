"""
Pins the deployed models to one exact Hugging Face commit, so the demo, the API, the
regression test and the batch job all load the same weights, and a re-upload to `main`
cannot silently change what runs.

    python mlops/pin_models.py                 # pin to the current head of the model repo
    python mlops/pin_models.py --revision abc  # pin to a specific commit

Writes the commit hash to mlops/model_manifest.json and demo/model_revision.txt
(build_space.py copies the latter into the Space; app.py passes it as `revision=`).
Run it after every model upload, then rerun the regression test before committing.
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_manifest, save_manifest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--revision", help="commit hash to pin (default: current head of main)")
args = ap.parse_args()

m = load_manifest()
from huggingface_hub import HfApi  # noqa: E402

info = HfApi().model_info(m["model_repo"], revision=args.revision or "main", files_metadata=False)
files = {s.rfilename for s in info.siblings}
for name, spec in m["models"].items():
    if f"{spec['subfolder']}/config.json" not in files:
        sys.exit(f"{m['model_repo']}@{info.sha[:10]} has no {spec['subfolder']}/config.json ({name} model)")

old = m["revision"]
m["revision"] = info.sha
m["pinned_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
save_manifest(m)
(ROOT / "demo/model_revision.txt").write_text(info.sha + "\n", encoding="utf-8")
print(f"{m['model_repo']}: {old} -> {info.sha}")
print("wrote mlops/model_manifest.json and demo/model_revision.txt")
print("next: python demo/build_space.py, then python mlops/regression_test.py")
