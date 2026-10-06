"""
Assembles demo/space/ into a self-contained, deployable folder: copies the four
code modules next to app.py, writes examples.json from pipeline/sample_output.json,
and copies demo/model_repo.txt. Safe to run again after any code change.
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "demo/space"

COPY = {
    ROOT / "pipeline/incident_extractor.py": "incident_extractor.py",
    ROOT / "relation_extraction/extract_relations.py": "extract_relations.py",
    ROOT / "ner_model/predict.py": "predict.py",
    ROOT / "severity_model/severity_predict.py": "severity_predict.py",
}
for src, name in COPY.items():
    if not src.exists():
        sys.exit(f"Missing {src}")
    shutil.copy2(src, SPACE / name)
    print(f"copied {src.relative_to(ROOT)} -> demo/space/{name}")

sample = ROOT / "pipeline/sample_output.json"
if not sample.exists():
    sys.exit("pipeline/sample_output.json not found. Run: python pipeline\\run_pipeline.py")
records = json.loads(sample.read_text(encoding="utf-8"))
examples, seen = [], {}
for r in records:
    commodity = r.get("commodity_type") or "custom"
    title = f"{commodity.replace('_', ' ').title()} example"
    seen[title] = seen.get(title, 0) + 1
    if seen[title] > 1:
        title += f" {seen[title]}"
    examples.append({"title": title, "commodity_type": commodity, "narrative": r["narrative"]})
(SPACE / "examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"wrote demo/space/examples.json ({len(examples)} examples)")

repo_file = ROOT / "demo/model_repo.txt"
if repo_file.exists():
    shutil.copy2(repo_file, SPACE / "model_repo.txt")
    print(f"copied model repo id: {repo_file.read_text().strip()}")
else:
    print("WARNING: demo/model_repo.txt not found. Run demo\\upload_models.py before deploying.")