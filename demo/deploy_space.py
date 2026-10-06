"""Creates (or updates) the Hugging Face Space from demo/space/. Hugging Face then builds the Docker image."""

import sys
from pathlib import Path

from huggingface_hub import HfApi, login

try:
    from huggingface_hub import get_token
except ImportError:                                   # older huggingface_hub
    from huggingface_hub import HfFolder
    get_token = HfFolder.get_token

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "demo/space"
SPACE_NAME = "phmsa-incident-extraction"
NEEDED = ["app.py", "ui_helpers.py", "requirements.txt", "Dockerfile", "README.md",
          "incident_extractor.py", "extract_relations.py", "predict.py", "severity_predict.py",
          "examples.json", "model_repo.txt"]

missing = [n for n in NEEDED if not (SPACE / n).exists()]
if missing:
    sys.exit(f"demo/space is missing: {', '.join(missing)}. Run demo\\upload_models.py, then demo\\build_space.py.")

if not get_token():
    login()
api = HfApi()
user = api.whoami()["name"]
space_id = f"{user}/{SPACE_NAME}"
api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(folder_path=str(SPACE), repo_id=space_id, repo_type="space",
                  ignore_patterns=["__pycache__/*", "*.pyc", "test_*"],
                  commit_message="Deploy PHMSA incident extraction demo")
print(f"\nUploaded. Hugging Face is now building the image (a few minutes).")
print(f"Watch the build and open the app at: https://huggingface.co/spaces/{space_id}")