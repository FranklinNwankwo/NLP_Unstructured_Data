"""
Local inference for the fine-tuned cause classifier. The encoding is identical to the
severity classifier (head + tail truncation, then [CLS] / [SEP]), so this reuses
severity_model/severity_predict.py and only points it at a different model folder.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "severity_model"))
from severity_predict import SeverityPredictor  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "phmsa_cause_model"


class CausePredictor(SeverityPredictor):
    """predict(texts) -> list[{"label": str, "probabilities": {label: float}}]"""

    def __init__(self, model_dir=MODEL_DIR):
        super().__init__(model_dir)