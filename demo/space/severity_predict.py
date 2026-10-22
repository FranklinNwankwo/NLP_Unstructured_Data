"""
Local inference for the fine-tuned severity classifier (Phase 6 output).
Mirrors the Colab encoding exactly: no special tokens at tokenization time,
keep the start and the end of narratives longer than 510 tokens, then add
[CLS] / [SEP].

Calibration: if the model folder holds calibration.json ({"temperature": T}, written by
mlops/calibrate.py), probabilities are softmax(logits / T). Dividing by one positive number never
changes which class is largest, so labels and accuracy are identical with or without it.
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent / "phmsa_severity_model"
MAX_LEN = 512


def head_tail(ids, max_len=MAX_LEN):
    """Keep the first and last tokens when a narrative is too long."""
    room = max_len - 2
    if len(ids) > room:
        head = room // 2
        ids = ids[:head] + ids[-(room - head):]
    return ids


def pad_batch(seqs, pad_id):
    """Right-pad token id lists to the longest one; returns (input_ids, attention_mask)."""
    width = max(len(s) for s in seqs)
    input_ids = [s + [pad_id] * (width - len(s)) for s in seqs]
    mask = [[1] * len(s) + [0] * (width - len(s)) for s in seqs]
    return input_ids, mask


class SeverityPredictor:
    def __init__(self, model_dir=MODEL_DIR):
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.tokenizer.model_max_length = 1_000_000      # we truncate ourselves; silences the length warning
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        calib = Path(model_dir) / "calibration.json"
        self.temperature = float(json.loads(calib.read_text())["temperature"]) if calib.exists() else 1.0

    @torch.no_grad()
    def logits(self, texts, batch_size=16):
        """texts: list[str] -> raw logits, a float tensor of shape (len(texts), n_classes)."""
        out = []
        cls_id, sep_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        for i in range(0, len(texts), batch_size):
            ids = self.tokenizer(texts[i:i + batch_size], add_special_tokens=False,
                                 truncation=False)["input_ids"]
            seqs = [[cls_id] + head_tail(x) + [sep_id] for x in ids]
            input_ids, mask = pad_batch(seqs, self.tokenizer.pad_token_id)
            out.append(self.model(input_ids=torch.tensor(input_ids),
                                  attention_mask=torch.tensor(mask)).logits.float())
        return torch.cat(out) if out else torch.empty(0, len(self.id2label))

    def predict(self, texts, batch_size=16):
        """texts: list[str] -> list[{"label": str, "probabilities": {label: float}}] (calibrated if available)"""
        if not texts:
            return []
        probs = torch.softmax(self.logits(texts, batch_size) / self.temperature, dim=-1).tolist()
        results = []
        for p in probs:
            best = max(range(len(p)), key=p.__getitem__)
            results.append({"label": self.id2label[best],
                            "probabilities": {self.id2label[k]: round(v, 4) for k, v in enumerate(p)}})
        return results
