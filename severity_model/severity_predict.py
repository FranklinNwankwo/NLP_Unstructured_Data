"""
Local inference for the fine-tuned severity classifier (Phase 6 output).
Mirrors the Colab encoding exactly: no special tokens at tokenization time,
keep the start and the end of narratives longer than 510 tokens, then add
[CLS] / [SEP].
"""

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

    @torch.no_grad()
    def predict(self, texts, batch_size=16):
        """texts: list[str] -> list[{"label": str, "probabilities": {label: float}}]"""
        results = []
        cls_id, sep_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        for i in range(0, len(texts), batch_size):
            ids = self.tokenizer(texts[i:i + batch_size], add_special_tokens=False,
                                 truncation=False)["input_ids"]
            seqs = [[cls_id] + head_tail(x) + [sep_id] for x in ids]
            input_ids, mask = pad_batch(seqs, self.tokenizer.pad_token_id)
            logits = self.model(input_ids=torch.tensor(input_ids),
                                attention_mask=torch.tensor(mask)).logits
            for p in torch.softmax(logits, dim=-1).tolist():
                best = max(range(len(p)), key=p.__getitem__)
                results.append({"label": self.id2label[best],
                                "probabilities": {self.id2label[k]: round(v, 4) for k, v in enumerate(p)}})
        return results