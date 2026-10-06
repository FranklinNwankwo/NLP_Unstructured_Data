"""
Local inference for the fine-tuned DistilBERT NER model (Phase 4 output).

Mirrors training exactly: spaCy blank tokenizer -> words -> HF tokenizer with
is_split_into_words -> first-subword prediction per word -> BIO decode to
character spans. Using the same word segmentation as the training export keeps
predicted spans aligned with what the baselines were scored on.
"""

from pathlib import Path

import spacy
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent / "phmsa_ner_model"
_word_tokenizer = spacy.blank("en")


def decode_bio(doc, tags):
    """BIO tags over spaCy tokens -> [{label, start, end, text}] with char offsets."""
    entities, cur = [], None
    for tok, tag in zip(doc, tags):
        if tok.is_space:
            continue  # whitespace tokens are transparent, they don't break a span
        if tag == "O":
            if cur:
                entities.append(cur)
                cur = None
            continue
        prefix, label = tag.split("-", 1)
        end = tok.idx + len(tok.text)
        if prefix == "I" and cur and cur["label"] == label:
            cur["end"] = end
        else:
            if cur:
                entities.append(cur)
            cur = {"label": label, "start": tok.idx, "end": end}
    if cur:
        entities.append(cur)
    for e in entities:
        e["text"] = doc.text[e["start"]:e["end"]]
    return entities


class NERPredictor:
    def __init__(self, model_dir=MODEL_DIR):
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    @torch.no_grad()
    def predict(self, sentences, batch_size=32):
        """sentences: list[str] -> list[list[entity dict]], same order."""
        results = []
        for i in range(0, len(sentences), batch_size):
            docs = [_word_tokenizer(s) for s in sentences[i:i + batch_size]]
            enc = self.tokenizer(
                [[t.text for t in d] for d in docs],
                is_split_into_words=True, truncation=True, max_length=256,
                padding=True, return_tensors="pt",
            )
            inputs = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            pred_ids = self.model(**inputs).logits.argmax(dim=-1)

            for b, doc in enumerate(docs):
                tags = ["O"] * len(doc)
                seen = set()
                for pos, wid in enumerate(enc.word_ids(batch_index=b)):
                    if wid is None or wid in seen:
                        continue  # first subword of each word carries the tag
                    seen.add(wid)
                    tags[wid] = self.id2label[int(pred_ids[b, pos])]
                results.append(decode_bio(doc, tags))
        return results