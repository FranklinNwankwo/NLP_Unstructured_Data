"""
End-to-end incident extraction (Phase 7).

    narrative -> sentences -> NER -> negation filter -> merge split spans
              -> relations (per sentence, deduped per narrative)
    narrative -> severity classifier (whole narrative)

Returns one JSON-serializable record per narrative. All character offsets are
relative to the original narrative string.

Negation filter: measured negation was concentrated in CONSEQUENCE spans
(5.7% of weak-labelled spans, against 0.8% for EQUIPMENT and 1.2% for
FAILURE_MODE), so only CONSEQUENCE spans are dropped, and only when a negation
word sits just before them in the same clause ("NO REPORTABLE INJURY",
"NO FATALITIES OR INJURIES", "NOT CAUSED BY FIRE"). Dropped spans are returned
in `dropped_negated` so nothing disappears silently.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _sub in ("relation_extraction", "ner_model", "severity_model"):
    sys.path.insert(0, str(ROOT / _sub))

import spacy
from extract_relations import classify_quantity, dedupe, extract_relations, normalize_entities

MIN_NARRATIVE_WORDS = 5     # same cut-off as the severity training data
MIN_SENTENCE_WORDS = 3      # same cut-off as the annotated sentences

# "no/none/zero" reach up to 3 words ahead ("NO REPORTED FATALITIES OR INJURIES");
# "not/without/never" up to 2 ("NOT CAUSED BY FIRE") so "NOT ISOLATED UNTIL THE FIRE" survives.
_NEG = re.compile(
    r"\b(?:no|none|zero)\b(?:\s+[\w'/-]+){0,3}\s*$"
    r"|\b(?:not|without|never)\b(?:\s+[\w'/-]+){0,2}\s*$",
    re.I,
)


def is_negated(sentence, start):
    clause = re.split(r"[.;:,()]", sentence[:start])[-1]
    return bool(_NEG.search(clause))


class IncidentExtractor:
    def __init__(self, ner, severity):
        self.ner = ner
        self.severity = severity
        self.splitter = spacy.blank("en")
        self.splitter.add_pipe("sentencizer")

    @classmethod
    def load_default(cls):
        from predict import NERPredictor                    # ner_model/predict.py
        from severity_predict import SeverityPredictor      # severity_model/severity_predict.py
        return cls(NERPredictor(), SeverityPredictor())

    def extract(self, narrative, commodity_type=None):
        return self.extract_many([narrative], [commodity_type])[0]

    def extract_many(self, narratives, commodities=None):
        n = len(narratives)
        commodities = commodities or [None] * n
        texts = ["" if t is None else str(t) for t in narratives]

        # 1. sentences, keeping each one's character offset in the narrative
        sentences = []                      # (narrative index, sentence index, text, offset)
        for i, text in enumerate(texts):
            for si, sent in enumerate(self.splitter(text).sents):
                stripped = sent.text.strip()
                if len(stripped.split()) < MIN_SENTENCE_WORDS:
                    continue
                offset = sent.start_char + (len(sent.text) - len(sent.text.lstrip()))
                sentences.append((i, si, stripped, offset))

        # 2. NER over all sentences at once
        raw = self.ner.predict([s[2] for s in sentences]) if sentences else []

        records = [{"commodity_type": commodities[i], "narrative": texts[i], "severity": None,
                    "entities": [], "relations": [], "dropped_negated": []} for i in range(n)]
        rels = [[] for _ in range(n)]

        for (i, si, sent, offset), ents in zip(sentences, raw):
            kept = []
            for e in ents:
                if e["label"] == "CONSEQUENCE" and is_negated(sent, e["start"]):
                    records[i]["dropped_negated"].append(self._span(e, offset, si))
                else:
                    kept.append(e)
            kept = normalize_entities(sent, kept)
            for e in kept:
                span = self._span(e, offset, si)
                if e["label"] == "QUANTITY":
                    span["unit_type"] = classify_quantity(e["text"])
                records[i]["entities"].append(span)
            for r in extract_relations(sent, kept, si):
                r["_offset"] = offset
                rels[i].append(r)

        # 3. relations, deduped per narrative
        for i in range(n):
            for r in dedupe(rels[i]):
                off = r["_offset"]
                records[i]["relations"].append({
                    "type": r["relation"], "confidence": r["confidence"],
                    "sentence_index": r["sentence_index"],
                    "head": self._span(r["head"], off), "tail": self._span(r["tail"], off),
                })

        # 4. severity over the whole narrative
        eligible = [i for i, t in enumerate(texts) if len(t.split()) >= MIN_NARRATIVE_WORDS]
        if eligible:
            for i, p in zip(eligible, self.severity.predict([texts[i] for i in eligible])):
                records[i]["severity"] = p
        for rec in records:
            if rec["severity"] is None:
                rec["note"] = f"narrative has fewer than {MIN_NARRATIVE_WORDS} words, severity not predicted"
        return records

    @staticmethod
    def _span(e, offset, sentence_index=None):
        out = {"text": e["text"], "label": e["label"],
               "start": offset + e["start"], "end": offset + e["end"]}
        if sentence_index is not None:
            out["sentence_index"] = sentence_index
        if "unit_type" in e:
            out["unit_type"] = e["unit_type"]
        return out