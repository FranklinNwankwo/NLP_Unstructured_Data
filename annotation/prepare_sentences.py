"""
Splits each narrative into sentences and carries weak-label entities along,
re-offset to be relative to the sentence rather than the full narrative.
Entities that straddle a sentence boundary (rare — usually a misplit on an
abbreviation) are dropped rather than kept with wrong offsets.
"""

import spacy
import pandas as pd

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

df = pd.read_parquet("data/processed/weak_labeled_narratives.parquet")
sev = pd.read_parquet("data/processed/phmsa_combined_with_severity.parquet")
severity_by_position = sev["severity_weak_label"].reset_index(drop=True)

rows = []
dropped_straddling = 0

for _, row in df.iterrows():
    narrative = row["narrative"]
    if not isinstance(narrative, str) or not narrative.strip():
        continue  # empty/NaN narrative — nothing to split into sentences
    entities = row["entities"]
    doc = nlp(narrative)

    for sent in doc.sents:
        sent_start, sent_end = sent.start_char, sent.end_char
        sent_text = sent.text.strip()
        if len(sent_text.split()) < 3:
            continue  # skip fragments too short to annotate meaningfully

        sent_entities = []
        for e in entities:
            if e["start"] >= sent_start and e["end"] <= sent_end:
                sent_entities.append({
                    "text": e["text"], "label": e["label"],
                    "start": int(e["start"] - sent_start),
                    "end": int(e["end"] - sent_start),
                    "unit_type": e["unit_type"],
                })
            elif e["start"] < sent_end and e["end"] > sent_start:
                dropped_straddling += 1  # overlaps boundary, don't keep partial

        rows.append({
            "report_index": row["report_index"],
            "commodity_type": row["commodity_type"],
            "severity_weak_label": severity_by_position.iloc[row["report_index"]],
            "sentence_text": sent_text,
            "entities": sent_entities,
            "num_entities": len(sent_entities),
        })

out = pd.DataFrame(rows)
out.to_parquet("data/processed/sentences_weak_labeled.parquet", index=False)

print(f"Sentences: {len(out):,} from {df['report_index'].nunique():,} narratives")
print(f"Entities dropped for straddling a sentence boundary: {dropped_straddling}")
print(f"\nSentence count by commodity:")
print(out["commodity_type"].value_counts())