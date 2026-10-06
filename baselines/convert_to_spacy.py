import spacy
from spacy.tokens import DocBin
import pandas as pd

import sys
sys.path.insert(0, "annotation")
from bio_utils import resolve_overlaps

nlp = spacy.blank("en")

train_df = pd.read_parquet("data/processed/ner_train.parquet")
test_df = pd.read_parquet("data/processed/ner_test.parquet")

# Split train further into train/dev (spaCy's training loop needs a dev set
# for early stopping) — 85/15, same random_state as the original split
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
n_dev = int(len(train_df) * 0.15)
dev_df, fit_df = train_df.iloc[:n_dev], train_df.iloc[n_dev:]


def make_docbin(df, path):
    db = DocBin()
    skipped = 0
    for _, row in df.iterrows():
        doc = nlp.make_doc(row["sentence_text"])
        ents = []
        for e in resolve_overlaps(row["entities"]):
            span = doc.char_span(e["start"], e["end"], label=e["label"], alignment_mode="contract")
            if span is None:
                skipped += 1  # offset didn't align to a token boundary
                continue
            ents.append(span)
        doc.ents = ents
        db.add(doc)
    db.to_disk(path)
    print(f"{path}: {len(df)} docs, {skipped} entities skipped (bad token alignment)")


make_docbin(fit_df, "data/processed/spacy_train.spacy")
make_docbin(dev_df, "data/processed/spacy_dev.spacy")
make_docbin(test_df, "data/processed/spacy_test.spacy")