import sys
sys.path.insert(0, "annotation")
import pandas as pd
from bio_utils import spans_to_bio, resolve_overlaps

df = pd.read_parquet("data/processed/ner_test.parquet")
suspect = 0
for _, row in df.iterrows():
    ents = resolve_overlaps(row["entities"])
    tokens, tags = spans_to_bio(row["sentence_text"], ents)
    tagged_text = " ".join(t for t, g in zip(tokens, tags) if g != "O")
    gold_text = " ".join(e["text"].strip() for e in ents)
    # crude check: do the tagged tokens roughly match the gold entity text?
    if ents and not any(e["text"].strip().split()[0].upper() in tagged_text.upper() for e in ents if e["text"].strip()):
        suspect += 1
        print(f"[possible misalign] {row['sentence_text'][:80]}")
        print(f"  gold: {[e['text'] for e in ents]}")
        print(f"  tagged tokens: {tagged_text}")
        print()

print(f"\n{suspect} sentences flagged for review out of {len(df)}")