"""NER + relation rules on real narratives: worked examples plus how often each relation fires."""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ner_model"))
sys.path.insert(0, str(ROOT / "relation_extraction"))

import pandas as pd
import spacy

from predict import NERPredictor
from extract_relations import RULES, extract_relations, dedupe, describe

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=300, help="narratives to process")
args = parser.parse_args()

df = pd.read_parquet(ROOT / "data/processed/weak_labeled_narratives.parquet")
df = df[df["narrative"].notna()]
parts = [g.sample(min(len(g), args.n // 3), random_state=7) for _, g in df.groupby("commodity_type")]
sample = pd.concat(parts).reset_index(drop=True)

splitter = spacy.blank("en")
splitter.add_pipe("sentencizer")

records = []  # (narrative_idx, sentence_idx, sentence)
for ni, text in enumerate(sample["narrative"]):
    for si, s in enumerate(splitter(str(text)).sents):
        t = s.text.strip()
        if len(t.split()) >= 3:
            records.append((ni, si, t))

print(f"{len(sample)} narratives -> {len(records):,} sentences. Running NER on CPU, this takes a minute or two...")
predictor = NERPredictor()
entities = predictor.predict([r[2] for r in records])

ents_by_narr = defaultdict(list)
rels_raw = defaultdict(list)
for (ni, si, sent), ents in zip(records, entities):
    ents_by_narr[ni].extend(ents)
    rels_raw[ni].extend(extract_relations(sent, ents, si))
rels_by_narr = {ni: dedupe(r) for ni, r in rels_raw.items()}

n_ents = sum(len(v) for v in ents_by_narr.values())
rel_counts = Counter((r["relation"], r["confidence"]) for rels in rels_by_narr.values() for r in rels)
with_rel = sum(1 for ni in range(len(sample)) if rels_by_narr.get(ni))

print(f"\nPredicted entities: {n_ents:,} ({n_ents / len(sample):.1f} per narrative)")
print("\nRelations after per-narrative dedupe:")
for name, conf in dict.fromkeys((r["name"], r["confidence"]) for r in RULES):
    print(f"  {name:<16}{rel_counts[(name, conf)]:>6}   {conf}")
print(f"\nNarratives with at least one relation: {with_rel}/{len(sample)} ({with_rel / len(sample):.0%})")

for ni in sample.groupby("commodity_type").head(1).index:
    row = sample.loc[ni]
    print(f"\n{'=' * 70}\n{row['commodity_type'].upper()}\n{'=' * 70}")
    print(str(row["narrative"])[:600])
    print("\nEntities:")
    by = defaultdict(list)
    for e in ents_by_narr.get(ni, []):
        if e["text"] not in by[e["label"]]:
            by[e["label"]].append(e["text"])
    for label, texts in sorted(by.items()):
        print(f"  {label}: {texts[:8]}")
    print("\nRelations:")
    rels = rels_by_narr.get(ni, [])
    if not rels:
        print("  (none)")
    for r in rels:
        print(f"  s{r['sentence_index']}: {describe(r)}")