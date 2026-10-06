"""
Builds the relation precision-check sheet: up to N relations per relation type,
drawn only from sentences the NER model never saw in training (all 1,050
annotated sentences are excluded). Low-confidence relations are left out
because they never fire.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ner_model"))
sys.path.insert(0, str(ROOT / "relation_extraction"))

import pandas as pd
import spacy

from predict import NERPredictor
from extract_relations import extract_relations, dedupe

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=900, help="narratives to scan")
parser.add_argument("--per_type", type=int, default=15, help="rows per relation type")
args = parser.parse_args()

annotated = pd.read_parquet(ROOT / "data/processed/annotated_sentences.parquet")
seen = set(zip(annotated["report_index"].astype(int), annotated["sentence_text"].str.strip()))

df = pd.read_parquet(ROOT / "data/processed/weak_labeled_narratives.parquet")
df = df[df["narrative"].notna()]
parts = [g.sample(min(len(g), args.n // 3), random_state=11) for _, g in df.groupby("commodity_type")]
sample = pd.concat(parts).reset_index(drop=True)

splitter = spacy.blank("en")
splitter.add_pipe("sentencizer")

records, skipped_seen = [], 0
for _, row in sample.iterrows():
    for si, s in enumerate(splitter(str(row["narrative"])).sents):
        t = s.text.strip()
        if len(t.split()) < 3:
            continue
        if (int(row["report_index"]), t) in seen:
            skipped_seen += 1
            continue
        records.append((int(row["report_index"]), row["commodity_type"], si, t))

print(f"{len(sample)} narratives -> {len(records):,} unseen sentences "
      f"({skipped_seen} annotated sentences excluded). Running NER on CPU, a few minutes...")
predictor = NERPredictor()
entities = predictor.predict([r[3] for r in records])

by_narr = {}
for (ri, commodity, si, sent), ents in zip(records, entities):
    for r in extract_relations(sent, ents, si):
        r["_ctx"] = (ri, commodity, sent)
        by_narr.setdefault(ri, []).append(r)

data = []
for rels in by_narr.values():
    for r in dedupe(rels):
        ri, commodity, sent = r["_ctx"]
        data.append({
            "relation": r["relation"], "head": r["head"]["text"], "tail": r["tail"]["text"],
            "correct": "", "error_type": "",
            "sentence": sent, "head_label": r["head"]["label"], "tail_label": r["tail"]["label"],
            "commodity": commodity, "report_index": ri, "confidence": r["confidence"],
        })

allr = pd.DataFrame(data)
allr = allr[allr["confidence"] == "rule"]
print("\nRelations available per type:")
print(allr.groupby("relation").size().to_string())

picked = [g.sample(min(len(g), args.per_type), random_state=3) for _, g in allr.groupby("relation")]
sheet = pd.concat(picked).sort_values(["relation", "commodity"]).reset_index(drop=True)
out = ROOT / "relation_extraction/relation_precision_sheet.csv"
sheet.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\nWrote {len(sheet)} rows -> {out}")