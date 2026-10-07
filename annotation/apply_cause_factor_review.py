import shutil
from collections import Counter
from pathlib import Path
import pandas as pd

review = pd.read_csv("annotation/cause_factor_review.csv", encoding="utf-8-sig", keep_default_na=False)
review["decision"] = review["decision"].astype(str).str.strip().str.upper().replace({"KEEP": ""})

bad = review[~review["decision"].isin({"", "FM", "CON", "DROP"})]
if len(bad):
    print("Unrecognized decisions (use FM, CON, DROP, or leave blank):")
    print(bad[["span", "decision"]])
    raise SystemExit(1)

LABEL_FOR = {"FM": "FAILURE_MODE", "CON": "CONSEQUENCE"}
decisions = {
    (int(r.report_index), int(r.start), int(r.end), str(r.span).strip()): r.decision
    for r in review.itertuples() if r.decision
}


def apply(df):
    changed = Counter()
    new_ents = []
    for _, row in df.iterrows():
        kept = []
        for e in row["entities"]:
            e = dict(e)
            if e["label"] == "CAUSE_FACTOR":
                d = decisions.get((int(row["report_index"]), int(e["start"]),
                                   int(e["end"]), e["text"].strip()))
                if d == "DROP":
                    changed["DROP"] += 1
                    continue
                if d in LABEL_FOR:
                    e["label"] = LABEL_FOR[d]
                    changed[d] += 1
            kept.append(e)
        new_ents.append(kept)
    df = df.copy()
    df["entities"] = new_ents
    df["num_entities"] = [len(k) for k in new_ents]
    return df, changed

for name in ["annotated_sentences", "ner_train", "ner_test"]:
    path = Path(f"data/processed/{name}.parquet")
    backup = path.with_name(f"{name}_v1.parquet")
    if not backup.exists():
        shutil.copy(path, backup)          # untouched original, kept forever
    df, changed = apply(pd.read_parquet(backup))   # always start from v1
    df.to_parquet(path, index=False)
    n_cf = sum(e["label"] == "CAUSE_FACTOR" for ents in df["entities"] for e in ents)
    print(f"{name}: applied {dict(changed)}  |  CAUSE_FACTOR remaining: {n_cf}")

print(f"\nNon-blank decisions in CSV: {int((review['decision'] != '').sum())} "
      f"(should equal the applied total for annotated_sentences)")