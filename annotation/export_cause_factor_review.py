import pandas as pd

df = pd.read_parquet("data/processed/annotated_sentences.parquet")

rows = []
for _, row in df.iterrows():
    for e in row["entities"]:
        if e["label"] == "CAUSE_FACTOR":
            rows.append({
                "report_index": int(row["report_index"]),
                "start": int(e["start"]),
                "end": int(e["end"]),
                "commodity": row["commodity_type"],
                "span": e["text"].strip(),
                "sentence": row["sentence_text"],
                "decision": "",
            })

out = pd.DataFrame(rows).sort_values(["span", "report_index"]).reset_index(drop=True)
out.to_csv("annotation/cause_factor_review.csv", index=False, encoding="utf-8-sig")
print(f"{len(out)} CAUSE_FACTOR spans -> annotation/cause_factor_review.csv")