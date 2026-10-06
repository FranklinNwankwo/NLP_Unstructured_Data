import pandas as pd

for name, path in [("train", "data/processed/ner_train.parquet"),
                    ("test", "data/processed/ner_test.parquet")]:
    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        ents = sorted(row["entities"], key=lambda e: e["start"])
        for i in range(len(ents) - 1):
            a, b = ents[i], ents[i + 1]
            if a["end"] > b["start"]:  # overlap
                print(f"[{name}] report_index={row['report_index']} commodity={row['commodity_type']}")
                print(f"  sentence: {row['sentence_text']}")
                print(f"  overlap: ({a['label']}) '{a['text']}'  <->  ({b['label']}) '{b['text']}'")
                print()