import pandas as pd

df = pd.read_parquet("data/processed/annotated_sentences.parquet")

TEST_SHARE = 0.15
RANDOM_STATE = 42

train_parts, test_parts = [], []
for commodity in df["commodity_type"].unique():
    sub = df[df["commodity_type"] == commodity].sample(frac=1, random_state=RANDOM_STATE)
    n_test = int(len(sub) * TEST_SHARE)
    test_parts.append(sub.iloc[:n_test])
    train_parts.append(sub.iloc[n_test:])

train_df = pd.concat(train_parts).reset_index(drop=True)
test_df = pd.concat(test_parts).reset_index(drop=True)

train_df.to_parquet("data/processed/ner_train.parquet", index=False)
test_df.to_parquet("data/processed/ner_test.parquet", index=False)

print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")
print("\nTrain by commodity:")
print(train_df["commodity_type"].value_counts())
print("\nTest by commodity:")
print(test_df["commodity_type"].value_counts())