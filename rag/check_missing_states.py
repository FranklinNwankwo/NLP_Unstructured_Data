"""
Which incidents end up with no state in the RAG index, and why?
    python -m rag.check_missing_states --input data\processed\phmsa_combined_with_severity.parquet
"""
import argparse

import pandas as pd

NULLS = {"", "NONE", "NAN", "NAT", "NULL", "<NA>"}


def clean(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    return s.mask(s.str.upper().isin(NULLS) | s.isna())


def main(path: str) -> None:
    df = pd.read_parquet(path)
    df = df[clean(df["NARRATIVE"]).notna()].copy()          # same rows as the RAG index

    state = clean(df["ONSHORE_STATE_ABBREVIATION"]).fillna(clean(df["LOCATION_STATE_ABBREVIATION"]))
    missing = df[state.isna()].copy()
    print(f"Incidents with a narrative: {len(df):,}")
    print(f"No state after the RAG lookup: {len(missing):,}\n")

    print("By commodity:")
    print(missing["commodity_type"].value_counts().to_string(), "\n")

    for col in ["ON_OFF_SHORE", "OFFSHORE_STATE_ABBREVIATION", "OCS_TYPE"]:
        if col in missing.columns:
            print(f"{col}:")
            print(clean(missing[col]).fillna("(empty)").value_counts().to_string(), "\n")

    if "ON_OFF_SHORE" in missing.columns:
        onshore = missing[clean(missing["ON_OFF_SHORE"]).str.upper().str.startswith("ON", na=False)]
        print(f"Marked ONSHORE but still no state: {len(onshore)}")
        if len(onshore):
            cols = [c for c in ["commodity_type", "REPORT_NUMBER", "IYEAR", "ONSHORE_CITY_NAME",
                                "ONSHORE_COUNTY_NAME", "LOCATION_STATE_ABBREVIATION"] if c in onshore.columns]
            print(onshore[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    main(ap.parse_args().input)