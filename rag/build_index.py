"""
Build the RAG index from the extracted incident table. Run offline (Colab or local)
whenever the extracted data changes; the Streamlit app only downloads the result.

    python -m rag.build_index --input data/processed/phmsa_combined_with_severity.parquet
    python -m rag.build_index --input data/processed/phmsa_combined_with_severity.parquet --push

Outputs (in rag_index/):
    incidents.parquet   one row per incident: filter + stats table
    chunks.parquet      chunk_id, report_id, text
    embeddings.npy      float16, row-aligned with chunks.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from rag.config import (
    CAUSE_ALIASES, CHUNK_OVERLAP, CHUNK_WORDS, COLS, EMBED_MODEL, EXTRA_TEXT_COLS,
    INDEX_REPO, OUT_DIR, STAT_COLS,
)


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - overlap, step)]


def fmt_value(v) -> str:
    if isinstance(v, (list, tuple, np.ndarray)):
        return ", ".join(str(x) for x in v)
    return "" if pd.isna(v) else str(v)


NULL_STRINGS = {"", "NONE", "NAN", "NAT", "NULL", "<NA>"}


def clean_str(s: pd.Series) -> pd.Series:
    """Strip/upper-case, and turn null-like strings (the combined parquet forces
    object columns to str, so nulls arrive as 'None'/'nan') into real NaN."""
    s = s.astype("string").str.strip()
    return s.mask(s.str.upper().isin(NULL_STRINGS) | s.isna())


def load_table(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, low_memory=False)

    source_cols = []
    for v in COLS.values():
        source_cols += v if isinstance(v, list) else [v]
    needed = source_cols + EXTRA_TEXT_COLS + STAT_COLS
    missing = [c for c in dict.fromkeys(needed) if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns {missing}. Edit COLS / EXTRA_TEXT_COLS / STAT_COLS in rag/config.py")

    out = pd.DataFrame(index=df.index)
    for key, src in COLS.items():
        if isinstance(src, list):          # first non-empty value across the listed columns
            col = clean_str(df[src[0]])
            for extra in src[1:]:
                col = col.fillna(clean_str(df[extra]))
            out[key] = col
        else:
            out[key] = df[src]
    for c in dict.fromkeys(EXTRA_TEXT_COLS + STAT_COLS):
        out[c] = df[c]
    df = out

    df["narrative"] = clean_str(df["narrative"])
    df = df.dropna(subset=["narrative"]).copy()

    df["report_id"] = clean_str(df["report_id"]).str.replace(r"\.0$", "", regex=True)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for c in ("state", "system_type", "cause"):
        df[c] = (clean_str(df[c]).str.upper()
                 .str.replace("_", " ", regex=False)   # gas_distribution -> GAS DISTRIBUTION
                 .fillna("UNKNOWN"))
    df["cause"] = df["cause"].replace(CAUSE_ALIASES)

    # PHMSA numbers reports per form, so the same REPORT_NUMBER can exist in two commodities.
    # Prefix with the commodity initials (GD / GTG / HL) to make ids unique.
    prefix = df["system_type"].map(lambda s: "".join(w[0] for w in str(s).split()))
    df["report_id"] = prefix + "-" + df["report_id"]

    dupes = int(df["report_id"].duplicated().sum())
    if dupes:
        print(f"WARNING: {dupes} duplicate report ids within the same commodity dropped")
    df = df.drop_duplicates(subset="report_id").reset_index(drop=True)

    print("\nValues the RAG filters will use (check these look right):")
    print(df["system_type"].value_counts().to_string(), "\n")
    print(df["cause"].value_counts().to_string(), "\n")
    print(f"states: {df['state'].nunique()} distinct, UNKNOWN for {int((df['state'] == 'UNKNOWN').sum())} rows")
    print(f"years: {int(df['year'].min())}-{int(df['year'].max())}\n")
    return df


def build(input_path: str, push: bool) -> None:
    df = load_table(input_path)
    print(f"{len(df):,} incidents with narratives")

    chunk_rows = []
    for r in df.to_dict("records"):
        year = "" if pd.isna(r["year"]) else int(r["year"])
        header = f"Report {r['report_id']} | {r['state']} | {year} | {r['system_type']} | {r['cause']}"
        extracted = "; ".join(
            f"{c}={fmt_value(r[c])}" for c in EXTRA_TEXT_COLS if fmt_value(r[c])
        )
        if extracted:
            header += f"\nExtracted: {extracted}"
        for j, piece in enumerate(chunk_words(str(r["narrative"]), CHUNK_WORDS, CHUNK_OVERLAP)):
            chunk_rows.append({
                "chunk_id": f"{r['report_id']}_{j}",
                "report_id": r["report_id"],
                "text": f"{header}\nNarrative: {piece}",
            })

    chunks = pd.DataFrame(chunk_rows)
    print(f"{len(chunks):,} chunks; embedding with {EMBED_MODEL} ...")

    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(
        chunks["text"].tolist(), batch_size=64, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float16)

    out = Path(OUT_DIR)
    out.mkdir(exist_ok=True)
    keep = list(COLS.keys()) + list(dict.fromkeys(EXTRA_TEXT_COLS + STAT_COLS))
    df[keep].to_parquet(out / "incidents.parquet", index=False)
    chunks.to_parquet(out / "chunks.parquet", index=False)
    np.save(out / "embeddings.npy", emb)
    print(f"Saved index to {out.resolve()} ({emb.nbytes / 1e6:.1f} MB of embeddings)")

    if push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(INDEX_REPO, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=INDEX_REPO, repo_type="dataset")
        print(f"Pushed to https://huggingface.co/datasets/{INDEX_REPO}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Extracted incidents table (.parquet or .csv)")
    ap.add_argument("--push", action="store_true", help="Upload the index to the HF dataset repo")
    args = ap.parse_args()
    build(args.input, args.push)
