"""
Central config for the RAG layer, set up for data/processed/phmsa_combined_with_severity.parquet
(phmsa_combined_raw.parquet + severity_weak_label, written by weak_supervision/severity_weak_labels.py).

Keys on the left are the names the RAG code uses. Values on the right are column names in
your parquet. You only ever edit the right-hand side; you never rename columns in your data.
"""
from pathlib import Path

# Canonical name -> column name in the combined parquet.
# A list means "take the first non-empty value in this order": gas transmission and
# hazardous liquid store the state in ONSHORE_STATE_ABBREVIATION, gas distribution has no
# onshore/offshore split and stores it in LOCATION_STATE_ABBREVIATION.
COLS = {
    "report_id": "REPORT_NUMBER",
    "year": "IYEAR",
    "state": ["ONSHORE_STATE_ABBREVIATION", "LOCATION_STATE_ABBREVIATION"],
    "system_type": "commodity_type",
    "cause": "CAUSE",
    "narrative": "NARRATIVE",
}

# Gas distribution words two cause categories differently from the other two commodities.
# Map them onto one name so "material failure" questions cover all three commodities.
# build_index prints the final cause counts: check these two raw strings appear there
# and fix the left-hand side here if PHMSA spells them differently.
CAUSE_ALIASES = {
    "PIPE, WELD, OR JOINT FAILURE": "MATERIAL FAILURE OF PIPE OR WELD",
    "PIPE, WELD OR JOINT FAILURE": "MATERIAL FAILURE OF PIPE OR WELD",
    "OTHER ACCIDENT CAUSE": "OTHER INCIDENT CAUSE",
}

# Extra per-incident columns appended to the embedded text and shown to the LLM.
# severity_weak_label comes from PHMSA's fatality/injury/fire/explosion flags.
EXTRA_TEXT_COLS: list[str] = ["severity_weak_label"]

# Categorical columns tallied in the exact statistics block.
STAT_COLS: list[str] = ["severity_weak_label"]

# Embeddings: small, CPU-friendly, fits Streamlit Community Cloud memory.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Chunking (PHMSA narratives are usually short; long ones get split).
CHUNK_WORDS = 300
CHUNK_OVERLAP = 50

# Where the built index lives.
OUT_DIR = "rag_index"                          # local build output
INDEX_REPO = "Chinonso11/phmsa-rag-index"      # Hugging Face *dataset* repo

# Example questions shown as buttons on the demo page. Their answers are pre-computed by
#   python -m rag.cache_examples
# and saved to EXAMPLES_CACHE, so clicking them costs no LLM quota.
# Re-run cache_examples after changing this list or rebuilding the index.
EXAMPLES = [
    "What caused corrosion failures in Texas gas transmission lines since 2015?",
    "What equipment failures led to incidents in Louisiana between 2012 and 2018?",
    "How do excavation damage incidents typically happen in California?",
]
EXAMPLES_CACHE = str(Path(__file__).resolve().parent / "example_answers.json")  # rag/example_answers.json

# Retrieval / prompt budget.
MAX_REPORTS_IN_CONTEXT = 12
MAX_CHARS_PER_REPORT = 1500
