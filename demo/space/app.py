"""Streamlit demo: PHMSA incident narrative -> entities, relations, severity."""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from ui_helpers import (ABOUT_MD, SEVERITY_NOTE, SEVERITY_ORDER, entities_dataframe,
                        highlight_html, legend_html, relations_dataframe)

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title="PHMSA incident extraction", layout="wide")


@st.cache_resource(show_spinner="Loading models (the first start downloads about 500 MB)...")
def load_extractor():
    ner_dir, sev_dir = os.environ.get("PHMSA_NER_DIR"), os.environ.get("PHMSA_SEVERITY_DIR")
    repo_file = HERE / "model_repo.txt"
    repo = repo_file.read_text().strip() if repo_file.exists() else None
    if not (ner_dir and sev_dir):
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(repo_id=repo, allow_patterns=["ner/*", "severity/*"]))
        ner_dir, sev_dir = root / "ner", root / "severity"
    from incident_extractor import IncidentExtractor
    from predict import NERPredictor
    from severity_predict import SeverityPredictor
    return IncidentExtractor(NERPredictor(ner_dir), SeverityPredictor(sev_dir)), repo


st.title("PHMSA incident narrative extraction")
st.caption("Turns a free-text pipeline incident narrative into a structured record: "
           "entities, relations between them, and a severity prediction.")

examples = json.loads((HERE / "examples.json").read_text(encoding="utf-8"))
choice = st.selectbox("Start from an example, or choose the last option and paste your own",
                      [e["title"] for e in examples] + ["My own text"])
default = next((e["narrative"] for e in examples if e["title"] == choice), "")
text = st.text_area("Incident narrative", value=default, height=220, max_chars=8000,
                    key=f"text_{choice}")

if st.button("Extract", type="primary", disabled=not text.strip()):
    extractor, _ = load_extractor()
    with st.spinner("Running entity extraction, relations and severity..."):
        st.session_state["record"] = extractor.extract(text)

record = st.session_state.get("record")
if record:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Narrative with extracted entities")
        st.markdown(highlight_html(record["narrative"], record["entities"]), unsafe_allow_html=True)
        st.markdown(legend_html(record["entities"]), unsafe_allow_html=True)
    with right:
        st.subheader("Predicted severity")
        sev = record["severity"]
        if sev:
            st.markdown(f"### {sev['label'].upper()}")
            for label in SEVERITY_ORDER:
                p = sev["probabilities"].get(label, 0.0)
                st.write(f"{label}: {p:.0%}")
                st.progress(min(max(float(p), 0.0), 1.0))
        else:
            st.info(record.get("note", "Severity was not predicted."))
        st.caption(SEVERITY_NOTE)

    st.subheader("Relations")
    rel_df = relations_dataframe(record)
    if rel_df.empty:
        st.info("No relations found. The relation layer is rule-based and conservative: it only links "
                "entities that sit close together in one sentence.")
    else:
        st.dataframe(rel_df, hide_index=True)
        st.caption("'Hand-checked' counts come from a person reviewing 15 sampled relations of each type. "
                   "They are small samples, so treat them as rough.")

    with st.expander("All extracted entities"):
        st.dataframe(entities_dataframe(record), hide_index=True)
    if record["dropped_negated"]:
        with st.expander("Consequences dropped as negated (for example 'no injuries')"):
            st.write(", ".join(d["text"] for d in record["dropped_negated"]))
    with st.expander("Raw JSON record"):
        st.json(record)
    st.download_button("Download JSON", json.dumps(record, indent=2, ensure_ascii=False),
                       file_name="incident_record.json", mime="application/json")

with st.expander("About this demo and its limits"):
    st.markdown(ABOUT_MD)
    repo_file = HERE / "model_repo.txt"
    if repo_file.exists():
        repo_id = repo_file.read_text().strip()
        st.markdown(f"Model weights: [{repo_id}](https://huggingface.co/{repo_id})")