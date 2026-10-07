"""Streamlit demo: PHMSA incident narrative -> entities, relations, severity."""

import json
import os
import time
from pathlib import Path

import streamlit as st

from ui_helpers import (ABOUT_MD, CSS, SEVERITY_NOTE, accuracy_cards_html, entities_dataframe,
                        entity_guide_html, hero_html, highlight_html, legend_html,
                        probability_bars_html, relation_cards_html, relations_dataframe,
                        severity_badge_html, stat_cards_html, step_html)

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title="PHMSA incident extraction", page_icon="\U0001F50E", layout="wide",
                   menu_items={"About": "Entities, relations and severity from PHMSA pipeline incident narratives. "
                                        "A portfolio project, not for operational, safety or regulatory decisions."})
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


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


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### How accurate is it?")
    st.markdown(accuracy_cards_html(), unsafe_allow_html=True)
    st.caption("Measured on held-out data. Limits are listed below and in the repository README.")
    st.markdown("### Entity types")
    st.markdown(entity_guide_html(), unsafe_allow_html=True)
    with st.expander("Limits and details"):
        st.markdown(ABOUT_MD)
    repo_file = HERE / "model_repo.txt"
    if repo_file.exists():
        repo_id = repo_file.read_text().strip()
        st.markdown(f"Model weights: [{repo_id}](https://huggingface.co/{repo_id})")

# ---------------------------------------------------------------- header and input
st.markdown(hero_html(), unsafe_allow_html=True)
st.markdown(step_html(1, "Choose or paste a narrative"), unsafe_allow_html=True)

examples = json.loads((HERE / "examples.json").read_text(encoding="utf-8"))
with st.container(border=True):
    choice = st.selectbox("Try an example, or pick the last option and paste your own",
                          [e["title"] for e in examples] + ["My own text"])
    default = next((e["narrative"] for e in examples if e["title"] == choice), "")
    text = st.text_area("Incident narrative", value=default, height=200, max_chars=8000,
                        key=f"text_{choice}", placeholder="Paste an incident narrative here...")
    col_button, col_hint = st.columns([1, 5])
    with col_button:
        run = st.button("Extract", type="primary", disabled=not text.strip())
    with col_hint:
        st.caption(f"{len(text.split()):,} words. Up to 8,000 characters.")

if run:
    extractor, _ = load_extractor()
    with st.spinner("Running entity extraction, relations and severity..."):
        started = time.perf_counter()
        st.session_state["record"] = extractor.extract(text)
        st.session_state["elapsed"] = time.perf_counter() - started

# ---------------------------------------------------------------- results
record = st.session_state.get("record")
if not record:
    st.caption("Results appear here after you click Extract.")
else:
    st.markdown(step_html(2, "Review the extraction"), unsafe_allow_html=True)
    st.markdown(stat_cards_html([
        ("Words", f"{len(record['narrative'].split()):,}"),
        ("Entities", len(record["entities"])),
        ("Relations", len(record["relations"])),
        ("Dropped as negated", len(record["dropped_negated"])),
        ("Time", f"{st.session_state.get('elapsed', 0.0):.1f} s"),
    ]), unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Narrative with extracted entities")
        st.markdown(highlight_html(record["narrative"], record["entities"]), unsafe_allow_html=True)
        st.markdown(legend_html(record["entities"], with_counts=True), unsafe_allow_html=True)
    with right:
        st.subheader("Predicted severity")
        sev = record["severity"]
        if sev:
            st.markdown(severity_badge_html(sev["label"]) + probability_bars_html(sev["probabilities"]),
                        unsafe_allow_html=True)
        else:
            st.info(record.get("note", "Severity was not predicted."))
        st.caption(SEVERITY_NOTE)

    st.subheader("Relations")
    cards = relation_cards_html(record)
    if cards:
        st.markdown(cards, unsafe_allow_html=True)
        st.caption("'Sample review' counts come from reviewing 15 sampled relations of each type against a "
                   "written rubric (judgments drafted by an AI assistant, not independently re-checked). "
                   "They are small samples, so treat them as rough.")
    else:
        st.info("No relations found. The relation layer is rule-based and conservative: it only links "
                "entities that sit close together in one sentence.")

    tab_table, tab_entities, tab_json = st.tabs(["Relations table", "All entities", "Raw JSON"])
    with tab_table:
        st.dataframe(relations_dataframe(record), hide_index=True)
    with tab_entities:
        st.dataframe(entities_dataframe(record), hide_index=True)
        if record["dropped_negated"]:
            st.caption("Consequences dropped as negated (for example 'no injuries'): "
                       + ", ".join(d["text"] for d in record["dropped_negated"]))
    with tab_json:
        st.json(record)
        st.download_button("Download JSON", json.dumps(record, indent=2, ensure_ascii=False),
                           file_name="incident_record.json", mime="application/json")

st.caption("Measured accuracy and limits are in the sidebar (on a phone, tap the arrow at the top left). "
           "Portfolio project built on public PHMSA incident reports. Do not paste confidential text: "
           "this app runs on a third-party host.")
