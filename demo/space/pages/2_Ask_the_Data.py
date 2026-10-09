import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

# This file lives in demo/space/pages/; the rag package lives at the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.config import EXAMPLES, EXAMPLES_CACHE  # noqa: E402
from rag.engine import Result  # noqa: E402

st.set_page_config(page_title="Ask the incident data", layout="wide")
st.title("Ask the incident data")
st.caption(
    "Questions are turned into filters over every extracted incident (exact counts), "
    "then the most relevant narratives are retrieved and summarised with citations."
)


@st.cache_data
def load_example_cache() -> dict:
    path = Path(EXAMPLES_CACHE)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_resource(show_spinner="Loading index and embedding model...")
def get_engine():
    from rag.engine import RAGEngine  # heavy import only when a live question is asked
    return RAGEngine(
        api_key=st.secrets["LLM_API_KEY"],
        model=st.secrets["LLM_MODEL"],
        base_url=st.secrets.get("LLM_BASE_URL"),
        answer_model=st.secrets.get("LLM_ANSWER_MODEL"),
    )


def use_example(text: str):
    st.session_state.q = text
    st.session_state.run = True  # answer immediately, no extra click on Ask


def render(res: Result):
    st.markdown(res.answer)

    left, right = st.columns([1, 1])
    with left:
        st.metric("Matching incidents", res.stats["matching_incidents"])
        with st.expander("Filters the question was turned into"):
            st.write(res.filters.describe())
            st.json(asdict(res.filters))
    with right:
        if res.stats.get("by_year"):
            st.bar_chart(pd.Series(res.stats["by_year"], name="incidents").sort_index())

    if not res.sources.empty:
        st.subheader("Source reports")
        st.dataframe(
            res.sources[["report_id", "score", "year", "state", "system_type", "cause"]],
            hide_index=True, use_container_width=True,
        )
        for _, r in res.sources.iterrows():
            year = int(r["year"]) if pd.notna(r["year"]) else ""
            with st.expander(f"[{r['report_id']}] {r['state']} {year}"):
                st.write(r["narrative"])


cached = load_example_cache()

cols = st.columns(len(EXAMPLES))
for c, ex in zip(cols, EXAMPLES):
    c.button(ex, on_click=use_example, args=(ex,), use_container_width=True)

st.text_input("Your question", key="q")

ask_clicked = st.button("Ask", type="primary")
run_example = st.session_state.pop("run", False)

if (ask_clicked or run_example) and st.session_state.get("q", "").strip():
    question = st.session_state.q.strip()

    if question in cached:
        render(Result.from_dict(cached[question]))
        st.caption("Pre-computed answer for this example question.")
    else:
        try:
            engine = get_engine()
            with st.spinner("Filtering, retrieving and writing the answer..."):
                res = engine.ask(question)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "quota" in msg.lower():
                st.warning(
                    "The demo has used up its free daily LLM quota. The example questions "
                    "above still work; custom questions will work again after the daily reset."
                )
            elif "503" in msg or "overloaded" in msg.lower() or "high demand" in msg.lower():
                st.warning(
                    "The Gemini model is overloaded right now (a temporary problem on Google's side). "
                    "Please try again in a minute. The example questions above still work."
                )
            else:
                st.error(f"Something went wrong: {e}")
            st.stop()
        render(res)
