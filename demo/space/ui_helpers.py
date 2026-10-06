"""Pure helpers for the Streamlit demo (no model or Streamlit imports, so they can be unit-tested)."""

import html

import pandas as pd

LABEL_COLORS = {
    "EQUIPMENT": "#9ecae9", "FAILURE_MODE": "#f4a6a0", "CAUSE_FACTOR": "#e08a85",
    "ACTION_TAKEN": "#9be0b5", "CONSEQUENCE": "#f7c59a", "QUANTITY": "#cdb4e0",
    "LOCATION": "#8fded0", "DATE_TIME": "#aeb9c6", "MATERIAL_SPEC": "#f4e08a",
    "INSPECTION_FINDING": "#c4cbd0", "PARTY_ROLE": "#eba87a", "REGULATORY_REF": "#b5c0c4",
}

# hand-checked relations per type: (correct, sampled). Small samples; see ABOUT_MD.
RELATION_PRECISION = {
    "CAUSED_BY": (10, 15), "HAS_QUANTITY": (11, 15), "LOCATED_AT": (13, 15),
    "MADE_OF": (12, 15), "REMEDIATED_BY": (6, 15), "RESULTED_IN": (9, 15),
}

SEVERITY_ORDER = ["minor", "moderate", "severe", "critical"]

SEVERITY_NOTE = (
    "Predicted from the narrative text. The training labels came from PHMSA's structured flags "
    "(fatality, injury requiring inpatient hospitalization, ignition, explosion), so this can "
    "differ from what the operator filed, and it says nothing about spill size or environmental damage."
)

ABOUT_MD = """
**What this is.** Two fine-tuned DistilBERT models plus a rule-based relation layer, trained on PHMSA
pipeline incident reports (gas distribution, gas transmission and gathering, hazardous liquid; 2010 to present).

**Entities** (12 types). Micro-F1 0.608 on 840 held-out spans, against labels from a single annotator.
Reasonably reliable: quantities (F1 0.85), materials (0.71), dates (0.70), equipment (0.67).
Weak: failure modes (0.45), locations (0.39), regulatory references (0.08).
Root causes (`CAUSE_FACTOR`) are **not** extracted: the model scores 0 on them.

**Relations** are rules applied to the entities within one sentence. A person checked 15 sampled relations
per type (90 in total): 68% were correct overall (95% interval 58 to 77%). The per-type counts shown in the
table are small samples, so treat them as rough. `REMEDIATED_BY` is the least reliable type.

**Severity** (minor, moderate, severe, critical) is predicted from the whole narrative: macro-F1 0.777 on
1,921 held-out narratives. The labels come from structured flags, not from the text. By a rough keyword
estimate about 40% of critical narratives never mention a death, and the model detected none of those in testing.

**Limits.** Tested only on PHMSA incident narratives. Narratives are short, upper-case, English reports.
Not for operational, safety or regulatory decisions.
"""


def _chip(label):
    color = LABEL_COLORS.get(label, "#ddd")
    return (f'<span style="background:{color};color:#111;padding:2px 8px;border-radius:10px;'
            f'font-size:0.8rem;margin-right:6px;white-space:nowrap">{html.escape(label)}</span>')


def _safe(text):
    """Escape for HTML and for Streamlit markdown ($ would start a LaTeX block)."""
    return html.escape(text).replace("$", "&#36;").replace("\n", "<br>")


def highlight_html(text, entities):
    """Narrative as one HTML block with each entity span highlighted and labelled."""
    out, pos = [], 0
    for e in sorted(entities, key=lambda e: (e["start"], -e["end"])):
        if e["start"] < pos:            # overlapping span: keep the first
            continue
        color = LABEL_COLORS.get(e["label"], "#ddd")
        out.append(_safe(text[pos:e["start"]]))
        out.append(
            f'<mark style="background:{color};color:#111;padding:1px 3px;border-radius:3px" '
            f'title="{html.escape(e["label"])}">{_safe(text[e["start"]:e["end"]])}'
            f'<sup style="font-size:0.6em;margin-left:2px">{html.escape(e["label"])}</sup></mark>'
        )
        pos = e["end"]
    out.append(_safe(text[pos:]))
    return '<div style="line-height:2.2;font-size:0.95rem">' + "".join(out) + "</div>"


def legend_html(entities):
    labels = sorted({e["label"] for e in entities})
    return "".join(_chip(l) for l in labels) if labels else ""


def relations_dataframe(record):
    rows = []
    for r in record["relations"]:
        tail = r["tail"]["text"] + (f" ({r['tail']['unit_type']})" if "unit_type" in r["tail"] else "")
        measured = RELATION_PRECISION.get(r["type"])
        rows.append({
            "Relation": r["type"],
            "From": r["head"]["text"],
            "To": tail,
            "Sentence": r["sentence_index"] + 1,
            "Hand-checked": f"{measured[0]} of {measured[1]} correct" if measured else "not measured",
        })
    return pd.DataFrame(rows, columns=["Relation", "From", "To", "Sentence", "Hand-checked"])


def entities_dataframe(record):
    if not record["entities"]:
        return pd.DataFrame(columns=["Label", "Text", "Count"])
    df = pd.DataFrame(record["entities"])
    out = df.groupby(["label", "text"]).size().reset_index(name="Count")
    out.columns = ["Label", "Text", "Count"]
    return out.sort_values(["Label", "Count"], ascending=[True, False]).reset_index(drop=True)