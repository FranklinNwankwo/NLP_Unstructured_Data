"""Pure helpers for the Streamlit demo (no model or Streamlit imports, so they can be unit-tested)."""

import html

import pandas as pd

NAVY = "#0b3a53"
TEAL = "#0b6e99"
BORDER = "#dfe7ee"

LABEL_COLORS = {
    "EQUIPMENT": "#9ecae9", "FAILURE_MODE": "#f4a6a0", "CAUSE_FACTOR": "#e08a85",
    "ACTION_TAKEN": "#9be0b5", "CONSEQUENCE": "#f7c59a", "QUANTITY": "#cdb4e0",
    "LOCATION": "#8fded0", "DATE_TIME": "#aeb9c6", "MATERIAL_SPEC": "#f4e08a",
    "INSPECTION_FINDING": "#c4cbd0", "PARTY_ROLE": "#eba87a", "REGULATORY_REF": "#b5c0c4",
}

ENTITY_HELP = {
    "EQUIPMENT": "Pipeline components: valve, main, regulator",
    "FAILURE_MODE": "What broke and how: corrosion, rupture, leak",
    "CAUSE_FACTOR": "Root cause (rarely detected, see limits)",
    "ACTION_TAKEN": "Response after the incident: isolated, repaired",
    "CONSEQUENCE": "Outcome: fire, explosion, release, injury",
    "QUANTITY": "A number with its unit",
    "LOCATION": "Where it happened",
    "DATE_TIME": "When it happened",
    "MATERIAL_SPEC": "Materials and substances: steel, gas",
    "INSPECTION_FINDING": "What an inspection found",
    "PARTY_ROLE": "Who did what",
    "REGULATORY_REF": "Rules or procedures cited",
}

# hand-reviewed relations per type: (correct, sampled). Small samples; see ABOUT_MD.
RELATION_PRECISION = {
    "CAUSED_BY": (10, 15), "HAS_QUANTITY": (11, 15), "LOCATED_AT": (13, 15),
    "MADE_OF": (12, 15), "REMEDIATED_BY": (6, 15), "RESULTED_IN": (9, 15),
}

SEVERITY_ORDER = ["minor", "moderate", "severe", "critical"]
SEVERITY_STYLE = {                      # (background, text colour)
    "minor": ("#2f855a", "#ffffff"), "moderate": ("#f6ad55", "#1f2937"),
    "severe": ("#dd6b20", "#ffffff"), "critical": ("#c53030", "#ffffff"),
}

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

**Relations** are rules applied to the entities within one sentence. 15 sampled relations per type (90 in
total) were reviewed against a written rubric, with the judgments drafted by an AI assistant and not
independently re-checked: 61 were correct (68%, 95% interval 58 to 77%). The per-type counts in the table
are small samples, so treat them as rough. `REMEDIATED_BY` is the least reliable type.

**Severity** (minor, moderate, severe, critical) is predicted from the whole narrative: macro-F1 0.777 on
1,921 held-out narratives. The labels come from structured flags, not from the text. By a rough keyword
estimate about 40% of critical narratives never mention a death, and the model detected none of those in testing.

**Limits.** Tested only on PHMSA incident narratives. Narratives are short, upper-case, English reports.
Not for operational, safety or regulatory decisions.
"""

# Injected once with <style>. Plain selectors only; if a selector stops matching, the page just looks plainer.
CSS = """
.block-container{max-width:1180px;padding-top:1.4rem;padding-bottom:3rem}
footer{visibility:hidden}
header[data-testid="stHeader"]{background:transparent}
section[data-testid="stSidebar"]{border-right:1px solid #dfe7ee}
.stButton>button[kind="primary"]{border-radius:10px;font-weight:600;padding:.45rem 1.6rem}
button[data-baseweb="tab"]{font-weight:600}
h2,h3{color:#0b3a53}
"""


def _safe(text):
    """Escape for HTML and for Streamlit markdown ($ would start a LaTeX block)."""
    return html.escape(text).replace("$", "&#36;").replace("\n", "<br>")


def _chip(label, text=None):
    color = LABEL_COLORS.get(label, "#ddd")
    return (f'<span style="background:{color};color:#111;padding:2px 9px;border-radius:10px;'
            f'font-size:0.8rem;margin-right:6px;white-space:nowrap">{html.escape(text or label)}</span>')


def hero_html():
    pill = ('<span style="display:inline-block;background:rgba(255,255,255,.16);border:1px solid '
            'rgba(255,255,255,.3);color:#fff;padding:.2rem .8rem;border-radius:999px;font-size:.8rem;'
            'margin:0 .45rem .35rem 0">{}</span>')
    pills = "".join(pill.format(t) for t in
                    ["12 entity types", "6 relation types", "4 severity levels", "2 fine-tuned DistilBERT models"])
    return (f'<div style="background:linear-gradient(135deg,{NAVY} 0%,{TEAL} 100%);border-radius:16px;'
            'padding:1.5rem 1.8rem;color:#fff;margin-bottom:.6rem">'
            '<div style="font-size:.76rem;letter-spacing:.16em;text-transform:uppercase;opacity:.78">'
            'NLP on pipeline incident reports</div>'
            '<div style="font-size:2.05rem;font-weight:800;line-height:1.15;margin:.25rem 0 .4rem 0">'
            'PHMSA Incident Extractor</div>'
            '<div style="font-size:1.02rem;opacity:.93;max-width:44rem;margin-bottom:.9rem">'
            'Paste a free-text pipeline incident narrative and get back a structured record: what failed, '
            'what was done about it, what resulted, and how severe it was.</div>'
            f'{pills}</div>')


def step_html(number, title):
    return ('<div style="display:flex;align-items:center;gap:.6rem;margin:1.1rem 0 .5rem 0">'
            f'<span style="display:inline-flex;align-items:center;justify-content:center;width:1.7rem;'
            f'height:1.7rem;border-radius:50%;background:{TEAL};color:#fff;font-weight:700;font-size:.9rem">'
            f'{int(number)}</span>'
            f'<span style="font-size:1.2rem;font-weight:700;color:{NAVY}">{html.escape(title)}</span></div>')


def stat_cards_html(items):
    """items: list of (label, value) shown as a row of small cards."""
    cards = "".join(
        f'<div style="flex:1;min-width:7rem;background:#fff;border:1px solid {BORDER};border-radius:12px;'
        f'padding:.7rem .95rem"><div style="font-size:1.5rem;font-weight:800;color:{NAVY};line-height:1.1">'
        f'{html.escape(str(value))}</div><div style="font-size:.72rem;color:#5b6b78;text-transform:uppercase;'
        f'letter-spacing:.07em;margin-top:.2rem">{html.escape(label)}</div></div>'
        for label, value in items)
    return f'<div style="display:flex;gap:.7rem;flex-wrap:wrap;margin:.2rem 0 .9rem 0">{cards}</div>'


def severity_badge_html(label):
    bg, fg = SEVERITY_STYLE.get(label, ("#4a5568", "#ffffff"))
    return (f'<div style="display:inline-block;background:{bg};color:{fg};padding:.4rem 1.3rem;'
            'border-radius:999px;font-weight:800;letter-spacing:.08em;font-size:1.05rem;margin:.2rem 0 .8rem 0">'
            f'{html.escape(label.upper())}</div>')


def probability_bars_html(probs):
    top = max(SEVERITY_ORDER, key=lambda k: probs.get(k, 0.0))
    rows = []
    for k in SEVERITY_ORDER:
        p = min(max(float(probs.get(k, 0.0)), 0.0), 1.0)
        bg = SEVERITY_STYLE[k][0]
        weight = "700" if k == top else "400"
        rows.append(
            f'<div style="display:flex;align-items:center;gap:.6rem;margin:.3rem 0;font-weight:{weight}">'
            f'<span style="width:5.2rem;font-size:.85rem">{k}</span>'
            f'<span style="flex:1;height:.6rem;background:#e6edf3;border-radius:6px;overflow:hidden">'
            f'<span style="display:block;height:100%;width:{p * 100:.1f}%;background:{bg};border-radius:6px"></span></span>'
            f'<span style="width:2.8rem;text-align:right;font-size:.85rem">{p:.0%}</span></div>')
    return "".join(rows)


def relation_cards_html(record):
    """One row per relation: [head] TYPE -> [tail], with how reliable that type was measured to be."""
    rows = []
    for r in record["relations"]:
        tail = r["tail"]["text"] + (f" ({r['tail']['unit_type']})" if "unit_type" in r["tail"] else "")
        measured = RELATION_PRECISION.get(r["type"])
        meta = f"sample review: {measured[0]} of {measured[1]} correct" if measured else "not measured"
        if r["confidence"] == "low":
            meta += " · low confidence"
        rows.append(
            f'<div style="display:flex;align-items:center;gap:.55rem;flex-wrap:wrap;padding:.55rem .3rem;'
            f'border-bottom:1px solid {BORDER}">'
            f'<span style="background:{LABEL_COLORS.get(r["head"]["label"], "#ddd")};color:#111;padding:2px 10px;'
            f'border-radius:10px;font-size:.88rem">{_safe(r["head"]["text"])}</span>'
            f'<span style="color:{NAVY};font-weight:700;font-size:.78rem;letter-spacing:.05em">'
            f'{html.escape(r["type"])} &#8594;</span>'
            f'<span style="background:{LABEL_COLORS.get(r["tail"]["label"], "#ddd")};color:#111;padding:2px 10px;'
            f'border-radius:10px;font-size:.88rem">{_safe(tail)}</span>'
            f'<span style="margin-left:auto;color:#6b7a86;font-size:.76rem">{html.escape(meta)}</span></div>')
    if not rows:
        return ""
    rows[-1] = rows[-1].replace(f"border-bottom:1px solid {BORDER}", "border-bottom:0", 1)   # no rule under the last row
    return (f'<div style="background:#fff;border:1px solid {BORDER};border-radius:12px;padding:.3rem .8rem">'
            + "".join(rows) + "</div>")


def accuracy_cards_html():
    items = [("Entity tagging", "F1 0.61", "840 held-out spans"),
             ("Relations", "68% precision", "61 of 90 sampled relations"),
             ("Severity", "Macro-F1 0.78", "1,921 held-out narratives")]
    return "".join(
        f'<div style="background:#fff;border:1px solid {BORDER};border-radius:10px;padding:.55rem .8rem;margin-bottom:.5rem">'
        f'<div style="font-size:.72rem;color:#5b6b78;text-transform:uppercase;letter-spacing:.07em">{a}</div>'
        f'<div style="font-size:1.15rem;font-weight:800;color:{NAVY}">{b}</div>'
        f'<div style="font-size:.78rem;color:#6b7a86">{c}</div></div>' for a, b, c in items)


def entity_guide_html():
    return "".join(
        f'<div style="margin:.45rem 0 .6rem 0">{_chip(label)}'
        f'<div style="font-size:.78rem;color:#44525d;margin:.2rem 0 0 .15rem">{html.escape(desc)}</div></div>'
        for label, desc in ENTITY_HELP.items())


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
    return (f'<div style="line-height:2.2;font-size:0.95rem;background:#fff;border:1px solid {BORDER};'
            'border-radius:12px;padding:.9rem 1.1rem;max-height:440px;overflow-y:auto">' + "".join(out) + "</div>")


def legend_html(entities, with_counts=False):
    counts = {}
    for e in entities:
        counts[e["label"]] = counts.get(e["label"], 0) + 1
    return "".join(_chip(l, f"{l} · {counts[l]}" if with_counts else None) for l in sorted(counts))


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
            "Sample review": f"{measured[0]} of {measured[1]} correct" if measured else "not measured",
        })
    return pd.DataFrame(rows, columns=["Relation", "From", "To", "Sentence", "Sample review"])


def entities_dataframe(record):
    if not record["entities"]:
        return pd.DataFrame(columns=["Label", "Text", "Count"])
    df = pd.DataFrame(record["entities"])
    out = df.groupby(["label", "text"]).size().reset_index(name="Count")
    out.columns = ["Label", "Text", "Count"]
    return out.sort_values(["Label", "Count"], ascending=[True, False]).reset_index(drop=True)
