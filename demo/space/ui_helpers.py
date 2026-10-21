"""Pure helpers for the Streamlit demo (no model or Streamlit imports, so they can be unit-tested)."""

import html
from urllib.parse import quote

import pandas as pd

# Control-room palette: crude-black steel, flare orange accent, gas-line blue, marker-post yellow.
BG = "#0d1215"
PANEL = "#141c21"
PANEL_2 = "#1a242a"
LINE = "#2a3740"
INK = "#dbe2e6"
MUTED = "#8c9ba4"
FLARE = "#f28c28"
GAS = "#5fb3d9"
MARKER = "#f2c230"
DISPLAY = "'Barlow Condensed','Arial Narrow',sans-serif"
MONO = "'IBM Plex Mono',ui-monospace,Menlo,Consolas,monospace"

LABEL_COLORS = {
    "EQUIPMENT": "#8cc4e6", "FAILURE_MODE": "#f09a92", "CAUSE_FACTOR": "#d9837d",
    "ACTION_TAKEN": "#8fd6a8", "CONSEQUENCE": "#f5b77f", "QUANTITY": "#c3a9de",
    "LOCATION": "#7fd3c4", "DATE_TIME": "#a9b6c3", "MATERIAL_SPEC": "#eed676",
    "INSPECTION_FINDING": "#bcc5cb", "PARTY_ROLE": "#e49c6b", "REGULATORY_REF": "#aab7bd",
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
    "CAUSED_BY": (10, 15), "HAS_QUANTITY": (12, 15), "LOCATED_AT": (13, 15),
    "MADE_OF": (12, 15), "REMEDIATED_BY": (6, 15), "RESULTED_IN": (9, 15),
}

SEVERITY_ORDER = ["minor", "moderate", "severe", "critical"]
SEVERITY_STYLE = {                      # (background, text colour): alarm-panel colours on a dark ground
    "minor": ("#3fa66b", "#06140c"), "moderate": ("#f2c230", "#1a1404"),
    "severe": ("#f28c28", "#1a0d02"), "critical": ("#e5484d", "#ffffff"),
}

CAUSE_NOTE = (
    "Predicted from the narrative text. The training labels are PHMSA's apparent cause as chosen by the "
    "operator on the report form, not an independent root-cause investigation. The weakest category is the "
    "catch-all 'other incident cause'; 'incorrect operation' is often confused with 'equipment failure'."
)

SEVERITY_NOTE = (
    "Predicted from the narrative text. The training labels came from PHMSA's structured flags "
    "(fatality, injury requiring inpatient hospitalization, ignition, explosion), so this can "
    "differ from what the operator filed, and it says nothing about spill size or environmental damage."
)

ABOUT_MD = """
**What this is.** Three fine-tuned DistilBERT models plus a rule-based relation layer, trained on PHMSA
pipeline incident reports (gas distribution, gas transmission and gathering, hazardous liquid; 2010 to present).

**Entities** (12 types). Micro-F1 0.608 on 840 held-out spans, against labels from a single annotator.
Reasonably reliable: quantities (F1 0.85), materials (0.71), dates (0.70), equipment (0.67).
Weak: failure modes (0.45), locations (0.39), regulatory references (0.08).
Root-cause spans (`CAUSE_FACTOR`) are **not** extracted (the model scores 0 on them); the cause is predicted
for the whole narrative instead (below).

**Relations** are rules applied to the entities within one sentence. 15 sampled relations per type (90 in
total) were reviewed by hand against a written rubric, starting from AI-drafted judgments:
62 were correct (69%, 95% interval 59 to 78%). The per-type counts in the table
are small samples, so treat them as rough. `REMEDIATED_BY` is the least reliable type.

**Severity** (minor, moderate, severe, critical) is predicted from the whole narrative: macro-F1 0.777 on
1,921 held-out narratives. The labels come from structured flags, not from the text. By a rough keyword
estimate about 40% of critical narratives never mention a death, and the model detected none of those in testing.

**Cause** (PHMSA's 8 apparent-cause categories) is predicted from the whole narrative: macro-F1 0.748, accuracy
0.818 on 1,921 held-out narratives, against 0.706 for a TF-IDF baseline. Excavation damage (F1 0.92), corrosion
(0.89) and equipment failure (0.86) are reliable; the catch-all "other incident cause" is not (0.38).

**Limits.** Tested only on PHMSA incident narratives. Narratives are short, upper-case, English reports.
Not for operational, safety or regulatory decisions.
"""

# Injected once with <style>. Plain selectors only; if a selector stops matching, the page just looks plainer.
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
.stApp{background-color:#0d1215;background-image:linear-gradient(rgba(95,179,217,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(95,179,217,.045) 1px,transparent 1px);background-size:28px 28px}
.block-container{max-width:1180px;padding-top:1.4rem;padding-bottom:3rem}
footer{visibility:hidden}
header[data-testid="stHeader"]{background:transparent}
section[data-testid="stSidebar"]{background:#11181c;border-right:1px solid #2a3740}
h1,h2,h3{font-family:'Barlow Condensed','Arial Narrow',sans-serif !important;letter-spacing:.02em}
h2,h3{color:#dbe2e6;text-transform:uppercase;font-weight:600 !important}
.stButton>button[kind="primary"]{border-radius:4px;font-weight:700;padding:.45rem 1.6rem;letter-spacing:.06em;text-transform:uppercase;color:#1a0d02}
.stButton>button{border-radius:4px}
button[data-baseweb="tab"]{font-weight:600;letter-spacing:.04em;text-transform:uppercase;font-size:.85rem}
div[data-testid="stTextArea"] textarea{font-family:'IBM Plex Mono',ui-monospace,monospace;font-size:.86rem;background:#10171b;border-color:#2a3740}
div[data-testid="stExpander"] details{border-color:#2a3740;background:#141c21}
div[data-testid="stMetricValue"]{font-family:'IBM Plex Mono',ui-monospace,monospace}
"""


def _safe(text):
    """Escape for HTML and for Streamlit markdown ($ would start a LaTeX block)."""
    return html.escape(text).replace("$", "&#36;").replace("\n", "<br>")


def _chip(label, text=None):
    color = LABEL_COLORS.get(label, "#ccc")
    return (f'<span style="background:{color};color:#0d1215;padding:2px 8px;border-radius:3px;'
            f'font:600 .74rem/1.5 {MONO};letter-spacing:.02em;margin-right:6px;white-space:nowrap">'
            f'{html.escape(text or label)}</span>')

_SCHEMATIC_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 560 150' fill='none' font-family='monospace"
    "'><g stroke='#5fb3d9' stroke-width='2' opacity='.55'><path d='M0 86H560'/><path d='M330 86V128H5"
    "60'/><path d='M88 74V98M96 74V98M268 74V98M276 74V98M436 116V140M444 116V140'/><path d='M150 74L"
    "174 98V74L150 98Z' fill='#0d1215'/><path d='M380 116L404 140V116L380 140Z' fill='#0d1215'/><path"
    " d='M486 78L506 86L486 94Z' fill='#5fb3d9'/><path d='M500 120L520 128L500 136Z' fill='#5fb3d9'/>"
    "</g><g stroke='#f28c28' stroke-width='1.6' opacity='.8'><path d='M218 86V54' stroke-dasharray='3"
    " 3'/><circle cx='218' cy='36' r='18' fill='#0d1215'/><path d='M200 36H236'/><path d='M346 86V58'"
    " stroke-dasharray='3 3'/><circle cx='346' cy='40' r='18' fill='#0d1215'/><path d='M328 40H364'/>"
    "</g><g fill='#f28c28' font-size='10' text-anchor='middle' opacity='.9'><text x='218' y='32'>PI</"
    "text><text x='218' y='49'>101</text><text x='346' y='36'>FT</text><text x='346' y='53'>102</text"
    "></g><g fill='#5fb3d9' font-size='9' opacity='.6'><text x='14' y='78'>6IN-NG-101</text><text x='"
    "420' y='78'>FLOW</text></g></svg>"
)
# A P&ID-style pipe run (valves, flanges, PI/FT instrument balloons) behind the hero text. Streamlit strips
# image URLs from inline styles, so it is a stylesheet class, URL-encoded. Wide screens only, in its own column.
CSS += ("@media (min-width:1000px){.phx-schem{"
        'background-image:url("data:image/svg+xml,' + quote(_SCHEMATIC_SVG, safe="") + '");'
        "background-repeat:no-repeat;background-position:right .4rem center;background-size:40% auto;"
        "padding-right:44% !important}}")


def hero_html():
    tag = (f'<span style="display:inline-block;border:1px solid {LINE};background:{PANEL_2};color:{INK};'
           f'padding:.18rem .6rem;border-radius:3px;font:600 .72rem/1.4 {MONO};letter-spacing:.04em;'
           'margin:0 .4rem .4rem 0">{}</span>')
    tags = "".join(tag.format(t) for t in
                   ["12 ENTITY TYPES", "6 RELATION TYPES", "4 SEVERITY LEVELS", "8 CAUSE CATEGORIES",
                    "3 DISTILBERT MODELS"])
    return (f'<div style="position:relative;background:{PANEL};border:1px solid {LINE};border-radius:6px;'
            f'overflow:hidden;margin-bottom:.6rem">'
            '<div class="phx-schem" style="padding:1.4rem 1.7rem 1.1rem 1.7rem">'
            f'<div style="font:600 .72rem/1 {MONO};letter-spacing:.16em;text-transform:uppercase;color:{FLARE}">'
            'PHMSA incident reports &middot; 49 CFR Parts 191 and 195</div>'
            f'<div style="font:700 2.5rem/1.05 {DISPLAY};letter-spacing:.02em;text-transform:uppercase;color:{INK};'
            'margin:.45rem 0 .5rem 0">PHMSA Incident Extractor</div>'
            f'<div style="font-size:1rem;color:{MUTED};max-width:40rem;margin-bottom:.9rem">'
            'Paste a free-text pipeline incident narrative and get back a structured record: what failed, '
            'what was done about it, what resulted, why it happened, and how severe it was.</div>'
            f'{tags}</div>'
            f'<div style="height:6px;background:repeating-linear-gradient(-45deg,{MARKER} 0 10px,{BG} 10px 20px)"></div>'
            '</div>')

def step_html(number, title):
    """Step marker drawn as a P&ID instrument balloon: a circle split by a line, tag above, number below."""
    balloon = (f'<span style="display:inline-flex;flex-direction:column;align-items:center;justify-content:center;'
               f'width:2.5rem;height:2.5rem;border-radius:50%;border:1.6px solid {FLARE};background:{BG};'
               f'font:600 .62rem/1 {MONO};color:{FLARE}">'
               '<span style="padding-bottom:2px">STEP</span>'
               f'<span style="border-top:1.4px solid {FLARE};width:100%;text-align:center;padding-top:3px;'
               f'font-size:.74rem">{int(number):02d}</span></span>')
    return ('<div style="display:flex;align-items:center;gap:.75rem;margin:1.2rem 0 .55rem 0">'
            f'{balloon}<span style="font:600 1.45rem/1 {DISPLAY};letter-spacing:.04em;text-transform:uppercase;'
            f'color:{INK}">{html.escape(title)}</span>'
            f'<span style="flex:1;height:1px;background:{LINE};margin-left:.4rem"></span></div>')

def stat_cards_html(items):
    """items: list of (label, value) shown as a row of instrument readouts."""
    cards = "".join(
        f'<div style="flex:1;min-width:7rem;background:{PANEL};border:1px solid {LINE};border-top:2px solid {GAS};'
        f'border-radius:4px;padding:.65rem .9rem"><div style="font:600 1.45rem/1.1 {MONO};color:{INK};'
        f'font-variant-numeric:tabular-nums">{html.escape(str(value))}</div>'
        f'<div style="font:600 .66rem/1.3 {MONO};color:{MUTED};text-transform:uppercase;'
        f'letter-spacing:.1em;margin-top:.3rem">{html.escape(label)}</div></div>'
        for label, value in items)
    return f'<div style="display:flex;gap:.6rem;flex-wrap:wrap;margin:.2rem 0 .9rem 0">{cards}</div>'

def severity_badge_html(label):
    bg, fg = SEVERITY_STYLE.get(label, ("#4a5568", "#ffffff"))
    return (f'<div style="display:inline-flex;align-items:center;gap:.55rem;background:{bg};color:{fg};'
            f'padding:.38rem 1.1rem;border-radius:3px;font:700 1.15rem/1.2 {DISPLAY};letter-spacing:.12em;'
            'margin:.2rem 0 .8rem 0"><span style="width:.55rem;height:.55rem;border-radius:50%;'
            f'background:{fg};opacity:.8"></span>{html.escape(label.upper())}</div>')

def probability_bars_html(probs):
    top = max(SEVERITY_ORDER, key=lambda k: probs.get(k, 0.0))
    rows = []
    for k in SEVERITY_ORDER:
        p = min(max(float(probs.get(k, 0.0)), 0.0), 1.0)
        bg = SEVERITY_STYLE[k][0]
        weight = "700" if k == top else "400"
        rows.append(
            f'<div style="display:flex;align-items:center;gap:.6rem;margin:.32rem 0;font-weight:{weight};color:{INK}">'
            f'<span style="width:5.2rem;font:inherit;font-size:.85rem">{k}</span>'
            f'<span style="flex:1;height:.55rem;background:{PANEL_2};border:1px solid {LINE};border-radius:2px;overflow:hidden">'
            f'<span style="display:block;height:100%;width:{p * 100:.1f}%;background:{bg}"></span></span>'
            f'<span style="width:2.8rem;text-align:right;font-family:{MONO};font-size:.82rem">{p:.0%}</span></div>')
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
            f'border-bottom:1px solid {LINE}">'
            f'<span style="background:{LABEL_COLORS.get(r["head"]["label"], "#ccc")};color:{BG};padding:2px 9px;'
            f'border-radius:3px;font-size:.88rem">{_safe(r["head"]["text"])}</span>'
            f'<span style="color:{FLARE};font:600 .74rem/1 {MONO};letter-spacing:.05em">'
            f'{html.escape(r["type"])} &#8594;</span>'
            f'<span style="background:{LABEL_COLORS.get(r["tail"]["label"], "#ccc")};color:{BG};padding:2px 9px;'
            f'border-radius:3px;font-size:.88rem">{_safe(tail)}</span>'
            f'<span style="margin-left:auto;color:{MUTED};font:.72rem/1.3 {MONO}">{html.escape(meta)}</span></div>')
    if not rows:
        return ""
    rows[-1] = rows[-1].replace(f"border-bottom:1px solid {LINE}", "border-bottom:0", 1)   # no rule under the last row
    return (f'<div style="background:{PANEL};border:1px solid {LINE};border-radius:4px;padding:.3rem .8rem">'
            + "".join(rows) + "</div>")

def cause_html(pred, top=3):
    """The predicted cause as a badge, then the top classes as probability bars."""
    probs = sorted(pred["probabilities"].items(), key=lambda kv: -kv[1])[:top]
    rows = []
    for i, (label, p) in enumerate(probs):
        p = min(max(float(p), 0.0), 1.0)
        weight = "700" if i == 0 else "400"
        rows.append(
            f'<div style="display:flex;align-items:center;gap:.6rem;margin:.32rem 0;font-weight:{weight};color:{INK}">'
            f'<span style="width:11rem;font-size:.85rem">{html.escape(label.capitalize())}</span>'
            f'<span style="flex:1;height:.55rem;background:{PANEL_2};border:1px solid {LINE};border-radius:2px;overflow:hidden">'
            f'<span style="display:block;height:100%;width:{p * 100:.1f}%;background:{GAS}"></span></span>'
            f'<span style="width:2.8rem;text-align:right;font-family:{MONO};font-size:.82rem">{p:.0%}</span></div>')
    badge = (f'<div style="display:inline-block;border:1.5px solid {GAS};color:{GAS};background:{BG};'
             f'padding:.34rem 1.1rem;border-radius:3px;font:700 1.15rem/1.2 {DISPLAY};letter-spacing:.08em;'
             f'text-transform:uppercase;margin:.2rem 0 .8rem 0">{html.escape(pred["label"].capitalize())}</div>')
    return badge + "".join(rows)

def accuracy_cards_html():
    items = [("Entity tagging", "F1 0.61", "840 held-out spans"),
             ("Relations", "69% precision", "62 of 90 sampled relations"),
             ("Severity", "Macro-F1 0.78", "1,921 held-out narratives"),
             ("Cause", "Macro-F1 0.75", "1,921 held-out narratives")]
    return "".join(
        f'<div style="background:{PANEL};border:1px solid {LINE};border-left:3px solid {FLARE};border-radius:3px;'
        f'padding:.5rem .75rem;margin-bottom:.5rem">'
        f'<div style="font:600 .64rem/1.3 {MONO};color:{MUTED};text-transform:uppercase;letter-spacing:.1em">{a}</div>'
        f'<div style="font:600 1.08rem/1.3 {MONO};color:{INK}">{b}</div>'
        f'<div style="font-size:.76rem;color:{MUTED}">{c}</div></div>' for a, b, c in items)

def entity_guide_html():
    return "".join(
        f'<div style="margin:.45rem 0 .6rem 0">{_chip(label)}'
        f'<div style="font-size:.78rem;color:{MUTED};margin:.25rem 0 0 .15rem">{html.escape(desc)}</div></div>'
        for label, desc in ENTITY_HELP.items())

def highlight_html(text, entities):
    """Narrative as one HTML block with each entity span highlighted and labelled."""
    out, pos = [], 0
    for e in sorted(entities, key=lambda e: (e["start"], -e["end"])):
        if e["start"] < pos:            # overlapping span: keep the first
            continue
        color = LABEL_COLORS.get(e["label"], "#ccc")
        out.append(_safe(text[pos:e["start"]]))
        out.append(
            f'<mark style="background:{color};color:{BG};padding:1px 3px;border-radius:2px" '
            f'title="{html.escape(e["label"])}">{_safe(text[e["start"]:e["end"]])}'
            f'<sup style="font:600 0.58em {MONO};margin-left:3px;opacity:.75">{html.escape(e["label"])}</sup></mark>'
        )
        pos = e["end"]
    out.append(_safe(text[pos:]))
    return (f'<div style="line-height:2.25;font-size:0.95rem;color:{INK};background:{PANEL};border:1px solid {LINE};'
            'border-radius:4px;padding:.9rem 1.1rem;max-height:440px;overflow-y:auto">' + "".join(out) + "</div>")


def legend_html(entities, with_counts=False):
    counts = {}
    for e in entities:
        counts[e["label"]] = counts.get(e["label"], 0) + 1
    return "".join(_chip(l, f"{l} · {counts[l]}" if with_counts else None) for l in sorted(counts))


def page_header_html(eyebrow, title, lead):
    """Compact header for secondary pages, in the same style as the hero."""
    return (f'<div style="background:{PANEL};border:1px solid {LINE};border-radius:6px;overflow:hidden;margin-bottom:.8rem">'
            '<div style="padding:1.1rem 1.5rem">'
            f'<div style="font:600 .72rem/1 {MONO};letter-spacing:.16em;text-transform:uppercase;color:{FLARE}">'
            f'{html.escape(eyebrow)}</div>'
            f'<div style="font:700 2rem/1.1 {DISPLAY};letter-spacing:.02em;text-transform:uppercase;color:{INK};'
            f'margin:.4rem 0 .35rem 0">{html.escape(title)}</div>'
            f'<div style="font-size:.95rem;color:{MUTED};max-width:44rem">{html.escape(lead)}</div></div>'
            f'<div style="height:5px;background:repeating-linear-gradient(-45deg,{MARKER} 0 10px,{BG} 10px 20px)"></div>'
            '</div>')


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