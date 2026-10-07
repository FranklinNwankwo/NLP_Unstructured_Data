"""
Offline checks for the demo (no model files, no network).

1. UI helper functions: HTML escaping, highlighting, tables.
2. The built Space folder has everything the Dockerfile and app need.
3. The Streamlit app runs end to end (select example, click Extract) with fake models.

Run build_space.py first, then:  python demo\test_demo.py
"""

import json
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "demo/space"
sys.path.insert(0, str(SPACE))

from ui_helpers import (RELATION_PRECISION, entities_dataframe, highlight_html, legend_html,
                        relations_dataframe)

failures = 0


def check(name, ok, detail=""):
    global failures
    failures += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n        {detail}" if not ok else ""))


# ---- 1. helpers ------------------------------------------------------------
text = 'COST WAS $50,000 & <b>SPILL</b>\nSECOND LINE $9'
ents = [{"text": "$50,000", "label": "QUANTITY", "start": 9, "end": 16},
        {"text": "SPILL", "label": "CONSEQUENCE", "start": text.index("SPILL"), "end": text.index("SPILL") + 5}]
h = highlight_html(text, ents)
check("highlight: '$' is escaped so markdown cannot start LaTeX", "$" not in h and "&#36;50,000" in h)
check("highlight: HTML in the narrative is escaped", "<b>" not in h and "&lt;b&gt;" in h and "&amp;" in h)
check("highlight: newline becomes <br>, no raw newline (would end the HTML block)", "\n" not in h and "<br>" in h)
check("highlight: both spans marked with their labels", h.count("<mark") == 2 and "QUANTITY" in h and "CONSEQUENCE" in h)
overlap = highlight_html("ABCDEF", [{"text": "ABCD", "label": "EQUIPMENT", "start": 0, "end": 4},
                                    {"text": "CDEF", "label": "LOCATION", "start": 2, "end": 6}])
check("highlight: overlapping span is skipped, not duplicated", overlap.count("<mark") == 1 and "EF" in overlap)
check("highlight: no entities returns plain text", "<mark" not in highlight_html("PLAIN TEXT", []))
check("legend: one chip per label present", legend_html(ents).count("<span") == 2 and legend_html([]) == "")

rec = {"entities": ents + [{"text": "SPILL", "label": "CONSEQUENCE", "start": 30, "end": 35, "sentence_index": 1}],
       "relations": [{"type": "HAS_QUANTITY", "confidence": "rule", "sentence_index": 0,
                      "head": {"text": "SPILL", "label": "CONSEQUENCE"},
                      "tail": {"text": "2 BARRELS", "label": "QUANTITY", "unit_type": "volume_liquid"}},
                     {"type": "INVOLVES_PARTY", "confidence": "low", "sentence_index": 2,
                      "head": {"text": "A", "label": "CAUSE_FACTOR"}, "tail": {"text": "B", "label": "PARTY_ROLE"}}]}
rdf = relations_dataframe(rec)
check("relations table: columns and measured counts",
      list(rdf.columns) == ["Relation", "From", "To", "Sentence", "Sample review"]
      and rdf.loc[0, "Sample review"] == "11 of 15 correct" and rdf.loc[0, "To"] == "2 BARRELS (volume_liquid)"
      and rdf.loc[1, "Sample review"] == "not measured" and rdf.loc[0, "Sentence"] == 1)
edf = entities_dataframe(rec)
check("entities table: duplicates are counted", int(edf.loc[edf["Text"] == "SPILL", "Count"].iloc[0]) == 2)
check("entities table: empty record gives an empty table", entities_dataframe({"entities": []}).empty)
check("measured counts match the precision check (61 of 90)", sum(c for c, _ in RELATION_PRECISION.values()) == 61
      and sum(n for _, n in RELATION_PRECISION.values()) == 90)

# ---- 1b. visual helpers -----------------------------------------------------
from ui_helpers import (SEVERITY_ORDER, accuracy_cards_html, entity_guide_html, hero_html,
                        probability_bars_html, relation_cards_html, severity_badge_html,
                        stat_cards_html, step_html)

check("severity badge: label uppercased for every level", all(l.upper() in severity_badge_html(l) for l in SEVERITY_ORDER))
bars = probability_bars_html({"minor": 0.1, "moderate": 0.7, "severe": 0.15, "critical": 0.05})
check("probability bars: four rows, widths match, top class bold",
      bars.count("width:70.0%") == 1 and bars.count("font-weight:700") == 1 and all(k in bars for k in SEVERITY_ORDER))
cards = relation_cards_html({"relations": [{"type": "LOCATED_AT", "confidence": "rule", "sentence_index": 0,
                                            "head": {"text": "A <b> & $5", "label": "EQUIPMENT"},
                                            "tail": {"text": "X", "label": "LOCATION"}}]})
check("relation cards: escaped, $ cannot start LaTeX, measured count shown",
      "<b>" not in cards and "&#36;5" in cards and "13 of 15 correct" in cards)
check("relation cards: empty string when there are no relations", relation_cards_html({"relations": []}) == "")
check("legend with counts shows how many of each",
      "EQUIPMENT · 2" in legend_html([{"label": "EQUIPMENT"}, {"label": "EQUIPMENT"}], with_counts=True))
blocks = [hero_html(), step_html(2, "x"), stat_cards_html([("A", 1)]), severity_badge_html("minor"), bars, cards,
          accuracy_cards_html(), entity_guide_html(), h]
check("HTML blocks have no blank line (it would end the block in Streamlit markdown)",
      all("\n\n" not in b for b in blocks))

# ---- 2. the Space folder ---------------------------------------------------
needed = ["app.py", "ui_helpers.py", "requirements.txt", "Dockerfile", "README.md", "incident_extractor.py",
          "extract_relations.py", "predict.py", "severity_predict.py", "examples.json"]
missing = [n for n in needed if not (SPACE / n).exists()]
check("space folder has every file the app needs", not missing, f"missing: {missing}")
check("README front matter selects the Docker SDK on port 8501",
      "sdk: docker" in (SPACE / "README.md").read_text(encoding="utf-8")
      and "app_port: 8501" in (SPACE / "README.md").read_text(encoding="utf-8"))
check("Dockerfile serves the app on 8501 and installs CPU torch",
      "8501" in (SPACE / "Dockerfile").read_text() and "whl/cpu" in (SPACE / "Dockerfile").read_text())
examples = json.loads((SPACE / "examples.json").read_text(encoding="utf-8"))
titles = [e["title"] for e in examples]
check("examples: at least one, titles unique, narratives non-empty",
      len(examples) >= 1 and len(set(titles)) == len(titles) and all(e["narrative"].strip() for e in examples))
if not (SPACE / "model_repo.txt").exists():
    print("[note] demo/space/model_repo.txt is missing: run demo\\upload_models.py, then build_space.py, before deploying.")

# ---- 3. the app, with fake models -----------------------------------------
from streamlit.testing.v1 import AppTest


class FakeNER:
    """Tags the first word as EQUIPMENT and the last word as LOCATION in every sentence."""
    def __init__(self, model_dir=None):
        pass

    def predict(self, sentences):
        out = []
        for s in sentences:
            words, ents, pos = s.split(), [], 0
            for i, w in enumerate(words):
                start = s.index(w, pos)
                pos = start + len(w)
                if i in (0, len(words) - 1) and len(words) > 1:
                    ents.append({"text": w, "label": "EQUIPMENT" if i == 0 else "LOCATION",
                                 "start": start, "end": start + len(w)})
            out.append(ents)
        return out


class FakeSeverity:
    def __init__(self, model_dir=None):
        pass

    def predict(self, texts):
        return [{"label": "moderate", "probabilities": {"minor": 0.1, "moderate": 0.7, "severe": 0.15, "critical": 0.05}}
                for _ in texts]


sys.modules["predict"] = types.SimpleNamespace(NERPredictor=FakeNER)
sys.modules["severity_predict"] = types.SimpleNamespace(SeverityPredictor=FakeSeverity)
tmp = tempfile.mkdtemp()
os.environ["PHMSA_NER_DIR"] = os.environ["PHMSA_SEVERITY_DIR"] = tmp

at = AppTest.from_file(str(SPACE / "app.py"), default_timeout=120).run()
check("app starts without an exception", not at.exception, str([e.value for e in at.exception]))
check("example selector lists the examples plus 'My own text'",
      list(at.selectbox[0].options) == titles + ["My own text"])
check("text box is pre-filled with the first example", at.text_area[0].value == examples[0]["narrative"])

at.button[0].click().run()
check("clicking Extract runs without an exception", not at.exception, str([e.value for e in at.exception]))
md = " ".join(m.value for m in at.markdown)
check("result shows highlighted entities", "<mark" in md and "EQUIPMENT" in md)
check("result shows the predicted severity", "MODERATE" in md and len(at.subheader) >= 2)

at.selectbox[0].select("My own text").run()
check("choosing 'My own text' empties the box", at.text_area[0].value == "" and not at.exception)
at.text_area[0].set_value("THE METER SET AT 5064 JENNIFER CIRCLE WAS DAMAGED BY FIRE.").run()
at.button[0].click().run()
check("pasting your own text works", not at.exception and "<mark" in " ".join(m.value for m in at.markdown),
      str([e.value for e in at.exception]))

print(f"\n{'ALL PASSED' if not failures else str(failures) + ' FAILED'}")
