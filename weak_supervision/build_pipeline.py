"""
Assembles the full weak-supervision pipeline: EntityRuler (EQUIPMENT +
FAILURE_MODE, from equipment_gazetteer.py and cause_gazetteer.py) plus a
custom Matcher-based component (QUANTITY, from quantity_matcher.py) in one
spaCy pipeline.

Default en_core_web_sm NER is disabled — we're not using its PERSON/ORG/etc.
labels, and leaving it enabled would create competing, overlapping entity
spans on the same tokens as our custom labels.
"""

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

from equipment_gazetteer import build_equipment_patterns
from cause_gazetteer import build_cause_patterns
from quantity_matcher import build_quantity_matcher, UNIT_TYPE_MAP


@Language.component("quantity_component")
def quantity_component(doc: Doc) -> Doc:
    """
    Runs the QUANTITY Matcher and merges its spans into doc.ents alongside
    whatever EntityRuler already placed there (EQUIPMENT, FAILURE_MODE).
    Uses filter_spans to resolve overlaps — existing ents (from EntityRuler,
    which runs earlier in the pipeline) take priority over new QUANTITY
    matches when they overlap, since equipment/cause spans were already
    confirmed via verbatim/derived gazetteer terms and QUANTITY spans are
    numeric and rarely should overlap with them anyway.
    """
    matcher = doc._.quantity_matcher
    matches = matcher(doc)

    quantity_spans = []
    for match_id, start, end in matches:
        rule_name = doc.vocab.strings[match_id]
        span = Span(doc, start, end, label="QUANTITY")
        span._.unit_type = UNIT_TYPE_MAP[rule_name]
        quantity_spans.append(span)

    # Existing ents first (priority), then new quantity spans — filter_spans
    # keeps longer/earlier spans and drops overlapping shorter ones
    all_spans = list(doc.ents) + quantity_spans
    doc.ents = filter_spans(all_spans)
    return doc


def build_pipeline():
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    # Register the unit_type extension on Span, so quantity_component can
    # attach it without erroring on re-registration if this is called twice
    if not Span.has_extension("unit_type"):
        Span.set_extension("unit_type", default=None)

    # EntityRuler — EQUIPMENT + FAILURE_MODE, phrase-based
    ruler = nlp.add_pipe(
    "entity_ruler",
    config={"phrase_matcher_attr": "LOWER"},
    before="parser" if "parser" in nlp.pipe_names else None,
)
    patterns = build_equipment_patterns() + build_cause_patterns()
    ruler.add_patterns(patterns)
    print(f"EntityRuler loaded with {len(patterns)} patterns "
          f"({len(build_equipment_patterns())} EQUIPMENT, "
          f"{len(build_cause_patterns())} FAILURE_MODE)")

    # Quantity Matcher, stored as a doc extension so the pipeline component
    # can access it per-doc without rebuilding it every call
    quantity_matcher = build_quantity_matcher(nlp)
    if not Doc.has_extension("quantity_matcher"):
        Doc.set_extension("quantity_matcher", default=quantity_matcher)

    nlp.add_pipe("quantity_component", last=True)

    return nlp


if __name__ == "__main__":
    nlp = build_pipeline()
    print(f"\nFull pipeline: {nlp.pipe_names}")

    # Real narratives from your Step 11 spot-check sample — testing all
    # three entity types together, across commodities
    test_narratives = [
        "ON MAY 29, 2014 A THIRD PARTY CONTRACTOR STRUCK A 4-INCH PLASTIC "
        "GAS DISTRIBUTION MAIN WITH A BACKHOE. THE CONTRACTOR HELD A VALID "
        "USA TICKET FOR THE AREA.",

        "A 4-IN RELIEF VALVE HAD FAILED WHERE IT RELIEVED NATURAL GAS INTO "
        "ATMOSPHERE. TOTAL AMOUNT OF NATURAL GAS RELEASED WAS CALCULATED "
        "TO BE 9,303 MCF.",

        "THE RELEASE WAS DUE TO ICE THAT FORMED IN A 1.5 INCH GLOBE VALVE. "
        "THE VALVE WAS LOCATED ON A DEHYDRATION UNIT.",

        "APPROXIMATELY 2.5 BARRELS OF CRUDE OIL WERE RELEASED FROM AN "
        "AUXILIARY VALVE. VISUAL EXAMINATION REVEALED EXTERNAL CORROSION "
        "AT THE FAILURE POINT.",
    ]

    for text in test_narratives:
        doc = nlp(text)
        print(f"\n{'='*70}\n{text}\n")
        if not doc.ents:
            print("  NO ENTITIES FOUND")
        for ent in doc.ents:
            extra = f" (unit_type={ent._.unit_type})" if ent.label_ == "QUANTITY" else ""
            print(f"  [{ent.label_}] '{ent.text}'{extra}")