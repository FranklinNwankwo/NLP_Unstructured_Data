"""
Assembles the full weak-supervision pipeline: EntityRuler (EQUIPMENT,
FAILURE_MODE, CONSEQUENCE, ACTION_TAKEN) + fire_service_filter + a Matcher-
based QUANTITY component, in one spaCy pipeline.

Default en_core_web_sm NER is disabled — its PERSON/ORG/etc. labels would
compete with our custom labels on the same tokens.
"""

import spacy
from collections import Counter
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

from equipment_gazetteer import build_equipment_patterns
from cause_gazetteer import build_cause_patterns
from consequence_action_gazetteer import (
    build_consequence_patterns, build_action_patterns, FIRE_SERVICE_WORDS, FIRE_SERVICE_STEMS,
)
from quantity_matcher import build_quantity_matcher, UNIT_TYPE_MAP

import re

# Scoped to the terms measure_negation.py showed were actually a problem
# (>=25% negated). Other CONSEQUENCE/ACTION_TAKEN terms measured clean and
# are left alone — a general negation model would be overkill for three words.
NEGATED_TERMS = {"fatality", "fatalities", "injury", "injuries", "leaks"}
NEGATION_WINDOW = re.compile(
    r"\b(no|not|without|never|none)\b\s*$"                    # "NO $"
    r"|\b(no|not|without|never|none)\b(\s+\S+){1,3}\s+(or|nor)\s*$",  # "NO X OR $"
    re.I,
)

@Language.component("negation_filter")
def negation_filter(doc: Doc) -> Doc:
    kept = []
    for ent in doc.ents:
        if ent.label_ == "CONSEQUENCE" and ent.text.lower() in NEGATED_TERMS:
            preceding = doc[max(0, ent.start - 5):ent.start].text
            if NEGATION_WINDOW.search(preceding):
                continue
        kept.append(ent)
    doc.ents = kept
    return doc

@Language.component("fire_service_filter")
def fire_service_filter(doc: Doc) -> Doc:
    """
    Drops CONSEQUENCE 'FIRE' spans that actually refer to the fire service
    (FIRE DEPARTMENT, FIRE DEPT., FIRE-FIGHTERS, FIRE/POLICE, ...).
    """
    kept = []
    for ent in doc.ents:
        if ent.label_ == "CONSEQUENCE" and ent.text.lower() == "fire":
            i = ent.end
            nxt = doc[i] if i < len(doc) else None
            if nxt is not None and nxt.lower_ in ("-", "/") and i + 1 < len(doc):
                nxt = doc[i + 1]
            if nxt is not None and (nxt.lower_ in FIRE_SERVICE_WORDS or nxt.lower_.startswith(FIRE_SERVICE_STEMS)):
                continue
        kept.append(ent)
    doc.ents = kept
    return doc


@Language.component("quantity_component")
def quantity_component(doc: Doc) -> Doc:
    """
    Runs the QUANTITY Matcher and merges its spans into doc.ents alongside
    what the EntityRuler already placed there. Existing ents take priority
    on overlap (filter_spans keeps longer/earlier spans).
    """
    matcher = doc._.quantity_matcher
    quantity_spans = []
    for match_id, start, end in matcher(doc):
        rule_name = doc.vocab.strings[match_id]
        span = Span(doc, start, end, label="QUANTITY")
        span._.unit_type = UNIT_TYPE_MAP[rule_name]
        quantity_spans.append(span)

    doc.ents = filter_spans(list(doc.ents) + quantity_spans)
    return doc


def build_pipeline():
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    if not Span.has_extension("unit_type"):
        Span.set_extension("unit_type", default=None)

    ruler = nlp.add_pipe(
        "entity_ruler",
        config={"phrase_matcher_attr": "LOWER"},
        before="parser" if "parser" in nlp.pipe_names else None,
    )
    equipment = build_equipment_patterns()
    cause = build_cause_patterns()
    consequence = build_consequence_patterns()
    action = build_action_patterns()
    patterns = equipment + cause + consequence + action
    ruler.add_patterns(patterns)
    print(f"EntityRuler loaded with {len(patterns)} patterns "
          f"({len(equipment)} EQUIPMENT, {len(cause)} FAILURE_MODE, "
          f"{len(consequence)} CONSEQUENCE, {len(action)} ACTION_TAKEN)")

    quantity_matcher = build_quantity_matcher(nlp)
    if not Doc.has_extension("quantity_matcher"):
        Doc.set_extension("quantity_matcher", default=quantity_matcher)

    # Order matters: filter first, so quantity overlap resolution sees the
    # already-filtered entities.
    nlp.add_pipe("fire_service_filter", last=True)
    nlp.add_pipe("negation_filter", last=True)
    nlp.add_pipe("quantity_component", last=True)

    return nlp


def run_checks(nlp):
    """(text, {(label, TEXT): expected_count}) — count 0 means must NOT appear."""
    cases = [
        ("THE FIRE DEPARTMENT ARRIVED AND EXTINGUISHED THE FIRE.",
         {("CONSEQUENCE", "FIRE"): 1, ("ACTION_TAKEN", "EXTINGUISHED"): 1}),
        ("FIRE DEPT. WAS NOTIFIED.",
         {("CONSEQUENCE", "FIRE"): 0}),
        ("FIRE-FIGHTERS ISOLATED THE LEAK.",
         {("CONSEQUENCE", "FIRE"): 0, ("ACTION_TAKEN", "ISOLATED"): 1,
          ("CONSEQUENCE", "LEAK"): 1}),
        ("THE FIRE/POLICE UNITS ARRIVED.",
         {("CONSEQUENCE", "FIRE"): 0}),
        ("THE RELEASE RESULTED IN AN EXPLOSION AND ONE FATALITY. "
         "TWO RESIDENTS WERE EVACUATED.",
         {("CONSEQUENCE", "RELEASE"): 1, ("CONSEQUENCE", "EXPLOSION"): 1,
          ("CONSEQUENCE", "FATALITY"): 1, ("CONSEQUENCE", "EVACUATED"): 1}),
        ("THE LINE WAS SHUT-IN AND BLOWN DOWN, THEN THE SECTION WAS REPLACED.",
         {("ACTION_TAKEN", "SHUT-IN"): 1, ("ACTION_TAKEN", "BLOWN DOWN"): 1,
          ("ACTION_TAKEN", "REPLACED"): 1}),
        ("REPAIR COSTS WERE $50,000 AND THE 4-INCH MAIN WAS REPAIRED.",
         {("ACTION_TAKEN", "REPAIR"): 0, ("ACTION_TAKEN", "REPAIRED"): 1,
          ("QUANTITY", "$50,000"): 1, ("QUANTITY", "4-INCH"): 1}),
        ("APPROXIMATELY 2.5 BARRELS OF CRUDE OIL WERE RELEASED FROM AN AUXILIARY "
         "VALVE. VISUAL EXAMINATION REVEALED EXTERNAL CORROSION.",
         {("QUANTITY", "2.5 BARRELS"): 1, ("EQUIPMENT", "AUXILIARY VALVE"): 1,
          ("FAILURE_MODE", "EXTERNAL CORROSION"): 1,
          ("CONSEQUENCE", "RELEASED"): 1}),
        ("THE FDNY FIRE MARSHALS AND FIRE CHIEFS ARRIVED.",
         {("CONSEQUENCE", "FIRE"): 0}),
        ("THE FIRE DEPARTMENT'S TRUCKS AND THE FIRE DEPTARTMENT ARRIVED.",
         {("CONSEQUENCE", "FIRE"): 0}),
        ("A FIRE REPORTED AT 3 AM DAMAGED THE HOUSE.",
         {("CONSEQUENCE", "FIRE"): 1}),
        ("THERE WERE NO FATALITIES OR INJURIES REPORTED. NO LEAKS WERE FOUND.",
         {("CONSEQUENCE", "FATALITIES"): 0, ("CONSEQUENCE", "INJURIES"): 0,
          ("CONSEQUENCE", "LEAKS"): 0}),
        ("THE INCIDENT RESULTED IN TWO FATALITIES AND SEVERAL INJURIES.",
         {("CONSEQUENCE", "FATALITIES"): 1, ("CONSEQUENCE", "INJURIES"): 1}),
        ("THE REPORT NOTED NO FATALITIES OR INJURIES AT THE SCENE.",
         {("CONSEQUENCE", "FATALITIES"): 0, ("CONSEQUENCE", "INJURIES"): 0}),
    ]

    failures = 0
    for text, expected in cases:
        doc = nlp(text)
        counts = Counter((e.label_, e.text.upper()) for e in doc.ents)
        ok = all(counts[key] == n for key, n in expected.items())
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {text}")
        if not ok:
            for key, n in expected.items():
                if counts[key] != n:
                    print(f"        expected {key} x{n}, got x{counts[key]}")
            print(f"        all ents: {[(e.label_, e.text) for e in doc.ents]}")
    print(f"\n{len(cases) - failures}/{len(cases)} checks passed")


if __name__ == "__main__":
    nlp = build_pipeline()
    print(f"\nFull pipeline: {nlp.pipe_names}\n")
    run_checks(nlp)