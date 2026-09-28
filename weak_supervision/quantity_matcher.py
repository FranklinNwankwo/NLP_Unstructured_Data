"""
QUANTITY span detection via spaCy's token-based Matcher — number+unit
patterns rather than a gazetteer.

Units confirmed from field_notes.md release/cost/dimension fields:
  - Gas commodities: UNINTENTIONAL_RELEASE/INTENTIONAL_RELEASE in mcf
  - hazardous_liquid: *_BBLS / barrels fields
  - Pressure: ACCIDENT_PSIG, MOP_PSIG (psig)
  - Dimensions: PIPE_DIAMETER, PIPE_WALL_THICKNESS, PUNCTURE_AXIAL/CIRCUM (inches)
  - Currency: EST_COST_* fields ($)
  - Percentage: general pattern (schema 2.10 inspection findings)

DISTANCE NOTE: bare "in" is also a preposition and a state abbreviation
("HIGHWAY 75, IN GEISMAR", "$50,000 IN DAMAGES"). It is therefore only
accepted directly after a short plain number (1-3 digits, optional decimal),
never after comma-formatted or 4+ digit numbers, and never across punctuation.
"""

import spacy
from spacy.matcher import Matcher

# 1-3 digit number with optional decimal: 42, 0.25, 16.5 — plausible pipe sizes
SHORT_NUM = {"TEXT": {"REGEX": r"^\d{1,3}(\.\d+)?$"}}
UNAMBIGUOUS_LENGTH_UNITS = ["inch", "inches", "ft", "ft.", "feet", "\""]


def build_quantity_matcher(nlp):
    matcher = Matcher(nlp.vocab)

    # Volume — gas (mcf)
    matcher.add("QUANTITY_VOLUME_GAS", [
        [{"LIKE_NUM": True}, {"IS_PUNCT": True, "OP": "?"}, {"LIKE_NUM": True, "OP": "?"},
         {"LOWER": {"IN": ["mcf", "mmcf", "scf"]}}],
    ])

    # Volume — liquid (barrels, gallons)
    matcher.add("QUANTITY_VOLUME_LIQUID", [
        [{"LIKE_NUM": True}, {"IS_PUNCT": True, "OP": "?"}, {"LIKE_NUM": True, "OP": "?"},
         {"LOWER": {"IN": ["barrel", "barrels", "bbl", "bbls", "gallon", "gallons", "gal"]}}],
    ])

    # Pressure
    matcher.add("QUANTITY_PRESSURE", [
        [{"LIKE_NUM": True}, {"IS_PUNCT": True, "OP": "?"}, {"LIKE_NUM": True, "OP": "?"},
         {"LOWER": {"IN": ["psig", "psi"]}}],
    ])

    # Distance/dimension
    matcher.add("QUANTITY_DISTANCE", [
        # 1. any number + unambiguous unit: 20 FEET, 10", 4 INCHES
        [{"LIKE_NUM": True}, {"LOWER": {"IN": UNAMBIGUOUS_LENGTH_UNITS}}],
        # 2. bare "in"/"in." ONLY directly after a short plain number: 42 IN
        [SHORT_NUM, {"LOWER": {"IN": ["in", "in."]}}],
        # 3. number + separator punctuation + number + unambiguous unit: 1-1/4"
        [{"LIKE_NUM": True}, {"IS_PUNCT": True}, {"LIKE_NUM": True},
         {"LOWER": {"IN": UNAMBIGUOUS_LENGTH_UNITS}}],
        # 4. hyphenated with NO whitespace around the hyphen (SPACY: False),
        #    so "15648 - IN" cannot match: 36-INCH, EIGHT-INCH, 4-IN
        [{"LIKE_NUM": True, "SPACY": False}, {"TEXT": "-", "SPACY": False},
         {"LOWER": {"IN": ["inch", "inches"]}}],
        [{"TEXT": {"REGEX": r"^\d{1,3}(\.\d+)?$"}, "SPACY": False},
         {"TEXT": "-", "SPACY": False}, {"LOWER": "in"}],
    ])

    # Currency
    matcher.add("QUANTITY_CURRENCY", [
        [{"TEXT": "$"}, {"LIKE_NUM": True}],
        [{"LIKE_NUM": True}, {"LOWER": {"IN": ["dollars", "usd"]}}],
    ])

    # Percentage
    matcher.add("QUANTITY_PERCENTAGE", [
        [{"LIKE_NUM": True}, {"TEXT": "%"}],
    ])

    return matcher


UNIT_TYPE_MAP = {
    "QUANTITY_VOLUME_GAS": "volume_gas",
    "QUANTITY_VOLUME_LIQUID": "volume_liquid",
    "QUANTITY_PRESSURE": "pressure",
    "QUANTITY_DISTANCE": "distance",
    "QUANTITY_CURRENCY": "currency",
    "QUANTITY_PERCENTAGE": "percentage",
}


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    matcher = build_quantity_matcher(nlp)

    # (sentence, should_match_something)
    tests = [
        ("TOTAL AMOUNT OF NATURAL GAS RELEASED WAS CALCULATED TO BE 9,303 MCF.", True),
        ("5 BARRELS OF OIL WAS DISCOVERED IN THE TANK 5 CONTAINMENT AREA.", True),
        ("APPROXIMATELY 2.5 BARRELS OF CRUDE OIL WERE RELEASED.", True),
        ("A VERY SMALL LEAK WAS FOUND, APPROXIMATELY 10 GALLONS OF PROPANE.", True),
        ("ABOUT A 10\" LONG X 0.25\" CRACK WAS FOUND.", True),
        ("STRUCK A 4-INCH PLASTIC GAS DISTRIBUTION MAIN.", True),
        ("THE PRESSURE WAS INITIALLY LOWERED TO 215 PSIG.", True),
        ("THE 42 IN LINE AND THE 36 IN LINE WERE ISOLATED.", True),
        # negatives — none of these should produce a distance match
        ("ON HIGHWAY 75, IN GEISMAR, LOUISIANA.", False),
        ("DAMAGES EXCEEDED 50,000 IN REPAIR COSTS.", False),
        ("ON DECEMBER 30, 2025 IN HOUSTON.", False),
        ("REPORT 15648 - IN HOUSTON, TX.", False),
        ("STRUCK A 4-IN VALVE.", True),
    ]

    for sent, should_match in tests:
        doc = nlp(sent)
        hits = [(nlp.vocab.strings[m], doc[s:e].text) for m, s, e in matcher(doc)]
        dist_hits = [h for h in hits if h[0] == "QUANTITY_DISTANCE"]
        ok = bool(hits) if should_match else not dist_hits
        print(f"[{'PASS' if ok else 'FAIL'}] {sent}")
        for rule, text in hits:
            print(f"        {rule} -> '{text}'")