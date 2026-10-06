"""
Rule-based relation extraction (Phase 5), per schema section 4, applied within
a single sentence over NER output (spans with char offsets).

Design notes
- Same-sentence only: NER is trained and scored per sentence, and same-sentence
  links are the highest-precision option.
- Proximity = number of whitespace-separated words between the two spans.
- normalize_entities() merges adjacent same-label spans the model split in two
  (e.g. "INTERNAL" + "CORROSION"). It only affects relation/demo output, never
  the NER evaluation.
- LOCATED_AT needs a locative preposition between equipment and location.
- MADE_OF only accepts construction materials. MATERIAL_SPEC in the labels also
  covers substances (gas, crude oil), which are not what a component is made of.
- CAUSED_BY (consequence -> failure mode) needs an explicit cue phrase
  ("due to", "caused by", ...). CAUSED_BY (failure mode -> cause factor) is kept
  but is low-confidence: the NER model does not emit CAUSE_FACTOR (test F1 ~0).
- Generic tails (INCIDENT) and investigative actions (EXAMINATION) are not
  accepted as consequences/remediation.
- HAS_QUANTITY also accepts EQUIPMENT as head and checks unit compatibility;
  this extends schema section 4 slightly (record as schema v0.2).
"""

import re

CAUSE_CUES = ("due to", "caused by", "attributed to", "result of",
              "resulting from", "because of")
LOCATIVE_PREPS = {"at", "near", "on", "in", "along", "off", "by"}
GENERIC_CONSEQUENCES = {"incident", "event", "accident"}
INVESTIGATIVE_ACTIONS = {
    "examination", "examined", "investigation", "investigated", "inspection",
    "inspected", "notified", "notification", "reported", "review", "reviewed",
    "analysis", "analyses", "testing", "tested", "survey", "surveys", "checked",
}
# "...WAS REPORTED TO DOT DUE TO GAS RELEASE" explains why it was reported, not
# what caused the incident. A reporting word between the spans blocks CAUSED_BY.
REPORTING_WORDS = {"reported", "notified"}

# Construction materials: values from the PHMSA forms (plastic types, steel,
# stainless steel, copper, cast/wrought/ductile iron) plus abbreviations seen
# in narratives (HDPE, POLY).
_MATERIAL_RE = re.compile(
    r"\b(steel|plastic|polyethylene|polypropylene|polybutylene|polyamide|pvc|pex|"
    r"pe|abs|hdpe|poly|copper|iron)\b", re.I)


def is_construction_material(text):
    return bool(_MATERIAL_RE.search(text))


RULES = [
    {"name": "CAUSED_BY", "head": ("FAILURE_MODE",), "tail": ("CAUSE_FACTOR",),
     "pick": "head", "max_gap": 12, "confidence": "low"},
    {"name": "CAUSED_BY", "head": ("CONSEQUENCE",), "tail": ("FAILURE_MODE",),
     "pick": "head", "max_gap": 10, "head_first": True, "cues": CAUSE_CUES,
     "blockers": REPORTING_WORDS, "confidence": "rule"},
    {"name": "RESULTED_IN", "head": ("FAILURE_MODE",), "tail": ("CONSEQUENCE",),
     "pick": "head", "max_gap": 12, "tail_stop": GENERIC_CONSEQUENCES,
     "confidence": "rule"},
    {"name": "REMEDIATED_BY", "head": ("FAILURE_MODE",), "tail": ("ACTION_TAKEN",),
     "pick": "head", "max_gap": 15, "tail_stop": INVESTIGATIVE_ACTIONS,
     "confidence": "rule"},
    {"name": "LOCATED_AT", "head": ("EQUIPMENT",), "tail": ("LOCATION",),
     "pick": "head", "max_gap": 6, "head_first": True,
     "connector": LOCATIVE_PREPS, "confidence": "rule"},
    {"name": "HAS_QUANTITY", "head": ("CONSEQUENCE", "INSPECTION_FINDING", "EQUIPMENT"),
     "tail": ("QUANTITY",), "pick": "tail", "max_gap": 4, "confidence": "rule"},
    {"name": "INVOLVES_PARTY", "head": ("CAUSE_FACTOR",), "tail": ("PARTY_ROLE",),
     "pick": "head", "max_gap": 10, "confidence": "low"},
    {"name": "MADE_OF", "head": ("EQUIPMENT",), "tail": ("MATERIAL_SPEC",),
     "pick": "head", "max_gap": 4, "tail_ok": is_construction_material,
     "confidence": "rule"},
]

_UNIT_PATTERNS = [
    ("currency", re.compile(r"\$|\bdollars?\b|\busd\b", re.I)),
    ("percentage", re.compile(r"%")),
    ("volume_gas", re.compile(r"\b(mcf|mmcf|scf)\b", re.I)),
    ("volume_liquid", re.compile(r"\b(barrels?|bbls?|gallons?|gal)\b", re.I)),
    ("pressure", re.compile(r"\bpsig?\b", re.I)),
    ("distance", re.compile(r"\b(inch|inches|in|ft|feet)\b|\"|'", re.I)),
]

QUANTITY_COMPAT = {
    "CONSEQUENCE": {"volume_gas", "volume_liquid", "currency", "percentage"},
    "EQUIPMENT": {"distance", "pressure"},
    "INSPECTION_FINDING": None,
}


def classify_quantity(text):
    for kind, pattern in _UNIT_PATTERNS:
        if pattern.search(text):
            return kind
    return "unknown"


def _compatible(head, tail):
    allowed = QUANTITY_COMPAT.get(head["label"])
    return allowed is None or tail.get("unit_type") in allowed


def normalize_entities(sentence, entities):
    """Merge adjacent same-label spans separated only by whitespace or a hyphen."""
    ents = sorted((dict(e) for e in entities), key=lambda e: e["start"])
    merged = []
    for e in ents:
        if merged:
            p = merged[-1]
            if (p["label"] == e["label"] and e["label"] != "QUANTITY"
                    and sentence[p["end"]:e["start"]].strip(" -") == ""):
                p["end"] = e["end"]
                p["text"] = sentence[p["start"]:p["end"]]
                continue
        merged.append(e)
    return merged


def _between(sentence, a, b):
    left, right = (a, b) if a["start"] <= b["start"] else (b, a)
    return sentence[left["end"]:right["start"]]


def _gap(sentence, a, b):
    return len(_between(sentence, a, b).split())


def _accept(rule, sentence, head, tail):
    if head["start"] < tail["end"] and head["end"] > tail["start"]:
        return False
    if rule.get("head_first") and head["start"] >= tail["start"]:
        return False
    between = _between(sentence, head, tail).lower()
    if "connector" in rule and not (set(re.findall(r"[a-z0-9']+", between)) & rule["connector"]):
        return False
    if "cues" in rule and not any(cue in between for cue in rule["cues"]):
        return False
    if "blockers" in rule and set(re.findall(r"[a-z0-9']+", between)) & rule["blockers"]:
        return False
    if tail["text"].strip().lower() in rule.get("tail_stop", ()):
        return False
    check = rule.get("tail_ok")
    if check and not check(tail["text"]):
        return False
    if rule["name"] == "HAS_QUANTITY" and not _compatible(head, tail):
        return False
    return True


def extract_relations(sentence, entities, sentence_index=0):
    ents = normalize_entities(sentence, entities)
    for e in ents:
        if e["label"] == "QUANTITY":
            e["unit_type"] = classify_quantity(e["text"])

    by_label = {}
    for e in ents:
        by_label.setdefault(e["label"], []).append(e)

    rels = []
    for rule in RULES:
        heads = [e for lab in rule["head"] for e in by_label.get(lab, [])]
        tails = [e for lab in rule["tail"] for e in by_label.get(lab, [])]
        if not heads or not tails:
            continue

        pairs = []
        if rule["pick"] == "head":      # each head picks its nearest valid tail
            for h in heads:
                best = None
                for t in tails:
                    if not _accept(rule, sentence, h, t):
                        continue
                    g = _gap(sentence, h, t)
                    if g <= rule["max_gap"] and (best is None or g < best[1]):
                        best = (t, g)
                if best:
                    pairs.append((h, best[0], best[1]))
        else:                           # each tail picks its nearest valid head
            for t in tails:
                best = None
                for h in heads:
                    if not _accept(rule, sentence, h, t):
                        continue
                    g = _gap(sentence, h, t)
                    if g <= rule["max_gap"] and (best is None or g < best[1]):
                        best = (h, g)
                if best:
                    pairs.append((best[0], t, best[1]))

        for h, t, g in pairs:
            rels.append({"relation": rule["name"], "head": h, "tail": t, "gap": g,
                         "confidence": rule["confidence"], "sentence_index": sentence_index})

    # a cue-based CAUSED_BY already says the same thing as RESULTED_IN reversed
    caused = {(r["head"]["start"], r["tail"]["start"]) for r in rels
              if r["relation"] == "CAUSED_BY" and r["head"]["label"] == "CONSEQUENCE"}
    return [r for r in rels
            if not (r["relation"] == "RESULTED_IN"
                    and (r["tail"]["start"], r["head"]["start"]) in caused)]


def dedupe(relations):
    seen, out = set(), []
    for r in relations:
        key = (r["relation"], r["head"]["text"].lower(), r["tail"]["text"].lower())
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def describe(r):
    tail = r["tail"]["text"]
    if "unit_type" in r["tail"]:
        tail += f" ({r['tail']['unit_type']})"
    flag = "  (low confidence)" if r["confidence"] == "low" else ""
    return f'{r["head"]["text"]} [{r["head"]["label"]}] --{r["relation"]}--> {tail} [{r["tail"]["label"]}]{flag}'


if __name__ == "__main__":
    def build(sentence, specs):
        return [{"text": t, "label": lab, "start": sentence.index(t),
                 "end": sentence.index(t) + len(t)} for t, lab in specs]

    tests = [
        ("THE 4-INCH PLASTIC MAIN LEAKED 2.5 BARRELS DUE TO EXTERNAL CORROSION.",
         [("4-INCH", "QUANTITY"), ("PLASTIC", "MATERIAL_SPEC"), ("MAIN", "EQUIPMENT"),
          ("LEAKED", "CONSEQUENCE"), ("2.5 BARRELS", "QUANTITY"),
          ("EXTERNAL CORROSION", "FAILURE_MODE")],
         {("HAS_QUANTITY", "MAIN", "4-INCH"), ("HAS_QUANTITY", "LEAKED", "2.5 BARRELS"),
          ("MADE_OF", "MAIN", "PLASTIC"), ("CAUSED_BY", "LEAKED", "EXTERNAL CORROSION")}),
        ("THE METER SET AT 5064 JENNIFER CIRCLE WAS DAMAGED.",
         [("METER SET", "EQUIPMENT"), ("5064 JENNIFER CIRCLE", "LOCATION")],
         {("LOCATED_AT", "METER SET", "5064 JENNIFER CIRCLE")}),
        ("THE CRACK WAS DUE TO FROST HEAVE.",
         [("CRACK", "FAILURE_MODE"), ("FROST HEAVE", "CAUSE_FACTOR")],
         {("CAUSED_BY", "CRACK", "FROST HEAVE")}),
        ("THE METER WAS REPLACED AT A COST OF $50,000.",
         [("METER", "EQUIPMENT"), ("REPLACED", "ACTION_TAKEN"), ("$50,000", "QUANTITY")],
         set()),
        ("THE CAUSE OF THE RELEASE IS ATTRIBUTED TO INTERNAL CORROSION.",
         [("RELEASE", "CONSEQUENCE"), ("INTERNAL", "FAILURE_MODE"), ("CORROSION", "FAILURE_MODE")],
         {("CAUSED_BY", "RELEASE", "INTERNAL CORROSION")}),
        ("THE GAS MAIN WAS MADE OF STEEL.",
         [("GAS", "MATERIAL_SPEC"), ("MAIN", "EQUIPMENT"), ("STEEL", "MATERIAL_SPEC")],
         {("MADE_OF", "MAIN", "STEEL")}),
        ("THE GRASS STRIP, A SIDEWALK AND THE METER SET.",
         [("GRASS STRIP", "EQUIPMENT"), ("SIDEWALK", "LOCATION"), ("METER SET", "EQUIPMENT")],
         set()),
        ("CORROSION RESULTED IN THE INCIDENT AND EXAMINATION FOLLOWED.",
         [("CORROSION", "FAILURE_MODE"), ("INCIDENT", "CONSEQUENCE"), ("EXAMINATION", "ACTION_TAKEN")],
         set()),
        ("THE METER SETTING AND BUILDING AT 3050 LAKECREST CIRCLE.",
         [("METER SETTING", "EQUIPMENT"), ("BUILDING", "LOCATION"), ("3050 LAKECREST CIRCLE", "LOCATION")],
         {("LOCATED_AT", "METER SETTING", "3050 LAKECREST CIRCLE")}),
        ("THIS INCIDENT WAS REPORTED TO DOT DUE TO GAS RELEASE.",
         [("INCIDENT", "CONSEQUENCE"), ("RELEASE", "FAILURE_MODE")],
         set()),
        ("A LEAK SURVEY WAS COMPLETED AFTER THE LEAK WAS ISOLATED.",
         [("LEAK", "FAILURE_MODE"), ("SURVEY", "ACTION_TAKEN"), ("ISOLATED", "ACTION_TAKEN")],
         {("REMEDIATED_BY", "LEAK", "ISOLATED")}),
    ]

    failures = 0
    for sentence, specs, expected in tests:
        rels = extract_relations(sentence, build(sentence, specs))
        got = {(r["relation"], r["head"]["text"], r["tail"]["text"]) for r in rels}
        ok = got == expected
        if sentence.endswith("FROST HEAVE."):
            ok = ok and all(r["confidence"] == "low" for r in rels)
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {sentence}")
        if not ok:
            print(f"        expected: {sorted(expected)}")
            print(f"        got:      {sorted(got)}")
    print(f"\n{len(tests) - failures}/{len(tests)} checks passed")