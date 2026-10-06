"""
Fills `correct` and `error_type` in relation_precision_sheet.csv by matching each
row on (relation, head, tail, report_index). Row order and anything pasted into
the sheet earlier do not matter: both columns are overwritten.

correct    = Y when the sentence supports the relation between those two spans
             (slightly truncated or over-long spans still count as Y).
error_type = NER  when a span itself is wrong (not an entity of that type, or a
                  negated mention that should not be tagged)
             RULE when both spans are fine but the sentence does not link them
                  that way (boilerplate "reported due to", investigative actions
                  mistaken for remediation, reversed cause/effect)
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "relation_extraction/relation_precision_sheet.csv"


def norm(s):
    return " ".join(str(s).split())


# (relation, head, tail, report_index, correct, error_type)
D = [
    # CAUSED_BY
    ("CAUSED_BY", "INCIDENT", "RELEASE", 461, "N", "RULE"),
    ("CAUSED_BY", "INCIDENT", "RELEASE", 201, "N", "RULE"),
    ("CAUSED_BY", "INCIDENT", "OUTAGE", 786, "N", "RULE"),
    ("CAUSED_BY", "INCIDENT", "DEBRIS", 834, "Y", ""),
    ("CAUSED_BY", "INCIDENT", "LEAKING", 1550, "Y", ""),
    ("CAUSED_BY", "INJURIES", "RELEASE", 371, "N", "NER"),
    ("CAUSED_BY", "INCIDENT", "RELEASE", 1961, "N", "RULE"),
    ("CAUSED_BY", "INCIDENT", "RUPTURE", 2615, "Y", ""),
    ("CAUSED_BY", "ACCIDENT", "FAILURE", 3700, "Y", ""),
    ("CAUSED_BY", "RELEASE", "INTERNAL CORROSION", 4296, "Y", ""),
    ("CAUSED_BY", "RELEASE", "INTERNAL CORROSION", 4287, "Y", ""),
    ("CAUSED_BY", "RELEASE", "LEAK", 8669, "Y", ""),
    ("CAUSED_BY", "RELEASE", "FUSION DEFECT", 5042, "Y", ""),
    ("CAUSED_BY", "SPILL", "CRACK", 9589, "Y", ""),
    ("CAUSED_BY", "RELEASE", "LEAK", 6005, "Y", ""),
    # HAS_QUANTITY
    ("HAS_QUANTITY", "DISTRIBUTION MAIN", "4-INCH", 1091, "Y", ""),
    ("HAS_QUANTITY", "TRANSMISSION COUPLING", '12"', 1261, "Y", ""),
    ("HAS_QUANTITY", "DAMAGES", "$139,700", 201, "Y", ""),
    ("HAS_QUANTITY", "PIPE", "10-FT", 262, "Y", ""),
    ("HAS_QUANTITY", "DAMAGES", "$50,000", 1091, "Y", ""),
    ("HAS_QUANTITY", "SERVICE LINE", "8-FT", 333, "Y", ""),
    ("HAS_QUANTITY", "EQUIPMENT", "6-FEET", 220, "N", "RULE"),
    ("HAS_QUANTITY", "ELECTRIC REPAIR", "7 INCHES", 255, "N", "NER"),
    ("HAS_QUANTITY", "CORE BARREL AUGER", "16'", 745, "Y", ""),
    ("HAS_QUANTITY", "PIPE", "750 PSIG", 2825, "Y", ""),
    ("HAS_QUANTITY", "ANCHORS", "4,600 FEET", 1937, "N", "RULE"),
    ("HAS_QUANTITY", "COUPON", "36-INCHES", 1878, "Y", ""),
    ("HAS_QUANTITY", "BLUNT", '10"', 1657, "N", "NER"),
    ("HAS_QUANTITY", "RELEASED", "1 BBL", 7783, "Y", ""),
    ("HAS_QUANTITY", "PIPE", "16-INCH", 4607, "Y", ""),
    # LOCATED_AT
    ("LOCATED_AT", "SERVICE LINE TUBING", "SCENIC DRIVE", 1286, "Y", ""),
    ("LOCATED_AT", "ELECTRONIC FLOW COMPUTER", "GEORGETOWN", 1110, "Y", ""),
    ("LOCATED_AT", "TRACK LOADER", "629 HWY D", 1098, "Y", ""),
    ("LOCATED_AT", "MAIN", "DAMAGE", 745, "N", "NER"),
    ("LOCATED_AT", "BELLHOLE", "1501 S. 4TH ST W ALLEY", 164, "Y", ""),
    ("LOCATED_AT", "TRAFFIC SIGNAL CONDUIT LINES", "2049 SYLVAN ROAD SW, ATLANTA", 195, "Y", ""),
    ("LOCATED_AT", "DISTRIBUTION SYSTEM", "ALICE, TEXAS", 556, "Y", ""),
    ("LOCATED_AT", "MAIN", "DOWNING STREET", 1448, "Y", ""),
    ("LOCATED_AT", "PE MAIN", "WEST JERSEY STREET NEAR UNION ST", 1417, "Y", ""),
    ("LOCATED_AT", "GRS-824", "OSWEGO, NY", 592, "Y", ""),
    ("LOCATED_AT", "GAS TRANSMISSION LINE", "CADDO PARISH LOUISIANA", 2286, "Y", ""),
    ("LOCATED_AT", "PIGGING", "STATION", 2387, "N", "NER"),
    ("LOCATED_AT", "VENT VALVE", "TRINITY", 2946, "Y", ""),
    ("LOCATED_AT", "FISHER CONTROL VALVE", "STATION", 6254, "Y", ""),
    ("LOCATED_AT", "ABSORBENT PADS", "AREA", 5908, "Y", ""),
    # MADE_OF
    ("MADE_OF", "MAIN PIPE", "PLASTIC", 612, "Y", ""),
    ("MADE_OF", "SAW", "STEEL", 1316, "N", "RULE"),
    ("MADE_OF", "LINE", "STEEL", 745, "Y", ""),
    ("MADE_OF", "MAIN", "LOW PRESSURE STEEL", 1316, "Y", ""),
    ("MADE_OF", "MAIN", "PLASTIC", 658, "Y", ""),
    ("MADE_OF", "MAIN", "PLASTIC NATURAL GAS", 456, "Y", ""),
    ("MADE_OF", "MAIN", "STEEL COATED", 1149, "Y", ""),
    ("MADE_OF", "WATER SERVICE", "STEEL", 1304, "N", "RULE"),
    ("MADE_OF", "DISTRIBUTION LINE", "STEEL FBE COATED HIGH PRESSURE", 460, "Y", ""),
    ("MADE_OF", "SERVICE LINE", "PLASTIC NATURAL GAS", 1009, "Y", ""),
    ("MADE_OF", "MAIN", "PLASTIC GAS", 174, "Y", ""),
    ("MADE_OF", "BACKHOE", "STEEL", 1094, "N", "RULE"),
    ("MADE_OF", "MAIN", "STEEL", 537, "Y", ""),
    ("MADE_OF", "LINE", "PE GAS", 340, "Y", ""),
    ("MADE_OF", "SERVICE LINE", "PLASTIC NATURAL GAS", 972, "Y", ""),
    # REMEDIATED_BY
    ("REMEDIATED_BY", "PUNCTURED", "MECHANICAL", 941, "N", "NER"),
    ("REMEDIATED_BY", "LEAK", "SURVEY", 1304, "N", "RULE"),
    ("REMEDIATED_BY", "LEAKS", "CUT", 360, "Y", ""),
    ("REMEDIATED_BY", "LEAKS", "SURVEY", 1113, "N", "RULE"),
    ("REMEDIATED_BY", "LEAKS", "SURVEYS", 26, "N", "RULE"),
    ("REMEDIATED_BY", "RELEASE", "ISOLATED", 1990, "Y", ""),
    ("REMEDIATED_BY", "LEAK", "ISOLATE", 3466, "Y", ""),
    ("REMEDIATED_BY", "MECHANICAL", "ANALYSES", 1672, "N", "RULE"),
    ("REMEDIATED_BY", "VENTING", "RECONFIGURED", 2169, "Y", ""),
    ("REMEDIATED_BY", "RUPTURE", "PRESSURED", 3359, "N", "RULE"),
    ("REMEDIATED_BY", "CRACK", "EXCAVATED", 5264, "N", "RULE"),
    ("REMEDIATED_BY", "FAILURES", "ANALYSES", 7196, "N", "RULE"),
    ("REMEDIATED_BY", "LEAKING", "REMOVED", 9254, "Y", ""),
    ("REMEDIATED_BY", "LEAK", "ISOLATING", 4054, "Y", ""),
    ("REMEDIATED_BY", "LEAKS", "CHECKED", 8113, "N", "NER"),
    # RESULTED_IN
    ("RESULTED_IN", "LEAK", "IGNITED", 1149, "Y", ""),
    ("RESULTED_IN", "RELEASE", "FIRE", 1320, "N", "RULE"),
    ("RESULTED_IN", "HEAT", "FLASH FIRE", 863, "Y", ""),
    ("RESULTED_IN", "MELTED", "DAMAGED", 433, "N", "RULE"),
    ("RESULTED_IN", "RELEASED", "IGNITED", 1286, "Y", ""),
    ("RESULTED_IN", "RELEASED", "FIRE", 1537, "Y", ""),
    ("RESULTED_IN", "ELECTRICAL SURGE", "FIRE", 1135, "Y", ""),
    ("RESULTED_IN", "LEAKING", "IGNITED", 609, "Y", ""),
    ("RESULTED_IN", "LEAKAGE", "FIRE", 3478, "Y", ""),
    ("RESULTED_IN", "BOOM", "FIREBALL", 2672, "N", "NER"),
    ("RESULTED_IN", "FLOOD", "FLASH", 3179, "N", "NER"),
    ("RESULTED_IN", "BLOWING", "DAMAGE", 2198, "N", "RULE"),
    ("RESULTED_IN", "MALFUNCTION", "DEBRIS", 2402, "N", "NER"),
    ("RESULTED_IN", "CRASHED", "RELEASE", 6300, "Y", ""),
    ("RESULTED_IN", "CORROSION", "RELEASE", 7477, "Y", ""),
]

decisions = {(r, norm(h), norm(t), int(i)): (c, e) for r, h, t, i, c, e in D}
assert len(D) == 90 and len(decisions) == 90, "decision table must hold 90 unique rows"

sheet = pd.read_csv(PATH, encoding="utf-8-sig", keep_default_na=False, dtype=str)
keys = [(r, norm(h), norm(t), int(i)) for r, h, t, i in
        zip(sheet["relation"], sheet["head"], sheet["tail"], sheet["report_index"])]

missing = [k for k in keys if k not in decisions]
unused = set(decisions) - set(keys)
if missing or unused:
    print(f"Sheet has {len(keys)} rows; {len(missing)} rows have no decision and "
          f"{len(unused)} decisions found no row. This is not the sheet these decisions were made for.")
    for k in missing[:5]:
        print("  no decision for:", k)
    raise SystemExit(1)

sheet["correct"] = [decisions[k][0] for k in keys]
sheet["error_type"] = [decisions[k][1] for k in keys]
try:
    sheet.to_csv(PATH, index=False, encoding="utf-8-sig")
except PermissionError:
    print("Cannot write the CSV. Close it in Excel (without saving) and run this again.")
    raise SystemExit(1)

print(f"Filled {len(sheet)} rows in {PATH.name}\n")
print(f"{'RELATION':<16}{'N':>4}{'Y':>4}{'NER':>5}{'RULE':>6}")
for name, g in sheet.groupby("relation"):
    print(f"{name:<16}{len(g):>4}{int((g['correct'] == 'Y').sum()):>4}"
          f"{int((g['error_type'] == 'NER').sum()):>5}{int((g['error_type'] == 'RULE').sum()):>6}")
print(f"\nOverall: {int((sheet['correct'] == 'Y').sum())}/{len(sheet)} correct")