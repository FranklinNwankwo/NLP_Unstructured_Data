"""Offline tests for the pipeline logic. Uses fake NER/severity models, so no model files are needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from incident_extractor import IncidentExtractor, is_negated


class FakeNER:
    """Returns the listed (text, label) spans for a sentence, located by string search."""
    def __init__(self, table):
        self.table = table

    def predict(self, sentences):
        out = []
        for s in sentences:
            ents = []
            for text, label in self.table.get(s, []):
                start = s.index(text)
                ents.append({"text": text, "label": label, "start": start, "end": start + len(text)})
            out.append(ents)
        return out


class FakeSeverity:
    def predict(self, texts):
        return [{"label": "minor", "probabilities": {"minor": 1.0}} for _ in texts]


S1 = "THE METER SET AT 5064 JENNIFER CIRCLE WAS DAMAGED."
S2 = "THERE WAS NO REPORTABLE INJURY."
S3 = "ONE FATALITY AND NO INJURIES WERE REPORTED."
S4 = "THE LINE WAS NOT ISOLATED UNTIL THE FIRE WAS OUT."
S5 = "THE CAUSE OF THE RELEASE IS ATTRIBUTED TO INTERNAL CORROSION."
TABLE = {
    S1: [("METER SET", "EQUIPMENT"), ("5064 JENNIFER CIRCLE", "LOCATION")],
    S2: [("INJURY", "CONSEQUENCE")],
    S3: [("FATALITY", "CONSEQUENCE"), ("INJURIES", "CONSEQUENCE")],
    S4: [("LINE", "EQUIPMENT"), ("FIRE", "CONSEQUENCE")],
    S5: [("RELEASE", "CONSEQUENCE"), ("INTERNAL", "FAILURE_MODE"), ("CORROSION", "FAILURE_MODE")],
}
ex = IncidentExtractor(FakeNER(TABLE), FakeSeverity())
failures = 0


def check(name, ok, detail=""):
    global failures
    failures += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n        {detail}" if not ok else ""))


# 1. negation rule on its own
check("negation: 'NO REPORTABLE INJURY'", is_negated(S2, S2.index("INJURY")))
check("negation: 'NO ... OR INJURIES'", is_negated(S3, S3.index("INJURIES")))
check("negation: FATALITY in the same sentence is not negated", not is_negated(S3, S3.index("FATALITY")))
check("negation: 'NOT ISOLATED UNTIL THE FIRE' keeps FIRE", not is_negated(S4, S4.index("FIRE")))
_s = "IT WAS NOT CAUSED BY FIRE."
check("negation: 'NOT CAUSED BY FIRE' drops FIRE", is_negated(_s, _s.index("FIRE")))
check("negation: comma ends the clause", not is_negated("NO LEAK WAS FOUND, BUT A FIRE STARTED.", 32))

# 2. offsets map back to the original narrative, even with leading whitespace
narr = "   " + S1 + "  " + S2 + " " + S3
rec = ex.extract(narr, "gas_distribution")
bad = [e for e in rec["entities"] if narr[e["start"]:e["end"]] != e["text"]]
check("offsets: every entity matches the narrative text", not bad and rec["entities"], str(bad))
check("offsets: dropped spans also map back",
      all(narr[d["start"]:d["end"]] == d["text"] for d in rec["dropped_negated"]) and rec["dropped_negated"])

# 3. negated consequences are removed, real ones kept
texts = [e["text"] for e in rec["entities"]]
dropped = [d["text"] for d in rec["dropped_negated"]]
check("negation: INJURY and INJURIES dropped, FATALITY kept",
      "FATALITY" in texts and "INJURY" not in texts and "INJURIES" not in texts
      and sorted(dropped) == ["INJURIES", "INJURY"], f"kept {texts}, dropped {dropped}")

# 4. relations carry absolute offsets
rel = [r for r in rec["relations"] if r["type"] == "LOCATED_AT"]
check("relations: LOCATED_AT found with absolute offsets",
      len(rel) == 1 and narr[rel[0]["head"]["start"]:rel[0]["head"]["end"]] == "METER SET"
      and narr[rel[0]["tail"]["start"]:rel[0]["tail"]["end"]] == "5064 JENNIFER CIRCLE", str(rel))

# 5. split spans are merged and linked
rec5 = ex.extract(S5)
caused = [r for r in rec5["relations"] if r["type"] == "CAUSED_BY"]
check("merge: INTERNAL + CORROSION become one span",
      any(e["text"] == "INTERNAL CORROSION" for e in rec5["entities"]), str(rec5["entities"]))
check("relations: RELEASE CAUSED_BY INTERNAL CORROSION",
      len(caused) == 1 and caused[0]["tail"]["text"] == "INTERNAL CORROSION", str(caused))

# 6. 'NOT ISOLATED UNTIL THE FIRE' keeps FIRE end to end
rec6 = ex.extract(S4 + " " + S1)
check("pipeline keeps FIRE after 'NOT ISOLATED UNTIL THE'",
      any(e["text"] == "FIRE" for e in rec6["entities"]), str(rec6["entities"]))

# 7. short narratives skip severity; batches keep their order
recs = ex.extract_many(["SEE ABOVE", S1 + " " + S1 + " " + S1], ["gas_distribution", "hazardous_liquid"])
check("short narrative: no severity, has a note", recs[0]["severity"] is None and "note" in recs[0])
check("batch order and commodity preserved",
      recs[1]["severity"]["label"] == "minor" and [r["commodity_type"] for r in recs] == ["gas_distribution", "hazardous_liquid"])
check("empty and None narratives do not crash", ex.extract_many([None, ""])[0]["entities"] == [])

# 8. records are JSON-serializable
import json
try:
    json.dumps(rec)
    ok = True
except TypeError as e:
    ok = False
check("record is JSON-serializable", ok)

# 9. severity helpers (pure python; copied out of the torch-dependent module so they can be tested here)
import re
src = (Path(__file__).resolve().parent.parent / "severity_model/severity_predict.py").read_text()
ns = {"MAX_LEN": 512}
exec(re.search(r"def head_tail.*?return ids\n", src, re.S).group(0), ns)
exec(re.search(r"def pad_batch.*?return input_ids, mask\n", src, re.S).group(0), ns)
long_ids = list(range(1000))
cut = ns["head_tail"](long_ids)
check("head_tail: 510 tokens, starts and ends like the original",
      len(cut) == 510 and cut[0] == 0 and cut[-1] == 999 and cut[254] == 254)
check("head_tail: short input untouched", ns["head_tail"]([5, 6, 7]) == [5, 6, 7])
ids, mask = ns["pad_batch"]([[1, 2, 3], [4]], 0)
check("pad_batch: padded to the longest, mask matches", ids == [[1, 2, 3], [4, 0, 0]] and mask == [[1, 1, 1], [1, 0, 0]])

print(f"\n{'ALL PASSED' if not failures else str(failures) + ' FAILED'}")