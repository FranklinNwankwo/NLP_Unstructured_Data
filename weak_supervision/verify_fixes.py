import pandas as pd

r = pd.read_parquet("data/processed/weak_labeled_narratives.parquet")

all_ents = [(row["commodity_type"], e) for _, row in r.iterrows() for e in row["entities"]]

# Bug 1: "N, IN" style false-positive distances (comma + the word "in")
import re
short_in = re.compile(r"^\d{1,3}(\.\d+)?\s+IN\.?$", re.I)
hyphen_in = re.compile(r"^\S+-IN\.?$", re.I)
comma_in = [e["text"] for _, e in all_ents
            if e["label"] == "QUANTITY" and e["unit_type"] == "distance"
            and re.search(r"\bIN\.?$", e["text"], re.I)
            and not short_in.match(e["text"]) and not hyphen_in.match(e["text"])]

# Bug 2 + 3: components/activity words mislabeled as FAILURE_MODE
bad_fm = [e["text"] for _, e in all_ents
          if e["label"] == "FAILURE_MODE"
          and e["text"].upper() in ("O-RING", "PIPE NIPPLE", "VALVE THREADS",
                                    "THREADED FITTING", "CONSTRUCTION", "COMMISSIONING")]

print(f"Bug 1 - comma+IN distances remaining: {len(comma_in)} {comma_in[:5]}")
print(f"Bug 2/3 - misfiled FAILURE_MODE terms remaining: {len(bad_fm)} {bad_fm[:5]}")

# Confirm the moved components now tag as EQUIPMENT instead
moved = [e["text"] for _, e in all_ents
         if e["label"] == "EQUIPMENT"
         and e["text"].upper() in ("O-RING", "PIPE NIPPLE", "VALVE THREADS", "THREADED FITTING")]
print(f"Moved components now tagged EQUIPMENT: {len(moved)}")