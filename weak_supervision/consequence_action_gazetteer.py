"""
CONSEQUENCE (schema 2.5) and ACTION_TAKEN (schema 2.4) gazetteers.

PROVENANCE — read this before trusting the lists:
  VERBATIM : "Valve Closure", "Operational Control" (UPSTREAM/DOWNSTREAM_
             ACTION_TAKEN checkbox text seen on the gas transmission and
             hazardous liquid forms).
  DERIVED  : everything else. field_notes.md only has COLUMN NAMES for these
             fields (FATALITY_IND, INJURY_IND, IGNITE_IND, EXPLODE_IND,
             NUM_PUB_EVACUATED, SHUTDOWN_*, HOW_EXTINGUISHED, *_IMPACT_IND),
             not checkbox text, so the terms below are plain domain vocabulary
             for those concepts, not transcribed form values.

Matching is on exact word forms (case-insensitive), so inflections are
enumerated explicitly.

DELIBERATE OMISSIONS (too ambiguous as bare words):
  "repair" (noun: REPAIR COSTS), "blowdown" (also equipment: BLOWDOWN STACK),
  "installed" (equipment history), "closed", "killed" (KILLED THE WELL),
  "fatal" (NON-FATAL), "shutdown" IS included as an action, not a consequence.

KNOWN LIMITATIONS (decide in Phase 2 annotation guidelines, not here):
  - Negated mentions ("NO INJURIES", "NO FIRE") are still tagged.
  - "FIRE" followed by a fire-service word (DEPARTMENT, DEPT, ALARM, ...) is
    removed by the fire_service_filter component in build_pipeline.py, since
    string patterns cannot express "fire, unless followed by X".
"""

# ============================================================
# CONSEQUENCE — DERIVED from FATALITY_IND, INJURY_IND, IGNITE_IND,
# EXPLODE_IND, NUM_PUB_EVACUATED, *_IMPACT_IND, release fields
# ============================================================
CONSEQUENCE_FIRE = ["Fire", "Fireball", "Flash Fire", "Ignition", "Ignited", "Burned"]
CONSEQUENCE_EXPLOSION = ["Explosion", "Exploded"]
CONSEQUENCE_FATALITY = ["Fatality", "Fatalities", "Death"]
CONSEQUENCE_INJURY = ["Injury", "Injuries", "Injured", "Hospitalized"]
CONSEQUENCE_EVACUATION = ["Evacuation", "Evacuated", "Evacuate"]
CONSEQUENCE_RELEASE = [
    "Release", "Released", "Spill", "Spilled",
    "Leak", "Leaks", "Leaked", "Leaking", "Leakage",
]
CONSEQUENCE_IMPACT = [
    "Contamination", "Contaminated", "Property Damage", "Outage", "Outages",
]

# ============================================================
# ACTION_TAKEN
# ============================================================
ACTION_VERBATIM = ["Valve Closure", "Operational Control"]  # VERBATIM
ACTION_ISOLATE = ["Isolated", "Isolate", "Isolating", "Isolation"]
ACTION_SHUT = [
    "Shut Down", "Shut-Down", "Shut Off", "Shut-Off",
    "Shut In", "Shut-In", "Shutdown", "Shutoff",
]
ACTION_DEPRESSURIZE = [
    "Blown Down", "Blow Down", "Depressurized", "Depressurised",
    "Depressurizing", "Purged", "Purging",
]
ACTION_REPAIR_FIX = [
    "Repaired", "Repairs", "Replaced", "Replacing", "Replacement",
    "Excavated", "Clamped", "Plugged", "Squeezed Off", "Capped",
    "Abandoned", "Restored", "Blocked In", "Flared",
    "Extinguished", "Extinguishing",
]
ACTION_CLEANUP = [
    "Cleaned Up", "Clean Up", "Cleanup", "Clean-Up", "Remediation", "Remediated",
]

# Words that, directly after "FIRE" (or after "FIRE-" / "FIRE/"), mean the
# fire SERVICE rather than a fire consequence. Used by fire_service_filter.
FIRE_SERVICE_WORDS = {
    "department", "departments", "dept", "dept.", "depts", "alarm", "alarms",
    "fighter", "fighters", "fighting", "marshal", "marshall", "chief",
    "captain", "crew", "crews", "truck", "trucks", "engine", "engines",
    "personnel", "official", "officials", "service", "services", "station",
    "stations", "hydrant", "hydrants", "protection", "response", "responder",
    "responders", "district", "company", "code", "extinguisher",
    "extinguishers", "suppression", "police", "ems", "rescue", "emergency",
    "battalion", "unit", "units", "command", "investigator", "investigators",
    "investigation", "inspector", "report", "reports",
}

# Prefix matches on the word after FIRE, catching plurals, possessives and
# typos: DEPARTMENT'S, DEPTARTMENT, MARSHALS, CHIEFS, FIGHTERS, ...
# Deliberately NOT prefix-matched: "report" ("FIRE REPORTED AT 3 AM" is a real
# fire), so report/reports stay exact-match in FIRE_SERVICE_WORDS above.
FIRE_SERVICE_STEMS = (
    "dept", "depart", "fighter", "fighting", "marshal", "chief", "alarm",
    "hydrant", "extinguish", "suppress", "crew", "truck", "engine",
    "personnel", "official",
)


def _build(label, groups):
    patterns = []
    for source_id, terms in groups.items():
        for term in terms:
            patterns.append({"label": label, "pattern": term, "id": source_id})
    return patterns


def build_consequence_patterns():
    return _build("CONSEQUENCE", {
        "derived_CONSEQUENCE_FIRE": CONSEQUENCE_FIRE,
        "derived_CONSEQUENCE_EXPLOSION": CONSEQUENCE_EXPLOSION,
        "derived_CONSEQUENCE_FATALITY": CONSEQUENCE_FATALITY,
        "derived_CONSEQUENCE_INJURY": CONSEQUENCE_INJURY,
        "derived_CONSEQUENCE_EVACUATION": CONSEQUENCE_EVACUATION,
        "derived_CONSEQUENCE_RELEASE": CONSEQUENCE_RELEASE,
        "derived_CONSEQUENCE_IMPACT": CONSEQUENCE_IMPACT,
    })


def build_action_patterns():
    return _build("ACTION_TAKEN", {
        "verbatim_ACTION_TAKEN_FORM_VALUES": ACTION_VERBATIM,
        "derived_ACTION_ISOLATE": ACTION_ISOLATE,
        "derived_ACTION_SHUT": ACTION_SHUT,
        "derived_ACTION_DEPRESSURIZE": ACTION_DEPRESSURIZE,
        "derived_ACTION_REPAIR_FIX": ACTION_REPAIR_FIX,
        "derived_ACTION_CLEANUP": ACTION_CLEANUP,
    })


if __name__ == "__main__":
    from collections import Counter

    for name, patterns in (("CONSEQUENCE", build_consequence_patterns()),
                           ("ACTION_TAKEN", build_action_patterns())):
        print(f"Total {name} patterns: {len(patterns)}")
        for source_id, count in Counter(p["id"] for p in patterns).items():
            print(f"  {source_id}: {count} terms")
        print()