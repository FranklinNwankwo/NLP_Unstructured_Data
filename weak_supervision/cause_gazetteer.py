"""
FAILURE_MODE gazetteer patterns, built from confirmed PHMSA cause taxonomy
across all three commodities. Every term is tagged with a confidence level:

  VERBATIM  - exact checkbox text confirmed via PDF screenshot or explicit
              prose in field_notes.md
  DERIVED   - not screenshotted, but directly derivable from a
              self-describing _IND field code
  GAP       - field code exists, no reliable checkbox text; excluded

CORRECTIONS carried across versions:
- PARTY_TYPE / VEHICLE_SUBTYPE values are PARTY_ROLE (schema 2.11), not
  FAILURE_MODE.
- Materials (Steel, Plastic, ...) are MATERIAL_SPEC (schema 2.9), not
  FAILURE_MODE.
- Component names (Relief Valve, SCADA, Block Valve, Pipe Nipple, O-Ring,
  Valve Threads, Threaded Fitting, ...) are EQUIPMENT, not FAILURE_MODE.
  They live in equipment_gazetteer.py.
- G7 CATEGORY_TYPE (Construction, Commissioning, Routine Maintenance, ...)
  describes the activity underway, not a failure, and is made of generic
  English words. Removed entirely.
"""

# ============================================================
# Top-level G1-G8 category names — VERBATIM, all three commodities
# ============================================================
CAUSE_CATEGORIES = [
    "Corrosion Failure", "Natural Force Damage", "Excavation Damage",
    "Other Outside Force Damage",
    "Pipe, Weld, or Joint Failure",        # gas_distribution's G5 name
    "Material Failure of Pipe or Weld",    # gas_transmission/hazardous_liquid's G5 name
    "Equipment Failure", "Incorrect Operation", "Other Incident Cause",
]

# ============================================================
# G1 Corrosion — VERBATIM
# ============================================================
G1_INTERNAL_EXTERNAL_SELECTOR = ["External Corrosion", "Internal Corrosion"]
G1_EXTERNAL_CORROSION = [
    "Galvanic Corrosion", "Atmospheric Corrosion", "Stray Current Corrosion",
    "Microbiological Corrosion", "Selective Seam Corrosion",
    "Localized Pitting", "General Corrosion",
]
G1_INTERNAL_CORROSION = [
    "Corrosive Commodity", "Water Drop-out", "Acid",
    "Microbiological", "Erosion",
]

# ============================================================
# G2 Natural Force Damage — VERBATIM
# ============================================================
G2_NATURAL_FORCE = [
    "Earth Movement", "Heavy Rains", "Floods", "Lightning", "Temperature",
    "High Winds", "Tree Roots", "Vegetation Roots", "Snow", "Ice Impact",
    "Ice Accumulation", "Earthquake", "Subsidence", "Landslide",
    "Washouts", "Scouring", "Flotation", "Mudslide",
    "Thermal Stress", "Frost Heave", "Frozen Components",
]

# G3 Excavation Damage — category label only; PARTY_TYPE excluded.

# ============================================================
# G4 Other Outside Force Damage — VERBATIM
# ============================================================
G4_OUTSIDE_FORCE = [
    "Nearby Industrial Fire", "Nearby Industrial Explosion",
    "Damage by Car", "Damage by Truck", "Motorized Vehicle Damage",
    "Damage by Boats", "Damage by Barges", "Damage by Drilling Rigs",
    "Maritime Equipment", "Vessels Set Adrift",
    "Fishing Activity", "Maritime Activity",
    "Electrical Arcing", "Previous Mechanical Damage",
]
G4_INTENTIONAL = ["Vandalism", "Terrorism"]

# ============================================================
# G5 Pipe/Weld/Joint Failure — DERIVED from self-describing field codes
# ============================================================
G5_PIPE_WELD_FAILURE_DERIVED = [
    "Dent", "Gouge", "Pipe Bend", "Arc Burn", "Crack", "Lack of Fusion",
    "Lamination", "Buckle", "Wrinkle", "Misalignment", "Burnt Steel",
]

# ============================================================
# G6 Equipment Failure — VERBATIM. Failure descriptors and category
# labels only; component names live in equipment_gazetteer.py.
# ============================================================
G6_CATEGORIES_GAS_DISTRIBUTION = [
    "Malfunction of Control/Relief Equipment", "Threaded Connection Failure",
    "Non-threaded Connection Failure", "Other Equipment Failure",
]
G6_CATEGORIES_GTG = [
    "Malfunction of Control/Relief Equipment",
    "Compressor or Compressor-related Equipment",
    "Threaded Connection/Coupling Failure", "Non-threaded Connection Failure",
    "Defective or Loose Tubing or Fitting",
    "Failure of Equipment Body", "Other Equipment Failure",
]
G6_CATEGORIES_HL = [
    "Malfunction of Control/Relief Equipment",
    "Pump or Pump-related Equipment",
    "Threaded Connection/Coupling Failure", "Non-threaded Connection Failure",
    "Defective or Loose Tubing or Fitting",
    "Failure of Equipment Body", "Other Equipment Failure",
]

G6_CONTROL_RELIEF_GAS_DISTRIBUTION = ["Power Failure"]
G6_CONTROL_RELIEF_GTG = ["Power Failure", "ESD System Failure"]
G6_CONTROL_RELIEF_HL = ["Power Failure", "ESD System Failure"]

G6_COMPRESSOR_GTG = [
    "Seal/Packing Failure", "Body Failure", "Crack in Body",
    "Appurtenance Failure", "Pressure Vessel Failure",
]
G6_PUMP_HL = [
    "Seal/Packing Failure", "Body Failure", "Crack in Body",
    "Appurtenance Failure",
]

G6_VALVE_GAS_DISTRIBUTION = ["Manufacturing Defect"]

# GTG-only confirmed (gas_distribution / hazardous_liquid screenshots were
# cut off before this section)
G6_ADDITIONAL_FACTORS_GTG = [
    "Excessive Vibration", "Overpressurization", "No Support",
    "Loss of Support", "Manufacturing Defect", "Loss of Electricity",
    "Improper Installation", "Improper Maintenance", "Mismatched Items",
    "Dissimilar Metals",
]

# ============================================================
# G7 Incorrect Operation — sub-cause categories, VERBATIM
# (CATEGORY_TYPE activity list removed — see docstring)
# ============================================================
G7_CATEGORIES_GAS_DISTRIBUTION = [
    "Damage by Operator", "Damage by Operator's Contractor",
    "Valve Left in Wrong Position", "Valve Placed in Wrong Position",
    "Pipeline Overpressured", "Equipment Overpressured",
    "Equipment Not Installed Properly", "Wrong Equipment Specified",
    "Wrong Equipment Installed",
]
G7_CATEGORIES_GTG = G7_CATEGORIES_GAS_DISTRIBUTION + [
    "Underground Gas Storage Overpressure", "Pressure Vessel Overpressure",
    "Cavern Overpressure",
]
G7_CATEGORIES_HL = G7_CATEGORIES_GAS_DISTRIBUTION + [
    "Tank Overfill", "Tank Overflow", "Vessel Overfill", "Vessel Overflow",
    "Sump/Separator Overfill", "Sump/Separator Overflow",
]
G7_OVERPRESSURE_DETAIL_GTG = [
    "Valve Misalignment", "Incorrect Reference Data", "Incorrect Calculation",
    "Miscommunication", "Inadequate Monitoring",
]
G7_OVERFLOW_DETAIL_HL = G7_OVERPRESSURE_DETAIL_GTG

# ============================================================
# RELEASE_TYPE (Part C, "Type of release/accident involved") — VERBATIM
# from field_notes.md RELEASE_TYPE lists and G-section screenshots.
# Bare "Leak" deliberately omitted: it names the consequence, not the
# mechanism, and appears in nearly every narrative. Decide it when the
# CONSEQUENCE gazetteer is built. "Crack" is already covered under G5.
# ============================================================
RELEASE_TYPE_TERMS = [
    "Mechanical Puncture", "Pinhole", "Rupture", "Connection Failure",
]

# G8 Other Incident Cause — free text only, no fixed vocabulary


def build_cause_patterns():
    patterns = []
    groups = {
        "all_commodities_CAUSE_CATEGORY": CAUSE_CATEGORIES,
        "all_commodities_G1_SELECTOR": G1_INTERNAL_EXTERNAL_SELECTOR,
        "gas_distribution_G1_EXTERNAL": G1_EXTERNAL_CORROSION,
        "gas_distribution_G1_INTERNAL": G1_INTERNAL_CORROSION,
        "gas_distribution_G2_NATURAL_FORCE": G2_NATURAL_FORCE,
        "gas_distribution_G4_OUTSIDE_FORCE": G4_OUTSIDE_FORCE,
        "gas_distribution_G4_INTENTIONAL": G4_INTENTIONAL,
        "gas_transmission_gathering_G5_DERIVED": G5_PIPE_WELD_FAILURE_DERIVED,

        "gas_distribution_G6_CATEGORIES": G6_CATEGORIES_GAS_DISTRIBUTION,
        "gas_distribution_G6_CONTROL_RELIEF": G6_CONTROL_RELIEF_GAS_DISTRIBUTION,
        "gas_distribution_G6_VALVE": G6_VALVE_GAS_DISTRIBUTION,

        "gas_transmission_gathering_G6_CATEGORIES": G6_CATEGORIES_GTG,
        "gas_transmission_gathering_G6_CONTROL_RELIEF": G6_CONTROL_RELIEF_GTG,
        "gas_transmission_gathering_G6_COMPRESSOR": G6_COMPRESSOR_GTG,
        "gas_transmission_gathering_G6_ADDITIONAL_FACTORS": G6_ADDITIONAL_FACTORS_GTG,

        "hazardous_liquid_G6_CATEGORIES": G6_CATEGORIES_HL,
        "hazardous_liquid_G6_CONTROL_RELIEF": G6_CONTROL_RELIEF_HL,
        "hazardous_liquid_G6_PUMP": G6_PUMP_HL,

        "gas_distribution_G7_CATEGORIES": G7_CATEGORIES_GAS_DISTRIBUTION,
        "gas_transmission_gathering_G7_CATEGORIES": G7_CATEGORIES_GTG,
        "gas_transmission_gathering_G7_OVERPRESSURE_DETAIL": G7_OVERPRESSURE_DETAIL_GTG,
        "hazardous_liquid_G7_CATEGORIES": G7_CATEGORIES_HL,
        "hazardous_liquid_G7_OVERFLOW_DETAIL": G7_OVERFLOW_DETAIL_HL,

        "all_commodities_RELEASE_TYPE": RELEASE_TYPE_TERMS,
    }

    for source_id, terms in groups.items():
        for term in terms:
            patterns.append({"label": "FAILURE_MODE", "pattern": term, "id": source_id})

    return patterns


if __name__ == "__main__":
    patterns = build_cause_patterns()
    print(f"Total FAILURE_MODE patterns: {len(patterns)}")
    unique_terms = set(p["pattern"] for p in patterns)
    print(f"Unique surface terms: {len(unique_terms)}")

    from collections import Counter
    counts = Counter(p["id"] for p in patterns)
    for source_id, count in counts.items():
        print(f"  {source_id}: {count} terms")

    print("\nKnown gaps / caveats:")
    print("  - gas_distribution's own G5 checkbox text unconfirmed; using")
    print("    gas_transmission's derived list as a stand-in")
    print("  - G6_ADDITIONAL_FACTORS confirmed for gas_transmission_gathering")
    print("    only")
    print("  - G8 Other Incident Cause has no fixed vocabulary (free text only)")
    print("  - Bare 'Leak' intentionally excluded (consequence, not mechanism)")