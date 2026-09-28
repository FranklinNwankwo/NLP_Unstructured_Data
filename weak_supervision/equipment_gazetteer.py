"""
EQUIPMENT gazetteer patterns, built from confirmed PHMSA form field values
in field_notes.md. Every term below is transcribed from a specific confirmed
value list — not paraphrased from memory. Source line/section noted per group.
"""

# gas_distribution — SYSTEM_PART_INVOLVED (Part C, Q2)
# field_notes.md line 567, verbatim full value list
GAS_DISTRIBUTION_EQUIPMENT = [
    "Main", "Main Valve", "Service Line", "Service Valve", "Service Riser",
    "Outside Meter/Regulator set", "Inside Meter/Regulator set",
    "Farm Tap Meter/Regulator set", "District Regulator/Metering Station",
]

# gas_transmission_gathering — SYSTEM_PART_INVOLVED (Part A, Q14), coarse facility type
# field_notes.md line 1322-1328, verbatim A14 value list
GTG_SYSTEM_PART = [
    "Belowground Storage", "Aboveground Storage",
    "Onshore Compressor Station", "Onshore Regulator/Metering Station",
    "Onshore Pipeline", "Offshore Platform", "Offshore Pipeline",
]

# gas_transmission_gathering — ITEM_INVOLVED (Part C, Q3), granular equipment
# field_notes.md line 1372-1401, verbatim full value list including sub-items
GTG_ITEM_INVOLVED = [
    "Pipe", "Pipe Body", "Pipe Seam",
    "Weld", "Fusion", "Girth Weld", "Plastic Fusion", "Butt Weld", "Fillet Weld",
    "Valve", "Mainline Valve", "Butterfly", "Check Valve", "Gate Valve",
    "Plug Valve", "Ball Valve", "Globe Valve", "Relief Valve", "Auxiliary Valve",
    "Compressor", "Meter", "Scraper", "Pig Trap",
    "Odorization System", "Filter", "Strainer", "Separator",
    "Dehydrator", "Drier", "Treater", "Scrubber",
    "Regulator", "Control Valve",
    "Pulsation Bottle", "Drip", "Drip Collection Device",
    "Cooler", "Heater",
    "Repair Sleeve", "Clamp", "Hot Tap Equipment",
    "Tap Fitting", "Stopple", "Thread-o-let", "Weld-o-let",
    "Flange Assembly", "Gasket", "ESD System", "Drain Lines", "Tubing",
    "Instrumentation", "Programmable Logic Controller",
    "Underground Gas Storage", "Cavern",
]

# hazardous_liquid — SYSTEM_PART_INVOLVED (Part A, Q14), coarse facility type
# field_notes.md line 2157-2165, verbatim A14 value list
HL_SYSTEM_PART = [
    "Onshore Breakout Tank", "Onshore Storage Vessel",
    "Onshore Terminal", "Onshore Tank Farm",
    "Onshore Equipment and Piping Associated with Belowground Storage",
    "Onshore Pump Station", "Onshore Meter Station",
    "Onshore Pipeline",
    "Offshore Platform", "Offshore Deepwater Port", "Offshore Pipeline",
]

# hazardous_liquid — ITEM_INVOLVED (Part C, Q3), granular equipment
# field_notes.md line 2118-2153, verbatim full value list including sub-items
HL_ITEM_INVOLVED = [
    "Pipe", "Pipe Body", "Pipe Seam",
    "Weld", "Girth Weld", "Butt Weld", "Fillet Wel d",
    "Valve", "Mainline Valve", "Butterfly", "Check Valve", "Gate Valve",
    "Plug Valve", "Ball Valve", "Globe Valve", "Relief Valve", "Auxiliary Valve",
    "Pump", "Positive Displacement Pump", "Centrifugal Pump", "Gear Pump",
    "Meter", "Prover", "Scraper", "Pig Trap", "Sump",
    "Filter", "Strainer", "Separator",
    "Repair Sleeve", "Clamp", "Tapping Equipment",
    "Tap Fitting", "Stopple", "Thread-o-ring", "Weld-o-let",
    "Flange Assembly", "Gasket", "Relief Lines", "Relief Equipment",
    "Drain Lines", "Tubing", "Instrumentation",
    "Tank", "Vessel", "Tank/Vessel",
    "Single Bottom System", "Double Bottom System", "Tank Shell", "Chime",
    "Roof Seal", "Roof Drain System", "Tank Mixer",
    "Pressure Vessel Head", "Pressure Vessel Wall", "Appurtenance",
]

# G6 "Malfunction of Control/Relief Equipment" checkbox terms — confirmed
# via PDF screenshot (G6 images), NOT from SYSTEM_PART_INVOLVED/ITEM_INVOLVED.
# These name specific control/relief components that were removed from
# cause_gazetteer.py (they're equipment, not failure modes) — added here so
# they have coverage somewhere rather than none.
GAS_DISTRIBUTION_G6_CONTROL_RELIEF_EQUIPMENT = [
    "SCADA", "Communications", "Block Valve",
    "Stopple/Control Fitting", "Pressure Regulator",
]
GTG_G6_CONTROL_RELIEF_EQUIPMENT = [
    "SCADA", "Communications", "Block Valve",
    "Stopple/Control Fitting", "Pressure Regulator",
]
# hazardous_liquid's confirmed G6 screenshot has SCADA/Communications/Block
# Valve/Stopple but no "Pressure Regulator" option — do not add it here,
# that omission is a real confirmed difference, not an oversight
HL_G6_CONTROL_RELIEF_EQUIPMENT = [
    "SCADA", "Communications", "Block Valve", "Stopple/Control Fitting",
]
# G6 threaded/non-threaded connection components — VERBATIM from G6
# screenshots. Moved here from cause_gazetteer.py: these name WHICH part
# failed (equipment), not the failure mechanism.
GAS_DISTRIBUTION_G6_CONNECTION_COMPONENTS = [
    "Pipe Nipple", "Valve Threads", "Threaded Pipe Collar", "Threaded Fitting",
    "O-Ring",
]
GTG_G6_CONNECTION_COMPONENTS = GAS_DISTRIBUTION_G6_CONNECTION_COMPONENTS + ["Mechanical Coupling"]
HL_G6_CONNECTION_COMPONENTS = GAS_DISTRIBUTION_G6_CONNECTION_COMPONENTS + ["Mechanical Coupling"]


def build_equipment_patterns():
    patterns = []
    groups = {
        "gas_distribution_SYSTEM_PART_INVOLVED": GAS_DISTRIBUTION_EQUIPMENT,
        "gas_distribution_G6_CONTROL_RELIEF": GAS_DISTRIBUTION_G6_CONTROL_RELIEF_EQUIPMENT,
        "gas_distribution_G6_CONNECTION_COMPONENTS": GAS_DISTRIBUTION_G6_CONNECTION_COMPONENTS,

        "gas_transmission_gathering_SYSTEM_PART_INVOLVED": GTG_SYSTEM_PART,
        "gas_transmission_gathering_ITEM_INVOLVED": GTG_ITEM_INVOLVED,
        "gas_transmission_gathering_G6_CONTROL_RELIEF": GTG_G6_CONTROL_RELIEF_EQUIPMENT,
        "gas_transmission_gathering_G6_CONNECTION_COMPONENTS": GTG_G6_CONNECTION_COMPONENTS,

        "hazardous_liquid_SYSTEM_PART_INVOLVED": HL_SYSTEM_PART,
        "hazardous_liquid_ITEM_INVOLVED": HL_ITEM_INVOLVED,
        "hazardous_liquid_G6_CONTROL_RELIEF": HL_G6_CONTROL_RELIEF_EQUIPMENT,
        "hazardous_liquid_G6_CONNECTION_COMPONENTS": HL_G6_CONNECTION_COMPONENTS,
        
    }

    for source_id, terms in groups.items():
        for term in terms:
            patterns.append({
                "label": "EQUIPMENT",
                "pattern": term,
                "id": source_id,
            })

    return patterns


if __name__ == "__main__":
    patterns = build_equipment_patterns()
    print(f"Total EQUIPMENT patterns: {len(patterns)}")
    unique_terms = set(p["pattern"] for p in patterns)
    print(f"Unique surface terms: {len(unique_terms)}")

    from collections import Counter
    counts = Counter(p["id"] for p in patterns)
    for source_id, count in counts.items():
        print(f"  {source_id}: {count} terms")