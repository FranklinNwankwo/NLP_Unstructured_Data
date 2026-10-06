# PHMSA Incident Extraction — Annotation Guidelines

## Correcting pre-annotations, not drawing from scratch
Every sentence has weak-label pre-annotations for EQUIPMENT, FAILURE_MODE,
CONSEQUENCE, ACTION_TAKEN, and QUANTITY. Fix what's wrong, add what's
missing, delete what shouldn't be there. CAUSE_FACTOR, LOCATION, DATE_TIME,
MATERIAL_SPEC, INSPECTION_FINDING, PARTY_ROLE, and REGULATORY_REF have no
pre-annotations — tag these from scratch wherever they appear.

## EQUIPMENT vs FAILURE_MODE — the core boundary rule
EQUIPMENT names WHICH component is involved. FAILURE_MODE describes WHAT
physically happened to it. "The **relief valve** [EQUIPMENT] **failed**
[not tagged — verb, not a span] due to **external corrosion**
[FAILURE_MODE]." If you're unsure which one a term is, ask: does this name
a physical part (EQUIPMENT), or does it describe a mechanism/defect
(FAILURE_MODE)? Materials (steel, plastic, cast iron) are neither — tag
them MATERIAL_SPEC.

## FAILURE_MODE vs CAUSE_FACTOR
FAILURE_MODE is the mechanism — what broke and how ("external corrosion",
"crack in girth weld"). CAUSE_FACTOR is the root cause — why it happened
("inadequate cathodic protection", "third-party excavation damage").
"The pipe **cracked** [FAILURE_MODE] due to **frost heave** [CAUSE_FACTOR]."

## Negation — do not tag negated mentions
"There were **no fatalities** or **injuries**" → do not tag "fatalities" or
"injuries" as CONSEQUENCE. The pre-annotations sometimes get this wrong
(weak labels miss most negation) — delete any pre-tagged span that is
actually being negated in context. This applies to all entity types, not
just CONSEQUENCE.

## "Fire" — service vs. actual fire
Tag CONSEQUENCE only when "fire" describes an actual fire at the incident.
Do not tag it when it's naming the fire service: "fire department", "fire
dept.", "fire marshal", "fire chief", "fire crew" are not a CONSEQUENCE
mention. "A **fire** [CONSEQUENCE] started" vs. "the **fire department**
[not tagged] responded".

## Corrosion — internal vs. external
The form makes operators choose one or the other per incident; a narrative
describing corrosion should get either "internal corrosion" or "external
corrosion" tagged as FAILURE_MODE, not both, unless the narrative
genuinely describes two separate corrosion findings.

## Excavation damage — party role, not failure mode
"Third-party contractor" causing excavation damage is PARTY_ROLE, not
FAILURE_MODE. The category label "Excavation Damage" itself is
FAILURE_MODE; who did the digging is PARTY_ROLE.

## Terms the weak labels intentionally under-tag
"Service" was narrowed out of the EQUIPMENT gazetteer after QA found it was
80% false positives in the general-English sense ("restored service to
customers"). It's still a real EQUIPMENT term when it means the physical
service line ("cut 1/2" service") — tag it when you see the equipment
sense, even though the pre-annotation won't have caught it. Same applies
to "roof" (usually a building, occasionally a tank roof) and "mixer"
(usually a concrete mixer, occasionally a tank mixer) — judge by context,
since these words were removed from the gazetteer specifically because
they're ambiguous, not because the equipment sense doesn't exist.

## Quantities
Tag the full number+unit span as one QUANTITY ("2.5 barrels", "9,303 MCF"),
never split. Bare numbers with no unit are not QUANTITY.

## CONSEQUENCE 

This the outcome, not the triggering event. This is what #88 got wrong — "the truck struck the house" is the initiating event, not a consequence. CONSEQUENCE means what resulted (fire, injury, spill, demolition). An event like a vehicle strike, a structural failure, or a weld crack that caused the incident belongs under CAUSE_FACTOR or FAILURE_MODE, not CONSEQUENCE. This was the recurring mistake across your batch — worth rereading before your next session.

## CAUSE_FACTOR

A stated cause under active investigation isn't a CAUSE_FACTOR yet. #89 had "source of the leak" tagged CAUSE_FACTOR while the sentence said the cause was still under investigation. If the narrative explicitly says the cause is unknown or pending, don't force a CAUSE_FACTOR tag onto the phrase describing the search for it.

## INVESTIGATION

"Investigation" as ACTION_TAKEN — judgment call, pick one and stick to it. Your schema's ACTION_TAKEN examples (isolate, replace, repair, reduce pressure) are physical remediation. Whether administrative activity like "an investigation is ongoing" counts is genuinely ambiguous — not a hard rule violation, but worth deciding once now so it's applied consistently rather than differently each time it comes up.

## When truly unsure
Flag the sentence rather than guessing, and note it for group review. Don't
resolve boundary ambiguity silently — a written tie-break rule that gets
skipped defeats its own purpose.7