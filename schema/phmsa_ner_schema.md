# PHMSA Incident Report Extraction Schema

> Original design document, v0.1.0. Sections 1 to 8 are unchanged. Section 9 at the end records where the finished project differs from this design.

**Version:** 0.1.0
**Scope:** Named Entity Recognition + relation extraction schema for pulling structured risk/failure information out of free-text narrative fields in PHMSA pipeline incident reports (hazardous liquid, gas transmission, gas distribution).
**Tooling assumption:** spaCy / HuggingFace token classification pipelines, Label Studio for annotation. No from-scratch tagger — schema is designed to be directly loadable as a spaCy `EntityRuler`/`ner` config or a HuggingFace `datasets` NER split (BIO tags).

---

## 1. Design principles (read before annotating)

1. **Flat NER, no nesting.** Nested/overlapping spans are excluded by design. Standard `ner` pipelines (spaCy, BERT-token-classification) assume one label per token. Where a real nesting conflict exists (e.g., a quantity inside a consequence phrase — *"released **1,200 barrels**"*), the **more specific, lower-level entity wins** (`QUANTITY` beats `CONSEQUENCE`), and the outer span is not tagged at all — it's reconstructed later via relation/proximity rules, not via nested tags. This trades a small amount of information for compatibility with every standard library, which matters more than completeness given the "no from-scratch" constraint.
2. **Entities are extractive spans only.** No entity is inferred or normalized at tagging time (that happens in a separate normalization layer, §6). If the text says "corroded," tag it as the surface form; don't silently map it to a canonical `EXTERNAL_CORROSION` code inside the span.
3. **Schema tracks PHMSA's own structured taxonomy on purpose.** PHMSA reports already carry structured fields (`CAUSE`, `SUBCAUSE`, `ITEM_INVOLVED`, release volume, etc.) alongside the free-text narrative. Aligning entity types to these fields is deliberate — it lets you **weakly supervise** (auto-label narrative spans by string-matching against the report's own structured fields) before ever touching Label Studio by hand. This is the single biggest labeling-cost reduction available for this dataset.
4. **Coarse-to-fine label hierarchy**, not a flat 40-class list. Coarse types keep the token-classification head small and the class-imbalance problem tractable; fine-grained subtypes live as an *attribute* on the entity (via a lookup table / secondary classifier), not as a separate BIO tag. Rationale in §3.4.
5. **BIO tagging scheme** (not BIOES). BIO is the default for HuggingFace `token-classification` examples and `seqeval` scoring; BIOES buys marginal accuracy on short spans at the cost of annotation/debugging complexity that isn't worth it here — most of your spans (equipment names, failure phrases) are 1–4 tokens.

---

## 2. Entity types

### 2.1 `EQUIPMENT` — the physical asset involved
Physical pipeline system component named in text.
- **Examples:** "16-inch pipeline," "compressor station," "pressure relief valve," "girth weld," "flange," "pig launcher," "meter set," "casing"
- **Boundary rule:** include size/material modifiers directly attached ("6-inch cast iron main"), exclude standalone location modifiers ("the pipeline **near the river crossing**" → tag only "pipeline"; "river crossing" is `LOCATION`).
- **Maps to PHMSA field:** `ITEM_INVOLVED` / `SYSTEM_PART_INVOLVED` — used for weak-supervision bootstrap.

### 2.2 `FAILURE_MODE` — the physical defect/failure mechanism
The *what broke*, described narratively.
- **Examples:** "external corrosion," "stress corrosion cracking," "seam weld failure," "mechanical puncture," "gasket seal failure," "overpressure rupture"
- **Boundary rule:** tag the failure phrase, not the consequence. "corroded and **leaked**" → `FAILURE_MODE`="corroded", `CONSEQUENCE`="leaked" — these get linked via a relation (§4), not merged into one span.
- **Maps to PHMSA field:** `CAUSE` (top level: corrosion / excavation damage / material-weld-equipment failure / natural force / other outside force / incorrect operation / other) + `SUBCAUSE`.

### 2.3 `CAUSE_FACTOR` — the underlying contributing cause, when distinct from the failure mode
Distinguishes *mechanism* (`FAILURE_MODE`) from *root cause* (`CAUSE_FACTOR`). Example: failure mode = "rupture"; cause factor = "third-party excavation damage" or "inadequate cathodic protection."
- **Examples:** "excavation damage," "inadequate cathodic protection," "manufacturing defect," "operator error," "third-party contractor error," "frost heave," "lightning strike"
- **Why separate from FAILURE_MODE:** conflating these two loses the causal chain that's the entire point of the extraction (failure ← caused by ← root cause). Keeping them separate entity types, linked by a `CAUSED_BY` relation, is what makes the output "structured risk info" rather than a bag of keywords.

### 2.4 `ACTION_TAKEN` — remediation / operator response
- **Examples:** "isolated the section," "replaced the valve," "excavated and repaired," "reduced operating pressure," "purged the line," "installed a repair sleeve," "shut down the compressor"
- **Boundary rule:** include the object of the action when it's the same equipment span already tagged elsewhere isn't required — action spans stand alone; equipment linkage happens via relation, not shared tagging (avoids double-tagging ambiguity).

### 2.5 `CONSEQUENCE` — observed outcome/impact
- **Examples:** "release," "spill," "fire," "explosion," "evacuation," "injury," "fatality," "environmental contamination," "product loss"
- **Boundary rule:** severity adjectives ("significant," "minor") are *not* part of the span — they're handled by the severity classifier (§5), not tagged as text.

### 2.6 `QUANTITY` — numeric measurement with unit
- **Examples:** "1,200 barrels," "3.5 miles," "450 psi," "12% wall loss," "$2.1 million"
- **Boundary rule:** number + unit is one span, always. Never split number and unit into separate entities — downstream normalization needs them atomic. Currency, volume, distance, pressure, and percentage all share this one type; the *kind* of quantity is an attribute (§3.4), not a separate entity type, because distinguishing them by regex/unit-lookup post-hoc is trivial and doesn't need the tagger to learn 5 near-identical numeric-span classes.

### 2.7 `LOCATION` — where the incident occurred
- **Examples:** "Harris County, Texas," "milepost 214," "Gulf of Mexico," "the compressor station," "residential area"
- **Boundary rule:** overlaps with `EQUIPMENT` are resolved by preferring `EQUIPMENT` when the span *is* the asset ("the compressor station" as the failed unit = `EQUIPMENT`), and `LOCATION` when it's describing where something else happened ("failure occurred **at the compressor station**" — still ambiguous in practice; default to `EQUIPMENT` since it's the higher-value field for this project and ties are resolved toward the entity type with a clearer downstream use).
- **Maps to PHMSA field:** `LOCATION_COUNTY`, `LOCATION_STATE`, `ONSHORE_OFFSHORE` — usually already structured, so this entity type is lower annotation priority; narrative mentions mainly catch specifics not in the structured fields (e.g., "near a residential subdivision").

### 2.8 `DATE_TIME` — temporal references
- **Examples:** "on March 3, 2024," "at approximately 2:15 p.m.," "within 24 hours of discovery"
- **Priority: low.** PHMSA structured fields (`INCIDENT_DATE`, `TIME_INITIAL_NOTIFICATION`) already cover this almost completely. Only annotate when narrative gives a *relative* time not in structured fields ("48 hours after the previous inspection").

### 2.9 `MATERIAL_SPEC` — pipe/material specification details
- **Examples:** "Grade X52 steel," "0.25-inch wall thickness," "polyethylene," "cast iron," "API 5L"
- **Rationale for separate type from EQUIPMENT:** material specs are often the actual causal detail in corrosion/material-failure cases ("thin-wall polyethylene" explains the failure mode); collapsing into `EQUIPMENT` would bury this signal.

### 2.10 `INSPECTION_FINDING` — measurement/result from inspection or testing
- **Examples:** "in-line inspection (ILI) detected metal loss," "hydrostatic test failed at 1,100 psi," "18% wall loss recorded," "anomaly identified during smart pig run"
- **Boundary rule:** the numeric result inside this span is *also* separately tagged `QUANTITY` (this is the one intentional exception to the no-nesting rule in §1, because inspection findings are a compound entity of [method] + [quantity] that's genuinely useful to keep both levels of).

### 2.11 `PARTY_ROLE` — who was involved
- **Examples:** "the operator," "third-party contractor," "excavation crew," "pipeline company personnel," "local fire department"
- **Priority: medium.** Mainly matters for excavation-damage and incorrect-operation cause categories where liability/root-cause attribution hinges on which party acted.

### 2.12 `REGULATORY_REF` — code/standard/CFR citation
- **Examples:** "49 CFR 192.611," "ASME B31.8," "API 1160"
- **Priority: low, opportunistic.** Rare in narrative text (more common in the formal cause-analysis sections some reports include), but high-value when present — don't spend early labeling budget here; add a rule-based `EntityRuler` regex pass for CFR citation patterns instead of training the model on rare examples.

### 2.13 Explicitly out of scope (don't tag)
- Generic pipeline-industry boilerplate ("in accordance with company procedures")
- Severity/opinion adjectives (handled by classifier, §5, not NER)
- Report metadata (form numbers, operator ID numbers — already structured fields, redundant to extract from narrative)

---

## 3. Attributes, normalization, and the coarse/fine split

### 3.1 Why attributes instead of more entity types
A 12-type flat schema is already borderline for a few-hundred-example fine-tuning set. Rather than expanding to 30+ fine-grained types (`EXTERNAL_CORROSION`, `INTERNAL_CORROSION`, `SCC`, ... as separate NER labels), each coarse entity carries a **post-hoc attribute** resolved by:
1. Direct string/lookup match against a domain gazetteer (fast, covers ~70-80% of cases), or
2. A lightweight secondary classifier (e.g., a scikit-learn or HF text-classification head) run only on the extracted span text, or
3. Falling back to the PHMSA structured `SUBCAUSE` field when the narrative span was originally weak-labeled from it.

### 3.2 Attribute schema per entity type

| Entity type | Attribute | Values (non-exhaustive, mirrors PHMSA codebook) |
|---|---|---|
| `FAILURE_MODE` | `mechanism_category` | corrosion / cracking / weld_failure / material_defect / mechanical_damage / equipment_malfunction |
| `CAUSE_FACTOR` | `cause_category` | corrosion / excavation_damage / natural_force / material_weld_equipment / incorrect_operation / other_outside_force |
| `CONSEQUENCE` | `severity_class` | near_miss / release_no_ignition / fire / explosion / injury / fatality |
| `QUANTITY` | `unit_type` | volume / distance / pressure / percentage / currency / count |
| `PARTY_ROLE` | `party_category` | operator / contractor / third_party / regulatory / public |

### 3.3 Canonicalization layer (post-NER, separate step)
Raw spans are normalized after extraction, not during tagging:
- Units → SI or a fixed reporting unit (barrels retained as-is; PHMSA convention, don't convert to liters)
- Dates → ISO 8601
- Equipment names → mapped to a controlled vocabulary (build this gazetteer from PHMSA's own `ITEM_INVOLVED` code list — it's public)

### 3.4 Rationale for coarse/fine split (engineering tradeoff, stated explicitly)
Fine-grained multiclass NER directly (e.g., tagging `EXTERNAL_CORROSION` vs `INTERNAL_CORROSION` as different B-/I- tags) would need the tagger to distinguish subtle context per class with very few training examples per subclass — a recipe for a degenerate model that just predicts the majority coarse class anyway. Decoupling detection (coarse NER, well-supported by data) from classification (attribute lookup/classifier, can lean on gazetteer + weak labels) gets you the fine-grained output without needing fine-grained-labeled training data for the hardest part.

---

## 4. Relation schema

Given the "no from-scratch, high-level libraries" constraint, full joint relation extraction (e.g., training a relation classifier over entity pairs) is available but not required for v1 — **default to a rule-based/proximity relation layer**, documented here so it's swappable for a trained relation classifier later without changing the entity schema.

| Relation | From → To | Extraction rule (v1, rule-based) |
|---|---|---|
| `CAUSED_BY` | `FAILURE_MODE` → `CAUSE_FACTOR` | nearest `CAUSE_FACTOR` span within same sentence, preferring one joined by "due to," "caused by," "as a result of" |
| `RESULTED_IN` | `FAILURE_MODE` → `CONSEQUENCE` | nearest `CONSEQUENCE` span within same or next sentence |
| `REMEDIATED_BY` | `FAILURE_MODE` → `ACTION_TAKEN` | nearest `ACTION_TAKEN` span, same paragraph, no cause-marker phrase between them |
| `LOCATED_AT` | `EQUIPMENT` → `LOCATION` | nearest `LOCATION` span in same sentence |
| `HAS_QUANTITY` | `CONSEQUENCE` or `INSPECTION_FINDING` → `QUANTITY` | `QUANTITY` span immediately adjacent (≤3 tokens) |
| `INVOLVES_PARTY` | `CAUSE_FACTOR` → `PARTY_ROLE` | same-sentence co-occurrence |
| `MADE_OF` | `EQUIPMENT` → `MATERIAL_SPEC` | same-sentence, no verb boundary between spans |

**Escalation path if rule-based relations underperform:** treat entity-pair classification as a text-classification problem (concatenate the two spans + surrounding context window, feed to a HF sequence classifier) — still a "high-level library" approach, no custom architecture needed.

---

## 5. Severity / risk classification layer

Separate from NER entirely — a document- or paragraph-level classifier, since "how bad was this" is a judgment over the whole incident, not a span.
- **Labels:** `near_miss`, `minor` (no injury, contained), `moderate` (release, no injury), `severe` (injury/large release/fire), `critical` (fatality/explosion)
- **Weak supervision source:** PHMSA reports already carry `FATALITY`, `INJURY_IND`, `IGNITE_IND`, `EXPLODE_IND`, and `UNINTENTIONAL_RELEASE_BBLS` as structured fields — build the label from these directly rather than hand-labeling severity from narrative text. This is nearly free ground truth.

---

## 6. Output data model (final structured record)

```json
{
  "report_id": "PHMSA-2024-XXXXXX",
  "entities": [
    {"text": "external corrosion", "type": "FAILURE_MODE", "attr": {"mechanism_category": "corrosion"}, "span": [142, 161]},
    {"text": "16-inch pipeline", "type": "EQUIPMENT", "span": [98, 115]},
    {"text": "1,200 barrels", "type": "QUANTITY", "attr": {"unit_type": "volume"}, "span": [210, 223]}
  ],
  "relations": [
    {"type": "CAUSED_BY", "from": "external corrosion", "to": "inadequate cathodic protection"},
    {"type": "HAS_QUANTITY", "from": "release", "to": "1,200 barrels"}
  ],
  "severity": "severe",
  "source_fields_used_for_weak_label": ["CAUSE", "SUBCAUSE", "UNINTENTIONAL_RELEASE_BBLS"]
}
```

This is the target shape for both the weak-supervision bootstrap output and the final model's inference output — keeping them structurally identical means you can diff model predictions against weak labels directly for error analysis.

---

## 7. Annotation / labeling decisions

- **Tagging tool:** Label Studio, NER span-labeling template, exported in CoNLL/BIO or HF `datasets` JSON — both are directly consumable by `spacy train` and HF `Trainer` without custom conversion code.
- **Annotation unit:** sentence-level, not full-report — PHMSA narratives run long; sentence-splitting first (spaCy sentencizer) keeps annotation spans manageable and matches how token-classification models are typically windowed.
- **Inter-annotator agreement:** if more than one person labels, compute span-level Cohen's kappa on a shared 10% sample before scaling up; `FAILURE_MODE` vs `CAUSE_FACTOR` boundary is the most likely disagreement point — resolve with a written tie-break rule (mechanism = *what physically happened to the equipment*; cause = *why it happened*) before annotating further.
- **Class imbalance:** `EQUIPMENT`, `FAILURE_MODE`, and `CONSEQUENCE` will dominate; `REGULATORY_REF` and `MATERIAL_SPEC` will be sparse. Don't oversample synthetic examples for rare types until you've measured actual eval F1 per type — same discipline as your fraud-detection precision/recall work, applied per entity class instead of per prediction class.
- **Weak-label QA:** since bootstrap labels come from string-matching PHMSA structured fields into the narrative, spot-check a sample for **field/narrative mismatch** (structured field says "corrosion" but narrative describes something else — happens when operators fill forms loosely) before trusting weak labels as training data without human review.

---

## 8. Versioning

Schema changes (adding/renaming entity types, changing boundary rules) should bump the version number at the top of this file and be accompanied by a note on whether existing annotations need re-review. Treat this file as the single source of truth the Label Studio config, the HF label list, and the eval script's label set are all generated from — not maintained separately in three places.

---

## 9. Implementation status: where the build differs from v0.1.0

Added after the project was finished. The version number above was not bumped; the differences are recorded here instead. Numbers are from the project's own evaluations (see the main README).

| Schema item | v0.1.0 said | As built |
|---|---|---|
| Flat NER, with one nesting exception (section 2.10: a `QUANTITY` inside an `INSPECTION_FINDING`) | the exception keeps both levels | **No exception.** Every overlap is resolved by "narrower span wins" (`resolve_overlaps`). The rule cost at most 8% of any label except `INSPECTION_FINDING`, which lost 10 of 48 spans (21%) |
| `CAUSE_FACTOR` (2.3) | extracted and linked by `CAUSED_BY` | The annotated labels were inconsistent. 80 of 165 spans were relabeled or dropped by a rubric. The final model still scores **F1 0.00**, so root causes are **not extracted** |
| `MATERIAL_SPEC` (2.9) | pipe and material specifications | The gold labels also contain substances (gas, crude oil, CO2, hydrocarbons, soil, ice). The `MADE_OF` relation therefore accepts only a construction-material whitelist |
| `REGULATORY_REF` (2.12) | add a rule-based regex pass, do not spend labeling budget | It was annotated by hand and learned by the model. F1 is 0.08 on 16 test spans, and the regex pass was not built |
| Attributes and canonicalization (section 3) | per-entity attributes (`mechanism_category`, `cause_category`, `severity_class`, `party_category`) and a normalization layer | **Only `QUANTITY.unit_type`** exists (volume_gas, volume_liquid, pressure, distance, currency, percentage), computed by regex in the relation and demo layer. The other attributes and the canonicalization layer were not built |
| Relations (section 4) | windows of the same or next sentence, or the same paragraph | **Same sentence only.** `HAS_QUANTITY` also accepts `EQUIPMENT` heads, with unit compatibility. `CAUSED_BY` from `FAILURE_MODE` to `CAUSE_FACTOR` and `INVOLVES_PARTY` never fire (no `CAUSE_FACTOR`). A cue-word `CAUSED_BY` from `CONSEQUENCE` to `FAILURE_MODE` was added. `MADE_OF`, `LOCATED_AT`, `RESULTED_IN` and `REMEDIATED_BY` gained whitelist, preposition and stoplist conditions. Hand-reviewed precision: 61 of 90 sampled relations (68%) |
| Severity (section 5) | five labels including `near_miss`, using `UNINTENTIONAL_RELEASE_BBLS` | **Four labels** (minor, moderate, severe, critical) from the four yes/no flags `FATALITY_IND`, `INJURY_IND`, `IGNITE_IND`, `EXPLODE_IND`. No `near_miss` and no release-volume term, so 97% of hazardous-liquid reports are "minor" |
| Output record (section 6) | `report_id`, `attr`, `source_fields_used_for_weak_label` fields | The record has `commodity_type`, `narrative`, `severity` (label and probabilities), `entities` (with offsets, sentence index and `unit_type`), `relations` (with confidence), `dropped_negated`, and an optional `note` |
| Annotation (section 7) | inter-annotator agreement if more than one person labels | One annotator, so no agreement figure exists. Sentence-level annotation in Label Studio with pre-annotations from the weak labeler, as specified. Guideline refinements made during annotation live in `annotation/ANNOTATION_GUIDELINES.md` |
