# PHMSA Incident Extraction

[![tests](https://github.com/FranklinNwankwo/NLP_Unstructured_Data/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/FranklinNwankwo/NLP_Unstructured_Data/actions/workflows/tests.yml)

**Turning free-text pipeline incident narratives into structured records: entities, relations between them, and a severity prediction, plus cited question answering over all incidents.**

## TL;DR

- **Problem.** PHMSA's 9,654 U.S. pipeline incident reports (2010 to present) each carry a free-text narrative that says what failed, why, what was done and what resulted, but none of it is structured or searchable.
- **What I built.** A pipeline that extracts 12 entity types, links them with relation rules, predicts severity from the narrative, and answers questions over every incident with exact counts and cited reports.
- **Results on held-out data:**
  - **Entities:** micro-F1 **0.608** (fine-tuned DistilBERT), vs 0.553 spaCy, 0.564 an 8-shot LLM and 0.371 rules
  - **Relations:** **69%** precision on 90 hand-reviewed relations (recall not measured)
  - **Severity:** macro-F1 **0.777**, vs 0.646 for TF-IDF + logistic regression
  - **Question answering:** **18/18** filters and counts correct; 18/18 answers with valid citations after a code check
- **What does not work.** Root causes (`CAUSE_FACTOR`) are not extracted (F1 0.00); severity cannot see a death the narrative never mentions.
- **Try it:** [live demo](https://nlp-on-oil-and-gas-unstructured-data.streamlit.app) (sleeps when idle, so the first load is slow) · [models on Hugging Face](https://huggingface.co/Chinonso11/phmsa-incident-models) · [full engineering log](docs/ENGINEERING_LOG.md)
- **What I'd do next:**
  - Predict root causes per narrative from PHMSA's structured `CAUSE` field (a free label for every report) instead of span tagging
  - Add a second annotator to measure label agreement, and gold relations to measure relation recall
  - Link cause and effect expressed by verbs and across sentences, where most of it lives

![Dashboard](docs/Demo1.png)
![Dashboard](docs/Demo2.png)

## Where to read more

Everything else (data audit, weak supervision, annotation, every training run, baselines, relation precision, severity analysis, deployment, decision register, mistakes and limitations) is in the **[engineering log](docs/ENGINEERING_LOG.md)**. Good starting points:

- [Results at a glance](docs/ENGINEERING_LOG.md#1-results-at-a-glance)
- [Architecture](docs/ENGINEERING_LOG.md#3-architecture)
- [The CAUSE_FACTOR investigation](docs/ENGINEERING_LOG.md#104-the-cause_factor-investigation)
- [Prompted-LLM baseline](docs/ENGINEERING_LOG.md#107-baseline-c-a-prompted-llm-baselinesbaseline_llmpy) and [bigger model](docs/ENGINEERING_LOG.md#108-does-a-bigger-model-help-distilbert-vs-bert-base-notebooksner_model_comparisonipynb)
- [Limitations and threats to validity](docs/ENGINEERING_LOG.md#15-limitations-and-threats-to-validity)
- [Decision register](docs/ENGINEERING_LOG.md#16-decision-register) and [mistakes, corrections and dead ends](docs/ENGINEERING_LOG.md#17-mistakes-corrections-and-dead-ends)

## Quickstart

```python
# download the two models from the Hugging Face Hub
from huggingface_hub import snapshot_download
path = snapshot_download("Chinonso11/phmsa-incident-models")
# copy <path>/ner      -> ner_model/phmsa_ner_model
#      <path>/severity -> severity_model/phmsa_severity_model
```

```
python pipeline/run_pipeline.py --text "YOUR NARRATIVE HERE"
streamlit run demo/space/app.py
```

Setup details, offline tests and the full reproduction order are in the [engineering log](docs/ENGINEERING_LOG.md#5-setup-and-quickstart).

**Status:** a portfolio project built on public U.S. government records. Not validated for operational, safety or regulatory use.