---
title: PHMSA Incident Extraction
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
short_description: Structured extraction from pipeline incident narratives
---

# PHMSA incident narrative extraction

Paste a free-text pipeline incident narrative and get back a structured record:

- **Entities** such as equipment, failure modes, consequences, actions taken, quantities, locations and dates
- **Relations** between them (what was located where, what was caused by what, what was done about it)
- **Severity** (minor, moderate, severe, critical) predicted from the whole narrative

Two fine-tuned DistilBERT models (entity tagging and severity) plus a rule-based relation layer,
trained on public PHMSA incident reports for gas distribution, gas transmission and gathering,
and hazardous liquid pipelines.

The app lists its measured accuracy and its main limits under "About this demo and its limits".
It is a portfolio project, not a tool for operational, safety or regulatory decisions.