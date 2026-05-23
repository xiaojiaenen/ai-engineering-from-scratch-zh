---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
description-zh: # LLM Evaluation Plan: Calibrated Judge + CI Gates

## 1. Evaluation Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────┐
│  Test Suite  │────▶│ Calibrated   │────▶│  Statistical │────▶│ CI Gate  │
│  Generation  │     │   Judge      │     │  Aggregation │     │ Decision │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────┘
       │                   │                    │                    │
   N samples          Rubric-based         Bootstrap CI        Ship / Block
   Per capability      Scoring             Per metric           / Iterate
```

---

## 2. Define Evaluation Taxonomy

```yaml
capabilities:
  - name: factual_accuracy
    weight: 0.30
    sub_metrics: [recall
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
