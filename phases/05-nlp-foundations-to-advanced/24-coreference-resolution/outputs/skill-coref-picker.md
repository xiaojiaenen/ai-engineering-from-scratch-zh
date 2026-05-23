---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
description-zh: ## Coreference Resolution: Selected Approach, Evaluation & Integration

### 1. Coreference Approach: **Neural End-to-End (Lee et al., 2017 style)**

- **Why:** Jointly learns mention detection and coreference linking, avoiding pipeline error propagation.
- **Mechanism:**
  - **Mention Detection:** Score all spans up to a max width (e.g., 10 tokens) using learned start/end representations.
  - **Pairwise Scoring:** Compute antecedent scores for each mention candidate pair using a learned bilinear attention over span embeddings.
  - **Clustering:** Greedily link each mention to its highest-scoring antecedent (or ∅) to form coreference clusters.
- **Backbone Encoder:** Transformer-based (e.g., SpanBERT or similar contextual encoder) for rich contextualized representations.
- **Key Features:** Span-level representations, learned entity linking scores, no hand-crafted features.

---

### 2. Evaluation Plan

| **Aspect** | **Detail** |
|---|---|
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
