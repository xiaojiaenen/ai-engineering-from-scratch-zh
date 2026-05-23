---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
description-zh: # Multilingual NLP Task Plan

## 1. Source Language

**Recommended: English**

- Largest volume of annotated data available
- Most pre-trained models are English-centric
- Serves as a strong pivot language for cross-lingual transfer

**Secondary pivot languages to consider:**
- **Chinese / Spanish / Arabic** — high-resource, typologically diverse
- Use these if the target domain/region demands it (e.g., medical NLP in Arabic)

---

## 2. Target Model

| Option | Model | Why |
|--------|-------|-----|
| **Cross-lingual Transformer** | **XLM-RoBERTa (XLM-R)** | Pre-trained on 100 languages; strong zero-shot transfer |
| **Multilingual LLM** | **mT5 / BLOOM / LLaMA-multilingual** | Generative tasks (summarization, QA, translation) |
| **Language-specific** | **FastText + fine-tuned BERT** |
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or a byte-level tokenizer like GPT-2).
