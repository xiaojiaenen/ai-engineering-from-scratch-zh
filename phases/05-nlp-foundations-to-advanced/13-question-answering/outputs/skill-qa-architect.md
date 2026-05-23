---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
description-zh: # QA System Design Guide

## 1. QA Architecture

| Architecture | When to Use |
|---|---|
| **Extractive QA** (e.g., BERT-based span extraction) | Factual answers verbatim from source documents |
| **Generative QA** (e.g., seq2seq like T5, Flan-T5) | Abstractive/compositional answers, multi-hop reasoning |
| **Retrieval-Augmented Generation (RAG)** | Large knowledge bases, up-to-date info needed, reducing hallucination |
| **Knowledge Graph QA** | Structured data, relational queries |
| **Hybrid** (retrieval + generation + verification) | Production-grade systems needing high accuracy |

**Recommendation for most cases → RAG pipeline:**
```
Query → Retriever → Reranker → LLM Generator → Answer
```

---

## 2. Retrieval Strategy

### Sparse Retrieval (Lexical)
- **BM2
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder like `all-MiniLM-L6-v2`), or hybrid.
3. Reader. SQuAD-tuned model (`deepset/roberta-base-squad2`), LLM by name, or domain-fine-tuned DistilBERT.
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
