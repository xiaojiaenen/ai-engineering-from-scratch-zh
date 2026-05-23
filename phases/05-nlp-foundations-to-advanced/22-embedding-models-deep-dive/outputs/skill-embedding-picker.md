---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
description-zh: # Selecting Embedding Model, Dimension & Retrieval Mode

## 1. Pick the Embedding Model

### Key Factors

| Factor | Consideration |
|---|---|
| **Domain** | General text → `text-embedding-3-small/large`, `bge-large`, `e5-large`; Code → `CodeBERT`, `starcoder`; Multilingual → `multilingual-e5-large`, `mBERT` |
| **Quality vs Speed** | Higher quality: `text-embedding-3-large`, `bge-large-en-v1.5`; Faster/smaller: `text-embedding-3-small`, `all-MiniLM-L6-v2` |
| **Max Tokens** | Short docs (≤512 tokens) → most models; Long docs → `jina-embeddings-v2` (8K), `nomic-embed-text` (8K), Cohere v3 (128K) |
| **License/Cost** | Open-source (self-hosted): `bge`, `
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
