---
name: skill-bpe-vs-wordpiece
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
description-zh: # Tokenizer Selection Guide for Corpus & Deployment Target

## Step 1: Analyze Your Corpus

| Corpus Characteristic | Consideration |
|---|---|
| Language | Multilingual → larger vocab; CJK → subword-aware needed |
| Domain (code, medical, legal) | Domain-specific tokens matter |
| Size (MB/GB) | Larger corpus → supports larger vocab reliably |
| Text structure | URLs, code, formulas → need pre-tokenizer rules |

---

## Step 2: Choose Tokenizer Algorithm

| Algorithm | Best For | Trade-off |
|---|---|---|
| **BPE** (Byte Pair Encoding) | General-purpose, multilingual, code | Fast encode/decode; good balance |
| **WordPiece** | English-centric NLU (BERT-style) | Slightly slower; deterministic merges |
| **Unigram** (SentencePiece) | Multilingual, CJK, resource-constrained inference | Prunable; probabilistic |
| **Byte-level BPE**
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
