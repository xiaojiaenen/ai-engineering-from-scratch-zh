---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
description-zh: # Machine Translation Evaluation – Shipping Domain

I'd be happy to help evaluate a machine translation output for the shipping industry. However, I need a few key details to proceed:

## Please Provide:

1. **Source text** – The original text in the source language
2. **MT output** – The machine-translated text to be evaluated
3. **Language pair** – e.g., English → Chinese, Spanish → French, etc.
4. **Purpose** – e.g., shipping documentation, customer-facing content, internal logistics communication, regulatory compliance

## Evaluation Criteria I Will Apply:

| Criterion | Description |
|---|---|
| **Accuracy** | Correct translation of meaning |
| **Terminology** | Proper use of shipping/logistics terms (e.g., Bill of Lading, FOB, CIF, demurrage, freight forwarding) |
| **Fluency** | Natural readability in the target language |
| **Consistency** | Uniform use of terms throughout |
| **Omissions/Additions** | Any missing or extra content |
| **
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable checklist: content preservation (no hallucinations), correct target language, register / formality match, terminology consistency with glossary if provided, no truncation or length explosion.
3. One domain-specific issue to probe. Legal: named entities, statute citations. Medical: drug names, dosages. UI: placeholder variables like `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to severity of issues found.

Refuse to ship without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
