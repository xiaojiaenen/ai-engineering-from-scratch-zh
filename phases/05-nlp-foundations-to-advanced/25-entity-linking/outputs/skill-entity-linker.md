---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
description-zh: # Entity Linking Pipeline Design

## 1. Knowledge Base (KB)

### Structure

```
┌─────────────────────────────────────────────────────┐
│                   Knowledge Base                     │
│                                                      │
│  Entity: Q42 (Douglas Adams)                         │
│  ├── Surface Forms: ["Douglas Adams", "Doug Adams",  │
│  │     "D Adams", "Douglas Noel Adams"]              │
│  ├── Type: Person > Author                           │
│  ├── Aliases: [multilingual variants]                │
│  ├── Description: "English author and humorist"      │
│  ├── Relations:                                      │
│  │     ├── (born_in, Cambridge)                      │
│  │     ├── (wrote, The Hitchhiker's Guide...)        │
│  │     └── (occupation, screenwriter)                │
│  └── Incoming Links: 12,048 (popularity prior)       │
│                                                      │
│
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
