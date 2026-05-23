---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
description-zh: # Dialogue State Tracker (DST) Design

## 1. Schema Design

The schema defines **what** the tracker tracks — the ontology of belief states.

### Slot-Value Ontology

```yaml
# Schema definition (e.g., restaurant booking domain)
domain: restaurant
slots:
  - name: food_type
    type: categorical
    values: [italian, chinese, indian, japanese, mexican, ...]
    description: "Type of cuisine the user wants"

  - name: price_range
    type: categorical
    values: [cheap, moderate, expensive]
    description: "Desired price range"

  - name: area
    type: categorical
    values: [north, south, east, centre, west]
    description: "Geographic preference"

  - name: name
    type: freeform          # open-vocabulary value
    description: "Restaurant name"

  - name: book_time
    type: freeform          # constrained by parser
    description: "Reservation
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
