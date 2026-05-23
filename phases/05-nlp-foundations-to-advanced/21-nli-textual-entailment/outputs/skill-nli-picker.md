---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
description-zh: # NLI Setup for Zero-Shot Faithfulness Classification

---

## 1. Model

**`ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli`**

- A `roberta-large` model fine-tuned on the union of **SNLI, MNLI, FEVER-NLI, and ANLI (R1–R3)**.
- **Why this model?** It covers a broad, diverse set of NLI training data (including adversarial examples), making it robust for zero-shot transfer to unseen faithfulness domains.

---

## 2. Label Template (Entailment ↔ Faithfulness Mapping)

| NLI Label | Faithfulness Interpretation |
|---|---|
| **entailment** | **Faithful** – the hypothesis (generated output) is fully supported by the premise (source). |
| **neutral** | **Not Faithful (unverifiable)** – the output contains information not present in or inferrable from the source.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
