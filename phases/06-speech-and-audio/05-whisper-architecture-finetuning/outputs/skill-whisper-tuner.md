---
name: whisper-tuner
description: Design a Whisper fine-tune or inference pipeline for a given language, domain, and latency budget.
description-zh: # Whisper Fine-Tune & Inference Pipeline Design

## 1. Requirement Analysis (Inputs)

```
┌─────────────────────────────────────────────────┐
│  LANGUAGE   │ e.g., Japanese, Arabic, Swahili    │
│  DOMAIN     │ e.g., medical, legal, call-center  │
│  LATENCY    │ e.g., < 500ms / < 2s / batch-only  │
│  WER TARGET │ e.g., < 8% on domain test set      │
└─────────────────────────────────────────────────┘
```

---

## 2. Pipeline Overview

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Data        │──▶│  Fine-Tune   │──▶│  Optimize &  │──▶│  Deploy &    │
│  Preparation │   │  Whisper     │   │
version: 1.0.0
phase: 6
lesson: 05
tags: [audio, whisper, asr, fine-tuning, lora]
---

Given a target (language set, domain, clip length distribution, latency budget, hardware) and data (hours available, quality), output:

1. Variant. Tiny / Base / Small / Medium / Large-v3 / Turbo. Reason.
2. Runtime. vanilla / faster-whisper / whisperx / whisper-streaming. Reason.
3. Fine-tune plan. Full-FT vs LoRA (r, target_modules), freeze-encoder policy, epoch count.
4. Inference guards. VAD (Silero or Whisper's own), `temperature=0`, `condition_on_previous_text=False`, `no_speech_threshold`.
5. Evaluation. Domain WER target, text normalization rules, hallucination-rate check on silence clips.

Refuse to deploy Whisper on arbitrary audio without VAD. Refuse to set `condition_on_previous_text=True` for multi-chunk jobs without a runaway guard. Flag any fine-tune that swaps Whisper's tokenizer or mel pipeline.
