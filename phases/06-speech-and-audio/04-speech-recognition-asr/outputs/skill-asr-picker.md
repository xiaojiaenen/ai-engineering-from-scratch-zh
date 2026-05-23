---
name: asr-picker
description: Pick ASR model, decoding strategy, chunking, and LM fusion for a given deployment target.
description-zh: # ASR System Configuration Guide by Deployment Target

Below is a decision framework mapping deployment constraints to recommended ASR components.

---

## 1. ASR Model

| Deployment Target | Recommended Model | Rationale |
|---|---|---|
| **Cloud (GPU)** | Conformer-CTC/Transducer (large, 100M+ params) | Accuracy priority; no latency/size constraints |
| **Edge GPU (Jetson, etc.)** | Streaming Conformer-Transducer (medium, 20–50M) | Balances accuracy and real-time factor |
| **Mobile / Embedded CPU** | Zipformer-Tiny, Emformer, or Pruned RNN-T (<10M) | Meets strict memory &
version: 1.0.0
phase: 6
lesson: 04
tags: [audio, asr, speech-recognition]
---

Given a deployment target (language list, domain, latency budget, hardware, offline / streaming, clip duration), output:

1. Model. Whisper-large-v3-turbo / Parakeet-TDT / Canary-Flash / wav2vec 2.0 / Moonshine. Reason in one sentence.
2. Decoding. Greedy / beam width / temperature fallback / LM fusion weight. Reason tied to the quality budget.
3. Chunking and VAD. Chunk length, stride, whether to gate with Silero-VAD or Whisper's own.
4. Language policy. Force language vs auto-LID; how to handle cross-lingual frames.
5. Eval plan. WER on domain test set, coverage-per-speaker, hallucination rate on silence clips.

Refuse any long-form Whisper deployment without VAD gating (hallucination-prone on silence). Refuse to report WER without text normalization (lower, punct strip). Flag any beam-width > 16 without an LM; raw beams over blanks do not help.
