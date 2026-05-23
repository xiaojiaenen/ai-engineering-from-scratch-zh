---
name: vad-tuner
description: Pick VAD model, threshold, silence hangover, pre-roll, and turn-detection strategy for a voice agent.
description-zh: # Voice Agent VAD Configuration Guide

## 1. VAD Model

| Option | Latency | Accuracy | Best For |
|---|---|---|---|
| **Silero VAD** ⭐ | ~1ms | Very high | Production agents (recommended) |
| WebRTC VAD | <1ms | Moderate | Ultra-low-resource / embedded |
| pyannote.audio | ~5ms | Highest | Offline / diarization-heavy |
| FunASR VAD | ~2ms | High | Chinese-language agents |

**Recommendation:** **Silero VAD v5** — best trade-off of accuracy, latency, and ease of deployment. Runs on CPU in ~1ms per frame.

---

## 2. Thresholds

```
Speech threshold:     0.5  (probability ≥ 0.5 → speech)
Noise threshold:      0.35 (used for noise-aware gate)
Energy gate (optional): -40 dBFS (skip VAD if below this)
```

| Parameter |
version: 1.0.0
phase: 6
lesson: 14
tags: [vad, silero, cobra, turn-detection, flush-trick]
---

Given the workload (consumer / call-center / edge / accessibility; noise profile; language mix; latency), output:

1. VAD. Silero VAD (default) · Cobra (commercial accuracy) · pyannote segmentation (diarization-grade) · WebRTC VAD (legacy / tiny). One-sentence reason.
2. Parameters. Threshold (0.3-0.5), min speech (200-300 ms), silence hangover (400-800 ms), pre-roll (250-500 ms).
3. Semantic turn detection. Enabled (LiveKit turn-detector or custom MLP) or not. Reason tied to expected user speech patterns.
4. Flush trick. Enabled (if STT supports it — Kyutai / Deepgram) or not. Expected latency savings.
5. Guards. Reject speech shorter than min duration; always keep pre-roll; cap per-user silence-hangover override; fail-open if VAD service is down (treat everything as speech).

Refuse energy-only VAD for production — too noisy. Refuse zero silence-hangover — will interrupt users. Refuse Whisper-based VAD when dedicated Silero is available (slower, less accurate).

Example input: "Call-center IVR for airline rebooking. Noisy background (airport). English + Spanish. &lt; 500 ms turn detection."

Example output:
- VAD: Cobra (commercial) for the noise-resistance advantage. Fall-back to Silero if cost prohibitive.
- Parameters: threshold 0.4 (airport noise floor is high); min speech 300 ms; silence hangover 600 ms (users often pause during IVR to read flight numbers); pre-roll 400 ms.
- Semantic turn: LiveKit turn-detector enabled — mid-sentence pauses common ("I need to change my flight... to tomorrow").
- Flush trick: enabled on Deepgram streaming. Expected savings: 400 ms → 150 ms turn-end latency.
- Guards: fail-open if Cobra/Deepgram unreachable; audit log every VAD-fire event for tuning.
