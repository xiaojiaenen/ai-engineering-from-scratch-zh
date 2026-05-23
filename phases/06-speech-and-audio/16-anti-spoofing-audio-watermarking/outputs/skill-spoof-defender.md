---
name: spoof-defender
description: Pick detection model, watermark, provenance manifest, and operational playbook for a voice-generation / voice-auth deployment.
description-zh: # Voice-Gen / Voice-Auth Deployment — Key Components

---

## 1. Detection Model

| Aspect | Recommendation |
|---|---|
| **Primary model** | **AASIST2** (graph attention + raw waveform) for spoofing detection on ASVspoof 5 protocol |
| **Ensemble layer** | Pair with **RawNet3** to capture complementary spectral artifacts |
| **Input features** | Raw waveform (no handcrafted front-end); optional LFCC branch for channel-robust mode |
| **Threshold strategy** | EER-calibrated per-locale; operate at **≤1 % EER with ≤5 % FRR** on enrollment audio |
| **Update cadence** | Retrain quarterly on new TTS / voice-clone samples (GPT-4o-audio, VALL-E XT, ElevenLabs v3, etc.) |
| **Hard-negative mining** | Adversarial vocoder outputs (HiFi
version: 1.0.0
phase: 6
lesson: 16
tags: [anti-spoofing, watermark, audioseal, asvspoof, c2pa, voice-fraud]
---

Given the workload (voice-gen vs voice-auth, deploy scale, compliance region, adversary profile), output:

1. Detection (CM). AASIST · RawNet2 · NeXt-TDNN + WavLM · commercial (Pindrop, Validsoft). Training data: ASVspoof 2019 / ASVspoof 5 / domain-specific. Target EER.
2. Watermarking (outbound gen). AudioSeal 16-bit payload encoding `(model_id, user_id, generation_ts)` · WaveVerify (alt) · none (with justification). Detector runs in CI on every output pre-ship.
3. Provenance. C2PA manifest signed with deployer's key · IPTC metadata · none (for non-consumer audio).
4. Voice-auth guards (if applicable). Liveness challenge (random phrase TTS' + transcribe), replay attack detection (AASIST + PA model), biometric threshold calibration per channel.
5. Operational. Audit log retention, consent artifact retention (7+ years), abuse-detection signals (sudden volume burst, named-entity prompts), kill-switch procedure.

Refuse voice-gen deploys without AudioSeal (or equivalent watermark). Refuse voice biometric deploys without anti-spoofing detection — voice cloning makes cosine-only auth trivially bypassable. Refuse deploys that depend on provenance manifest alone (strippable). Refuse detection thresholds trained on ASVspoof 2019 for real-world deploys without a channel-calibration sweep.

Example input: "Bank customer-service IVR. Voice biometric unlock + AI-generated voice agent. 10M calls/month. US + EU."

Example output:
- Detection: Pindrop commercial (preferred) or NeXt-TDNN + WavLM open. Training on ASVspoof 5 + 100k bank-specific call samples. Target EER &lt; 0.5% on in-domain data.
- Watermarking: AudioSeal 16-bit payload on every outbound TTS utterance; payload encodes bank_id + session_id + timestamp. Detector verifies before transmit.
- Provenance: C2PA manifest on audio-export-to-customer workflows; internal-only calls skip.
- Voice-auth: liveness challenge at every auth (TTS random 4-digit phrase; user repeats + detector + transcriber). Anti-spoofing runs on every inbound auth attempt. Biometric threshold at FAR 0.1%, FRR 1%.
- Operational: 7-year retention on consent + audit log in region (EU data EU-resident). Alert on sudden clone-request volume &gt; 2σ; kill-switch on abuse detection.
