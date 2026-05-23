---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
description-zh: # Comprehensive LLM Alignment Pipeline: RLHF / DPO / GRPO

---

## 1. Overview & Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ALIGNMENT PIPELINE OVERVIEW                  │
│                                                                 │
│  Phase 0: Base Model & Supervised Fine-Tuning (SFT)            │
│  Phase 1: Preference Data Collection                            │
│  Phase 2: Reward Model (RM) Training                            │
│  Phase 3: Policy Optimization (RLHF / DPO / GRPO)              │
│  Phase 4: Evaluation & Iteration                                │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │  SFT     │──▶│  Data    │──▶│  RM or   │──▶│  Policy  │    │
│  │  Model
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
