---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
description-zh: # Game-RL / Reasoning-RL Training Pipeline

## A Unified Framework Inspired by AlphaZero, MuZero & GRPO

---

## 1. Problem Statement & Domain Specification

### 1.1 Domain Abstraction

Before designing the pipeline, we formalize the domain as a **Markov Decision Process (MDP)** or a **Partially Observable MDP (POMDP)**:

```
Domain = (S, A, T, R, γ, Ω, O)
```

| Symbol | Meaning | Example (Chess) | Example (Math Reasoning) |
|--------|---------|-----------------|--------------------------|
| **S** | State space | Board position + history | Problem + partial proof steps |
| **
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
