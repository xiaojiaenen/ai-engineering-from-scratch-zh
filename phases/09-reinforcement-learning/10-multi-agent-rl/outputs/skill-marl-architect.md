---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
description-zh: # Choosing the Right Multi-Agent RL Regime

## Quick Decision Flowchart

```
Are agents cooperative or competitive?
├── Competitive → Self-Play or League
├── Cooperative → CTDE or IPPO
└── Mixed/General-sum → CTDE or IPPO + communication

Is the agent count fixed and small?
├── Yes (2-8) → CTDE (QMIX, MAPPO, MADDPG)
└── No / Scalable → IPPO or mean-field

Do agents need to share global state?
├── Yes → CTDE (centralized critic)
└── No → IPPO / independent learners

Is the environment symmetric?
├── Yes → Self-Play
└── No → League / asymmetric CTDE
```

---

## The Four Regimes Compared

### 1. **IPPO (Independent PPO)**
| Aspect | Details |
|---|---|
| **Core idea** | Each agent runs its own PPO independently; treats others as part of the environment |
| **Best for** | Large-scale, homogeneous agents,
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
