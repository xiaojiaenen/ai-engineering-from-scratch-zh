---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
description-zh: # DQN Training Configuration

## Replay Buffer

| Parameter | Value |
|---|---|
| **Buffer size** | 1,000,000 transitions |
| **Min samples before training** | 50,000 |
| **Batch size** | 32 |
| **Priority** | None (uniform sampling) |
| **Storage** | `(s, a, r, s', done)` tuples |

```python
buffer = ReplayBuffer(
    capacity=1_000_000,
    min_size=50_000,
    batch_size=32,
    priority=False  # set True for PER variant
)
```

---

## Target Network Sync

| Parameter | Value | Rationale |
|---|---|---|
| **Update method** | Hard copy | Original DQN style |
| **Sync frequency** | Every 10,000 steps | Stable bootstrapping |
| **τ (soft update)** | N/A | Use hard copy (set τ=0
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
