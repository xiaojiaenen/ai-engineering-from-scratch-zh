---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
description-zh: 为给定环境生成PPO训练配置和诊断计划。

---

## 1. PPO Training Configuration (YAML)

Below is a general-purpose PPO config. Adjust hyperparameters to match your specific environment.

```yaml
# =============================================================
# PPO Training Configuration
# =============================================================

# --- Environment ---
env:
  name: "MyEnv-v0"          # Gymnasium-registered env ID
  num_envs: 8               # Number of parallel environments
  seed: 42
  max_episode_steps: 1000   # Truncation limit per episode
  normalize_obs: true       # Running mean/std normalization
  normalize_reward: true    # Reward normalization (return/running std)
  clip_obs: 10.0
  clip_reward: 10.0
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
