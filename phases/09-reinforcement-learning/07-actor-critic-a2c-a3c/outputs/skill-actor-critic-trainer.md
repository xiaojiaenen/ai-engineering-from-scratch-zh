---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
description-zh: # A2C / A3C / GAE Configuration

Below is a comprehensive, modular configuration you can adapt for any Gym-style environment.

---

## 1. General Hyperparameters

```yaml
# ─── Environment ───
env_name: "CartPole-v1"          # or any Gym/Gymnasium env
num_envs: 8                      # parallel environments (A2C/GAE)
seed: 42

# ─── Algorithm Selection ───
algorithm: "A2C"                 # "A2C" | "A3C" | "PPO_GAE"
distributed:
  enabled: false                 # true → A3C (multi-process)
  num_workers: 4                 # A3C worker processes
  update_master: "global"        # "global" | "shared_grad"
```

---

## 2. Network Architecture
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
