---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
description-zh: 针对给定任务生成REINFORCE/actor-critic/PPO训练配置并诊断方差问题。

## 1. Algorithm Configs

### REINFORCE (Baseline Config)

```yaml
# config_reinforce.yaml
algorithm: REINFORCE

environment:
  name: "CartPole-v1"         # swap for your task
  num_envs: 1
  max_episode_steps: 500

policy:
  type: "discrete"
  architecture: "mlp"
  hidden_sizes: [128, 128]
  activation: "tanh"

optimizer:
  type: "Adam"
  lr: 3.0e-4
  eps: 1.0e-5

gamma: 0.99
entropy_coeff: 0.01
max_grad_norm: 0.5

# --- Variance reduction knobs ---
baseline:
  enabled: true                # value-function baseline
  type: "value_network"        # "value_network"
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
