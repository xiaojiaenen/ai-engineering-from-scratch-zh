# Actor-Critic — A2C 与 A3C

> REINFORCE 噪声很大。引入一个学习 `V̂(s)` 的 critic，并从回报中减去它，你就能得到具有相同期望但方差低得多的优势量。这就是 actor-critic。A2C 同步运行它；A3C 跨线程运行。它们都是 2015 年后所有现代深度强化学习方法的思维模型。

**类型：** 构建
**语言：** Python
**前置知识：** 第 9 阶段 · 04（TD 学习），第 9 阶段 · 06（REINFORCE）
**时间：** 约 75 分钟

## 问题所在

Vanilla REINFORCE 可以工作，但其方差非常糟糕。蒙特卡洛回报 `G_t` 在不同回合之间可以波动十倍。将该噪声乘以 `∇ log π` 并取平均，会产生一个梯度估计器，需要数千个回合才能把策略移动到你可以用更少次 DQN 更新移动的距离。

方差源于使用原始回报。如果你减去一个基线 `b(s_t)`——任何状态函数，包括学习到的价值——期望不变，方差降低。最优的可处理基线是 `V̂(s_t)`。现在乘以 `∇ log π` 的量为*优势*：

`A(s, a) = G - V̂(s)`

如果动作产生了高于平均的回报，则是好的；低于平均则为差。带有学习到的 critic 的 REINFORCE 就是*actor-critic*。Critic 给 actor 提供了一个低方差的教师。这是 2015 年之后的所有深度策略方法（A2C、A3C、PPO、SAC、IMPALA）的架构。

## 概念

![Actor-critic：策略网络加上价值网络，TD 残差作为优势](../assets/actor-critic.svg)

**两个网络，一个联合损失：**

- **Actor** `π_θ(a | s)`：策略。采样以执行动作。通过策略梯度训练。
- **Critic** `V_φ(s)`：估计从状态开始的期望回报。通过最小化 `(V_φ(s) - target)²` 训练。

**优势。** 两种标准形式：

- *MC 优势：* `A_t = G_t - V_φ(s_t)`。无偏，方差较高。
- *TD 优势：* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏（使用了 `V_φ`），方差低得多。也称为 *TD 残差* `δ_t`。

**n 步优势。** 在两者之间插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 是纯 TD。`n = ∞` 是 MC。大多数实现在 Atari 中使用 `n = 5`，在 MuJoCo 上的 PPO 中使用 `n = 2048`。

**广义优势估计（GAE）。** Schulman 等人（2016）提出对所有 n 步优势进行指数加权平均：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。`λ = 0` 是 TD（低方差，高偏差）。`λ = 1` 是 MC（高方差，无偏）。`λ = 0.95` 是 2026 年的默认值——调整直到偏差/方差旋钮到达你想要的位置。

**A2C：同步优势 actor-critic。** 在 `N` 个并行环境中收集 `T` 步。计算每一步的优势。在合并的批次上更新 actor 和 critic。重复。A3C 更简单、可扩展的兄弟。

**A3C：异步优势 actor-critic。** Mnih 等人（2016）。生成 `N` 个工作线程，每个运行一个环境。每个工作线程在自己的 rollout 上本地计算梯度，然后异步应用到共享参数服务器。不需要重放缓冲区——工作者通过运行不同的轨迹来去相关。A3C 证明了你可以在 CPU 上大规模训练。在 2026 年，基于 GPU 的 A2C（批处理并行环境）占主导地位，因为 GPU 需要大批量。

**联合损失。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三项：策略梯度损失、价值回归、熵正则项。`c_v ~ 0.5`，`c_e ~ 0.01` 是规范起始点。

```figure
actor-critic
```

## 构建它

### 步骤 1：一个 critic

线性 critic `V_φ(s) = w · features(s)`，使用 MSE 更新：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在表格环境中，critic 在几百个回合内收敛。在 Atari 上，用共享 CNN 主干 + 价值头替换线性 critic。

### 步骤 2：n 步优势

给定长度为 `T` 的 rollout 和引导的最终 `V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是 critic 的目标。`advantages` 是乘以 `∇ log π` 的量。

### 步骤 3：联合更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

on-policy，每次更新一个 rollout，actor 和 critic 使用不同的学习率。

### 步骤 4：并行化（A3C vs A2C）

- **A3C：** 启动 `N` 个线程。每个运行自己的环境和自己的前向传播。定期将梯度更新推送到共享主节点。主节点上不需要锁——竞争是可以接受的，它们只是增加了噪声。
- **A2C：** 在单个进程中运行 `N` 个环境实例，将观测堆叠到 `[N, obs_dim]` 批次中，进行批处理前向传播、批处理反向传播。更高的 GPU 利用率，确定性，更容易推理。2026 年的默认选择。

我们的玩具代码是单线程的，以保持清晰；重写为批处理 A2C 只需三行 numpy 代码。

## 陷阱

- **actor 梯度前的 critic 偏差。** 如果 critic 是随机的，其基线没有信息量，你在纯噪声上训练。在开启策略梯度之前预热 critic 几百步，或使用慢 actor 学习率。
- **优势归一化。** 按批次将优势归一化为零均值/单位标准差。以接近零的成本大幅稳定训练。
- **共享主干。** 在图像输入上使用共享特征提取器供 actor 和 critic 使用。分离的头部。共享特征从两个损失中搭便车。
- **on-policy 契约。** A2C 对数据恰好复用一次更新。更多则你的梯度是有偏的（重要性采样修正就是 PPO 添加的）。
- **熵坍塌。** 没有 `c_e > 0`，策略在几百次更新后变得接近确定性并停止探索。
- **奖励尺度。** 优势幅度取决于奖励尺度。归一化奖励（如使用运行标准差除法）以获得跨任务的一致梯度幅度。

## 使用它

A2C/A3C 在 2026 年很少是最终选择，但它们是后来所有方法 refin 的架构：

| 方法 | 与 A2C 的关系 |
|------|---------------|
| PPO | A2C + 裁剪的重要性比率用于多 epoch 更新 |
| IMPALA | A3C + V-trace off-policy 修正 |
| SAC（第 9 阶段 · 07） | 带软价值 critic 的 off-policy A2C（下节课） |
| GRPO（第 9 阶段 · 12） | 不带 critic 的 A2C——组相对优势 |
| DPO | A2C 坍缩到偏好排序损失，无采样 |
| AlphaStar / OpenAI Five | 带联盟训练 + 模仿预训练的 A2C |

如果你在 2026 年的论文中看到"advantage"，想想 actor-critic。

## 交付

保存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: 为给定环境生成 A2C / A3C / GAE 配置，包含指定的优势估计和损失权重。
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

给定环境和计算预算，输出：

1. 并行方式。A2C（GPU 批处理）vs A3C（CPU 异步）及工作者数量。
2. Rollout 长度 T。每次更新的每环境步数。
3. 优势估计器。n 步或 GAE(λ)；指定 λ。
4. 损失权重。`c_v`（价值）、`c_e`（熵）、梯度裁剪。
5. 学习率。Actor 和 critic（如果使用则分开设置）。

拒绝在 horizon > 1000 的环境上使用单 worker A2C（过于 on-policy，太慢）。拒绝在缺少优势归一化的情况下交付。对 `c_e = 0` 且观测熵 < 0.1 的运行标记为熵坍塌。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上使用 MC 优势（`G_t - V(s_t)`）训练 actor-critic。与第 6 课的带运行均值基线的 REINFORCE 比较样本效率。
2. **中等。** 切换到 TD 残差优势（`r + γ V(s') - V(s)`）。测量优势批次的方差。下降了多少？
3. **困难。** 实现 GAE(λ)。 Sweep `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`。绘制最终回报 vs 样本效率。该任务的偏差/方差甜蜜点在哪里？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Actor | "策略网络" | `π_θ(a\|s)`，通过策略梯度更新。 |
| Critic | "价值网络" | `V_φ(s)`，通过 MSE 回归更新到回报/TD 目标。 |
| Advantage | "比平均水平好多少" | `A(s, a) = Q(s, a) - V(s)` 或其估计量。`∇ log π` 的乘子。 |
| TD 残差 | "δ" | `δ_t = r + γ V(s') - V(s)`；一步优势估计。 |
| GAE | "插值旋钮" | n 步优势的指数加权求和，由 `λ` 参数化。 |
| A2C | "同步 actor-critic" | 跨环境批处理；每次 rollout 一次梯度步。 |
| A3C | "异步 actor-critic" | 工作线程将梯度推送到共享参数服务器。原始论文；2026 年较少见。 |
| Bootstrap | "在终点使用 V" | 截断 rollout，添加 `γ^n V(s_{t+n})` 闭合求和。 |

## 延伸阅读

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C，原始异步 actor-critic 论文。
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE。
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 基础；当 critic 是神经网络时，结合第 9 章函数近似阅读。
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — 带 V-trace off-policy 修正的可扩展分布式 actor-critic。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 值得阅读的 production A2C/PPO 实现。
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — 两时间尺度 actor-critic 分解的基础收敛结果。
