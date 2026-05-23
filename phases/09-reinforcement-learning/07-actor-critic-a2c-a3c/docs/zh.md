# Actor-Critic — A2C and A3C

> REINFORCE 算法噪声很大。添加一个学习 `V̂(s)` 的评论家，将其从回报中减去，你将获得一个期望相同但方差低得多的优势函数。这就是演员-评论家方法。A2C 以同步方式运行；A3C 在多个线程上运行。这两者是所有现代深度强化学习方法的核心思想。

**类型:** 构建
**语言:** Python
**前提知识:** 第9阶段 · 04 (时序差分学习), 第9阶段 · 06 (REINFORCE)
**时间:** 约75分钟

## 问题所在

普通的 REINFORCE 可以工作，但其方差极其糟糕。蒙特卡洛回报 `G_t` 在不同回合之间的波动幅度可能超过10倍。将这种噪声乘以 `∇ log π` 并进行平均，会产生一个需要成千上万回合才能将策略推动到同等距离的梯度估计器，而使用少得多的 DQN 更新就能达到相同的效果。

方差来源于使用原始回报。如果你减去一个基线 `b(s_t)` —— 任何关于状态的函数，包括学习到的价值 —— 期望不变，方差会降低。最佳的易处理基线是 `V̂(s_t)`。现在，乘以 `∇ log π` 的量就是*优势*：

`A(s, a) = G - V̂(s)`

如果一个动作产生了高于平均的回报，它就是好的；如果低于平均，就是坏的。带有学习到的评论家的 REINFORCE 就是*演员-评论家*方法。评论家为演员提供了一个低方差的教师。这是2015年之后每一种深度策略方法（A2C, A3C, PPO, SAC, IMPALA）的基础。

## 核心概念

![演员-评论家：策略网络加价值网络，TD残差作为优势](../assets/actor-critic.svg)

**两个网络，一个共享损失：**

- **演员** `π_θ(a | s)`：策略。通过采样来执行动作。使用策略梯度进行训练。
- **评论家** `V_φ(s)`：估计从状态开始的预期回报。训练目标是最小化 `(V_φ(s) - target)²`。

**优势函数。** 两种标准形式：

- *蒙特卡洛优势：* `A_t = G_t - V_φ(s_t)`。无偏估计，方差较高。
- *TD优势：* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。有偏估计（使用了 `V_φ`），方差远低于前者。也称为 *TD残差* `δ_t`。

**n步优势函数。** 在两者之间进行插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1` 是纯TD。`n = ∞` 是蒙特卡洛。大多数实现对雅达利游戏使用 `n = 5`，对 MuJoCo 上的 PPO 使用 `n = 2048`。

**广义优势估计 (GAE)。** Schulman 等人 (2016) 提出了一种对所有 n 步优势函数进行指数加权平均的方法：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。`λ = 0` 是 TD（低方差，高偏差）。`λ = 1` 是 MC（高方差，无偏）。`λ = 0.95` 是 2026 年的默认选择——调整到你想要的偏差/方差平衡点。

**A2C：同步优势演员-评论家。** 在 `N` 个并行环境中收集 `T` 步数据。计算每一步的优势。在合并的批次上更新演员和评论家。重复。它是 A3C 的更简单、更具可扩展性的版本。

**A3C：异步优势演员-评论家。** Mnih 等人 (2016)。启动 `N` 个工作线程，每个线程运行一个环境。每个工作线程在自己的轨迹上本地计算梯度，然后异步地将梯度应用到共享的参数服务器。不需要经验回放缓冲区——工作线程通过运行不同的轨迹来去相关。A3C 证明了可以在大规模 CPU 上进行训练。在2026年，基于 GPU 的 A2C（批量并行环境）占主导地位，因为 GPU 需要大批量数据。

**组合损失函数。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

三个项：策略梯度损失、价值回归损失、熵奖励。`c_v ~ 0.5`，`c_e ~ 0.01` 是标准的起始点。

## 动手实现

### 第一步：构建评论家

使用均方误差更新的线性评论家 `V_φ(s) = w · features(s)`：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在表格环境中，评论家会在几百回合内收敛。在雅达利游戏中，用共享的 CNN 主干网络 + 价值头替换线性评论家。

### 第二步：n步优势函数

给定长度为 `T` 的轨迹和一个自举的最终值 `V(s_T)`：

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

`returns` 是评论家的目标。`advantages` 是乘以 `∇ log π` 的量。

### 第三步：组合更新

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

在线策略，每次更新使用一次轨迹，为演员和评论家设置独立的学习率。

### 第四步：并行化 (A3C vs A2C)

- **A3C：** 启动 `N` 个线程。每个线程运行自己的环境和自己的前向传播。定期将梯度更新推送到共享的主服务器。主服务器上没有锁——竞争是允许的，它们只是增加了噪声。
- **A2C：** 在单个进程中运行 `N` 个环境实例，将观测值堆叠成一个 `[N, obs_dim]` 的批次，进行批量前向传播，批量反向传播。GPU 利用率更高，结果确定性更强，更容易推理。这是2026年的默认选择。

我们的示例代码为了清晰是单线程的；将其重写为批量 A2C 只需要三行 numpy 代码。

## 注意事项

- **评论家在演员梯度前的偏差。** 如果评论家是随机初始化的，它的基线将毫无信息量，你将在纯噪声上进行训练。在开启策略梯度之前，先将评论家预热几百步，或者为演员使用较低的学习率。
- **优势归一化。** 对每个批次的优势进行零均值/单位方差归一化。能以接近零的代价极大地稳定训练。
- **共享主干。** 对于图像输入，为演员和评论家使用共享的特征提取器，但使用独立的输出头。共享特征可以搭两个损失的便车。
- **在线策略契约。** A2C 恰好将数据重用一次。重用更多次会导致梯度有偏（重要性采样修正正是 PPO 所添加的）。
- **熵坍塌。** 没有 `c_e > 0`，策略会在几百次更新后变得近乎确定性，并停止探索。
- **奖励尺度。** 优势的幅度取决于奖励的尺度。对奖励进行归一化（例如，除以运行标准差）可以保持跨任务的梯度幅度一致性。

## 应用场景

A2C/A3C 在2026年很少是最终选择，但它们是后续所有方法进行改进的基础架构：

| 方法 | 与 A2C 的关系 |
|--------|----------------|
| PPO | A2C + 裁剪的重要性比率，用于多轮次更新 |
| IMPALA | A3C + V-trace 离策略修正 |
| SAC (第9阶段 · 07) | 具有软价值评论家的离策略 A2C (下一课) |
| GRPO (第9阶段 · 12) | 没有评论家的 A2C —— 组相对优势 |
| DPO | 将 A2C 压缩为偏好排序损失，无需采样 |
| AlphaStar / OpenAI Five | 具有联赛训练 + 模仿预训练的 A2C |

如果你在2026年的论文中看到“优势”，请想到演员-评论家方法。

## 部署它

保存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
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
```

## 练习

1. **简单。** 在 4×4 网格世界上，使用 MC 优势 (`G_t - V(s_t)`) 训练演员-评论家。与第06课的带有运行均值基线的 REINFORCE 比较样本效率。
2. **中等。** 切换到 TD残差优势 (`r + γ V(s') - V(s)`)。测量优势批次的方差。方差降低了多少？
3. **困难。** 实现 GAE(λ)。扫描 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`。绘制最终回报 vs 样本效率的图表。对于此任务，偏差/方差的最佳平衡点在哪里？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 演员 | "策略网络" | `π_θ(a|s)`，通过策略梯度更新。 |
| 评论家 | "价值网络" | `V_φ(s)`，通过均方误差回归到回报/TD目标来更新。 |
| 优势 | "比平均水平好多少" | `A(s, a) = Q(s, a) - V(s)` 或其估计器。`∇ log π` 的乘数。 |
| TD残差 | "δ" | `δ_t = r + γ V(s') - V(s)`；一步优势估计。 |
| GAE | "插值旋钮" | n步优势的指数加权和，由 `λ` 参数化。 |
| A2C | "同步演员-评论家" | 跨环境批量处理；每次轨迹更新一次梯度。 |
| A3C | "异步演员-评论家" | 工作线程将梯度推送到共享参数服务器。原始论文；在2026年较少见。 |
| 自举 | "在时间步末端使用 V" | 截断轨迹，加上 `γ^n V(s_{t+n})` 来闭合求和。 |

## 延伸阅读

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) —— A3C，原始的异步演员-评论家论文。
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) —— GAE。
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) —— 基础；当评论家是神经网络时，将此与第9章的函数逼近结合阅读。
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) —— 具有 V-trace 离策略修正的可扩展分布式演员-评论家。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) —— 值得一读的生产级 A2C/PPO 实现。
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) —— 两时间尺度演员-评论家分解的奠基性收敛结果。