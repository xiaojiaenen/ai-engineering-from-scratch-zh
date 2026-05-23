# 策略梯度 —— 从零实现 REINFORCE

> 停止估计价值。直接参数化策略，计算期望回报的梯度，沿梯度上升方向前进。Williams (1992) 用一个定理就写明白了。这就是 PPO、GRPO 以及所有大语言模型强化学习循环存在的原因。

**类型：** 构建
**语言：** Python
**先决条件：** 第 3 阶段 · 03 (反向传播), 第 9 阶段 · 03 (蒙特卡洛), 第 9 阶段 · 04 (时序差分学习)
**时长：** 约 75 分钟

## 问题所在

Q-learning 和 DQN 参数化*价值*函数。你通过 `argmax Q` 选择动作。这在离散动作和离散状态空间中尚可。但当动作是连续的时（`argmax` 遍历 10 维扭矩空间？），或者当你想要一个随机性策略时（`argmax` 在结构上就是确定性的），它就失效了。

策略梯度方法则参数化*策略*。`π_θ(a | s)` 是一个神经网络，输出动作的分布。从中采样来执行动作。计算期望回报关于 `θ` 的梯度。沿梯度上升方向前进。没有 `argmax`。没有贝尔曼递推。只有在 `J(θ) = E_{π_θ}[G]` 上的梯度上升。

REINFORCE 定理 (Williams 1992) 告诉你这个梯度是可计算的：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。运行一个回合。计算回报。在每一步乘以 `∇ log π_θ(a | s)`。求平均。梯度上升。完成。

2026 年的每一个大语言模型-强化学习算法——PPO, DPO, GRPO——都是 REINFORCE 的变体。深刻理解它是本阶段后续内容以及第 10 阶段 · 07 (RLHF 实现) 和第 10 阶段 · 08 (DPO) 的先决条件。

## 核心概念

![策略梯度：softmax 策略，log-π 梯度，回报加权更新](../assets/policy-gradient.svg)

**策略梯度定理。** 对于任何由 `θ` 参数化的策略 `π_θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 是从步 `t` 开始的折扣回报。期望是对从 `π_θ` 采样的完整轨迹 `τ` 取的。

**证明很简短。** 对期望下的 `J(θ) = Σ_τ P(τ; θ) G(τ)` 求导。使用 `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)` (对数导数技巧)。分解 `log P(τ; θ) = Σ log π_θ(a_t | s_t) + environment terms that do not depend on θ`。环境项消失。两行代数运算就得到了这个定理。

**方差减小技巧。** 原始 REINFORCE 方差巨大——回报是噪声的，`∇ log π` 是噪声的，它们的乘积噪声非常大。两个标准解决方法：

1.  **基线减法。** 对于任何不依赖于 `a_t` 的基线 `b(s_t)`，用 `G_t - b(s_t)` 替换 `G_t`。因为 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`，这是无偏的。典型选择：由评论家学习的 `b(s_t) = V̂(s_t)` → Actor-Critic (第 07 课)。
2.  **回报到动作 (Reward-to-go)。** 用 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)` 替换 `Σ_t G_t · ∇ log π_θ(a_t | s_t)`。对于给定的动作，只有未来的回报才重要——过去的回报贡献零均值噪声。

两者结合，你得到：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

这就是带有基线的 REINFORCE——它是 A2C (第 07 课) 和 PPO (第 08 课) 的直接前身。

**Softmax 策略参数化。** 对于离散动作，标准选择：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中 `f_θ` 是为每个动作输出一个分数的神经网络。梯度具有简洁的形式：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

即，被采取动作的分数减去其在该策略下的期望值。

**用于连续动作的高斯策略。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 有解析形式。这就是第 9 阶段 · 07 的 SAC 所需要的全部。

## 动手构建

### 第 1 步：softmax 策略网络

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

对于表格环境，使用线性策略（每个动作一个权重向量）。对于 Atari 环境，替换为 CNN 并保留 softmax 头。

### 第 2 步：采样与对数概率

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### 第 3 步：捕获对数概率的轨迹采集

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### 第 4 步：REINFORCE 更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

梯度 `∇ log π(a|s) = e_a - π(·|s)`（`a` 的 onehot 向量减去概率）是 softmax 策略梯度的核心。请将其刻入肌肉记忆。

### 第 5 步：基线

在最近回合中 `G` 的运行平均值足以减小方差，让 4x4 网格世界运行起来；它大约需要 500 个回合来收敛。将基线升级为一个可学习的 `V̂(s)`，你就得到了 Actor-Critic。

## 陷阱

-   **梯度爆炸。** 回报可能非常大。在乘以 `∇ log π` 之前，务必对批次中的 `G` 进行归一化，使其均值为 0，方差为 1。
-   **熵坍缩。** 策略过早地收敛到接近确定性的动作，停止探索，陷入停滞。修复方法：在目标函数中添加熵奖励 `β · H(π(·|s))`。
-   **高方差。** 原始 REINFORCE 需要数千个回合。评论家基线（第 07 课）或 TRPO/PPO 的信任域（第 08 课）是标准的解决方法。
-   **样本效率低。** 在策略 (On-policy) 意味着每次更新后就丢弃所有转换数据。通过重要性采样的离策略 (Off-policy) 修正可以重新利用数据，但代价是增加了方差（PPO 的比率是一个裁剪后的重要性权重）。
-   **非平稳梯度。** 100 个回合前的相同梯度使用的是旧的 `π`。因此，在策略方法每隔几次轨迹采集就更新一次。
-   **信用分配问题。** 没有回报到动作 (Reward-to-go)，过去的回报会带来噪声。请务必使用回报到动作。

## 实际应用

在 2026 年，REINFORCE 很少被直接运行，但其梯度公式无处不在：

| 用例 | 派生方法 |
|----------|---------------|
| 连续控制 | 使用高斯策略的 PPO / SAC |
| 大语言模型 RLHF | 带有 KL 惩罚的 PPO，运行在 token 级别策略上 |
| 大语言模型推理 (DeepSeek) | GRPO —— 带有组相对基线的 REINFORCE，无需评论家 |
| 多智能体 | 集中化评论家 REINFORCE (MADDPG, COMA) |
| 离散动作机器人 | A2C, A3C, PPO |
| 仅有偏好设置 | DPO —— 重写为偏好似然损失的 REINFORCE，无需采样 |

当你在 2026 年的训练脚本中读到 `loss = -advantage * log_prob` 时，那就是带有基线的 REINFORCE。整篇论文（DPO, GRPO, RLOO）都是在这一行基础上进行的方差减小技巧。

## 交付

保存为 `outputs/skill-policy-gradient-trainer.md`：

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
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
```

## 练习

1.  **简单。** 在 4x4 网格世界上，使用线性 softmax 策略实现 REINFORCE。不加基线训练 1,000 个回合。绘制学习曲线；测量方差（回报的标准差）。
2.  **中等。** 添加一个运行均值基线。再次训练。比较样本效率和方差与原始运行的区别。基线将收敛步数减少了多少？
3.  **困难。** 添加熵奖励 `β · H(π)`。扫描 `β ∈ {0, 0.01, 0.1, 1.0}`。绘制最终回报和策略熵。在此任务上，最佳平衡点在哪里？

## 关键术语

| 术语 | 人们常说 | 其实际含义 |
|------|-----------------|-----------------------|
| 策略梯度 | "直接训练策略" | `∇J(θ) = E[G · ∇ log π_θ(a|s)]`；源于对数导数技巧。 |
| REINFORCE | "最初的策略梯度算法" | Williams (1992)；蒙特卡洛回报乘以对数策略梯度。 |
| 对数导数技巧 | "得分函数估计器" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`；使得期望的梯度易于处理。 |
| 基线 | "方差减小" | 从 `G` 中减去的任何 `b(s)`；因为 `E[b · ∇ log π] = 0`，所以是无偏的。 |
| 回报到动作 | "只有未来回报才重要" | 使用 `G_t^{from t}` 而非完整的 `G_0`；正确且方差更低。 |
| 熵奖励 | "鼓励探索" | `+β · H(π(·|s))` 项防止策略坍缩。 |
| 在策略 | "在刚刚看到的数据上训练" | 梯度期望是相对于当前策略的——不能直接重用旧数据。 |
| 优势 | "比平均好多少" | `A(s, a) = G(s, a) - V(s)`；带有基线的 REINFORCE 所乘的带符号量。 |

## 延伸阅读

-   [Williams (1992). 简单的连接主义强化学习梯度跟随算法](https://link.springer.com/article/10.1007/BF00992696) —— 原始 REINFORCE 论文。
-   [Sutton 等 (2000). 带有函数逼近的强化学习策略梯度方法](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) —— 带有函数逼近的现代策略梯度定理。
-   [Sutton & Barto (2018). 第 13 章 — 策略梯度方法](http://incompleteideas.net/book/RLbook2020.pdf) —— 教科书中的表述。
-   [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) —— 带有 PyTorch 代码的清晰教学讲解。
-   [Peters & Schaal (2008). 使用策略梯度的运动技能强化学习](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) —— 方差减小以及将 REINFORCE 与信任域家族（TRPO, PPO）联系起来的自然梯度视角。