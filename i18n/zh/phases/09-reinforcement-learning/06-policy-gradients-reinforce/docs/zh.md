# Policy Gradient — 从零实现 REINFORCE

> 停止估计价值。直接参数化策略，计算期望回报的梯度，沿梯度方向上升。Williams (1992) 用一个定理写下了它。这就是 PPO、GRPO 以及所有 LLM RL 循环存在的原因。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 3 · 03（反向传播）、Phase 9 · 03（蒙特卡洛）、Phase 9 · 04（TD 学习）
**时间：** ~75 分钟

## 问题所在

Q-learning 和 DQN 参数化的是*价值*函数。你通过 `argmax Q` 来选择动作。这在离散动作和离散状态下没有问题，但当动作是连续时就会崩溃（对 10 维扭矩做 `argmax`？），或者当你想要一个随机策略时（`argmax` 本质上是确定性的）。

策略梯度则参数化的是*策略*本身。`π_θ(a | s)` 是一个神经网络，输出动作上的分布。从中采样来执行动作。计算相对于 `θ` 的期望回报梯度。沿梯度上升。不需要 `argmax`。不需要 Bellman 递归。只是在 `J(θ) = E_{π_θ}[G]` 上做梯度上升。

REINFORCE 定理（Williams 1992）告诉你这个梯度是可计算的：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。运行一个回合。计算回报。在每一步乘以 `∇ log π_θ(a | s)`。取平均。梯度上升。完成。

2026 年的所有 LLM-RL 算法 —— PPO、DPO、GRPO —— 都是 REINFORCE 的改进版本。能够手搓理解它是本阶段其余内容，以及 Phase 10 · 07（RLHF 实现）和 Phase 10 · 08（DPO）的前提。

## 核心概念

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**策略梯度定理。** 对于任意由 `θ` 参数化的策略 `π_θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 是从步骤 `t` 开始的折现回报。期望是对从 `π_θ` 采样的完整轨迹 `τ` 取的。

**证明很短。** 对期望下的 `J(θ) = Σ_τ P(τ; θ) G(τ)` 求导。使用 `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`（log-derivative 技巧）。将 `log P(τ; θ) = Σ log π_θ(a_t | s_t) + 不依赖于 θ 的环境项` 展开。环境项消失。两行代数运算即可得到该定理。

**方差缩减技巧。**  vanilla REINFORCE 的方差极大 —— 回报有噪声，`∇ log π` 有噪声，它们的乘积非常 noisy。两个标准修复方法：

1. **基线相减。** 用 `G_t - b(s_t)` 替换 `G_t`，其中 `b(s_t)` 是任何不依赖于 `a_t` 的基线。由于 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`，该操作保持无偏。典型选择：`b(s_t) = V̂(s_t)`，由评论家学习 → actor-critic（Lesson 07）。
2. **奖励往后计。** 用 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)` 替换 `Σ_t G_t · ∇ log π_θ(a_t | s_t)`。对于一个给定动作，只有未来回报有影响 —— 过去的奖励贡献零均值噪声。

合并后得到：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

即带基线的 REINFORCE —— A2C（Lesson 07）和 PPO（Lesson 08）的直接祖先。

**Softmax 策略参数化。** 对于离散动作，标准选择：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中 `f_θ` 是任何输出每个动作得分的神经网络。梯度有简洁形式：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

即被选动作的得分减去策略下的期望得分。

**连续动作的高斯策略。** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 有闭式解。这就是 Phase 9 · 07 中 SAC 所需的全部内容。

```figure
policy-gradient-landscape
```

## 动手实现

### 步骤 1：softmax 策略网络

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

对于表格型环境使用线性策略（每个动作一个权重向量）。对于 Atari，换成 CNN 并保留 softmax 头部。

### 步骤 2：采样和对数概率

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

### 步骤 3：捕获 log-probs 的回放

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

### 步骤 4：REINFORCE 更新

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

梯度 `∇ log π(a|s) = e_a - π(·|s)`（`a` 的 onehot 减去概率分布）是 softmax 策略梯度的核心。把它变成肌肉记忆。

### 步骤 5：基线

在近期回合上对 `G` 维护一个滑动平均值，足以将 4×4 GridWorld 跑通；大约需要 ~500 个回合收敛。将基线升级为可学习的 `V̂(s)`，你就得到了 actor-critic。

## 常见陷阱

- **梯度爆炸。** 回报可能非常大。在乘以 `∇ log π` 之前，始终对批次内的 `G` 进行归一化到 `~N(0, 1)`。
- **熵坍塌。** 策略过早收敛到接近确定性的动作，停止探索，陷入局部最优。修复方法：在目标函数中添加熵 bonus `β · H(π(·|s))`。
- **高方差。** vanilla REINFORCE 需要数千个回合。评论家基线（Lesson 07）或 TRPO/PPO 的信任域（Lesson 08）是标准修复。
- **样本效率低。** on-policy 意味着你在一次更新后丢弃所有转换。通过重要性采样进行 off-policy 校正可以回收数据，但代价是方差（PPO 的 ratio 就是裁剪的重要性采样权重）。
- **非平稳梯度。** 100 回合前获得的同一个梯度使用的是旧的 `π`。on-policy 方法因此每隔几个 rollout 就更新一次。
- **信用分配。** 不使用 reward-to-go 时，过去的奖励会贡献噪声。始终使用 reward-to-go。

## 实际应用

在 2026 年，REINFORCE 很少直接运行，但其梯度公式无处不在：

| 用例 | 衍生方法 |
|------|----------|
| 连续控制 | 使用高斯策略的 PPO / SAC |
| LLM RLHF | 带 KL 惩罚的 PPO，在 token 级策略上运行 |
| LLM 推理（DeepSeek） | GRPO —— 带组相对基线的 REINFORCE，无需评论家 |
| 多智能体 | 集中式评论家 REINFORCE（MADDPG、COMA） |
| 离散动作机器人 | A2C、A3C、PPO |
| 仅偏好设置 | DPO —— 将 REINFORCE 重写为偏好似然损失，无需采样 |

当你在 2026 年的训练脚本中看到 `loss = -advantage * log_prob` 时，那就是带基线的 REINFORCE。整篇论文（DPO、GRPO、RLOO）都是建立在这一行之上的方差缩减技巧。

## 交付物

保存为 `outputs/skill-policy-gradient-trainer.md`：

```markdown
---
name: policy-gradient-trainer
description: 为给定任务生成 REINFORCE / actor-critic / PPO 训练配置，并诊断方差问题。
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

给定一个环境（离散/连续动作、回合长度、奖励统计），输出：

1. 策略头部。Softmax（离散）或高斯（连续），含参数量。
2. 基线。无（vanilla）、滑动平均、可学习 `V̂(s)`、或 A2C 评论家。
3. 方差控制。默认启用 reward-to-go、回报归一化、梯度裁剪值。
4. 熵 bonus。系数 β 及衰减调度。
5. 批次大小。每次更新的回合数；on-policy 数据新鲜度约定。

拒绝在回合长度 > 500 步时不使用基线的 REINFORCE。拒绝使用 softmax 头部进行连续动作控制。标记任何 `β = 0` 且观测到策略熵 < 0.1 的运行作为熵坍塌。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 REINFORCE，使用线性 softmax 策略。不带基线训练 1,000 个回合。绘制学习曲线；测量方差（回报的标准差）。
2. **中等。** 添加滑动平均基线。重新训练。与 vanilla 运行对比样本效率和方差。基线将收敛所需的步数减少了多少？
3. **困难。** 添加熵 bonus `β · H(π)`。遍历 `β ∈ {0, 0.01, 0.1, 1.0}`。绘制最终回报和策略熵。此任务上的最佳参数在哪里？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 策略梯度 | "直接训练策略" | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`；由 log-derivative 技巧推导而来。 |
| REINFORCE | "原始 PG 算法" | Williams (1992)；蒙特卡洛回报乘以策略对数梯度。 |
| Log-derivative 技巧 | "得分函数估计量" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`；使得期望的梯度可计算。 |
| 基线 | "方差缩减" | 从 `G` 中减去的任意 `b(s)`；无偏因为 `E[b · ∇ log π] = 0`。 |
| 奖励往后计 | "只有未来回报才算数" | 用 `G_t^{from t}` 而非完整的 `G_0`；更正确且方差更低。 |
| 熵 bonus | "鼓励探索" | `+β · H(π(·\|s))` 项，防止策略坍塌。 |
| On-policy | "用刚看到的数据训练" | 梯度期望是关于当前策略的 —— 不能直接复用旧数据。 |
| 优势函数 | "比平均水平好多少" | `A(s, a) = G(s, a) - V(s)`；带基线的 REINFORCE 所乘的有符号量。 |

## 延伸阅读

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — 原始 REINFORCE 论文。
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — 带函数逼近的现代策略梯度定理。
- [Sutton & Barto (2018). 第 13 章 —— 策略梯度方法](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书式讲解。
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — 清晰的教学性讲解，附带 PyTorch 代码。
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — 方差缩减和自然梯度视角，将 REINFORCE 与信任域系列（TRPO、PPO）联系起来。
