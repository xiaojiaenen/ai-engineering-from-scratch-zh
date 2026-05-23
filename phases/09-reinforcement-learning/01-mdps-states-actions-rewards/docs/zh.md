# 马尔可夫决策过程、状态、动作与奖励

> 马尔可夫决策过程包含五个要素：状态、动作、转移、奖励、折扣。强化学习中的一切算法——Q-learning、PPO、DPO、GRPO——都是围绕这个结构进行优化。学好它，就能免费阅读强化学习的后续所有内容。

**类型：** 学习
**语言：** Python
**先修课程：** 第一阶段 · 06（概率与分布），第二阶段 · 01（机器学习分类）
**时间：** ~45 分钟

## 问题

你正在编写一个国际象棋机器人。或者是一个库存规划器。或者是一个交易智能体。或者是用于训练推理模型的PPO循环。四个不同的领域，却有一个惊人的事实：它们都归结为同一个数学对象。

监督学习给你提供 `(x, y)` 对，并让你拟合一个函数。强化学习不提供标签——只有一连串的状态、你采取的动作和一个标量奖励。这一步棋赢了吗？补货决策节省了钱吗？这笔交易盈利了吗？大语言模型刚生成的这个token是否从评判那里获得了更高的奖励？

在你将其形式化之前，你无法从这串数据流中学习。“我看到的”、“我所做的”、“接下来发生了什么”、“这有多好”——每一项都必须变成一个你可以推理的对象。这个形式化过程就是一个马尔可夫决策过程。本阶段的每一个强化学习算法，包括结尾的RLHF和GRPO循环，都是围绕这个结构进行优化。

## 概念

![马尔可夫决策过程：状态、动作、转移、奖励、折扣](../assets/mdp.svg)

**五个对象。**

- **状态** `S`。智能体做决策所需的一切信息。在GridWorld中是网格位置。在国际象棋中是棋盘局面。在大语言模型中是上下文窗口加上任何记忆。
- **动作** `A`。选择项。上/下/左/右移动。走一步棋。生成一个token。
- **转移** `P(s' | s, a)`。给定状态 `s` 和动作 `a`，下一状态的分布。国际象棋中是确定性的，库存管理中是随机的，大语言模型解码中几乎是确定性的。
- **奖励** `R(s, a, s')`。标量信号。获胜=+1，失败=-1。收入减去成本。GRPO中的对数似然比项。
- **折扣** `γ ∈ [0, 1)`。未来奖励相对于当前奖励的重要程度。`γ = 0.99` 带来大约100步的视野；`γ = 0.9` 带来大约10步。

**马尔可夫性质** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`。未来仅取决于当前状态。如果不是这样，那么是状态表示不完整——这不是方法的失败，而是状态的失败。

**策略与回报。** 策略 `π(a | s)` 将状态映射到动作分布。回报 `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` 是未来奖励的折扣总和。价值 `V^π(s) = E[G_t | s_t = s]` 是在策略 `π` 下，从 `s` 开始的期望回报。Q值 `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` 是从特定动作开始的期望回报。每个强化学习算法都估计这两个中的一个，然后相应地改进 `π`。

**贝尔曼方程。** 本阶段所有算法都使用的不动点方程：

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

这些方程将期望回报分解为“这一步的奖励”加上“你到达位置的折扣价值”。递归。第9阶段的每个算法要么通过迭代这个方程直到收敛（动态规划），要么从中采样（蒙特卡洛），或者进行一步自举（时序差分）。

## 实现它

### 步骤1：一个微型确定性MDP

一个4×4的GridWorld。智能体从左上角开始，右下角为终止状态，每步奖励-1，动作集 `{up, down, left, right}`。见 `code/main.py`。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

五行代码。这就是整个环境。确定性转移，恒定步长惩罚，吸收性终止状态。

### 步骤2：执行一个策略

策略是一个从状态到动作分布的函数。最简单的：均匀随机。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

运行随机策略1000次。在这个4×4棋盘上，平均回报大约在-60到-80之间。最优回报是-6（沿右下角直线的路径）。缩小这个差距就是第9阶段的全部内容。

### 步骤3：通过贝尔曼方程精确计算 `V^π`

对于小型MDP，贝尔曼方程是一个线性系统。枚举状态，应用期望，迭代直到价值不再变化。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

这是迭代策略评估。它是Sutton & Barto中的第一个算法，也是后续所有强化学习方法的理论基础。

### 步骤4：`γ` 是一个具有物理意义的超参数

有效视野大约是 `1 / (1 - γ)`。`γ = 0.9` → 10步。`γ = 0.99` → 100步。`γ = 0.999` → 1000步。

太低会导致智能体目光短浅。太高会导致信用分配变得有噪声，因为许多早期步骤共同承担着对未来遥远奖励的责任。大语言模型的RLHF通常使用 `γ = 1`，因为情节短且有限。控制任务使用 `0.95–0.99`。长时程策略游戏使用 `0.999`。

## 陷阱

- **非马尔可夫状态。** 如果你需要最后三个观测值来做决策，那么“状态”不仅仅是当前观测值。解决方法：堆叠帧（DQN在Atari上堆叠4帧）或使用循环状态（基于观测的LSTM/GRU）。
- **稀疏奖励。** 仅在大状态空间中获胜的奖励几乎使学习不可能。塑造奖励（中间信号）或用模仿学习进行自举（第9阶段 · 09）。
- **奖励黑客。** 优化代理奖励常常导致病态行为。OpenAI的赛艇智能体转着圈收集无限增益道具，而不是完成比赛。始终根据目标结果定义奖励，而不是代理。
- **折扣误设。** 在无限时程任务中使用 `γ = 1` 会使每个价值变为无穷大。总是通过有限时程或 `γ < 1` 来限制。
- **奖励缩放。** {+100, -100} 与 {+1, -1} 的奖励给出相同的最优策略，但梯度幅度截然不同。在输入PPO/DQN之前，将其归一化到 `[-1, 1]` 左右。

## 使用它

2026年的技术栈在编写代码之前，将每个强化学习管道简化为一个MDP：

| 场景 | 状态 | 动作 | 奖励 | γ |
|------|-------|--------|--------|---|
| 控制（运动、操作） | 关节角度+速度 | 连续扭矩 | 特定任务塑造 | 0.99 |
| 游戏（国际象棋、围棋、扑克） | 棋盘+历史 | 合法走法 | 赢=+1 / 输=-1 | 1.0 (有限) |
| 库存/定价 | 库存+需求 | 订货量 | 收入 - 成本 | 0.95 |
| 大语言模型的RLHF | 上下文token | 下一个token | 最终的奖励模型分数 | 1.0 (情节约200 token) |
| 推理任务的GRPO | 提示+部分响应 | 下一个token | 最终验证器0/1 | 1.0 |

在编写任何训练循环之前，先写出这五元组。大多数“强化学习不管用”的错误报告，追溯起来都是一个在纸面上就已损坏的MDP公式。

## 部署它

保存为 `outputs/skill-mdp-modeler.md`：

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## 练习

1. **简单。** 在 `code/main.py` 中实现4×4 GridWorld和随机策略执行。运行10,000个情节。报告回报的均值和标准差。与最优回报（-6）进行比较。
2. **中等。** 对均匀随机策略运行 `policy_evaluation`，设置 `γ ∈ {0.5, 0.9, 0.99}`。将 `V` 作为4×4网格打印出来。解释为什么靠近终止状态的状态价值随着 `γ` 增大而增长更快。
3. **困难。** 将GridWorld随机化：每个动作有 `p = 0.1` 的概率滑向相邻方向。重新评估均匀策略。`V[start]` 是变好还是变差？为什么？

## 关键术语

| 术语 | 人们常说的 | 其实际含义 |
|------|-----------------|-----------------------|
| MDP | “强化学习设置” | 满足马尔可夫性质的元组 `(S, A, P, R, γ)`。 |
| 状态 | “智能体看到的” | 在所选策略类下，对未来动态的充分统计量。 |
| 策略 | “智能体的行为” | 条件分布 `π(a | s)` 或确定性映射 `s → a`。 |
| 回报 | “总奖励” | 从当前步开始的折扣总和 `Σ γ^t r_t`。 |
| 价值 | “一个状态有多好” | 在 `π` 下，从 `s` 开始的期望回报。 |
| Q值 | “一个动作有多好” | 在 `π` 下，从 `s` 开始，采取第一个动作 `a` 的期望回报。 |
| 贝尔曼方程 | “动态规划递归” | 将价值/Q值分解为一步奖励加上后继状态折扣价值的不动点分解。 |
| 折扣 `γ` | “未来 vs 现在” | 对未来遥远奖励的几何权重；有效视野 `~1/(1-γ)`。 |

## 延伸阅读

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 教材。第3章涵盖MDP和贝尔曼方程；第1章激励了奖励假设，这是后续所有课程的基础。
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — 贝尔曼方程的起源。
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — 从深度强化学习角度简明介绍MDP。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 运筹学领域关于MDP和精确求解方法的参考书。
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — 将MDP作为动态规划特例的最清晰推导。