# 时间差分 — Q-Learning 与 SARSA

> 蒙特卡洛等待回合结束。TD 在每个步骤后通过自举下一个值估计进行更新。Q-learning 是离策略且乐观的；SARSA 是策略内且谨慎的。两者都只需一行代码。它们构成了本阶段所有深度强化学习方法的基石。

**类型：** 构建
**语言：** Python
**先修知识：** 阶段 9 · 01（MDP）、阶段 9 · 02（动态规划）、阶段 9 · 03（蒙特卡洛）
**时间：** 约 75 分钟

## 问题所在

蒙特卡洛方法可行，但它有两个代价高昂的要求。它需要能终止的回合，且仅在最终回报确定后才更新。如果你的回合有 1,000 步，MC 要等 1,000 步才能更新任何东西。它在实践中是高方差、低偏差且缓慢的。

动态规划则有相反的特性——零方差的自举备份——但需要已知模型。

时间差分（TD）学习折中了两者。从单个转移 `(s, a, r, s')`，构造一步目标 `r + γ V(s')` 并将 `V(s)` 向它拉近。无需模型，无需完整回合。由于在右侧使用了近似 `V` 而带来偏差，但相比 MC 方差显著降低，且从第一步就能在线更新。

这正是所有现代 RL——DQN、A2C、PPO、SAC——所依托的转折点。阶段 9 的其余内容是构建在你将在本课中编写的一步 TD 更新之上的函数近似层和技巧。

## 概念

![Q-learning 与 SARSA：离策略 max vs 策略内 Q(s', a')](../assets/td.svg)

**TD(0) 对 V 的更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

方括号中的量是 TD 误差 `δ = r + γ V(s') - V(s)`。它是 MC 中 `G_t - V(s_t)` 的在线模拟。收敛需要 `α` 满足 Robbins-Monro 条件（`Σ α = ∞`，`Σ α² < ∞`）且所有状态被无限次访问。

**Q-learning。** 一种用于控制的离策略 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max` 假设从 `s'` 开始将遵循*贪婪*策略，无论智能体实际采取什么动作。这种解耦使 Q-learning 能够学习 `Q*`，同时智能体通过 ε-贪婪进行探索。Mnih 等人（2015）将其转换为 Ataris 上的深度 Q-learning（第 05 课）。

**SARSA。** 一种策略内 TD 方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

名称来自元组 `(s, a, r, s', a')`。SARSA 使用智能体*实际*采取的下一个动作 `a'`，而非贪婪的 `argmax`。收敛到当前运行的 ε-贪婪 `π` 对应的 `Q^π`，当 `ε → 0` 时变为 `Q*`。

**悬崖行走的差别。** 在经典的悬崖行走任务中（跌落悬崖 = 奖励 -100），Q-learning 学习沿悬崖边缘的最优路径，但在探索期间偶尔会承受惩罚。SARSA 学习一条距离悬崖一步的安全路径，因为它将探索噪声纳入了其 Q 值考量。随着训练，两者在 `ε → 0` 时都能达到最优。在实践中这很重要：当部署时实际发生探索，SARSA 的行为更为保守。

**期望 SARSA。** 用 `π` 下的期望值替换 `Q(s', a')`：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

比 SARSA 方差更低（无需采样 `a'`），相同的策略内目标。通常是现代教材中的默认选择。

**n步 TD 和 TD(λ)。** 通过在自举前等待 `n` 步来插值 TD(0) 和 MC。`n=1` 是 TD，`n=∞` 是 MC。TD(λ) 使用几何权重 `(1-λ)λ^{n-1}` 对所有 `n` 求平均。大多数深度 RL 使用 `n` 在 3 到 20 之间。

```figure
qlearning-gridworld
```

## 构建它

### 步骤 1：在 ε-贪婪策略上的 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行代码。与 Q-learning 的唯一区别在于目标行。

### 步骤 2：Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 将目标与行为解耦。这一个符号就是策略内与离策略的区别。

### 步骤 3：学习曲线

追踪每 100 个回合的平均回报。Q-learning 在简单确定性 GridWorld 上收敛更快；SARSA 在悬崖行走中更为保守。在 `code/main.py` 中的 4×4 GridWorld 上，两者在 `α=0.1, ε=0.1` 下经过约 2,000 个回合即接近最优。

### 步骤 4：与 DP 真实值比较

运行价值迭代（第 02 课）以获得 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。一个健康的表格型 TD 代理在 4×4 GridWorld 上经过 10,000 个回合后可达到 `~0.5` 以内。

## 陷阱

- **初始 Q 值很重要。** 乐观初始化（对于负奖励任务设置 `Q = 0`）鼓励探索。悲观初始化可能永远困住贪婪策略。
- **α 调度。** 常数 `α` 对于非平稳问题是可以的。衰减的 `α_n = 1/n` 理论上保证收敛，但实际上太慢——将 `α` 固定于 `[0.05, 0.3]` 并监控学习曲线。
- **ε 调度。** 从较高值开始（`ε=1.0`），衰减至 `ε=0.05`。"GLIE"（无限探索下的极限贪婪）是收敛条件。
- **Q-learning 中的最大值偏差。** 当 `Q` 有噪声时，`max` 算子会向上有偏。导致高估——Hasselt 的双 Q-learning（在第 05 课中用于 DDQN）通过使用两个 Q 表来修复此问题。
- **非终止回合。** TD 可以在没有终止的情况下学习，但你需要要么限制步数，要么正确处理截止处的自举。标准做法：将截止视为非终止，继续自举。
- **状态哈希。** 如果状态是元组/张量，使用可哈希的键（元组而非列表；浮点数取整后的元组，而非原始值）。

## 使用它

2026 年的 TD 格局：

| 任务 | 方法 | 原因 |
|------|--------|--------|
| 小型表格环境 | Q-learning | 直接学习最优策略。 |
| 策略内安全关键 | SARSA / 期望 SARSA | 探索期间保守。 |
| 高维状态 | DQN（阶段 9 · 05） | 带回放和目标网络的神经网络 Q 函数。 |
| 连续动作 | SAC / TD3（阶段 9 · 07） | 对 Q 网络的 TD 更新；策略网络输出动作。 |
| LLM RL（基于奖励模型） | PPO / GRPO（阶段 9 · 08, 12） | 带 TD 风格优势的 Actor-Critic，通过 GAE。 |
| 离线 RL | CQL / IQL（阶段 9 · 08） | 带保守正则化的 Q-learning。 |

你在 2026 年论文中读到的 90% 的"RL"是 Q-learning 或 SARSA 的某种扩展。在深入阅读之前，先在手指上掌握表格更新的细节。

## 交付它

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: 为表格或小型特征 RL 任务在 Q-learning、SARSA、期望 SARSA 之间做出选择。
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

给定一个表格或小型特征环境，输出：

1. 算法。Q-learning / SARSA / 期望 SARSA / n步变体。一句话理由，关联策略内与离策略及方差。
2. 超参数。α, γ, ε, 衰减调度。
3. 初始化。Q_0 值（乐观 vs 零）及理由。
4. 收敛诊断。目标学习曲线，如可能则检查 `|Q - Q*|`。
5. 部署注意事项。探索在推理时将如何表现？是否需要 SARSA 的保守性？

拒绝将表格 TD 应用于超过 10⁶ 的状态空间。拒绝在没有最大值偏差警告的情况下交付 Q-learning 代理。标记任何在整个训练过程中 ε 保持为 1.0 的代理（无利用阶段）。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上实现 Q-learning 和 SARSA。绘制学习曲线（每 100 个回合的平均回报）共 2,000 个回合。谁收敛更快？
2. **中等。** 构建悬崖行走环境（4×12，最后一行是奖励 -100 并重置到起点的悬崖）。比较 Q-learning 和 SARSA 的最终策略。截图各自的路径。哪个更接近悬崖？
3. **困难。** 实现 Double Q-learning。在有噪声奖励的 GridWorld 上（每步奖励添加高斯噪声 σ=5），展示 Q-learning 显著高估 `V*(0,0)`，而 Double Q-learning 不会。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-------------------------|------------------------------------------------------------|
| TD 误差 | "更新信号" | `δ = r + γ V(s') - V(s)`，自举残差。 |
| TD(0) | "一步 TD" | 每次转移后使用仅下一步估计的更新。 |
| Q-learning | "离策略 RL 入门" | 对下一步动作取 `max` 的 TD 更新；无论行为策略如何都学习 `Q*`。 |
| SARSA | "策略内 Q-learning" | 使用实际下一步动作的 TD 更新；学习当前 ε-贪婪 `π` 的 `Q^π`。 |
| 期望 SARSA | "低方差 SARSA" | 用 `π` 下的期望替换采样动作 `a'`。 |
| GLIE | "正确的探索调度" | 无限探索下的极限贪婪；Q-learning 收敛所需。 |
| 自举 | "在目标中使用当前估计" | 区分 TD 与 MC 的关键。偏差来源但大幅降低方差。 |
| 最大化偏差 | "Q-learning 高估" | 对噪声估计取 `max` 会上偏；Double Q-learning 可修复。 |

## 延伸阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始论文及收敛证明。
- [Sutton & Barto (2018). 第 6 章 — 时间差分学习](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q-learning、期望 SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 最大化偏差的修复方案。
- [Seijen, Hasselt, Whiteson, Wiering (2009). 期望 SARSA 的理论与实证分析](https://ieeexplore.ieee.org/document/4927542) — 期望 SARSA 的动机。
- [Rummery & Niranjan (1994). 使用联结主义系统的在线 Q-learning](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — 首次提出 SARSA（当时称为"修正联结主义 Q-learning"）的论文。
- [Sutton & Barto (2018). 第 7 章 — n步自举](http://incompleteideas.net/book/RLbook2020.pdf) — 将 TD(0) 推广至 TD(n)，是从 Q-learning 到资格迹线再到后来 PPO 中 GAE 的路径。
