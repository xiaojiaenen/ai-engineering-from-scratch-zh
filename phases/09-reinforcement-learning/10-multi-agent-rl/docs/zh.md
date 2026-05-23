# 多智能体强化学习

> 单智能体强化学习假设环境是静态的。将两个学习中的智能体放入同一世界，这个假设便被打破：每个智能体都是对方环境的一部分，而两者都在不断变化。多智能体强化学习便是在马尔可夫假设不再成立时，仍能使学习收敛的一系列技巧。

**类型：** 构建
**语言：** Python
**先修课程：** 第 9 阶段 · 04 (Q-learning)、第 9 阶段 · 06 (REINFORCE)、第 9 阶段 · 07 (Actor-Critic)
**时间：** 约 45 分钟

## 问题

一个学习在房间里导航的机器人是单智能体强化学习问题。一个足球队则不是。AlphaStar 对战《星际争霸》对手不是。竞价智能体构成的市场不是。两辆车协商通过四向停车标志不是。众多现实中的多人问题都不是。

在每一个多智能体场景中，从任何一个智能体的视角看，其他智能体*都是*环境的一部分。当它们学习并改变行为时，环境就变得非平稳。马尔可夫性质——“下一个状态仅取决于当前状态和我的动作”——被违反了，因为下一个状态还取决于*其他*智能体的选择，而它们的策略是移动的目标。

这打破了表格收敛证明（Q-learning 的保证假设环境是平稳的）。它也打破了朴素的深度强化学习：智能体陷入追逐循环，永远无法收敛到一个稳定的策略。你需要多智能体特定的技术：集中训练/分散执行、反事实基线、联赛博弈、自我博弈。

2026 年的应用：机器人集群、交通路由、自动驾驶车队、市场模拟器、多智能体 LLM 系统（第 16 阶段）以及任何包含多个智能玩家的游戏。

## 概念

![四种 MARL 模式：独立学习、集中式评论家、自我博弈、联赛](../assets/marl.svg)

**形式化：马尔可夫博弈。** MDP 的推广：状态 `S`、联合动作 `a = (a_1, …, a_n)`、转移函数 `P(s' | s, a)`、以及每个智能体的奖励 `R_i(s, a, s')`。每个智能体 `i` 在其自身的策略 `π_i` 下最大化其自身回报。如果奖励相同，则是**完全合作型**。如果是零和的，则是**对抗型**。如果是混合的，则是**一般和型**。

**核心挑战：**

- **非平稳性。** 从智能体 `i` 的视角看，`P(s' | s, a_i)` 取决于 `π_{-i}`，而后者正在变化。
- **信用分配。** 在共享奖励下，是哪个智能体导致的？
- **探索协调。** 智能体必须探索互补的策略，而非冗余地探索相同状态。
- **可扩展性。** 联合动作空间随 `n` 呈指数增长。
- **部分可观测性。** 每个智能体只能看到自己的观测；全局状态是隐藏的。

**四种主导模式：**

**1. 独立 Q-learning / 独立 PPO (IQL, IPPO)。** 每个智能体学习自己的 Q 函数或策略，将其他智能体视为环境的一部分。简单，有时有效（特别是当经验回放充当平滑的智能体建模技巧时）。理论收敛性：无。实践中：适用于松散耦合任务，对紧密耦合任务效果不佳。

**2. 集中训练，分散执行 (CTDE)。** 最常用的现代范式。每个智能体拥有自己的*策略* `π_i`，该策略以局部观测 `o_i` 为条件——部署时标准的分散执行。在*训练*期间，一个集中式评论家 `Q(s, a_1, …, a_n)` 以完整的全局状态和联合动作为条件。示例：
- **MADDPG** (Lowe et al. 2017)：每个智能体使用一个集中式评论家的 DDPG。
- **COMA** (Foerster et al. 2017)：反事实基线——询问“如果我当时采取了动作 `a'`，我的奖励会是多少？”——隔离我的贡献。
- **MAPPO** / 带共享评论家的 **IPPO** (Yu et al. 2022)：带有集中式价值函数的 PPO。2026 年合作型 MARL 的主流方法。
- **QMIX** (Rashid et al. 2018)：价值分解——`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))` 带有单调混合。

**3. 自我博弈。** 同一个智能体的两个副本相互对弈。对手的策略*就是*我过去某个快照的策略。AlphaGo / AlphaZero / MuZero。OpenAI Five。最适用于零和博弈；训练信号是对称的。

**4. 联赛博弈。** 自我博弈在一般和/对抗性环境下的扩展：维护一个过去和当前策略的群体，从联赛中采样一个对手，对他们进行训练。加入利用者（专注于打败当前最优者）和主利用者（专注于打败利用者）。AlphaStar（《星际争霸 II》）。当游戏存在“石头-剪刀-布”策略循环时需要。

**通信。** 允许智能体相互发送学习到的消息 `m_i`。在合作场景中有效。Foerster 等人 (2016) 证明，可微的智能体间通信可以端到端地训练。如今基于 LLM 的多智能体系统（第 16 阶段）本质上是用自然语言进行通信。

## 动手构建

本课程使用一个 6×6 的网格世界，包含两个合作型智能体。它们从相对角落出发，必须到达一个共享目标。共享奖励：当任一智能体仍在移动时，每步 `-1`；两者都到达时 `+10`。参见 `code/main.py`。

### 步骤 1：多智能体环境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*联合*动作空间是 `|A|² = 16`。全局状态是两个位置。

### 步骤 2：独立 Q-learning

每个智能体运行自己的 Q 表，以联合状态为键。每一步：两个智能体都选择 ε-贪婪动作，收集联合转移，每个智能体用共享奖励更新自己的 Q 值。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

在此任务中有效，因为奖励密集且对齐。在紧密耦合任务中会失败（例如，一个智能体必须*等待*另一个）。

### 步骤 3：集中式 Q 与分解值更新

使用一个关于联合动作 `Q(s, a_1, a_2)` 的 Q 函数。根据共享奖励更新。执行时通过边缘化进行分散：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。用指数级的联合动作空间交换一个*正确的*全局视图。

### 步骤 4：简单自我博弈（对抗型双智能体）

同一个智能体，两个角色。训练智能体 A 对抗智能体 B；经过 `K` 个回合后，将 A 的权重复制到 B。对称训练，持续进步。AlphaZero 方法的微缩版。

## 常见陷阱

- **非平稳回放。** 与独立智能体一起使用经验回放比单智能体更差，因为旧转移是由现已过时的对手生成的。解决方法：重新标记或按新近度加权。
- **信用分配模糊性。** 长回合后的共享奖励；无法明确指出是哪个智能体的贡献。解决方法：反事实基线 (COMA) 或为每个智能体设计奖励。
- **策略漂移/追逐。** 每个智能体的最佳响应随着其他智能体的更新而变化。解决方法：集中式评论家、降低学习率，或冻结一个智能体。
- **通过协调进行奖励黑客攻击。** 智能体找到了设计者未预料到的协调利用。竞价智能体收敛到出价为零。解决方法：谨慎设计奖励，添加行为约束。
- **探索冗余。** 两个智能体探索相同的状态-动作对。解决方法：为每个智能体添加熵奖励，或进行角色条件化。
- **联赛循环。** 纯自我博弈可能陷入主导循环。解决方法：使用多样对手的联赛博弈。
- **样本爆炸。** `n` 个智能体 × 状态空间 × 联合动作。使用函数近似进行近似；分解的动作空间（每个智能体一个策略输出头）。

## 应用

2026 年 MARL 应用图谱：

| 领域 | 方法 | 说明 |
|------|------|------|
| 合作导航 / 操作 | MAPPO / QMIX | CTDE；共享评论家 + 分散执行器。 |
| 双人博弈（国际象棋、围棋、扑克） | 结合 MCTS 的自我博弈 (AlphaZero) | 零和；对称训练。 |
| 复杂多人博弈（Dota、星际争霸） | 联赛博弈 + 模仿预训练 | OpenAI Five, AlphaStar。 |
| 自动驾驶车队 | CTDE MAPPO / 带注意力的 PPO | 部分可观测；可变团队规模。 |
| 拍卖市场 | 博弈论均衡 + RL | 当 `n` → ∞ 时使用平均场 RL。 |
| LLM 多智能体系统（第 16 阶段） | 自然语言通信 + 角色条件化 | 在智能体规划层的 RL 循环。 |

在 2026 年，MARL 最大的增长领域是基于 LLM 的：语言模型智能体集群进行谈判、辩论、构建软件。RL 出现在对*轨迹级*输出的偏好优化中，而不是 token 级别（第 16 阶段 · 03）。

## 完成交付

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## 练习

1. **简单。** 在 2-智能体合作网格世界上训练独立 Q-learning。需要多少个回合才能使平均回报 > 0？绘制联合学习曲线。
2. **中等。** 添加一个“协调”任务：只有当两个智能体在同一回合踏上目标时，才算到达目标。独立 Q 仍然能收敛吗？什么会失效？
3. **困难。** 实现一个用于 MAPPO 式训练的集中式评论家，并将其在协调任务上的收敛速度与独立 PPO 进行比较。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 马尔可夫博弈 | “多智能体 MDP” | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个智能体有自己的奖励。 |
| CTDE | “集中训练，分散执行” | 训练时使用联合评论家；每个智能体的策略只使用局部观测。 |
| IPPO | “独立 PPO” | 每个智能体分别运行 PPO。简单基线；常被低估。 |
| MAPPO | “多智能体 PPO” | 带有以全局状态为条件的集中式价值函数的 PPO。 |
| QMIX | “单调价值分解” | `Q_tot = f_monotone(Q_1, …, Q_n)` 允许分散的 argmax 操作。 |
| COMA | “反事实多智能体” | 优势值 = 我的 Q 值减去对我的动作边缘化后的期望 Q 值。 |
| 自我博弈 | “智能体 vs 过去的自己” | 单智能体，两个角色；零和博弈的标准方法。 |
| 联赛博弈 | “群体训练” | 缓存过去策略，从池中采样对手；处理策略循环。 |

## 延伸阅读

- [Lowe et al. (2017). 混合合作-竞争环境中的多智能体演员-评论家 (MADDPG)](https://arxiv.org/abs/1706.02275) — 带集中式评论家的 CTDE。
- [Foerster et al. (2017). 反事实多智能体策略梯度 (COMA)](https://arxiv.org/abs/1705.08926) — 用于信用分配的反事实基线。
- [Rashid et al. (2018). QMIX: 单调价值函数分解](https://arxiv.org/abs/1803.11485) — 具有单调性的价值分解。
- [Yu et al. (2022). PPO 在合作多智能体博弈中惊人的有效性 (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO 在 MARL 中出乎意料地强大。
- [Vinyals et al. (2019). 使用多智能体强化学习在《星际争霸 II》中达到宗师级水平 (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 大规模联赛博弈。
- [Silver et al. (2017). 在无人类知识的情况下掌握围棋游戏 (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 零和博弈中的纯自我博弈。
- [Sutton & Barto (2018). 第 15 章 — 神经科学 & 第 17 章 — 前沿](http://incompleteideas.net/book/RLbook2020.pdf) — 包括教科书对多智能体场景以及 CTDE 旨在解决的非平稳性问题的简要论述。
- [Zhang, Yang & Başar (2021). 多智能体强化学习：选择性概述](https://arxiv.org/abs/1911.10635) — 涵盖合作、竞争和混合 MARL 及收敛性结果的综述。