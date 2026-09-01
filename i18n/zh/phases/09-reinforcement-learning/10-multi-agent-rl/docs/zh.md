# 多智能体强化学习

> 单智能体强化学习假设环境是平稳的。将两个正在学习的智能体放入同一个世界，该假设就会被打破：每个智能体都是对方环境的一部分，且双方都在不断变化。多智能体强化学习是一组在马尔可夫假设不再成立时使学习收敛的技巧。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 9 · 04（Q-learning）、阶段 9 · 06（REINFORCE）、阶段 9 · 07（Actor-Critic）
**时间：** 约 45 分钟

## 问题所在

一个学习在房间里导航的机器人是一个单智能体 RL 问题。一支足球队则不是。AlphaStar 对战《星际争霸》对手不是。一个由竞价智能体组成的市场不是。两辆车在四向停车标志处协商也不是。许多现实世界中的多人问题都不是。

在多智能体的每一种场景中，从任何一个智能体的视角来看，其他智能体*都是*环境的一部分。随着它们的学习和行为改变，环境变得非平稳。马尔可夫性——"下一状态只取决于当前状态和我的动作"——被违反了，因为下一状态还取决于*其他*智能体的选择，而它们的策略是移动目标。

这打破了表格式收敛证明（Q-learning 的 guarantees 假设环境是平稳的）。它也击穿了朴素深度 RL：智能体互相追逐形成循环，永远无法收敛到稳定策略。你需要多智能体特定的技术：集中训练/分布式执行、反事实基线、联盟对战、自对弈。

2026 年应用：机器人集群、交通路由、自动驾驶车队、市场模拟器、多智能体 LLM 系统（第 16 阶段），以及任何拥有多于一个智能玩家的博弈。

## 核心概念

![四种 MARL 范式：独立、集中式 critic、自对弈、联盟](../assets/marl.svg)

**形式化：马尔可夫博弈。** MDP 的推广：状态 `S`、联合动作 `a = (a_1, …, a_n)`、转移 `P(s' | s, a)`，以及每个智能体的奖励 `R_i(s, a, s')`。每个智能体 `i` 在其自身策略 `π_i` 下最大化自身的回报。如果奖励相同，则是**完全合作**。如果是零和，则是**对抗**。如果是混合的，则是**一般和**。

**核心挑战：**

- **非平稳性。** 从智能体 `i` 的视角看 `P(s' | s, a_i)` 依赖于 `π_{-i}`，而 `π_{-i}` 在变化。
- **信用分配。** 共享奖励时，哪个智能体造成了它？
- **探索协调。** 智能体必须探索互补策略，而不是冗余地探索相同的状态。
- **可扩展性。** 联合动作空间随 `n` 指数增长。
- **部分可观测性。** 每个智能体只能看到自己的观测；全局状态被隐藏。

**四种主流范式：**

**1. 独立 Q-learning / 独立 PPO（IQL、IPPO）。** 每个智能体学习自己的 Q 或策略，将其他智能体视为环境的一部分。简单，有时有效（尤其是在经验回放起到平滑的智能体建模技巧时）。理论收敛性：无。实践中：松散耦合的任务尚可，紧密耦合的任务很差。

**2. 集中式训练，分布式执行（CTDE）。** 最常见的现代范式。每个智能体拥有自己的*策略* `π_i`，它根据局部观测 `o_i` 进行决策——在部署时标准地分布式执行。在*训练*期间，一个集中式 critic `Q(s, a_1, …, a_n)` 依赖完整的全局状态和联合动作。示例：
- **MADDPG**（Lowe 等 2017）：带每个智能体集中式 critic 的 DDPG。
- **COMA**（Foerster 等 2017）：反事实基线——问"如果我采取动作 `a'` 而不是当前动作，我的奖励会是多少？"——隔离我的贡献。
- **MAPPO** / **IPPO** 共享 critic（Yu 等 2022）：带集中式价值函数的 PPO。2026 年合作式 MARL 的主流方法。
- **QMIX**（Rashid 等 2018）：价值分解——`Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`，使用单调混合。

**3. 自对弈。** 同一智能体的两个副本相互对战。对手的策略*就是*我过去某个快照的策略。AlphaGo / AlphaZero / MuZero。OpenAI Five。在零和博弈中效果最佳；训练信号是对称的。

**4. 联盟对战。** 自对弈对一般和/对抗环境的扩展：维护一个过去和当前策略的人口池，从联盟中采样对手进行训练。加入破解者（专门针对当前最优策略的特化智能体）和主破解者（专门针对破解者的特化智能体）。AlphaStar（《星际争霸 II》）。当博弈存在"石头-剪刀-布"策略循环时需要此方法。

**通信。** 允许智能体之间发送学习型消息 `m_i`。在合作场景中有效。Foerster 等（2016）证明可微分的智能体间通信可以被端到端训练。今天的基于 LLM 的多智能体系统（第 16 阶段）本质上用自然语言通信。

```figure
f3-marl-orbit
```

## 动手实现

本课使用一个 6×6 网格世界，包含两个合作智能体。它们从对角出发，必须到达共同目标。共享奖励：任一智能体仍在移动时每一步 `-1`，两者都到达时 `+10`。参见 `code/main.py`。

### 步骤 1：多智能体环境

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # 两个智能体

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*联合*动作空间为 `|A|² = 16`。全局状态是两个位置。

### 步骤 2：独立 Q-learning

每个智能体维护自己的 Q 表，键为联合状态。每一步：双方各自选择 ε-贪婪动作，收集联合转移，各自用共享奖励更新自己的 Q 表。

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

在此任务上可以工作，因为奖励是稠密且对齐的。但在紧密耦合任务上会失败（例如其中一个智能体必须*等待*另一个的情况）。

### 步骤 3：带分解值更新的集中式 Q

使用覆盖联合动作的一个 Q 表 `Q(s, a_1, a_2)`。从共享奖励更新。在执行时通过边缘化实现去中心化：`π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`。用正确的全局视图换取对指数级联合动作空间的分解。

### 步骤 4：简单自对弈（对抗式双智能体）

同一智能体，两种角色。训练智能体 A 对战智能体 B；经过 `K` 轮后，将 A 的权重复制到 B。对称训练，持续进步。AlphaZero 配方的微缩版。

## 常见陷阱

- **非平稳经验回放。** 独立智能体的经验回放比单智能体更差，因为旧转移是由早已过时的对手生成的。修复：按近期性重新标记或加权。
- **信用分配歧义。** 长 episode 后的共享奖励；没有明确方式说明哪个智能体做出了贡献。修复：反事实基线（COMA），或按智能体进行奖励塑形。
- **策略漂移/追逐。** 每个智能体的最优响应随对方每次更新而变化。修复：集中式 critic、慢学习率，或交替固定一方。
- **通过协调作弊。** 智能体发现设计师未预见的协作漏洞。拍卖智能体收敛到零竞价。修复：精心设计的奖励、行为约束。
- **探索冗余。** 两个智能体探索相同的状态-动作对。修复：每个智能体的熵正则，或角色条件化。
- **联盟循环。** 纯自对弈可能陷入支配循环。修复：使用多样化对手的联盟对战。
- **样本爆炸。** `n` 个智能体 × 状态空间 × 联合动作。用函数近似近似；分解动作空间（每个智能体一个策略输出头）。

## 应用场景

2026 年 MARL 应用地图：

| 领域 | 方法 | 说明 |
|--------|--------|-------|
| 合作导航/操控 | MAPPO / QMIX | CTDE；共享 critic + 分布式 actor。 |
| 双人对战游戏（国际象棋、围棋、扑克） | 带 MCTS 的自对弈（AlphaZero） | 零和；对称训练。 |
| 复杂多人游戏（Dota、星际争霸） | 联盟对战 + 模仿预训练 | OpenAI Five、AlphaStar。 |
| 自动驾驶车队 | CTDE MAPPO / 带注意力的 PPO | 部分可观测；可变团队规模。 |
| 拍卖市场 | 博弈论均衡 + RL | 当 `n` → ∞ 时使用均值场 RL。 |
| LLM 多智能体系统（第 16 阶段） | 自然语言通信 + 角色条件化 | RL 循环位于智能体规划层。 |

2026 年，MARL 增长最快的领域是基于 LLM 的：谈判、辩论、构建软件的語言模型智能体集群。RL 表现为对*轨迹级*输出的偏好优化，而非 token 级（第 16 阶段 · 03）。

## 交付物

保存为 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: 为给定任务选择合适的多智能体 RL 范式（IPPO、CTDE、自对弈、联盟对战）。
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

给定一个包含 `n` 个智能体的任务，输出：

1. 范式分类。合作 / 对抗 / 一般和。给出理由。
2. 算法。IPPO / MAPPO / QMIX / 自对弈 / 联盟对战。理由与耦合紧密度和奖励结构相关联。
3. 信息访问。集中式训练（哪些全局信息进入 critic）？分布式执行？
4. 信用分配。反事实基线、价值分解或奖励塑形。
5. 探索计划。每智能体熵、基于种群的训练或联盟对战。

在紧密耦合的合作任务上拒绝推荐独立 Q-learning。在存在循环风险的一般和场景上拒绝推荐自对弈。标记任何缺乏固定对手评估的 MARL 流水线（ cherry-picked 自对弈数字很常见）。
```

## 练习

1. **简单。** 在 2 智能体合作网格世界上训练独立 Q-learning。平均回报 > 0 需要多少轮？绘制联合学习曲线。
2. **中等。** 增加一个"协调"任务：仅当两个智能体在同一回合踏入目标时才算到达。独立 Q 是否仍能收敛？什么会失败？
3. **困难。** 为 MAPPO 式训练实现一个集中式 critic，并与独立 PPO 在协调任务上的收敛速度进行比较。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 马尔可夫博弈 | "多智能体 MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`；每个智能体有自己的奖励。 |
| CTDE | "集中式训练，分布式执行" | 训练时有联合 critic；每个智能体的策略仅使用局部观测。 |
| IPPO | "独立 PPO" | 每个智能体独立运行 PPO。简单基线；常被低估。 |
| MAPPO | "多智能体 PPO" | 带依赖全局状态的集中式价值函数的 PPO。 |
| QMIX | "单调价值分解" | `Q_tot = f_monotone(Q_1, …, Q_n)` 允许去中心化 argmax。 |
| COMA | "反事实多智能体" | 优势 = 我的 Q 减去对我动作边缘化后的期望 Q。 |
| 自对弈 | "智能体对战过去的自己" | 单个智能体，两种角色；零和博弈的标准做法。 |
| 联盟对战 | "人口训练" | 缓存历史策略，从种群中采样对手；处理策略循环。 |

## 延伸阅读

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 带集中式 critic 的 CTDE。
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — 用于信用分配的反事实基线。
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — 带单调性的价值分解。
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO 在 MARL 中效果惊人地好。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 大规模联盟对战。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 零和博弈中的纯自对弈。
- [Sutton & Barto (2018). 第 15 章 — 神经科学 & 第 17 章 — 前沿](http://incompleteideas.net/book/RLbook2020.pdf) — 包含教科书对多智能体场景的简要介绍，以及 CTDE 旨在解决的"非平稳性"问题。
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — 涵盖合作、竞争和混合 MARL 及其收敛性结果的综述。
