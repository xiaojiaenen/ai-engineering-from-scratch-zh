# 蒙特卡洛方法 —— 从完整回合中学习

> 动态规划需要模型，而蒙特卡洛只需回合。运行策略，观察回报，求平均。这是强化学习中最朴素的想法——也是开启后续一切的关键。

**类型:** 构建
**语言:** Python
**前置要求:** 第 9 阶段 · 01 (马尔可夫决策过程), 第 9 阶段 · 02 (动态规划)
**时间:** ~75 分钟

## 问题所在

动态规划非常优雅，但它假设你可以查询`P(s' | s, a)`中任意状态和动作的转移概率。现实世界中几乎没有事情是这样的。机器人无法通过解析计算关节扭矩作用后，摄像头像素的分布。定价算法无法对所有可能的客户反应进行积分。语言模型在生成一个token后，无法枚举所有可能的续写内容。

你需要一种只需要从环境中*采样*的方法。运行策略，获得一条轨迹`s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`，用它来估计价值。这就是蒙特卡洛方法。

从动态规划到蒙特卡洛的转变在哲学意义上很重要：我们从*已知模型 + 精确回溯*转向*采样轨迹 + 平均回报*。方差会剧增，但适用性会爆炸式增长。本课之后的所有强化学习算法——时序差分、Q-learning、REINFORCE、PPO、GRPO——其核心都是蒙特卡洛估计器，有时会在上面叠加自举技术。

## 核心概念

![蒙特卡洛：轨迹展开，计算回报，求平均；首次访问 vs 每次访问](../assets/monte-carlo.svg)

**核心思想，一句话概括：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)` 其中`G^{(i)}(s)`是在策略`π`下访问`s`后观察到的回报。

**首次访问 vs 每次访问蒙特卡洛。** 给定一个多次访问状态`s`的回合，首次访问蒙特卡洛只计算首次访问后的回报；每次访问蒙特卡洛计算所有访问。两者在极限下都是无偏的。首次访问更易于分析（独立同分布样本）。每次访问每回合使用更多数据，实践中通常收敛更快。

**增量均值。** 无需存储所有回报，直接更新运行平均值：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

重组得：`V_new = V_old + α · (target - V_old)`，其中`α = 1/n`。将`1/n`替换为固定步长`α ∈ (0, 1)`，你就会得到一个非平稳蒙特卡洛估计器，能够追踪`π`的变化。这一步就是蒙特卡洛转向时序差分，乃至所有现代强化学习算法的全部跳跃。

**探索现在成为一个问题。** 动态规划通过枚举触及每个状态。蒙特卡洛只能看到策略访问的状态。如果`π`是确定性的，那么状态空间的整个区域永远不会被采样，其价值估计将永远停留在零。历史上有三种解决方法：

1.  **探索性起始。** 每个回合从一个随机的(s, a)对开始。保证覆盖性；实践中不现实（你无法将机器人"重置"到任意状态）。
2.  **ε-贪婪策略。** 在当前Q值的基础上采取贪婪动作，但以概率`ε`选择一个随机动作。渐近地，所有状态-动作对都会被采样。
3.  **离策略蒙特卡洛。** 在行为策略`μ`下收集数据，通过重要性采样来学习目标策略`π`。方差很高，但这是通往DQN等经验回放方法的桥梁。

**蒙特卡洛控制。** 评估→改进→评估，就像策略迭代一样，但评估是基于采样的：

1.  运行`π`，获得一个回合。
2.  根据观察到的回报更新`Q(s, a)`。
3.  使`π`相对于`Q`是ε-贪婪的。
4.  重复。

在温和条件下（每个状态-动作对被无限次访问，`α`满足罗宾斯-门罗条件），以概率1收敛到`Q*`和`π*`。

## 动手构建

### 步骤 1：轨迹展开 → (s, a, r) 列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

没有模型，只有`env.reset()`和`env.step(s, a)`。接口与gym环境相同，但更精简。

### 步骤 2：计算回报（反向扫描）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

一次遍历，`O(T)`。反向递推`G_t = r_{t+1} + γ G_{t+1}`避免了重复求和。

### 步骤 3：首次访问蒙特卡洛评估

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

三行代码完成工作：首次访问时标记状态为已见，增加计数，更新运行均值。

### 步骤 4：ε-贪婪蒙特卡洛控制（同策略）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 步骤 5：与动态规划黄金标准对比

当回合数 → ∞ 时，你对`V^π`的蒙特卡洛估计值应与第2课的动态规划结果一致。实践中：在4×4网格世界上运行50,000回合，你可以得到与动态规划答案相差`~0.1`以内的结果。

## 常见陷阱

-   **无限回合。** 蒙特卡洛要求回合能够*终止*。如果你的策略可能永远循环，请设定`max_steps`的上限，并将达到上限视为隐式失败。使用随机策略的网格世界经常超时——这是正常的，只需确保正确计数即可。
-   **方差。** 蒙特卡洛使用完整回报。对于长回合，方差巨大——结束时一个不幸的奖励会使`V(s_0)`发生同等幅度的变化。时序差分方法（第4课）通过自举技术降低了这种方差。
-   **状态覆盖。** 在一个全新的Q表上运行贪婪策略，在值相等的情况下只会尝试一个动作。你*必须*探索（ε-贪婪、探索性起始、UCB）。
-   **非平稳策略。** 如果`π`发生变化（如在蒙特卡洛控制中），旧的回报来自不同的策略。固定α的蒙特卡洛能处理此问题；样本均值的蒙特卡洛则不能。
-   **离策略重要性采样。** 权重`π(a|s)/μ(a|s)`沿轨迹相乘。方差随时间步长爆炸。可通过每步加权重要性采样进行限制，或改用时序差分。

## 应用场景

2026年蒙特卡洛方法的应用场景：

| 使用场景 | 为什么用蒙特卡洛 |
|----------|------------------|
| 短视界博弈（21点、扑克） | 回合自然终止；回报干净。 |
| 离线评估已记录的策略 | 对存储的轨迹计算平均折扣回报。 |
| 蒙特卡洛树搜索（AlphaZero） | 从树节点进行的蒙特卡洛展开指导选择。 |
| 语言模型强化学习评估 | 计算给定策略下，对采样完成结果的平均奖励。 |
| PPO 中的基线估计 | 优势目标`A_t = G_t - V(s_t)`使用了蒙特卡洛`G_t`。 |
| 强化学习教学 | 最简单且真正有效的算法——去除自举以观察核心。 |

现代深度强化学习算法（PPO、SAC）通过`n`步回报或广义优势估计，在纯蒙特卡洛（完整回报）和纯时序差分（一步自举）之间进行插值。两种端点都是同一估计器的实例。

## 部署运行

保存为`outputs/skill-mc-evaluator.md`：

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## 练习

1.  **简单。** 在4×4网格世界上实现对均匀随机策略的首次访问蒙特卡洛评估。运行10,000回合。绘制`V(0,0)`随回合数变化的曲线，并与动态规划答案对比。
2.  **中等。** 实现ε-贪婪蒙特卡洛控制，设`ε ∈ {0.01, 0.1, 0.3}`。比较20,000回合后的平均回报。曲线是什么样子？偏差-方差权衡发生在哪里？
3.  **困难。** 实现*离策略*蒙特卡洛并应用重要性采样：在均匀随机策略`μ`下收集数据，估计确定性最优策略`V^π`的`π`。比较普通重要性采样、每步重要性采样和加权重要性采样。哪种方差最低？

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 蒙特卡洛 | "随机采样" | 通过对来自分布的独立同分布样本求平均来估计期望值。 |
| 回报 `G_t` | "未来奖励" | 从步`t`到回合结束的折扣奖励总和：`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| 首次访问蒙特卡洛 | "每个状态只计数一次" | 回合中只有首次访问对价值估计有贡献。 |
| 每次访问蒙特卡洛 | "使用所有访问" | 每次访问都有贡献；略有偏差但样本效率更高。 |
| ε-贪婪 | "探索噪声" | 以概率`1-ε`选择贪婪动作；以概率`ε`选择随机动作。 |
| 重要性采样 | "纠正从错误分布采样" | 通过`π(a|s)/μ(a|s)`的乘积对回报重新加权，以从`μ`的数据估计`V^π`。 |
| 同策略 | "从我自己的数据中学习" | 目标策略 = 行为策略。原始蒙特卡洛、PPO、SARSA。 |
| 离策略 | "从别人的数据中学习" | 目标策略 ≠ 行为策略。重要性采样蒙特卡洛、Q-learning、DQN。 |

## 延伸阅读

-   [Sutton & Barto (2018). 第 5 章 — 蒙特卡洛方法](http://incompleteideas.net/book/RLbook2020.pdf) — 权威论述。
-   [Singh & Sutton (1996). 使用替换资格迹的强化学习](https://link.springer.com/article/10.1007/BF00114726) — 首次访问 vs 每次访问分析。
-   [Precup, Sutton, Singh (2000). 用于离策略策略评估的资格迹](http://incompleteideas.net/papers/PSS-00.pdf) — 离策略蒙特卡洛与方差控制。
-   [Mahmood et al. (2014). 用于离策略学习的加权重要性采样](https://arxiv.org/abs/1404.6362) — 现代低方差重要性采样估计器。
-   [Tesauro (1995). TD-Gammon, 一个自我教学的西洋双陆棋程序](https://dl.acm.org/doi/10.1145/203330.203343) — 首个大规模经验证明，蒙特卡洛/时序差分自我对弈可以收敛到超人类水平的演示；是本阶段后半部分所有课程的概念先驱。