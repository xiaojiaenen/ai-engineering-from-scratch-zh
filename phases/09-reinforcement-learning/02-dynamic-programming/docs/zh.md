# 动态规划 —— 策略迭代与值迭代

> 动态规划是“开挂”的强化学习。你已知转移函数和奖励函数；只需反复迭代贝尔曼方程直至 `V` 或 `π` 不再变化。它是所有基于采样的方法试图逼近的基准。

**类型：** 构建
**语言：** Python
**先修课程：** 第9阶段 · 01（马尔可夫决策过程）
**时长：** ~75分钟

## 问题描述

你有一个已知模型的马尔可夫决策过程：可以查询任何状态-动作对的 `P(s' | s, a)` 和 `R(s, a, s')`。库存管理者知道需求分布。棋盘游戏具有确定性转移。网格世界仅需四行Python代码。你拥有一个*模型*。

无模型强化学习（Q-learning、PPO、REINFORCE）是为没有模型的情况设计的——你只能从环境中采样。但当你确实拥有模型时，存在更快、更好的方法：动态规划。贝尔曼于1957年设计了它们。它们至今仍定义了正确性：当人们说“该MDP的最优策略”时，他们指的是动态规划会返回的策略。

在2026年你需要掌握它们有三个原因。首先，强化学习研究中的每个表格环境（GridWorld、FrozenLake、CliffWalking）都通过动态规划求解以生成黄金标准策略。其次，精确值让你能够*调试*采样方法：如果Q-learning对 `V*(s_0)` 的估计与动态规划答案相差30%，说明你的Q-learning有错误。第三，现代离线强化学习和规划方法（MCTS、AlphaZero的搜索、第9阶段·10中的基于模型的强化学习）都在学习或给定的模型上迭代贝尔曼备份。

## 核心概念

![策略迭代与值迭代并排对比](../assets/dp.svg)

**两种算法，都是基于贝尔曼方程的不动点迭代。**

**策略迭代**。交替执行两个步骤直至策略不再变化。

1. *评估：* 给定策略 `π`，通过反复应用 `V^π` 计算 `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` 直至收敛。
2. *改进：* 给定 `V^π`，使 `π` 关于 `V^π` 贪心：`π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

收敛性得到保证，因为（a）每一步改进要么保持 `π` 不变，要么严格增加某个状态的 `V^π`，（b）确定性策略的空间是有限的。即使对于大规模状态空间，通常也能在约5-20次外部迭代内收敛。

**值迭代**。将评估和改进合并为一次扫描。应用贝尔曼*最优性*方程：

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

重复直至 `max_s |V_{new}(s) - V(s)| < ε`。最后通过选择贪心动作提取策略。每次迭代严格更快——无需内部评估循环——但通常需要更多迭代才能收敛。

**广义策略迭代（GPI）**。统一的框架。价值函数和策略锁定在双向改进循环中；任何推动两者走向相互一致的方法（异步值迭代、修正策略迭代、Q-learning、演员-评论家、PPO）都是GPI的实例。

**为什么 `γ < 1` 很重要**。贝尔曼算子是上确界范数下的 `γ`-压缩：`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。压缩意味着唯一不动点和几何收敛。去掉 `γ < 1`，你将失去保证——你需要有限视野或吸收终止状态。

## 构建实现

### 步骤1：构建GridWorld MDP模型

使用与第01课相同的4×4 GridWorld。我们添加一个随机变体：智能体以概率 `0.1` 滑向随机垂直方向。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` 返回一个 `(s', r, p)` 列表。这就是完整的模型。

### 步骤2：策略评估

给定策略 `π(s) = {action: prob}`，迭代贝尔曼方程直至 `V` 不再移动：

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### 步骤3：策略改进

用关于 `V` 的贪心策略替换 `π`。如果 `π` 未改变，则返回——我们已达到最优。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### 步骤4：将它们串联起来

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

在4×4网格上的典型收敛：4-6次外部迭代。输出 `V*(0,0) ≈ -6` 和一个严格减少步数的策略。

### 步骤5：值迭代（单循环版本）

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

相同的不动点，更少的代码行。

## 常见陷阱

- **忘记处理终止状态**。如果你将贝尔曼方程应用于吸收状态，它仍会选取一个“最佳动作”但不会改变任何东西。用 `if s == terminal: V[s] = 0` 进行防护。
- **上确界范数与L2收敛**。使用 `max |V_new - V|`，而非平均值。理论保证针对上确界范数。
- **就地更新与同步更新**。就地更新 `V[s]`（高斯-赛德尔）比使用单独的 `V_new` 字典（雅可比）收敛更快。生产代码使用就地更新。
- **策略平局**。如果两个动作具有相同的Q值，`argmax` 在每次迭代中可能以不同方式打破平局，导致“策略稳定”检查振荡。使用稳定的平局打破法（固定顺序中的第一个动作）。
- **状态空间爆炸**。动态规划每次扫描的复杂度为 `O(|S| · |A|)`。适用于约10⁷个状态。超出此范围，你需要函数近似（第9阶段·05及以后）。

## 应用场景

在2026年，动态规划是正确性基准和规划器的内循环：

| 用例 | 方法 |
|----------|--------|
| 精确求解小型表格MDP | 值迭代（更简单）或策略迭代（外部步骤更少） |
| 验证Q-learning/PPO实现 | 在玩具环境中与DP最优的V*进行比较 |
| 基于模型的强化学习（第9阶段·10） | 在学习的转移模型上进行贝尔曼备份 |
| AlphaZero/MuZero中的规划 | 蒙特卡洛树搜索 = 异步贝尔曼备份 |
| 离线强化学习（CQL、IQL） | 保守Q迭代——对OOD动作施加惩罚的DP |

每当有人说“最优价值函数”，他们指的都是“DP不动点”。当你在论文中看到 `V*` 或 `Q*` 时，想想这个循环。

## 保存代码

保存为 `outputs/skill-dp-solver.md`：

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## 练习

1. **简单**。在4×4 GridWorld上对 `γ ∈ {0.9, 0.99}` 运行值迭代。需要多少次扫描才能使 `max |ΔV| < 1e-6`？将 `V*` 打印为4×4网格。
2. **中等**。在*随机*GridWorld（滑动概率 `0.1`）上比较策略迭代与值迭代。计算：扫描次数、实际耗时、最终 `V*(0,0)`。哪种方法在迭代次数上收敛更快？在实际耗时上呢？
3. **困难**。构建修正策略迭代：在评估步骤中，仅运行 `k` 次扫描而非直至收敛。绘制 `k ∈ {1, 2, 5, 10, 50}` 的 `V*(0,0)` 误差与 `k` 的关系图。这条曲线关于评估/改进权衡告诉你什么？

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|-----------------------|
| 策略迭代 | “DP算法” | 交替进行评估（`V^π`）和改进（关于 `V^π` 的贪心 `π`）直至策略不再变化。 |
| 值迭代 | “更快的DP” | 在一次扫描中应用贝尔曼最优性备份；几何收敛至 `V*`。 |
| 贝尔曼算子 | “递归关系” | `(T V)(s) = max_a Σ P (r + γ V(s'))`；上确界范数下的 `γ`-压缩。 |
| 压缩映射 | “DP收敛的原因” | 任何满足 `||T x - T y|| ≤ γ ||x - y||` 的算子 `T` 都有唯一不动点。 |
| 广义策略迭代（GPI） | “一切都是DP” | 推动 `V` 和 `π` 走向相互一致的任何方法。 |
| 同步更新 | “雅可比风格” | 在一次扫描中始终使用旧的 `V`；分析清晰但较慢。 |
| 就地更新 | “高斯-赛德尔风格” | 使用正在更新的 `V`；实践中收敛更快。 |

## 扩展阅读

- [Sutton & Barto (2018). 第4章 — 动态规划](http://incompleteideas.net/book/RLbook2020.pdf) — 策略迭代和值迭代的经典呈现。
- [Bertsekas (2019). 强化学习与最优控制](http://www.athenasc.com/rlbook.html) — 压缩映射论证的严谨处理。
- [Puterman (2005). 马尔可夫决策过程](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 修正策略迭代及其收敛分析。
- [Howard (1960). 动态规划与马尔可夫过程](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — 策略迭代的原始论文。
- [Bertsekas & Tsitsiklis (1996). 神经动态规划](http://www.athenasc.com/ndpbook.html) — 从DP到近似DP/深度强化学习的桥梁，后续课程均会用到。