# 时间差分 — Q-Learning 与 SARSA

> 蒙特卡洛方法必须等到回合结束。TD方法通过自举下一个价值估计值在每步后立即更新。Q-learning是离策略且乐观的；SARSA是在策略且谨慎的。两者都只需一行代码实现。两者都是本阶段所有深度强化学习方法的基石。

**类型：** 构建  
**语言：** Python  
**前置知识：** 第9阶段 · 01（MDP）、第9阶段 · 02（动态规划）、第9阶段 · 03（蒙特卡洛）  
**预计时间：** 约75分钟

## 问题所在

蒙特卡洛方法可行但有两大昂贵要求。它需要回合终止，并且只在最终回报返回后才更新。如果你的回合长达1000步，MC要等1000步才能更新任何值。它方差高、偏差低，实践中速度较慢。

动态规划具有相反特性——零方差的自举备份——但需要已知模型。

时间差分学习折中了二者。从单次转移 `(s, a, r, s')` 中，构建单步目标 `r + γ V(s')` 并将 `V(s)` 向其微调。无需模型。无需完整回合。右侧使用近似的 `V` 会带来偏差，但方差比MC低得多，并且从第一步开始就能进行在线更新。

这是现代强化学习（DQN、A2C、PPO、SAC）所有方法的转折点。第9阶段的其余内容都是基于你将在本课中编写的单步TD更新之上构建的函数近似层和技巧。

## 核心概念

![Q-learning 与 SARSA：离策略最大值 vs 在策略 Q(s', a')](../assets/td.svg)

**V函数的TD(0)更新：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

方括号内的量是TD误差 `δ = r + γ V(s') - V(s)`。它是MC中 `G_t - V(s_t)` 的在线类比。收敛要求 `α` 满足Robbins-Monro条件 (`Σ α = ∞`, `Σ α² < ∞`) 且所有状态被无限次访问。

**Q-learning.** 一种用于控制的离策略TD方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

其中 `max` 假设从 `s'` 开始将遵循*贪婪*策略，无论智能体实际采取什么动作。这种解耦使得Q-learning能够学习 `Q*`，同时智能体通过ε-贪婪进行探索。Mnih等人（2015）将其转换为Atari游戏上的深度Q-learning（第05课）。

**SARSA.** 一种在策略TD方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

其名称来源于元组 `(s, a, r, s', a')`。SARSA使用智能体*实际*采取的下一动作 `a'`，而不是贪婪的 `argmax`。它收敛到当前运行的任何ε-贪婪策略 `π` 的 `Q^π`，在极限情况下 `ε → 0` 变为 `Q*`。

**悬崖行走差异。** 在经典的悬崖行走任务（掉下悬崖奖励为-100）中，Q-learning学习到沿悬崖边缘的最优路径，但在探索期间偶尔会受到惩罚。SARSA学习到一条离悬崖远一步的更安全路径，因为它将探索噪声纳入了Q值计算。经过训练，两者在 `ε → 0` 时都能达到最优。实践中这一点很重要：当部署时实际进行探索时，SARSA的行为更为保守。

**期望SARSA.** 用其在 `π` 下的期望值替换 `Q(s', a')`：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

比SARSA方差更低（不采样 `a'`），相同在策略目标。通常是现代教材中的默认选择。

**n步TD与TD(λ).** 通过在自举前等待 `n` 步，在TD(0)和MC之间进行插值。`n=1` 是TD，`n=∞` 是MC。TD(λ) 使用几何权重 `(1-λ)λ^{n-1}` 对所有 `n` 进行平均。大多数深度RL使用介于3到20之间的 `n`。

## 动手实现

### 步骤1：基于ε-贪婪策略的SARSA

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

八行代码。与Q-learning的*唯一*区别在于目标行。

### 步骤2：Q-learning

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

其中 `max` 将目标与行为解耦。这一个符号就是在策略与离策略的区别。

### 步骤3：学习曲线

跟踪每100个回合的平均回报。在简单的确定性网格世界中，Q-learning收敛更快；在悬崖行走中，SARSA更为保守。在 `code/main.py` 的4×4网格世界中，使用 `α=0.1, ε=0.1` 时，两者经过约2000个回合后都接近最优。

### 步骤4：与DP真实值比较

运行价值迭代（第02课）得到 `Q*`。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`。一个健康表格TD智能体在4×4网格世界运行10000个回合后，误差在 `~0.5` 以内。

## 常见陷阱

- **初始Q值很重要。** 乐观初始化（对于负奖励任务设 `Q = 0`）鼓励探索。悲观初始化可能永远困住贪婪策略。
- **α调度。** 恒定 `α` 对于非平稳问题足够。衰减的 `α_n = 1/n` 在理论上能保证收敛但实践中太慢——将 `α` 固定在 `[0.05, 0.3]` 并监控学习曲线。
- **ε调度。** 从高值开始 (`ε=1.0`)，衰减到 `ε=0.05`。“GLIE”（在无限探索下极限处为贪婪）是收敛条件。
- **Q-learning中的最大化偏差。** 当 `Q` 有噪声时，`max` 算子会向上偏。导致过高估计——Hasselt的双重Q-learning（DDQN在第05课中使用）通过两个Q表解决此问题。
- **非终止回合。** TD无需终止就能学习，但你需要限制步数或在限制步数处正确处理自举。标准做法：将限制步视为非终止，继续自举。
- **状态哈希。** 如果状态是元组/张量，使用可哈希的键（用元组而非列表；浮点数元组要四舍五入，不要用原始值）。

## 应用场景

2026年TD领域应用概览：

| 任务 | 方法 | 原因 |
|------|------|------|
| 小型表格环境 | Q-learning | 直接学习最优策略。 |
| 在策略安全关键 | SARSA / 期望SARSA | 探索期间行为保守。 |
| 高维状态 | DQN（第9阶段 · 05） | 带经验回放和目标网络的神经网络Q函数。 |
| 连续动作 | SAC / TD3（第9阶段 · 07） | 对Q网络进行TD更新；策略网络输出动作。 |
| LLM RL（基于奖励模型） | PPO / GRPO（第9阶段 · 08, 12） | 通过GAE实现TD风格优势的演员-评论家架构。 |
| 离线RL | CQL / IQL（第9阶段 · 08） | 带保守正则化的Q-learning。 |

你在2026年论文中读到的“强化学习”百分之九十是Q-learning或SARSA的某种细化。在阅读更深层次内容前，请将表格更新内化于心。

## 提交代码

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## 练习

1. **简单.** 在4×4网格世界中实现Q-learning和SARSA。绘制2000个回合的学习曲线（每100个回合的平均回报）。哪个收敛更快？
2. **中等.** 构建一个悬崖行走环境（4×12，最后一行是奖励为-100并重置到起点的悬崖）。比较Q-learning和SARSA的最终策略。截图各自采取的路径。哪个更靠近悬崖？
3. **困难.** 实现双重Q-learning。在带噪声奖励的网格世界（每步奖励添加高斯噪声σ=5）中，展示Q-learning会明显过高估计 `V*(0,0)`，而双重Q-learning则不会。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------|----------|
| TD误差 | "更新信号" | `δ = r + γ V(s') - V(s)`，自举残差。 |
| TD(0) | "单步TD" | 每次转移后仅使用下一状态的估计进行更新。 |
| Q-learning | "离策略RL入门" | 对下一状态动作取 `max` 的TD更新；无论行为策略如何都学习 `Q*`。 |
| SARSA | "在策略Q-learning" | 使用实际下一动作的TD更新；学习当前ε-贪婪π的 `Q^π`。 |
| 期望SARSA | "低方差SARSA" | 用π下的期望替换采样的 `a'`。 |
| GLIE | "正确的探索调度" | 无限探索下极限处为贪婪；Q-learning收敛所需条件。 |
| 自举 | "在目标中使用当前估计" | 区分TD与MC的关键。偏差来源但大幅降低方差。 |
| 最大化偏差 | "Q-learning过高估计" | 对有噪声估计取 `max` 会向上偏；通过双重Q-learning修正。 |

## 扩展阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始论文及收敛证明。
- [Sutton & Barto (2018). 第6章 — 时间差分学习](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q-learning、期望SARSA。
- [Hasselt (2010). 双重Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 最大化偏差的修正方法。
- [Seijen, Hasselt, Whiteson, Wiering (2009). 期望SARSA的理论与实证分析](https://ieeexplore.ieee.org/document/4927542) — 期望SARSA的动机。
- [Rummery & Niranjan (1994). 使用连接系统在线Q-learning](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — 提出SARSA的论文（当时称为“改进的连接Q-learning”）。
- [Sutton & Barto (2018). 第7章 — n步自举](http://incompleteideas.net/book/RLbook2020.pdf) — 将TD(0)推广到TD(n)，从Q-learning到资格迹，再到PPO中的GAE的路径。