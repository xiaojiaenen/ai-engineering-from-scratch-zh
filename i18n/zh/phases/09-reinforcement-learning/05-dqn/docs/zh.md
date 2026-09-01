# 深度 Q 网络 (DQN)

> 2013年：Mnih 用原始像素训练了一个 Q-learning 网络，在七款 Atari 游戏中击败了所有经典强化学习智能体。2015年：扩展到 49 款游戏，发表在《Nature》，开启了深度强化学习时代。DQN 是 Q-learning 加上三个让函数近似稳定化的技巧。

**类型：** Build
**语言：** Python
**前置知识：** 第 3 阶段 · 第 03 课（反向传播），第 9 阶段 · 第 04 课（Q-learning，SARSA）
**时间：** 约 75 分钟

## 问题所在

表格 Q-learning 需要为每个 (状态，动作) 对保存单独的 Q 值。棋盘有约 10⁴³ 个状态。一帧 Atari 画面是 210×160×3 = 100,800 个特征。表格型 RL 在几千个状态时就崩溃了，更不用说数十亿的状态。

 hindsight 看来显而易见的解决方案：用神经网络 `Q(s, a; θ)` 替换 Q 表。但这显而近之的方案花了数十年才被实现。与 Q-learning 结合的朴素函数近似会在"致命三角"（deadly triad）下发散——函数近似 + 自举（bootstrapping）+ 离策略（off-policy）学习。Mnih 等人（2013，2015）确定了三个工程技巧来稳定训练：

1. **经验回放** 解相关转移样本。
2. **目标网络** 冻结自举目标。
3. **奖励裁剪** 归一化梯度量级。

在 Atari 上使用 DQN 是第一次：同一个架构、同一组超参，就能从原始像素解决数十个控制问题。此后所有的"深度强化学习"算法——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——都是基于这三个技巧堆叠而来。

## 核心概念

![DQN 训练循环：环境、经验回放缓冲区、在线网络、目标网络、Bellman TD 损失](../assets/dqn.svg)

**目标函数。** DQN 最小化神经 Q 函数的一步 TD 损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 在线网络，每一步通过梯度下降更新。`θ^-` = 目标网络，定期从 `θ` 复制（大约每 10,000 步）。`D` = 过去转移的经验回放缓冲区。

**三个技巧，按重要性排序：**

**经验回放。** 一个包含 `~10⁶` 条转移的环形缓冲区。每个训练步骤均匀随机采样一个小批量。这打破了时间相关性（连续帧几乎相同），让网络能多次从稀有的高奖励转移中学习，并解相关连续的梯度更新。没有它，在线 TD 配合神经网络在 Atari 上会发散。

**目标网络。** 在 Bellman 方程两侧使用同一个网络 `Q(·; θ)` 会导致目标函数每次更新都在移动——"追赶自己的尾巴"。解决方案：保留第二个网络 `Q(·; θ^-)`，权重冻结。每 `C` 步，将 `θ` 复制到 `θ^-`。这让回归目标在数千步梯度更新期间保持稳定。软更新 `θ^- ← τ θ + (1-τ) θ^-`（在 DDPG、SAC 中使用）是更平滑的变体。

**奖励裁剪。** Atari 奖励幅度从 1 到 1000+ 不等。裁剪到 `{-1, 0, +1}` 防止任何单局游戏主导梯度。当奖励量级重要时这会出错；但在 Atari 中只需关注符号即可。

**Double DQN。** Hasselt（2016）解决了最大化偏差：使用在线网络*选择*动作，使用目标网络*评估*它。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

可无缝替换，效果 consistently 更好。默认使用它。

**其他改进（Rainbow，2017）：** 优先回放（更频繁采样高 TD 误差转移）、Dueling 架构（分离 `V(s)` 和优势头）、噪声网络（学习式探索）、n 步回报、分布 Q（C51/QR-DQN）、多步自举。每项增加几个百分点；收益大致可叠加。

```figure
f3-dqn-stability
```

## 动手实现

这里的代码仅用标准库，无 numpy——我们在一个小型连续 GridWorld 上使用手工实现的单层 MLP，所以每个训练步骤仅需数微秒。该算法与大规模 Atari DQN 完全一致。

### 步骤 1：经验回放缓冲区

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 使用约 50,000 容量；我们的小型环境 5,000 就足够。

### 步骤 2：小型 Q 网络（手工 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传播：线性层 → ReLU → 线性层。这就是整个网络。

### 步骤 3：DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

结构与第 04 课的 Q-learning 相同，有两个区别：(a) 我们对可微的 `Q(·; θ)` 进行反向传播，而不是索引表格；(b) 目标使用 `Q(·; θ^-)`。

### 步骤 4：外层循环

每个episode中，对 `Q(·; θ)` 执行 ε-贪婪行动，将转移推入缓冲区，采样小批量，执行一步梯度下降，定期同步 `θ^- ← θ`。模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我们的小型 GridWorld（16维 one-hot 状态）上，智能体大约在 500 个 episode 内学会近乎最优的策略。在 Atari 上，将此扩展到 2 亿帧并添加 CNN 特征提取器。

## 常见陷阱

- **致命三角。** 函数近似 + 离策略 + 自举可能导致发散。DQN 通过目标网络 + 回放来缓解；不要移除其中任何一个。
- **探索。** ε 必须衰减，通常在前 ~10% 的训练中从 1.0 降到 0.01。如果早期探索不足，Q 网络会收敛到局部最优。
- **高估。** 对噪声 Q 取 `max` 会产生向上的偏差。在生产环境中始终使用 Double DQN。
- **奖励量级。** 裁剪或归一化奖励；梯度量级与奖励量级成正比。
- **回放缓冲区冷启动。** 缓冲区拥有几千条转移之前不要开始训练。早期在约 20 个样本上的梯度容易过拟合。
- **目标同步频率。** 太频繁 ≈ 没有目标网络；太稀疏 ≈ 目标过时。Atari DQN 使用 10,000 步环境交互。经验法则：每隔训练总时长的约 1/100 同步一次。
- **观测预处理。** Atari DQN 堆叠 4 帧使状态满足马尔可夫性。任何包含速度信息的.env 需要帧堆叠或循环状态。

## 使用场景

2026 年，DQN 不再是 SOTA，但仍然是离策略算法的参考实现：

| 任务 | 首选方法 | 为什么不选 DQN？ |
|------|------------------|--------------|
| 离散动作 Atari 类 | Rainbow DQN 或 Muesli | 同一框架，更多技巧。 |
| 连续控制 | SAC / TD3（第 9 阶段 · 第 07 课） | DQN 没有策略网络。 |
| 在线策略 / 高吞吐 | PPO（第 9 阶段 · 第 08 课） | 无需经验回放；更容易扩展。 |
| 离线 RL | CQL / IQL / Decision Transformer | 保守 Q 目标，无自举爆炸。 |
| 大型离散动作空间（推荐系统） | 带动作嵌入的 DQN，或 IMPALA | 可行；装饰更重要。 |
| LLM RL | PPO / GRPO | 序列级，非步级；损失不同。 |

这些原则仍然适用。回放和目标网络出现在 SAC、TD3、DDPG、SAC-X、AlphaZero 的自我对弈缓冲区以及所有离线 RL 方法中。奖励裁剪演变为 PPO 中的优势归一化。这个架构是蓝图。

## 交付物

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: 为离散动作强化学习任务生成 DQN 训练配置（缓冲区、目标同步、ε 调度、奖励裁剪）。
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

给定一个离散动作环境（观测形状、动作数量、回合长度、奖励量级），输出：

1. 网络。架构（MLP / CNN / Transformer）、特征维度、层深。
2. 经验回放缓冲区。容量、小批量大小、预热大小。
3. 目标网络。同步策略（每 C 步硬同步或软 τ）。
4. 探索。ε 起始 / 结束值 / 调度长度。
5. 损失函数。Huber vs MSE、梯度裁剪值、奖励裁剪规则。
6. Double DQN。默认开启，除非有明确理由关闭。

拒绝交付没有目标网络、没有经验回放缓冲区、或 ε 保持为 1 的 DQN。拒绝连续动作任务（转交 SAC / TD3）。标记任何奖励范围超过每步均值 10 倍的情况，提示需要裁剪或尺度归一化。
```

## 练习

1. **简单。** 运行 `code/main.py`。绘制每 episode 的回报曲线。运行均值超过 -10 需要多少个 episode？
2. **中等。** 禁用目标网络（在 Bellman 目标的两侧都使用在线网络）。测量训练不稳定性——回报是否震荡或发散？
3. **困难。** 添加 Double DQN：使用在线网络选择 `argmax a'`，目标网络评估。在有噪声奖励的 GridWorld 上，对比 1,000 个 episode 后 `Q(s_0, best_a)` 与真实 `V*(s_0)` 的偏差，有无 Double DQN。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------------|-----------------------|
| DQN | "深度 Q 学习" | 带神经网络 Q 函数、经验回放缓冲区和目标网络的 Q 学习。 |
| 经验回放 | "打乱的转移" | 环形缓冲区，每个梯度步均匀采样；解相关数据。 |
| 目标网络 | "冻结的自举" | 周期复制用于 Bellman 目标的 Q；稳定训练。 |
| 致命三角 | "为什么 RL 会发散" | 函数近似 + 自举 + 离策略 = 无收敛保证。 |
| Double DQN | "解决最大化偏差" | 在线网络选择动作，目标网络评估。 |
| Dueling DQN | "V 和 A 头" | 分解 Q = V + A - mean(A)；输出相同，梯度流更好。 |
| Rainbow | "所有技巧的集合" | DDQN + PER + dueling + n 步 + 噪声 + 分布式合二为一。 |
| PER | "优先回放" | 按 TD 误差量级比例采样转移。 |

## 延伸阅读

- [Mnih 等人 (2013)。Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 开启深度强化学习的 2013 年 NeurIPS 研讨会论文。
- [Mnih 等人 (2015)。Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — 《Nature》论文，49 款游戏的 DQN。
- [Hasselt, Guez, Silver (2016)。Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN。
- [Wang 等人 (2016)。Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — Dueling DQN。
- [Hessel 等人 (2018)。Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — 技巧堆叠论文。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 清晰的现代讲解。
- [Sutton & Barto (2018)。第 9 章 — 带近似的在线策略预测](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书对"致命三角"（函数近似 + 自举 + 离策略）的处理，DQN 的目标网络和回放缓冲区正是为了驯服它而设计。
- [CleanRL DQN 实现](https://docs.cleanrl.dev/rl-algorithms/dqn/) — 消融研究中使用的参考单文件 DQN；与本课的从零实现版本配合阅读效果更佳。
