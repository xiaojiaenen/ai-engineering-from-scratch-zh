# 深度Q网络 (DQN)

> 2013年：Mnih在原始像素上训练了一个Q学习网络，在七款Atari游戏上击败了所有经典强化学习智能体。2015年：扩展到49款游戏，发表于《自然》杂志，引发了深度强化学习时代。DQN是Q学习加上三个技巧，使得函数逼近变得稳定。

**类型：** 构建
**语言：** Python
**先决条件：** 第3阶段 · 03（反向传播），第9阶段 · 04（Q学习，SARSA）
**时长：** ~75分钟

## 问题所在

表格型Q学习需要为每个（状态，动作）对存储一个独立的Q值。一个棋盘有大约10⁴³种状态。一帧Atari画面是210×160×3 = 100,800个特征。表格型强化学习在成千上万种状态时就失效了，更不用说数十亿种。

事后看来，解决方案显而易见：用一个神经网络`Q(s, a; θ)`替代Q表。但这种“事后显而易见”的认识花了数十年时间。使用朴素函数逼近的Q学习在“致命三要素”下会发散——函数逼近 + 自举 + 离策略学习。Mnih等人（2013，2015）确定了三个稳定学习的工程技巧：

1. **经验回放** 去相关转换。
2. **目标网络** 冻结自举目标。
3. **奖励裁剪** 归一化梯度大小。

DQN在Atari上首次实现了用单一架构和单一超参数集，从原始像素解决数十个控制问题。自那以后构建的一切“深度强化学习”算法——DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57——都是建立在这个三技巧基础之上。

## 核心概念

![DQN训练循环：环境、回放缓冲、在线网络、目标网络、贝尔曼TD损失](../assets/dqn.svg)

**目标。** DQN最小化神经Q函数上的单步TD损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 在线网络，每步通过梯度下降更新。`θ^-` = 目标网络，定期从`θ`复制（每约10,000步）。`D` = 过去转换的回放缓冲。

**三个技巧，按重要性排序：**

**经验回放。** 一个包含`~10⁶`个转换的环形缓冲区。每个训练步骤均匀随机采样一个小批量。这打破了时间相关性（连续帧几乎相同），让网络能从罕见的有奖励转换中多次学习，并去相关连续的梯度更新。没有它，基于神经网络的在线策略TD在Atari上会发散。

**目标网络。** 在贝尔曼方程两侧使用同一个网络`Q(·; θ)`会使目标在每次更新时移动——“追逐自己的尾巴”。修复方法是：保持第二个网络`Q(·; θ^-)`，其权重冻结。每`C`步，复制`θ → θ^-`。这稳定了回归目标，使其能在数千次梯度更新中保持有效。软更新`θ^- ← τ θ + (1-τ) θ^-`（用于DDPG，SAC）是一个更平滑的变体。

**奖励裁剪。** Atari的奖励大小从1到1000+不等。裁剪到`{-1, 0, +1}`防止任何单个游戏主导梯度。当奖励大小重要时这是错误的；对于Atari来说没问题，因为只有符号重要。

**双DQN。** Hasselt (2016) 修复了最大化偏差：用在线网络*选择*动作，用目标网络*评估*它。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

即插即用，效果一致更好。默认使用它。

**其他改进（Rainbow，2017）：** 优先级回放（更频繁采样高TD误差的转换），决斗架构（分离`V(s)`和优势头），噪声网络（学习的探索），n步回报，分布式Q（C51/QR-DQN），多步自举。每个增加几个百分点；收益大致是相加的。

## 动手实现

这里的代码只使用标准库，不依赖numpy——我们手写一个单隐层MLP，在一个微小的连续GridWorld上运行，因此每个训练步骤在微秒内完成。该算法在规模上与Atari DQN相同。

### 步骤1：回放缓冲

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

Atari约50,000容量；我们的玩具环境5,000就足够了。

### 步骤2：一个微小的Q网络（手动MLP）

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

前向传播：线性 → ReLU → 线性。这就是整个网络。

### 步骤3：DQN更新

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

形式是第04课的Q学习，有两个区别：(a) 我们通过可微的`Q(·; θ)`反向传播，而不是索引一个表；(b) 目标使用`Q(·; θ^-)`。

### 步骤4：外层循环

对于每个episode，在`Q(·; θ)`上执行ε-greedy动作，将转换推入缓冲区，采样一个小批量，执行一次梯度步进，定期同步`θ^- ← θ`。模式如下：

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

在我们的微小GridWorld上，状态是16维的独热编码，智能体在约500个episode内学习到接近最优的策略。在Atari上，将此扩展到2亿帧并添加CNN特征提取器。

## 常见陷阱

- **致命三要素。** 函数逼近 + 离策略 + 自举可能发散。DQN通过目标网络 + 回放缓冲来缓解；不要移除任何一个。
- **探索。** ε必须衰减，通常在训练的前~10%从1.0降到0.01。没有足够的早期探索，Q网络会收敛到局部盆地。
- **过高估计。** `max`对有噪声的Q存在向上偏差。在生产环境中务必使用双DQN。
- **奖励尺度。** 裁剪或归一化奖励；梯度大小与奖励大小成正比。
- **回放缓冲冷启动。** 在缓冲区有几千个转换之前不要训练。在大约20个样本上的早期梯度会过拟合。
- **目标同步频率。** 太频繁≈没有目标网络；太不频繁≈过时目标。Atari DQN使用10,000个环境步。经验法则：每训练时间范围的约1/100同步一次。
- **观测预处理。** Atari DQN堆叠4帧以使状态具有马尔可夫性。任何包含速度信息的环境都需要帧堆叠或循环状态。

## 应用场景

在2026年，DQN很少是最先进的，但仍然是参考的离策略算法：

| 任务 | 首选方法 | 为何不用DQN？ |
|------|----------|--------------|
| 离散动作类Atari游戏 | Rainbow DQN 或 Muesli | 相同框架，更多技巧。 |
| 连续控制 | SAC / TD3（第9阶段 · 07） | DQN没有策略网络。 |
| 在线策略 / 高吞吐 | PPO（第9阶段 · 08） | 无回放缓冲；更易扩展。 |
| 离线强化学习 | CQL / IQL / Decision Transformer | 保守的Q目标，无自举爆炸。 |
| 大型离散动作空间（推荐系统） | 带动作嵌入的DQN，或 IMPALA | 没问题；嵌入很重要。 |
| 大语言模型RL | PPO / GRPO | 序列级别，非步级别；损失不同。 |

经验教训仍然适用。回放和目标网络出现在SAC、TD3、DDPG、SAC-X、AlphaZero的自对弈缓冲区以及每个离线强化学习方法中。奖励裁剪作为PPO中的优势归一化而存续。该架构是蓝图。

## 保存运行

保存为`outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 练习

1. **简单。** 运行`code/main.py`。绘制每个episode的回报曲线。多少个episode后，运行平均值超过-10？
2. **中等。** 禁用目标网络（在贝尔曼目标的两侧都使用在线网络）。衡量训练不稳定性——回报是否振荡或发散？
3. **困难。** 添加双DQN：用在线网络选择`argmax a'`，用目标网络评估。在具有噪声奖励的GridWorld上，比较有无双DQN时，经过1,000个episode后`Q(s_0, best_a)`与真实`V*(s_0)`的偏差。

## 关键术语

| 术语 | 人们常说 | 它的实际含义 |
|------|----------|-------------|
| DQN | “深度Q学习” | 使用神经Q函数、回放缓冲区和目标网络的Q学习。 |
| 经验回放 | “打乱的转换” | 每个梯度步均匀采样的环形缓冲区；去相关数据。 |
| 目标网络 | “冻结的自举” | 用于贝尔曼目标的Q的定期副本；稳定训练。 |
| 致命三要素 | “为什么强化学习会发散” | 函数逼近 + 自举 + 离策略 = 无收敛保证。 |
| 双DQN | “最大化偏差的修复” | 在线网络选择动作，目标网络评估它。 |
| 决斗DQN | “V和A头” | 将Q分解为V + A - mean(A)；输出相同，梯度流更好。 |
| Rainbow | “所有技巧” | DDQN + PER + 决斗 + n步 + 噪声 + 分布式，集于一身。 |
| PER | “优先级回放” | 根据TD误差大小按比例采样转换。 |

## 延伸阅读

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 2013年NeurIPS研讨会论文，开启了深度强化学习。
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — 《自然》论文，49款游戏的DQN。
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN。
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — 决斗DQN。
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — 堆叠技巧的论文。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 清晰的现代阐述。
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — 教科书对“致命三要素”（函数逼近 + 自举 + 离策略）的处理，DQN的目标网络和回放缓冲正是为了驯服它而设计。
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — 用于消融研究的参考单文件DQN实现；适合与本课从头构建的版本一同阅读。