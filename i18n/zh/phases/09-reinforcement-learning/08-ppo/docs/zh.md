# 近端策略优化 (PPO)

> A2C 在一次更新后就会丢弃每条轨迹。PPO 通过裁剪的重要性比率包装策略梯度，使得你可以在同一批数据上执行 10+ 个 epoch 而不会导致策略爆炸。Schulman 等人（2017）。即使在 2026 年，它仍是默认的策略梯度算法。

**类型：** 构建
**语言：** Python
**前置知识：** 第 9 阶段 · 06（REINFORCE），第 9 阶段 · 07（Actor-Critic）
**时间：** 约 75 分钟

## 问题所在

A2C（课程 07）是在线策略算法：梯度 `E_{π_θ}[A · ∇ log π_θ]` 需要从*当前* `π_θ` 采样的数据。进行一次更新后，`π_θ` 就会改变；你使用的数据就变成了离策略数据。重新使用它会导致梯度存在偏差。

轨迹的收集代价高昂。在 Atari 上，跨越 8 个环境的 128 步轨迹 = 1024 个转移，需要数秒的环境运行时间。在一步梯度更新后就把这些数据丢弃是浪费。

信任域策略优化（TRPO，Schulman 2015）是第一种解决方案：约束每次更新，使得新旧策略之间的 KL 散度保持在 `δ` 以下。理论清晰，但每次更新都需要共轭梯度求解。没有人会在 2026 年使用 TRPO。

PPO（Schulman 等人，2017）用简单的裁剪目标函数取代了硬性的信任域约束。多一行代码。每个轨迹 10 个 epoch。无需共轭梯度。足够的理论保证。九年过去了，它仍然是从 MuJoCo 到 RLHF 各领域的默认策略梯度算法。

## 核心概念

![PPO 裁剪代理目标函数：在 1 ± ε 处进行比率裁剪](../assets/ppo.svg)

**重要性比率。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略相对于收集数据的策略的似然比率。`r_t = 1` 表示没有变化。`r_t = 2` 表示新策略采取 `a_t` 的可能性是旧策略的两倍。

**裁剪代理目标。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两项：

- 如果优势 `A_t > 0` 且比率试图增长超过 `1 + ε`，裁剪会平滑梯度——不要将好动作进一步推高到旧概率的 `+ε` 以上。
- 如果优势 `A_t < 0` 且比率试图增长超过 `1 - ε`（意味着我们会让坏动作变得更可能，相对于其裁剪后的减少），裁剪会限制梯度——不要将坏动作推低到 `-ε` 以下。

`min` 处理另一个方向：如果比率向*有利*方向移动，你仍然获得梯度（在可能伤害你的方向上不裁剪）。

典型 `ε = 0.2`。绘制目标函数关于 `r_t` 的图像：一个分段线性函数，在"好侧"有平顶，在"坏侧"有平底。

**完整的 PPO 损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

与 A2C 相同的 actor-critic 结构。三个系数，通常为 `c_v = 0.5`，`c_e = 0.01`，`ε = 0.2`。

**训练循环。**

1. 在 `N` 个并行环境中收集 `N × T` 个转移，每个环境 `T` 步。
2. 计算优势（GAE），将其冻结为常量。
3. 冻结 `π_{θ_old}` 作为当前 `π_θ` 的快照。
4. 对 `K` 个 epoch，对每个小批量 `(s, a, A, V_target, log π_old(a|s))`：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + 价值损失 + 熵。
   - 梯度步。
5. 丢弃轨迹。返回步骤 1。

`K = 10` 和小批量大小 64 是标准超参数设置。PPO 很鲁棒：精确数值在 ±50% 内通常无关紧要。

**KL 惩罚变体。** 原始论文提出了一种使用自适应 KL 惩罚的替代方案：`L = L^{PG} - β · KL(π_θ || π_old)`，其中 `β` 根据观察到的 KL 进行调整。裁剪版本成为主流；KL 变体在 RLHF 中存活下来（在那里，KL 到参考策略是一个你始终想要的独立约束）。

```figure
ppo-clip
```

## 动手构建

### 步骤 1：在轨迹收集时捕获 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照在轨迹收集时取一次。在更新 epoch 期间不会改变。

### 步骤 2：计算 GAE 优势（课程 07）

与 A2C 相同。对批次进行归一化。

### 步骤 3：裁剪代理更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # 反向传播 -surrogate，加上价值损失，减去熵
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # 已裁剪
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

"裁剪 → 零梯度"模式是 PPO 的核心。如果新策略已经在有利方向上漂移太远，更新就会停止。

### 步骤 4：价值和熵

添加标准 MSE 到评论者目标，并在执行者上添加熵奖励，与 A2C 相同。

### 步骤 5：诊断

每次更新需要关注三件事：

- **平均 KL** `E[log π_old - log π_θ]`。应保持在 `[0, 0.02]` 范围内。如果超过 `0.1`，减少 `K_EPOCHS` 或 `LR`。
- **裁剪分数** —— 比率位于 `[1-ε, 1+ε]` 之外的样本比例。应为 `~0.1-0.3`。如果 `~0`，裁剪永远不会触发 → 提高 `LR` 或 `K_EPOCHS`。如果 `~0.5+`，你在过拟合轨迹 → 降低它们。
- **解释方差** `1 - Var(V_target - V_pred) / Var(V_target)`。评论者质量指标。随着评论者学习应趋近于 1。

## 常见陷阱

- **裁剪系数设置不当。** `ε = 0.2` 是事实标准。设置为 `0.1` 会使更新过于保守；`0.3+` 会引发不稳定性。
- **epoch 过多。** `K > 20` 通常会破坏稳定性，因为策略会远离 `π_old`。限制 epoch，尤其是对于大网络。
- **未进行奖励归一化。** 大的奖励尺度会侵蚀裁剪范围。在计算优势前对奖励进行归一化（运行标准差）。
- **忘记优势归一化。** 每批次零均值/单位标准差归一化是标准做法。跳过它会破坏 PPO 在大多数基准测试上的表现。
- **学习率未衰减。** PPO 受益于线性 LR 衰减到零。恒定 LR 通常更差。
- **重要性比率数学错误。** 始终使用 `exp(log_new - log_old)` 以获得数值稳定性，而非 `new / old`。
- **梯度符号错误。** 最大化代理目标 = *最小化* `-L^{CLIP}`。翻转符号是最常见的 PPO bug。

## 应用场景

PPO 是 2026 年跨多个领域的默认 RL 算法：

| 用例 | PPO 变体 |
|----------|-------------|
| MuJoCo / 机器人控制 | 带高斯策略的 PPO，GAE(0.95) |
| Atari / 离散游戏 | 带分类策略的 PPO，滚动 128 步轨迹 |
| LLM 的 RLHF | 带 KL 惩罚到参考模型的 PPO，奖励来自响应末尾的 RM |
| 大规模游戏智能体 | IMPALA + PPO（AlphaStar，OpenAI Five）|
| 推理 LLM | GRPO（课程 12）—— 无评论者的 PPO 变体 |
| 仅偏好数据 | DPO —— PPO+KL 的闭式坍缩，无需在线采样 |

PPO *损失形状* —— 裁剪代理 + 价值 + 熵 —— 是 DPO、GRPO 和几乎所有 RLHF 管道的骨架。

## 交付

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: 为给定环境生成 PPO 训练配置和诊断计划。
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

给定一个环境和训练预算，输出：

1. 轨迹大小。`N` 个环境 × `T` 步。
2. 更新计划。`K` 个 epoch，小批量大小，LR 计划。
3. 代理参数。`ε`（裁剪），`c_v`，`c_e`，优势归一化开启。
4. 优势。GAE(`λ`) 带有显式 `γ` 和 `λ`。
5. 诊断计划。KL、裁剪分数、解释方差阈值及警报。

拒绝 `K > 30` 或 `ε > 0.3`（不安全信任域）。拒绝任何没有优势归一化或 KL/裁剪监控的 PPO 运行。标记持续高于 0.4 的裁剪分数为漂移。
```

## 练习

1. **简单。** 在 4×4 GridWorld 上运行 PPO，`ε=0.2, K=4`。与 A2C（每个轨迹一个 epoch）在匹配的环境步数下比较样本效率。
2. **中等。** 扫描 `K ∈ {1, 4, 10, 30}`。绘制回报与环境步数的关系，并跟踪每次更新的平均 KL。在此任务上 KL 在哪个 `K` 值爆炸？
3. **困难。** 用自适应 KL 惩罚替换裁剪代理（如果 `KL > 2·target` 则 `β` 加倍，如果 `KL < target/2` 则减半）。比较最终回报、稳定性和无裁剪程度。

## 关键术语

| 术语 | 人们说什么 | 实际含义 |
|------|-----------------|-----------------------|
| 重要性比率 | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`；相对于收集数据的策略的偏差。 |
| 裁剪代理 | "PPO 的主要技巧" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；有利侧裁剪后梯度平坦。 |
| 信任域 | "TRPO / PPO 意图" | 限制每次更新的 KL 以保证单调改进。 |
| KL 惩罚 | "软信任域" | PPO 替代方案：`L - β · KL(π_θ \|\| π_old)`。自适应 `β`。 |
| 裁剪分数 | "裁剪触发的频率" | 诊断指标 —— 应为 0.1-0.3；超出范围表示设置不当。 |
| 多 epoch 训练 | "数据复用" | 每个轨迹 K 个 epoch；以方差成本换取样本效率。 |
| 类在线策略 | "主要是在线策略" | PPO 名义上是线策略，但 K>1 epoch 安全地使用了轻微离策略的数据。 |
| PPO-KL | "另一种 PPO" | KL 惩罚变体；用于 RLHF，其中 KL 到参考策略已经是约束。 |

## 延伸阅读

- [Schulman 等人 (2017)。近端策略优化算法](https://arxiv.org/abs/1707.06347) —— 论文原文。
- [Schulman 等人 (2015)。信任域策略优化](https://arxiv.org/abs/1502.05477) —— TRPO，PPO 的前身。
- [Andrychowicz 等人 (2021)。在线策略 RL 中什么很重要？一项大规模实证研究](https://arxiv.org/abs/2006.05990) —— 每个 PPO 超参数的消融实验。
- [Ouyang 等人 (2022)。训练语言模型遵循人类反馈的指令](https://arxiv.org/abs/2203.02155) —— InstructGPT；PPO-in-RLHF 配方。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) —— 带有 PyTorch 的干净现代阐述。
- [CleanRL PPO 实现](https://github.com/vwxyzjn/cleanrl) —— 许多论文使用的参考单文件 PPO。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) —— PPO 在语言模型上的生产配方；与课程 09（RLHF）一起阅读。
- [Engstrom 等人 (2020)。深度策略梯度中的实现很重要](https://arxiv.org/abs/2005.12729) —— "37 个代码级优化"论文；哪些 PPO 技巧是关键负载，哪些是民间传说。
