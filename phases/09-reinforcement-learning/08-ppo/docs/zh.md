# 近端策略优化 (PPO)

> A2C 在一次更新后就会丢弃每个轨迹数据。PPO 通过一个截断的重要性比率包裹策略梯度，从而能在同一数据上进行 10 次以上迭代而策略不会崩溃。Schulman 等人 (2017)。直到 2026 年它仍是默认的策略梯度算法。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段 9 · 06 (REINFORCE), 阶段 9 · 07 (Actor-Critic)
**时间：** 约 75 分钟

## 问题所在

A2C（第 07 课）是同策略的：梯度 `E_{π_θ}[A · ∇ log π_θ]` 要求数据采样自 *当前* 策略 `π_θ`。进行一次更新后，`π_θ` 发生变化；你所使用的数据就变成了异策略的。重复使用这些数据会导致梯度有偏。

轨迹采集是昂贵的。在 Atari 游戏中，一次轨迹采集（8 个环境 × 128 步）= 1024 个转移和十几秒的环境时间。一次梯度更新后就丢弃这些数据是浪费的。

信赖域策略优化（TRPO, Schulman 2015）是第一个解决方案：约束每次更新，使得新旧策略之间的 KL 散度保持在 `δ` 以下。理论上很简洁，但每次更新都需要共轭梯度求解。2026 年没人用 TRPO 了。

PPO（Schulman 等人 2017）用一个简单的截断目标替代了硬性信赖域约束。只需多加一行代码。每个轨迹可进行 10 次迭代。无需共轭梯度。理论保证足够好。九年过去了，它仍然是从 MuJoCo 到 RLHF 等所有领域的默认策略梯度算法。

## 核心概念

![PPO 截断代理目标：重要性比率在 1 ± ε 处截断](../assets/ppo.svg)

**重要性比率。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略与收集数据的策略之间的似然比。`r_t = 1` 表示没有变化。`r_t = 2` 表示新策略采取动作 `a_t` 的可能性是旧策略的两倍。

**截断代理目标。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

包含两项：

- 如果优势 `A_t > 0` 为正且比率试图增长超过 `1 + ε`，截断会使梯度变平——不要将一个好的动作推到比旧概率高 `+ε` 以上的地方。
- 如果优势 `A_t < 0` 为负且比率试图增长超过 `1 - ε`（意味着我们会使一个坏动作相对于其截断后的概率变得更具可能性），截断会限制梯度——不要将一个坏动作的概率推到比 `-ε` 更低的地方。

`min` 处理另一个方向：如果比率朝着 *有利* 方向移动，你仍然能获得梯度（在对你有害的一侧不截断）。

典型的 `ε = 0.2`。将目标函数绘制为 `r_t` 的函数：这是一个分段线性函数，在“好”的一侧有一个平顶，在“坏”的一侧有一个平底。

**完整的 PPO 损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

与 A2C 相同的 Actor-Critic 结构。三个系数，通常是 `c_v = 0.5`、`c_e = 0.01`、`ε = 0.2`。

**训练循环。**

1. 在 `N` 个并行环境中采集 `N × T` 个转移，每个环境运行 `T` 步。
2. 计算优势（GAE），将其冻结为常数。
3. 冻结 `π_{θ_old}` 作为当前 `π_θ` 的快照。
4. 进行 `K` 个迭代，对于每个大小为 `(s, a, A, V_target, log π_old(a|s))` 的小批量数据：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + 值损失 + 熵。
   - 进行梯度更新。
5. 丢弃轨迹数据。返回第 1 步。

`K = 10` 和 64 的小批量大小是一组标准的超参数。PPO 鲁棒性很好：确切的数值在 ±50% 范围内通常影响不大。

**KL 惩罚变体。** 原始论文提出了一个使用自适应 KL 惩罚的替代方案：`L = L^{PG} - β · KL(π_θ || π_old)`，其中 `β` 根据观测到的 KL 进行调整。截断版本成为了主流；KL 变体在 RLHF 中得以保留（因为对参考策略的 KL 散度本身就是一个你始终想要满足的独立约束）。

## 动手构建

### 第 1 步：在轨迹采集时捕获 `log π_old(a | s)`

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

快照在轨迹采集时获取一次。在更新迭代期间不会改变。

### 第 2 步：计算 GAE 优势（第 07 课）

与 A2C 相同。在批次内进行归一化。

### 第 3 步：截断代理更新

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
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

“截断 → 梯度为零”的模式是 PPO 的核心。如果新策略已经朝着有利方向漂移太远，更新就会停止。

### 第 4 步：值函数和熵

向 Critic 目标添加标准的 MSE，并对 Actor 添加熵奖励，与 A2C 相同。

### 第 5 步：诊断指标

每次更新需关注三件事：

- **平均 KL 散度** `E[log π_old - log π_θ]`。应保持在 `[0, 0.02]` 范围内。如果它远超 `0.1`，则需减小 `K_EPOCHS` 或 `LR`。
- **截断比例** —— 比率落在 `[1-ε, 1+ε]` 区间外的样本比例。应为 `~0.1-0.3`。如果 `~0`，则截断从未触发 → 提高 `LR` 或 `K_EPOCHS`。如果 `~0.5+`，说明对轨迹数据过拟合 → 降低它们。
- **解释方差** `1 - Var(V_target - V_pred) / Var(V_target)`。Critic 质量指标。随着 Critic 的学习，应趋向 1。

## 常见陷阱

- **截断系数未调优。** `ε = 0.2` 是事实上的标准。设为 `0.1` 会使更新过于保守；`0.3+` 则会引发不稳定性。
- **迭代次数过多。** `K > 20` 经常导致不稳定，因为策略会远离 `π_old`。限制迭代次数，尤其是对于大型网络。
- **未进行奖励归一化。** 大的奖励范围会侵蚀截断区间。在计算优势之前，对奖励进行归一化（使用运行时的标准差）。
- **忘记对优势归一化。** 按批次进行零均值/单位方差归一化是标准做法。跳过这一步会在大多数基准测试上破坏 PPO。
- **学习率未衰减。** PPO 受益于学习率线性衰减至零。恒定学习率通常效果更差。
- **重要性比率计算错误。** 为保证数值稳定性，始终使用 `exp(log_new - log_old)`，而非 `new / old`。
- **梯度符号错误。** 最大化代理目标 = *最小化* `-L^{CLIP}`。符号反转是最常见的 PPO bug。

## 使用场景

PPO 是 2026 年跨多个领域的默认 RL 算法，应用范围出奇地广：

| 使用场景 | PPO 变体 |
|----------|-------------|
| MuJoCo / 机器人控制 | 使用高斯策略的 PPO，GAE(0.95) |
| Atari / 离散游戏 | 使用类别策略的 PPO，滚动 128 步轨迹采集 |
| LLM 的 RLHF | 使用 KL 惩罚的 PPO，指向参考模型，奖励来自响应结束时的 RM |
| 大规模游戏智能体 | IMPALA + PPO (AlphaStar, OpenAI Five) |
| 推理 LLM | GRPO（第 12 课）—— 无 Critic 的 PPO 变体 |
| 仅偏好数据 | DPO —— PPO+KL 的封闭形式解，无需在线采样 |

PPO 的 *损失形状* —— 截断代理 + 值损失 + 熵 —— 是 DPO、GRPO 以及几乎所有 RLHF 流程的框架。

## 部署

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上运行 PPO，使用 `ε=0.2, K=4`。在相同的环境步数下，比较其与 A2C（每个轨迹一次迭代）的样本效率。
2. **中等。** 扫描 `K ∈ {1, 4, 10, 30}`。绘制回报 vs 环境步数的图表，并跟踪每次更新的平均 KL 散度。在本任务中，`K` 设为多少时 KL 散度会爆炸？
3. **困难。** 将截断代理替换为自适应 KL 惩罚（`β`：如果 `KL > 2·target` 则加倍，如果 `KL < target/2` 则减半）。比较最终回报、稳定性和无截断性。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|-----------------|-----------------------|
| 重要性比率 | "r_t(θ)" | `π_θ(a|s) / π_old(a|s)`；与收集数据的策略的偏差。 |
| 截断代理 | "PPO 的主要技巧" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；超过截断点后在有利侧梯度变平。 |
| 信赖域 | "TRPO / PPO 的意图" | 限制每次更新的 KL 散度以保证单调改进。 |
| KL 惩罚 | "软信赖域" | PPO 的替代方案：`L - β · KL(π_θ || π_old)`。自适应 `β`。 |
| 截断比例 | "截断触发的频率" | 诊断指标 —— 应为 0.1-0.3；超出则意味着调参不当。 |
| 多次迭代训练 | "数据复用" | 每个轨迹进行 K 次迭代；用方差成本换取样本效率。 |
| 准同策略 | "基本上是同策略" | PPO 名义上是同策略的，但 K>1 次迭代安全地使用了略微偏离策略的数据。 |
| PPO-KL | "另一种 PPO" | KL 惩罚变体；用于 RLHF，因为指向参考模型的 KL 本身已是约束。 |

## 延伸阅读

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 原始论文。
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO，PPO 的前身。
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — 对每个 PPO 超参数进行了消融实验。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT；RLHF 中的 PPO 方案。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — 使用 PyTorch 的清晰现代阐述。
- [CleanRL PPO 实现](https://github.com/vwxyzjn/cleanrl) — 许多论文使用的参考单文件 PPO 实现。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — 语言模型 PPO 的生产级方案；与第 09 课（RLHF）一同阅读。
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — “37 个代码级优化”论文；哪些 PPO 技巧是关键支撑，哪些只是经验之谈。