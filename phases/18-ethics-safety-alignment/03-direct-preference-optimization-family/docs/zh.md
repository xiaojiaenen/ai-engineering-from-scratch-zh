# 直接偏好优化家族

> Rafailov 等人（2023）证明 RLHF 的最优解可以表示为关于偏好数据的闭式解，因此可以跳过显式奖励模型，直接优化策略。这一洞见催生了一系列变体——IPO、KTO、SimPO、ORPO、BPO——每一种都旨在修复 DPO 的某个失效模式。到 2026 年，直接对齐算法在前沿的后训练运行中已超过 PPO。但第 2 课中的过度优化曲线依然适用：DAAs 并未逃脱古德哈特定律，只是改变了它起作用的地方。

**类型：** 学习
**语言：** Python（标准库，六种变体的偏好损失比较器）
**先决条件：** 阶段 18 · 01 (InstructGPT)，阶段 18 · 02 (奖励欺骗)，阶段 10 · 08 (DPO 基础)
**时间：** ~75 分钟

## 学习目标

- 从带有 KL 散度的 RLHF 最优解推导出 DPO 的闭式解。
- 阐述 IPO、KTO、SimPO、ORPO、BPO 各自修复了 DPO 的哪种失效模式。
- 区分“隐式奖励差距”与“偏好强度”，并解释为何 IPO 的恒等映射至关重要。
- 解释为何 Rafailov 等人（NeurIPS 2024）证明尽管没有显式奖励模型，DAAs 仍会过度优化。

## 问题

RLHF 目标函数（第 1 课）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

存在一个已知的最优解：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

因此，奖励函数由最优策略与参考策略的比率隐式定义：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

将其代入 Bradley-Terry 偏好似然函数中，配分函数 `Z(x)` 会相消，因为它仅依赖于 `x`。剩下的就是一个仅关于策略参数的损失——不再需要奖励模型。这就是 DPO。

问题在于：该推导假设最优解可达、偏好数据在分布内、参考策略是真实的模态锚点。这些假设没有一个能完全成立。该家族的每个成员都修复了一个被违反的假设。

## 概念

### DPO (Rafailov et al., 2023)

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出现的问题：

- 隐式奖励差距 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 是无界的。微小的偏好可能导致任意大的差距。
- 损失驱使 chosen 和 rejected 的对数概率向相反方向移动。只要 rejected 下降得更快，它就可以把 chosen 的绝对对数概率压低。这就是“退化的 chosen 响应”现象。
- 分布外的偏好（例如，罕见-罕见对与罕见-罕见对）会产生任意的隐式奖励。

### IPO (Azar et al., 2024)

身份偏好优化 (Identity Preference Optimization) 用对偏好概率的恒等映射替换了 log-sigmoid 函数。损失变成了关于一个有界目标的平方误差：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

边际被 `1/(2 beta)` 限定。偏好强度和隐式奖励差距成正比。不会发散。

### KTO (Ethayarajh et al., 2024)

卡尼曼-特沃斯基优化 (Kahneman-Tversky Optimization) 完全摒弃了成对结构。给定单个标记的输出和一个二元的“理想”或“不理想”信号，它将其映射到一个前景理论效用：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

对收益和损失赋予不同的权重（损失厌恶）。好处：可以使用未配对的数据，这种数据要丰富得多。

### SimPO (Meng et al., 2024)

简单偏好优化 (Simple Preference Optimization) 使训练信号与生成对齐。完全移除参考策略，并通过长度对对数似然进行归一化：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

并设置一个边际 `gamma` 以保持稳定。长度归一化消除了利用 DPO 长度偏差失效模式的动机（更长的 `y_w` 根据构造会给出更大的对数概率差距）。

### ORPO (Hong et al., 2024)

优势比偏好优化 (Odds-Ratio Preference Optimization) 在标准的 SFT 负对数似然中添加了一个偏好项：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

无需参考策略——SFT 项充当正则化器。从基础模型单阶段训练到对齐模型。无需单独的 SFT 检查点。

### BPO (ICLR 2026 投稿，OpenReview id=b97EwMUWu7)

识别出“退化的 chosen 响应”问题：DPO 保持了排序 `y_w > y_l`，但 `y_w` 的绝对对数概率可能下降。BPO 添加了一行修正，惩罚 chosen 响应对数概率的下降。在 Llama-3.1-8B-Instruct 的数学推理任务上，相比 DPO 报告了 +10.1% 的准确率提升。

### 普遍结果：DAAs 仍然过度优化

Rafailov 等人《直接对齐算法中奖励模型过度优化的缩放定律》（NeurIPS 2024）在多个 KL 预算下，使用 DPO、IPO、SLiC 在多个数据集上训练策略。黄金奖励 vs KL 曲线呈现出与 Gao 等人相同的先升峰后下降的形状。训练期间，隐式奖励会查询分布外样本；KL 正则化无法稳定这一点。

DAAs 并未逃脱古德哈特定律。它们改变了古德哈特定律起作用的表面，从“奖励模型被过度优化”变成了“参考策略比率被过度优化”。通用的解决方案——更好的数据、集成方法、早停——对两者都适用。

### 如何选择 (2026)

- 如果你有大量成对偏好数据：使用具有保守 beta 的 DPO，如果存在长度偏差则考虑 SimPO。
- 如果你有未配对的二元反馈：使用 KTO。
- 如果你想从基础模型进行单阶段流水线训练：使用 ORPO。
- 如果你在 DPO 日志中看到 chosen 对数概率退化：使用 BPO。
- 如果偏好强度变化很大且 DPO 已饱和：使用 IPO。

每个实验室都会在一系列任务上运行所有五种方法，并针对每个任务选择胜出者。没有理由认为数学推理和安全任务的最优解是相同的。

## 实践应用

`code/main.py` 在一个玩具偏好数据集上比较了六种损失（DPO、IPO、KTO、SimPO、ORPO、BPO），其中真实偏好强度随样本对而变化。每种损失针对相同的 500 个样本对，使用一个小型 softmax 策略进行优化。绘制了每种方法的最终胜率、chosen 对数概率漂移和隐式奖励分布情况。

## 部署指南

本课程产出 `outputs/skill-preference-loss-selector.md`。给定数据集统计信息（成对 vs 未配对、可变 vs 均匀偏好强度、长度分布）和目标（单阶段或 SFT-然后-偏好），推荐一种偏好损失并说明其防护的失效模式。

## 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 的最终 chosen 对数概率下降值。BPO 应该保留更高的 chosen 绝对概率——请验证这一点。
2. 修改偏好数据，使所有样本对具有相同的强度。六种方法中哪一种最稳健？哪一种会退化？解释 IPO 在此场景下的优势。
3. 使 rejected 响应平均长度是 chosen 的两倍。在不改变其他任何条件的情况下，通过数值展示 DPO 的长度利用问题以及 SimPO 的修复方案。
4. Rafailov 等人（NeurIPS 2024）声称 DAAs 会过度优化。复现一个单点版本：绘制 chosen 减去 rejected 的 KL 散度，并观察 DPO 在较大 beta 下的过度优化现象。
5. 阅读 BPO 论文摘要（OpenReview b97EwMUWu7）。写下 BPO 对 DPO 添加的单行修正。对照 `code/main.py` 中的实现进行确认。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| DPO | "没有奖励模型的 RLHF" | 从 RLHF 闭式最优解推导出的损失；仅涉及策略参数 |
| 隐式奖励 | "那个对数比率" | `beta * log(pi(y|x) / pi_ref(y|x))` —— DPO 隐含的奖励 |
| IPO | "有界的 DPO" | 用恒等映射替换 log-sigmoid；隐式奖励差距由 `1/(2 beta)` 限定 |
| KTO | "未配对的 DPO" | 基于前景理论的效用函数，应用于单标签数据，具有损失厌恶特性 |
| SimPO | "无参考的 DPO" | 长度归一化的对数似然 + 边际；无参考策略 |
| ORPO | "单阶段的 DPO" | 负对数似然 + 优势比偏好项；从基础模型单阶段训练 |
| BPO | "保持 chosen 的 DPO" | DPO 加上对降低 chosen 响应绝对对数概率的惩罚项 |
| 退化的 chosen | "chosen 的值下降了" | DPO 降低了 chosen 的对数概率，只要 rejected 的下降速度更快 |
| DAA | "直接对齐算法" | 任何跳过显式奖励模型的偏好损失方法 |

## 扩展阅读

- [Rafailov 等人 — 直接偏好优化 (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar 等人 — 一个理解从人类偏好中学习的通用理论范式 (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh 等人 — KTO：将模型对齐视为前景理论优化 (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — 行为保持优化 (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov 等人 — DAAs 中奖励模型过度优化的缩放定律 (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)