```python
# 直接偏好优化家族

> Rafailov 等人（2023）证明 RLHF 的最优解关于偏好数据具有闭合形式，因此你可以跳过显式奖励模型，直接优化策略。这一洞见催生了一个算法家族——IPO、KTO、SimPO、ORPO、BPO——每个算法都修复了 DPO 的某一种失效模式。在 2026 年，直接对齐算法 shipped 到生产环境的后训练任务数超过 PPO。但 Lesson 2 中的过优化曲线仍然适用：直接对齐算法并未逃离古德哈特定律，只是改变了它刺痛的地点。

**类型：** 学习
**语言：** Python（标准库，含 six-variant 偏好损失比较器）
**前置条件：** 阶段 18 · 01（InstructGPT）、阶段 18 · 02（奖励黑客攻击）、阶段 10 · 08（DPO 基础）
**时间：** 约 75 分钟

## 学习目标

- 从 RLHF-with-KL 最优解推导 DPO 的闭合形式。
- 阐述 IPO、KTO、SimPO、ORPO、BPO 各自修复了 DPO 的哪种失效模式。
- 区分"隐式奖励间隔"与"偏好强度"，并解释 IPO 的身份映射为何重要。
- 说明为何 Rafailov 等人（NeurIPS 2024）证明即使没有显式奖励模型，直接对齐算法仍会过优化。

## 问题所在

RLHF 目标函数（Lesson 1）：

```
max_π E_{x,y~π}[r(x, y)] - β * KL(π || π_ref)
```

存在已知最优解：

```
π*(y|x) = (1/Z(x)) * π_ref(y|x) * exp(r(x, y) / β)
```

因此奖励被隐式定义为最优策略与参考策略之比：

```
r(x, y) = β * log(π*(y|x) / π_ref(y|x)) + β * log Z(x)
```

将其代入 Bradley-Terry 偏好似然函数，且由于分区函数 `Z(x)` 仅依赖于 `x`，它在计算中被消去。剩余部分仅是策略参数的损失——无需奖励模型。这就是 DPO。

关键在于：该推导假设最优解可达、偏好数据处于分布内、且参考策略是真值模态锚点。这些假设均不完全成立。该家族的每个成员各自修复一个不同的被违背假设。

## 概念详解

### DPO（Rafailov 等，2023）

```
L_DPO = -log sigmoid(
  β * log(π(y_w | x) / π_ref(y_w | x))
  - β * log(π(y_l | x) / π_ref(y_l | x))
)
```

可能出现的问题：

- 隐式奖励间隔 `β * (log(π/π_ref)_w - log(π/π_ref)_l)` 是无界的。微小的偏好可能产生任意大的间隔。
- 损失函数推动 chosen 和 rejected 的 log-prob 向相反方向变化。只要 rejected 下降更快，它可以推动 chosen 的绝对 log-prob 下降。这就是"退化选择响应"现象。
- 分布外偏好（稀有对 vs 稀有对）产生任意隐式奖励。

### IPO（Azar 等，2024）

身份偏好优化（Identity Preference Optimization）用恒等映射替换 log-sigmoid，对偏好概率进行平方误差损失：

```
L_IPO = (log(π(y_w | x) / π_ref(y_w | x)) - log(π(y_l | x) / π_ref(y_l | x)) - 1/(2β))^2
```

间隔被限定在 `1/(2β)`。偏好强度与隐式奖励间隔成正比。不会爆炸。

### KTO（Ethayarajh 等，2024）

 Kahneman-Tversky 优化完全放弃了成对结构。给定单个标注输出和二元"可取"或"不可取"信号，映射为前景理论效用：

```
v(x, y) = σ(β * log(π(y|x) / π_ref(y|x)) - z_ref)
```

对收益和损失使用不同权重（损失厌恶）。好处是你可以使用非配对数据，这类数据丰富得多。

### SimPO（Meng 等，2024）

简单偏好优化让训练信号与生成过程对齐。完全移除参考策略，并按长度归一化 log-likelihood：

```
L_SimPO = -log sigmoid(
  (β / |y_w|) * log π(y_w | x)
  - (β / |y_l|) * log π(y_l | x)
  - γ
)
```

带有间隔 `γ` 以稳定训练。长度归一化消除了利用 DPO 的长度偏差失效模式（更长的 `y_w` 天然产生更大的 log-prob 间隔）的动机。

### ORPO（Hong 等，2024）

 odds-ratio 偏好优化在标准 SFT 负对数似然基础上添加偏好项：

```
L_ORPO = L_NLL(y_w) + λ * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

无需参考策略——SFT 项作为正则化项。从基础模型到对齐模型单阶段训练。无需单独的 SFT checkpoint。

### BPO（ICLR 2026 投稿，OpenReview id=b97EwMUWu7）

识别出退化选择响应问题：DPO 保留了排序 `y_w > y_l`，但 `y_w` 的绝对 log-prob 可能下降。BPO 添加了一行修正项，惩罚 chosen 响应的向下移动。在 Llama-3.1-8B-Instruct 的数学推理任务上，相比 DPO 报告了 +10.1% 的准确率提升。

### 普适性结论：直接对齐算法仍会过优化

Rafailov 等人《直接对齐算法的奖励模型过优化缩放律》（NeurIPS 2024）在多种数据集上、跨 KL 预算，训练了使用 DPO、IPO、SLiC 的策略。金标准奖励与 KL 的曲线呈现出同样的 Gao 等人峰值-坍塌形状。训练过程中隐式奖励在分布外样本上查询；KL 正则化无法稳定这一现象。

直接对齐算法并未逃离古德哈特定律。它们将痛苦的表面从"奖励模型过优化"转移到"参考策略比值过优化"。普适性解决方案——更好的数据、集成、早停——两者均适用。

### 2026 年如何选择

- 若你有大量成对偏好数据：保守 β 的 DPO；若存在长度偏差则用 SimPO。
- 若你有非配对二值反馈：KTO。
- 若你想要从基础模型开始的单阶段管道：ORPO。
- 若在 DPO 日志中看到退化的 chosen log-prob：BPO。
- 若偏好强度差异大且 DPO 饱和：IPO。

每个实验室都会在任务套件上跑全部五种算法，然后按任务选择获胜者。数学推理和安全对齐的最优解未必相同。

```figure
dpo-margin
```

## 实践

`code/main.py` 在玩具偏好数据集上比较六种损失（DPO、IPO、KTO、SimPO、ORPO、BPO），其中真实偏好强度随样本对变化。每种损失在相同 500 对样本上使用小型 softmax 策略进行优化。绘制最终的胜率、chosen-log-prob 漂移、以及每种方法的隐式奖励散布。

## 交付物

本课产出 `outputs/skill-preference-loss-selector.md`。给定数据集统计信息（成对 vs 非配对、偏好强度变化 vs 均匀、长度分布）和目标（单阶段还是 SFT-后偏好），推荐一种偏好损失并报告它保护的失效模式。

## 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 的最终 chosen-log-prob 下降量。BPO 应保留更高的 chosen 绝对概率——验证这一点。

2. 修改偏好数据使所有对的强度相等。哪种方法最稳健？哪种退化？解释 IPO 在此处的优势。

3. 使 rejected 响应平均比 chosen 长 2 倍。不改变其他任何东西，数值展示 DPO 的长度剥削和 SimPO 的修复。

4. Rafailov 等人（NeurIPS 2024）声称直接对齐算法会过优化。复现单点版本：绘制 chosen 减 rejected 的 KL 散度，观察 DPO 在较大 β 时的过优化现象。

5. 阅读 BPO 论文摘要（OpenReview b97EwMUWu7）。写出 BPO 向 DPO 添加的那一行修正。与 `code/main.py` 的实现对照确认。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| DPO | "无奖励模型的 RLHF" | 从 RLHF 最优解闭合形式推导的损失；仅含策略参数 |
| 隐式奖励 | "log 比值" | `β * log(π(y\|x) / π_ref(y\|x))`——DPO 隐含的奖励 |
| IPO | "有界的 DPO" | 用恒等映射替换 log-sigmoid；隐式奖励间隔上限为 `1/(2β)` |
| KTO | "非配对的 DPO" | 前景理论效用作用于单标签，含损失厌恶 |
| SimPO | "无参考的 DPO" | 长度归一化 log-likelihood + 间隔；无参考策略 |
| ORPO | "单阶段的 DPO" | NLL + odds-ratio 偏好项；从基础模型单遍训练 |
| BPO | "保留选择的 DPO" | DPO 加上对 chosen 响应绝对 log-prob 下降的惩罚 |
| 退化选择 | "chosen 下降" | DPO 推动 chosen 的 log-prob 下降，只要 rejected 下降更快 |
| DAA | "直接对齐算法" | 跳过显式奖励模型的偏好损失方法统称 |

## 延伸阅读

- [Rafailov 等 —— 直接偏好优化（NeurIPS 2023，arXiv:2305.18290）](https://arxiv.org/abs/2305.18290)
- [Azar 等 —— 理解人类偏好学习的一般理论范式（AISTATS 2024，arXiv:2310.12036）](https://arxiv.org/abs/2310.12036) —— IPO
- [Ethayarajh 等 —— KTO：模型对齐作为前景理论优化（arXiv:2402.01306）](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen —— SimPO（NeurIPS 2024，arXiv:2405.14734）](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne —— ORPO（EMNLP 2024，arXiv:2403.07691）](https://arxiv.org/abs/2403.07691)
- [BPO —— 行为保留优化（ICLR 2026，OpenReview b97EwMUWu7）](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov 等 —— 直接对齐算法的奖励模型过优化缩放律（NeurIPS 2024，arXiv:2406.02900）](https://arxiv.org/abs/2406.02900)
```
