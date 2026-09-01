# 评估 — FID、CLIP Score、人类偏好

> 每个生成模型排行榜都会引用 FID、CLIP score 和一个来自人类偏好竞技场（arena）的胜率。每个数字都有一个被研究者利用来操纵的失效模式。如果你不了解这些失效模式，你就无法区分真正的改进和操纵行为。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 8 · 01（分类学）、阶段 2 · 04（评估指标）
**时间：** 约 45 分钟

## 问题所在

生成模型根据*样本质量*和*条件遵循度*来评判。两者都没有闭式度量。你的模型需要生成 10,000 张图片；需要有人为它们打分；你需要在不同模型家族、不同分辨率、不同架构之间信任这些数字。三个指标熬过了 2014-2026 年的重重考验：

- **FID（Fréchet Inception Distance）**。Inception 网络特征空间中，真实分布与生成分布之间的距离。越低越好。
- **CLIP score**。生成图片的 CLIP-image 嵌入与提示词的 CLIP-text 嵌入之间的余弦相似度。越高越好。衡量提示词遵循程度。
- **人类偏好**。将两个模型在相同提示词下直接对决，让人类（或 GPT-4 级模型）选出更好的一个，聚合为 Elo 分数。

你还会看到：IS（inception score，已基本退役）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每个指标都修正了前一个指标的某项缺陷。

## 概念

![FID、CLIP 和偏好：三个轴，不同的失效模式](../assets/evaluation.svg)

### FID — 样本质量

Heusel 等人（2017）。步骤：

1. 对 N 张真实图片和 N 张生成图片提取 Inception-v3 特征（2048 维）。
2. 对每个池拟合高斯分布：计算均值 `μ_r, μ_g` 和协方差 `Σ_r, Σ_g`。
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解释：特征空间中两个多元高斯分布之间的 Fréchet 距离。越低表示分布越相似。

失效模式：
- **小 N 时的偏差。** FID 是特征分布上的均值平方——小 N 会低估协方差，导致虚假的低位 FID。始终使用 N ≥ 10,000。
- **依赖 Inception。** Inception-v3 是在 ImageNet 上训练的。与 ImageNet 差距较大的领域（人脸、艺术、文本图片）会产生无意义的 FID。使用领域特定的特征提取器。
- **可操纵性。** 针对 Inception 先验过拟合可以得到低位 FID，但视觉质量并未提升。用 CMMD（见下文）来应对。

### CLIP score — 提示词遵循度

Radford 等人（2021）。对于生成的图片 + 提示词：

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

对 30,000 张生成图片取平均 → 得到一个标量，可在模型间比较。

失效模式：
- **CLIP 自身的盲区。** CLIP 的组合推理能力较弱（"蓝色球体上的红色立方体"经常失败）。模型可以在 CLIP score 上排名靠前，但实际上并未遵循复杂提示词。
- **短提示词偏差。** 短提示词在现实中与更多 CLIP 图像匹配。长提示词在机械层面上会降低 CLIP score。
- **提示词操纵。** 在提示词中包含"高质量、4k、杰作"等词汇会虚增 CLIP score，但并未改善图像-文本绑定。

CMMD（Jayasumana 等人，2024）修正了其中一些问题：使用 CLIP 特征而非 Inception，用最大均值差异（MMD）代替 Fréchet 距离。在检测细微质量差异方面表现更好。

### 人类偏好 — 基准真相

选取一批提示词池。分别用模型 A 和模型 B 生成。将配对结果展示给人类（或强 LLM judge）。将胜场聚合为 Elo 或 Bradley-Terry 分数。基准测试：

- **PartiPrompts（Google）**：1,600 个多样化提示词，12 个类别。
- **HPSv2**：107k 个人类标注，广泛用作自动代理。
- **ImageReward**：137k 个提示词-图片偏好对，MIT 许可。
- **PickScore**：在 Pick-a-Pic 2.6M 偏好上训练。
- **类 Chatbot-Arena 图片竞技场**：https://imagearena.ai/ 及其他。

失效模式：
- **judge 方差。** 非专家与专家的偏好不同。两者都要用。
- **提示词分布。** 挑选的提示词有利于某个家族。必须记录。
- **LLM judge 奖励黑客。** GPT-4 judge 会被漂亮但错误的输出欺骗。与人进行三角验证。

## 联合使用

一份生产环境的评估报告应包含：

1. 在 10,000-30,000 样本上与保留的真实分布计算 FID（样本质量）。
2. 在同一批样本上，相对于其提示词计算 CLIP score / CMMD（遵循度）。
3. 与上一代模型的盲测竞技场中的胜率（整体偏好）。
4. 失效模式分析：50 个随机采样的输出，标记已知问题（手部解剖、文本渲染、物体数量一致性）。

单一指标都是谎言。三个相互验证的指标 + 定性审查才是主张。

```figure
gx-fid-distributions
```

## 构建它

`code/main.py` 在合成的"特征向量"上实现了 FID、类 CLIP score 和 Elo 聚合（我们使用 4 维向量作为 Inception 特征的替身）。你会看到：

- 在小 N 和大 N 上计算 FID —— 偏差的表现。
- 将"CLIP score"实现为特征池之间的余弦相似度。
- 从合成偏好流更新 Elo 的规则。

### 步骤 1：四行实现 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 步骤 2：类 CLIP 余弦相似度

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 步骤 3：Elo 聚合

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 陷阱

- **N=1000 时的 FID。** 启发式方法在 N=10k 以下不可靠。报告低 N FID 的论文是在操纵。
- **跨分辨率比较 FID。** Inception 的 299×299 重缩放会改变特征分布。仅在匹配分辨率下比较。
- **只报告一个随机种子。** 至少运行 3 个种子。报告标准差。
- **通过负向提示词膨胀 CLIP score。** 某些管道通过过度拟合提示词来提升 CLIP。检查是否有视觉饱和。
- **提示词重叠导致的 Elo 偏差。** 如果两个模型在训练时都见过基准提示词，Elo 就无意义。使用保留的提示词集。
- **人工评估付费众包偏差。** Prolific、MTurk 标注者偏向年轻/技术友好群体。混合招募艺术/设计专家。

## 使用它

2026 年的生产评估协议：

| 支柱 | 最低要求 | 推荐 |
|------|---------|------|
| 样本质量 | 10k 样本 vs 保留真实数据的 FID | + 5k 样本的 CMMD + 各类别子集的 FID |
| 提示词遵循 | 30k 样本的 CLIP score | + HPSv2 + ImageReward + VQA 式问答 |
| 偏好 | 200 对盲测 vs 基线 | + 2000 对人工 + LLM judge + Chatbot Arena |
| 失效分析 | 50 个手部标记样本 | 500 个手部标记样本 + 自动安全分类器 |

四项支柱全部包含 = 可信主张。仅凭任何一项 = 营销话术。

## 交付

保存 `outputs/skill-eval-report.md`。Skill 接受新模型检查点和基线，输出完整的评估计划：样本量、指标、失效模式探针、签字标准。

## 练习

1. **简单。** 运行 `code/main.py`。在同一批合成分布上比较 N=100 和 N=1000 时的 FID。报告偏差幅度。
2. **中等。** 从合成类 CLIP 特征实现 CMMD（参见 Jayasumana 等人，2024 的公式）。比较其对质量差异的敏感度与 FID。
3. **困难。** 复现 HPSv2 设置：从 Pick-a-Pic 的子集中取 1000 个图片-提示词对，在偏好数据上微调一个小型 CLIP 风格评分器，并测量其与保留数据集的一致性。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| FID | "Fréchet Inception Distance" | 真实与生成 Inception 特征的高斯拟合之间的 Fréchet 距离。 |
| CLIP score | "图文相似度" | CLIP 图像嵌入与文本嵌入之间的余弦相似度。 |
| CMMD | "FID 的替代品" | CLIP 特征的 MMD；偏差更小，无需高斯假设。 |
| IS | "Inception score" | Exp KL(p(y|x) || p(y))；与现代模型相关性差，已退役。 |
| HPSv2 / ImageReward / PickScore | "学习到的偏好代理" | 在人类偏好上训练的小模型；用作自动 judge。 |
| Elo | "棋类评分" | 成对胜场的 Bradley-Terry 聚合。 |
| PartiPrompts | "基准提示词集" | Google 策展的 1,600 个提示词，涵盖 12 个类别。 |
| FD-DINO | "自监督替代方案" | 使用 DINOv2 特征的 FID；对 ImageNet 外领域更好。 |

## 生产注意：评估本身也是一种推理负载

在 10,000 个样本上运行 FID 意味着要生成 10,000 张图片。对于 50 步 SDXL base 在单张 L4 上以 1024² 分辨率，这需要约 11 小时的单次请求推理。评估预算确实是真实的，且框架完全符合离线推理场景（最大化吞吐量，忽略 TTFT）：

- **批次处理，忘掉延迟。** 离线评估 = 静态批次处理，使用内存能容纳的最大批次大小。在 80GB H100 上使用 `pipe(...).images` 并设置 `num_images_per_prompt=8`，比单次请求快 4-6 倍墙钟时间。
- **缓存真实特征。** 在真实参考集上运行 Inception（FID）或 CLIP（CLIP score、CMMD）特征提取只需执行*一次*，存储为 `.npz` 文件。不要每次评估都重新计算。

对于 CI / 回归门禁：在每个 PR 上运行 500 样本子集的 FID + CLIP score（约 30 分钟）；每晚运行完整的 10k FID + HPSv2 + Elo。

## 延伸阅读

- [Heusel 等人（2017）。GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium（FID）](https://arxiv.org/abs/1706.08500) — FID 论文。
- [Jayasumana 等人（2024）。Rethinking FID: Towards a Better Evaluation Metric for Image Generation（CMMD）](https://arxiv.org/abs/2401.09603) — CMMD。
- [Radford 等人（2021）。Learning Transferable Visual Models from Natural Language Supervision（CLIP）](https://arxiv.org/abs/2103.00020) — CLIP。
- [Wu 等人（2023）。HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2。
- [Xu 等人（2023）。ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward。
- [Yu 等人（2023）。Scaling Autoregressive Models for Content-Rich Text-to-Image Generation（Parti + PartiPrompts）](https://arxiv.org/abs/2206.10789) — PartiPrompts。
- [Stein 等人（2023）。Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — 失效模式综述。
