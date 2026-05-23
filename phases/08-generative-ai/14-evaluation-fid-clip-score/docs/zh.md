# 评估 — FID、CLIP 得分、人类偏好

> 每个生成模型排行榜都会引用 FID、CLIP 得分以及来自人类偏好竞技场的胜率。每个指标都有其失效模式，执着的研究者可以利用这些漏洞。如果你不了解这些失效模式，就无法区分真实的改进和利用漏洞的“优化”。

**类型：** 构建
**语言：** Python
**先决条件：** 阶段 8 · 01（分类体系），阶段 2 · 04（评估指标）
**时间：** 约 45 分钟

## 问题所在

生成模型依据*样本质量*和*条件遵循度*来评判。两者都没有封闭形式的度量。你的模型需要渲染 10,000 张图像；需要某种方式给它们打分；你需要信任这些分数，跨模型家族、跨分辨率、跨架构地信任。在 2014-2026 的严苛考验下，有三个指标存活下来：

- **FID (Fréchet Inception Distance)。** 在 Inception 网络的特征空间中，真实分布与生成分布之间的距离。越低越好。
- **CLIP 得分。** 生成图像的 CLIP 图像嵌入与提示词的 CLIP 文本嵌入之间的余弦相似度。越高越好。衡量对提示词的遵循程度。
- **人类偏好。** 让两个模型在相同的提示词上直接对抗，让人类（或 GPT-4 级别的模型）选择更好的一个，然后聚合为 Elo 分数。

你还会看到：IS (inception score，基本退役)、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每一个都是为了修正前一个指标的某个失效模式。

## 核心概念

![FID、CLIP 和偏好：三个轴线，不同的失效模式](../assets/evaluation.svg)

### FID — 样本质量

Heusel 等人 (2017)。步骤：

1.  对 N 张真实图像和 N 张生成图像提取 Inception-v3 特征（2048 维）。
2.  拟合一个高斯分布到每个特征池：计算均值 `μ_r, μ_g` 和协方差 `Σ_r, Σ_g`。
3.  FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`。

解读：两个多元高斯分布在特征空间中的 Fréchet 距离。值越低 = 分布越相似。

失效模式：
- **对小 N 有偏。** FID 是特征分布上的均方误差——小 N 会低估协方差，导致虚假的低 FID。务必使用 N ≥ 10,000。
- **依赖 Inception。** Inception-v3 在 ImageNet 上训练。远离 ImageNet 的领域（人脸、艺术、文本图像）得出的 FID 毫无意义。使用特定领域的特征提取器。
- **可被操纵。** 过度拟合 Inception 先验知识可以获得低 FID，但视觉质量并未改善。用下面介绍的 CMMD 来应对。

### CLIP 得分 — 提示遵循度

Radford 等人 (2021)。对于一张生成图像 + 提示词：

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

对 3 万张生成图像取平均 → 得到一个可在模型间比较的标量。

失效模式：
- **CLIP 自身的盲点。** CLIP 的组合推理能力较弱（例如“一个红色立方体在一个蓝色球体上”经常失败）。模型可以在 CLIP 得分上排名靠前，但并未真正遵循复杂的提示词。
- **短提示词偏向。** 短提示词在 CLIP 图像匹配中天然优势更大。长提示词的 CLIP 得分在机制上就更低。
- **提示词操纵。** 在提示词中加入“高画质，4k，杰作”等内容会人为推高 CLIP 得分，但并未改善图文绑定。

CMMD (Jayasumana 等人，2024) 修正了其中一些问题：使用 CLIP 特征替代 Inception，使用最大均值差异替代 Fréchet 距离。在检测细微质量差异方面表现更好。

### 人类偏好 — 事实基准

选取一组提示词。用模型 A 和模型 B 分别生成。将图像对展示给人类（或一个强大的 LLM 评判器）。将胜率聚合为 Elo 或 Bradley-Terry 分数。基准测试：

- **PartiPrompts (谷歌)**：1,600 条多样化提示词，12 个类别。
- **HPSv2**：10 万条人类标注，广泛用作自动化代理。
- **ImageReward**：13.7 万对提示词-图像偏好数据，MIT 许可。
- **PickScore**：基于 Pick-a-Pic 260 万条偏好数据训练。
- **聊天机器人竞技场风格的图像竞技场**：https://imagearena.ai/ 及其他。

失效模式：
- **评判者差异。** 非专家与专家的偏好不同。两者都要使用。
- **提示词分布。** 精心挑选的提示词可能偏向某个模型家族。务必记录清楚。
- **LLM 评判奖励黑客。** GPT-4 评判器会被华而不实但错误的输出欺骗。结合人类评判进行三角验证。

## 综合使用

一份生产级的评估报告应包含：

1.  **FID**：针对预留的真实分布，在 1-3 万样本上计算（样本质量）。
2.  **CLIP 得分 / CMMD**：针对同一样本及其提示词计算（遵循度）。
3.  **盲评胜率**：在盲评竞技场中与上一版本模型对比（总体偏好）。
4.  **失效模式分析**：随机抽取 50 个输出样本，标记已知问题（手部解剖、文字渲染、物体数量一致性）。

任何单一指标都是片面的。三个相互印证的指标 + 定性审查才构成一个可靠的结论。

## 动手构建

`code/main.py` 实现了 FID、类似 CLIP 得分以及 Elo 聚合，使用的是合成“特征向量”（我们用 4 维向量作为 Inception 特征的替代）。你会看到：

- 在小 N 和大 N 下计算 FID —— 了解其偏差。
- 作为特征池之间余弦相似度的“CLIP 得分”。
- 来自合成偏好流的 Elo 更新规则。

### 第一步：四行代码实现 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 第二步：CLIP 风格的余弦相似度

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 第三步：Elo 聚合

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 常见陷阱

- **N=1000 时的 FID。** N<10k 时的经验法则是不可靠的。报告低 N FID 的论文可能在“钻空子”。
- **跨分辨率比较 FID。** Inception 的 299×299 缩放会改变特征分布。仅在匹配的分辨率下比较。
- **只报告单一随机种子。** 至少运行 3 个种子。报告标准差。
- **通过负面提示词夸大 CLIP 得分。** 一些流程通过过度拟合提示词来提升 CLIP 得分。检查是否存在视觉上的饱和现象。
- **因提示词重叠导致的 Elo 偏差。** 如果两个模型在训练时都见过某个基准提示词，那么 Elo 值就毫无意义。使用预留的提示词集。
- **众包人工评估的偏差。** Prolific、MTurk 上的标注员更年轻、更懂技术。要混合招募艺术/设计领域的专家。

## 生产实践

2026 年的生产级评估协议：

| 支柱 | 最低要求 | 推荐做法 |
|------|----------|----------|
| 样本质量 | 与预留真实分布对比，在 1 万样本上计算 FID | + 在 5k 样本上计算 CMMD + 按类别子集计算 FID |
| 提示遵循度 | 在 3 万样本上计算 CLIP 得分 | + HPSv2 + ImageReward + VQA 风格问答 |
| 偏好评估 | 与基线模型盲测 200 对 | + 2000 对人类评估 + LLM 评判 + 聊天机器人竞技场 |
| 失效分析 | 50 个手动标记样本 | 500 个手动标记样本 + 自动化安全分类器 |

报告中包含全部四个支柱 = 有根据的结论。只包含其中任何一个 = 营销说辞。

## 交付使用

保存 `outputs/skill-eval-report.md`。该技能接受一个新模型检查点 + 基线，并输出一个完整的评估计划：样本量、指标、失效模式探查、签署标准。

## 练习

1.  **简单。** 运行 `code/main.py`。在相同的合成分布上，比较 N=100 和 N=1000 时的 FID。报告偏差幅度。
2.  **中等。** 基于合成的 CLIP 风格特征实现 CMMD（公式见 Jayasumana 等人，2024）。比较其对质量差异的敏感度与 FID 的差异。
3.  **困难。** 复现 HPSv2 的设置：从 Pick-a-Pic 子集中获取 1000 对图像-提示词，基于偏好数据微调一个小型 CLIP 风格评分器，并测量其与预留集的一致性。

## 关键术语

| 术语 | 人们常说什么 | 它实际意味着什么 |
|------|--------------|------------------|
| FID | "Fréchet Inception Distance" | 对真实 vs 生成 Inception 特征进行高斯拟合后的 Fréchet 距离。 |
| CLIP 得分 | "文本-图像相似度" | CLIP 图像嵌入和文本嵌入之间的余弦相似度。 |
| CMMD | "FID 的替代者" | 基于 CLIP 特征的 MMD；偏差更小，无需高斯假设。 |
| IS | "Inception 得分" | Exp KL(p(y|x) || p(y))；在现代模型上相关性差，已退役。 |
| HPSv2 / ImageReward / PickScore | "学习的偏好代理" | 在人类偏好上训练的小型模型；用作自动评判器。 |
| Elo | "国际象棋评分" | 基于成对胜率的 Bradley-Terry 聚合。 |
| PartiPrompts | "基准提示词集" | 1,600 条谷歌策划的提示词，涵盖 12 个类别。 |
| FD-DINO | "自监督替代方案" | 使用 DINOv2 特征的 FD；更适合非 ImageNet 领域。 |

## 生产提示：评估本身也是一种推理工作负载

在 1 万样本上运行 FID 意味着生成 1 万张图像。对于一个在单 L4 上运行的 50 步 SDXL base（1024²），大约需要 11 小时的单请求推理。评估预算是真实存在的，其场景设定恰好是离线推理场景（最大化吞吐量，忽略 TTFT）：

- **激进批处理，忽略延迟。** 离线评估 = 使用内存允许的最大尺寸进行静态批处理。`pipe(...).images` 配合 `num_images_per_prompt=8` 在 80GB H100 上运行，实际墙钟时间比单请求快 4-6 倍。
- **缓存真实特征。** 针对真实参考集提取的 Inception (FID) 或 CLIP (CLIP-score, CMMD) 特征只需运行*一次*，并存储为 `.npz`。不要在每次评估时重新计算。

用于 CI/回归门控：每个 PR 在 500 样本子集上运行 FID + CLIP 得分（约 30 分钟）；每晚运行完整的 1 万 FID + HPSv2 + Elo。

## 延伸阅读

- [Heusel 等人 (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID 论文。
- [Jayasumana 等人 (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD。
- [Radford 等人 (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP。
- [Wu 等人 (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2。
- [Xu 等人 (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward。
- [Yu 等人 (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts。
- [Stein 等人 (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — 失效模式调查。