# 水印技术 — SynthID、Stable Signature、C2PA

> 三种技术构成了 2026 年 AI 生成内容的溯源框架。SynthID（Google DeepMind）— 图像水印于 2023 年 8 月发布，文本+视频水印于 2024 年 5 月发布（Gemini + Veo），文本水印于 2024 年 10 月通过 Responsible GenAI Toolkit 开源，多模态统一检测器于 2025 年 11 月随 Gemini 3 Pro 一同发布。文本水印通过微调下一个 token 的采样概率实现（人眼不可察觉）；图像/视频水印可抵抗压缩、裁剪、滤镜、帧率变化等攻击。Stable Signature（Fernandez 等，ICCV 2023，arXiv:2303.15435）— 对潜在扩散解码器进行微调，使每个输出包含嵌入的固定消息；即使裁剪至原始内容的 10%，生成图像的检测率仍 >90%（FPR<1e-6）。后续研究"Stable Signature is Unstable"（arXiv:2405.07145，2024 年 5 月）表明，微调可去除水印同时保持图像质量。C2PA — 基于密码学的防篡改元数据标准（C2PA 2.2 Explainer 2025）。水印与 C2PA 互为补充：元数据可能被剥离但承载更丰富的溯源信息；水印在转码后仍可存续但信息容量有限。

**类型：** 实践构建
**语言：** Python（标准库，token 水印嵌入与检测）
**前置知识：** Phase 10 · 04（采样）、Phase 01 · 09（信息论）
**预计时长：** ~75 分钟

## 学习目标

- 描述 token 级水印机制（SynthID-text 风格）及其可检测性原理。
- 描述 Stable Signature 及 2024 年破坏该方案的攻击方法。
- 说明 C2PA 的作用及其为何与水印技术互补。
- 描述关键技术局限性：模型专属信号、改写鲁棒性不足、以及保义攻击（arXiv:2508.20228）。

## 问题背景

2023-2024 年，深度伪造和 AI 生成内容大规模进入政治与消费领域。水印技术是提出的技术方案之一：在生成时嵌入溯源信号，在后续检测时识别。2025 年的证据表明：没有任何水印方案是无条件鲁棒的，但与 C2PA 元数据结合后，可提供可用的溯源链条。

## 核心概念

### 文本水印（SynthID-text 风格）

Kirchenbauer 等人在 2023 年提出的机制，由 Google 工程化落地：

1. 在每个解码步骤，对前 K 个 token 进行哈希，将词表伪随机划分为"绿色"和"红色"两个集合。
2. 通过向绿色集合的 logit 值添加 δ 来偏向采样。
3. 生成结果中绿色 token 的比例高于随机期望。

检测过程：对每个前缀重新哈希，统计生成文本中的绿色 token 数量，计算 z-score。水纹文本的 z-score >0，人类文本的 z-score ≈0。

特性：
- 对读者不可察觉（δ 足够小，质量损失轻微）。
- 已知词表划分函数时可检测。
- 不抵抗改写——重写文本会破坏信号。

SynthID-text 于 2024 年 10 月通过 Google Responsible GenAI Toolkit 开源。

### Stable Signature（图像水印）

Fernandez 等，ICCV 2023。通过对潜在扩散解码器进行微调，使每张生成图像在潜在表示中包含嵌入的固定二进制消息。检测时通过神经解码器从潜在空间中解码。即使裁剪至原始内容的 10%，检测率仍 >90%（FPR<1e-6）。

2024 年 5 月"Stable Signature is Unstable"（arXiv:2405.07145）：对解码器进行微调可去除水印同时保持图像质量。对抗性后生成微调成本较低，水印的对抗鲁棒性有限。

### SynthID 统一检测器（2025 年 11 月）

随 Gemini 3 Pro 发布：一个多模态检测器，通过单一 API 读取文本、图像、音频和视频中的 SynthID 信号。统一了 Google 的溯源技术栈。

### C2PA

内容溯源与真实性联盟（Coalition for Content Provenance and Authenticity）。基于密码学的防篡改元数据标准。C2PA 2.2 Explainer（2025）。C2PA 清单记录溯源声明（谁创建、何时创建、经过了什么变换），并由创建者的密钥签名。

与水印互补：
- 元数据可能被剥离；水印难以（轻易）去除。
- 元数据信息丰富（完整溯源链）；水印仅携带比特信息。
- C2PA 依赖平台采用；水印自动嵌入。

Google 已在 Search、Ads 和"关于此图像"功能中同时集成两者。

### 局限性

- **模型专属。** SynthID 水印仅标记启用 SynthID 的模型生成的内容。来自未启用 SynthID 模型的生成结果不会被水印标记，因此"无 SynthID 信号"不等于"非 AI 生成"。
- **改写脆弱。** 文本水印无法抵抗保义改写。
- **变换攻击。** arXiv:2508.20228（2025）展示了可同时破坏文本水印和许多图像水印的保义攻击方法。
- **微调去除。** 如"Stable Signature is Unstable"所示，生成后的微调可去除嵌入的水印。

### 欧盟 AI 法案第 50 条

AI 生成内容标注的透明度代码（初稿 2025 年 12 月，第二稿 2026 年 3 月，预计终稿 2026 年 6 月，见 [欧洲委员会状态页面](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)）。截至 2026 年 4 月，该代码仍处于草案阶段，时间线可能调整。监管层要求技术层实现。深度伪造必须被标注。

### 本课在 Phase 18 中的位置

第 22-23 课涉及模型输出的内容（私有数据、溯源信号）。第 27 课涵盖训练数据治理。第 24 课是要求这些技术措施的监管框架。

```figure
an-watermark-greenlist
```

## 实践使用

`code/main.py` 构建了一个简化版文本水印示例。token 为整数 0..N-1；水印采样偏向哈希定义的绿色集合。检测器计算绿色 token 的 z-score。你可以观察到 1000 token 生成的检测结果，见证改写如何破坏信号，并测量人类文本的误报率。

## 实战产出

本课产出 `outputs/skill-provenance-audit.md`。给定包含溯源声明的内容部署，你将审计：水印机制（如有）、C2PA 签名链（如有）、各自的对抗鲁棒性，以及各模态的覆盖情况。

## 练习

1. 运行 `code/main.py`。报告 1000 token 水印生成与人类撰写文本的 z-score。在 95% 置信度阈值下，识别误报率。

2. 实现一个改写攻击，用同义词替换 30% 的 token。重新测量 z-score。

3. 阅读 Kirchenbauer 等 2023 年第 6 节的鲁棒性分析。为什么文本水印在改写下失效而图像水印可抵抗裁剪？

4. 设计一个同时使用 SynthID-text + C2PA 元数据的部署方案。描述消费者看到的溯源链条。指出每个组件的一个失效模式。

5. 2024 年"Stable Signature is Unstable"结果表明微调可去除图像水印。设计一个部署控制措施来限制此类攻击——例如，要求微调检查点的签名发布。

## 关键术语

| 术语 | 人们通常的说法 | 实际含义 |
|------|----------------|----------|
| SynthID | "Google 的水印" | 跨模态溯源信号；文本、图像、音频、视频 |
| Token 水印 | "Kirchenbauer 风格" | 偏向采样的文本水印，通过绿色 token z-score 检测 |
| Stable Signature | "图像水印" | 微调解码器水印；ICCV 2023 |
| C2PA | "元数据标准" | 基于密码学的防篡改溯源元数据 |
| 改写鲁棒性 | "改写会破坏它吗" | 文本水印特性；目前能力有限 |
| 微调去除 | "对抗性去水印" | 通过对解码器微调去除图像水印的攻击 |
| 跨模态检测器 | "统一 SynthID" | 2025 年 11 月发布的跨模态统一 API |

## 延伸阅读

- [Kirchenbauer 等 — A Watermark for Large Language Models（ICML 2023，arXiv:2301.10226）](https://arxiv.org/abs/2301.10226) — token 水印机制
- [Fernandez 等 — Stable Signature（ICCV 2023，arXiv:2303.15435）](https://arxiv.org/abs/2303.15435) — 图像水印论文
- [《Stable Signature is Unstable》（arXiv:2405.07145）](https://arxiv.org/abs/2405.07145) — 水印去除攻击
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/) — 跨模态水印
- [C2PA 2.2 Explainer（2025）](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html) — 元数据标准
