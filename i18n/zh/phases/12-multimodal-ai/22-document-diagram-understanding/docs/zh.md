# 文档与图表理解

> 文档不是照片。PDF、科学论文、发票或手写表单具有布局、表格、图表、脚注、页眉以及语义结构，这是纯图像理解无法捕捉的。VLM 之前的方案是一个流水线：Tesseract OCR + LayoutLMv3 + 表格提取启发式规则。VLM 浪潮用免 OCR 模型取代了它——Donut（2022）、Nougat（2023）、DocLLM（2023）——直接输出结构化标记。到 2026 年，前沿做法只是"把页面图片以 2576px 原生分辨率喂给 Claude Opus 4.7"，结构化标记输出随之免费获得。本课梳理文档 AI 的三个时代演进。

**类型：** 构建
**语言：** Python（标准库、布局感知文档解析骨架）
**前置条件：** 第 12 阶段 · 05（LLaVA），第 5 阶段（NLP）
**耗时：** 约 180 分钟

## 学习目标

- 阐述文档 AI 的三个时代：OCR 流水线、免 OCR、VLM 原生。
- 描述 LayoutLMv3 的三条输入流：文本、布局（bbox）、图像 patch，以及统一掩码。
- 对比 Donut（免 OCR，图像 → 标记）、Nougat（科学论文 → LaTeX）、DocLLM（布局感知生成式）、PaliGemma 2（VLM 原生）。
- 为新任务（发票、科学论文、手写表单、中文收据）选择合适的文档模型。

## 问题

"理解这份 PDF"听起来简单，实则颇具挑战。信息分布在：

- 文本内容（占 90% 的信号）。
- 布局（页眉、脚注、侧边栏、双栏排版）。
- 表格（行、列、合并单元格）。
- 图表与示意图。
- 手写批注。
- 字体与排版（标题与正文的区别）。

原始 OCR 会丢失文本之外的所有信息。一套关心发票的系统需要知道"总计：$1,245"来自右下角，而非脚注。

## 概念

### 第一代 — OCR 流水线（2021 年以前）

经典方案：

1. PDF → 逐页转为图片。
2. Tesseract（或商业 OCR）提取带逐词边界框的文本。
3. 布局分析器识别块（页眉、表格、段落）。
4. 表格结构识别器解析表格。
5. 领域规则 + 正则表达式提取字段。

对清晰的印刷文本有效。遇到手写、倾斜扫描件、复杂表格、非英文脚本时崩溃。每种失败模式都需要定制异常路径。

### TrOCR（2021）

TrOCR（Li 等，arXiv:2109.10282）用 Transformer 编码器-解码器替换了 Tesseract 的经典 CNN-CTC，在合成文本图像 + 真实文本图像上训练。在手写和多语言文本上显著提升。仍是流水线（检测 → TrOCR → 布局），但 OCR 步骤大幅改进。

### 第二代 — 免 OCR（2022-2023）

最早的免 OCR 模型说：跳过检测，直接把图像像素映射为结构化输出。

Donut（Kim 等，arXiv:2111.15664）：
- 编码器-解码器 Transformer，编码器为 Swin-B。
- 输出为 JSON（表单理解）、markdown（摘要），或任意任务特定模式。
- 无需 OCR、无需布局、无需检测。

Nougat（Blecher 等，arXiv:2308.13418）：
- 专门在科学论文上训练。
- 输出为 LaTeX / markdown。
- 处理公式、多栏布局、插图。
- 每个 arXiv 解析器都会调用的模型。

它们是专家，而非通才。Donut 面对科学论文会失败；Nougat 面对发票会失败。

### LayoutLMv3（2022）

另一条路线。LayoutLMv3（Huang 等，arXiv:2204.08387）保留 OCR 但加入布局理解：

- 三条输入流：OCR 文本 token、逐 token 的 2D 边界框、图像 patch。
- 跨三种模态的掩码训练目标（掩码文本、掩码 patch、掩码布局）。
- 下游任务：分类、实体抽取、表格 QA。

LayoutLMv3 是 OCR 系文档理解的巅峰。对表单和发票效果强。需要上游 OCR。在标准化文档评测上表现最佳（VLM 之前）。

### DocLLM（2023）

DocLLM（Wang 等，arXiv:2401.00908）是 LayoutLM 的生成式兄弟。基于布局 token 条件生成自由形式回答。更适合文档 QA；仍依赖 OCR 输入。

### 第三代 — VLM 原生（2024+）

2024 年的 VLM 已足够好，可以完全取代流水线。将高分辨率全页图像直接喂给 VLM，提问，得到答案。

- LLaVA-NeXT 336-tile AnyRes 适用于小文档。
- Qwen2.5-VL 动态分辨率原生支持 2048+ 像素。
- Claude Opus 4.7 支持 2576px 文档。
- PaliGemma 2（2025 年 4 月）专门针对文档 + 手写训练。

VLM 原生与 OCR 流水线的差距迅速缩小。到 2026 年，VLM 原生在以下方面胜出：

- 场景文本（手写 + 印刷，混合脚本）。
- 含合并单元格的复杂表格。
- 嵌入文本中的数学公式。
- 带文字标注的图表。

OCR 流水线仍在以下方面胜出：

- 大规模纯扫描任务，逐页延迟至关重要。
- 流水线可靠性（确定性失败 vs VLM 幻觉）。
- 受监管环境要求可审计的 OCR 输出。

### Claude 4.7 / GPT-5 前沿

在 2576 像素原生输入下，前沿 VLM 接近人类精度完成文档理解。2026 年初的基准数字：

- DocVQA：Claude 4.7 ~95.1，PaliGemma 2 ~88.4，Nougat ~77.3，流水线 LayoutLMv3 ~83。
- ChartQA：Claude 4.7 ~92.2，GPT-4V ~78。
- VisualMRC：Claude 4.7 ~94。

闭源模型的差距主要在分辨率和基础 LLM 规模。7B 量级的开源模型落后几个百分点，但正在追赶。

### 数学公式与 LaTeX 输出

科学论文需要精确的 LaTeX 公式输出。Nougat 为此训练。带有 LaTeX 目标的 VLM（Qwen2.5-VL-Math、Nougat 衍生模型）可产出可用 LaTeX。未经显式 LaTeX 训练的 VLM 只能产生可读但不精确的转录。

2026 年的科学论文流水线：先用 Nougat 处理 PDF，再对疑难页面用 VLM。

### 手写

仍是最难的子任务。混合印刷 + 手写（医生笔记、填写表单）是当前 OCR 流水线在成本上仍优于 VLM 的领域。纯手写 VLM 正在改进（Claude 4.7、PaliGemma 2）。

### 2026 配方

新项目文档 AI 选型建议：

- 大规模纯印刷发票：LayoutLMv3 + 规则，成本效益高。
- 混合文档（科学 + 手写 + 表单）：VLM 原生（PaliGemma 2 或 Qwen2.5-VL）。
- 完整 arXiv 入库：Nougat 处理公式，VLM 处理图表。
- 受监管场景：OCR 流水线 + VLM 交叉校验。

```figure
mm-doc-layout
```

## 实践

`code/main.py`：

- 一个玩具级布局感知 token 化器：给定（文本、bbox）对，生成 LayoutLMv3 风格的输入。
- 一个 Donut 风格的任务模式生成器：表单 JSON 模板。
- 对比 OCR 流水线、Donut、Nougat 和 VLM 原生每页的 token 预算。

## 交付

本课产出 `outputs/skill-document-ai-stack-picker.md`。给定一个文档 AI 项目（领域、规模、质量、监管），在 OCR 流水线、免 OCR 专家模型和 VLM 原生之间做出选择。

## 练习

1. 你的项目每天处理 1000 万张发票。哪个方案能在不损失精度的前提下最小化每页成本？

2. 为什么 LayoutLMv3 在表单 QA 上优于纯 CLIP-VLM，却在场景文本上不如？bbox 流放弃了什么？

3. Nougat 生成 LaTeX。提出一个 VLM 原生输出在 LaTeX 保真度上胜过 Nougat 的测试用例，以及 Nougat 胜出的用例。

4. 阅读 PaliGemma 2 论文（Google，2024）。相比 PaliGemma 1，哪个关键训练数据增量提升了文档准确率？

5. 设计一个监管安全的混合方案：OCR 流水线为主，VLM 为次级交叉校验。如何处理分歧？

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------|---------|
| OCR 流水线 | "Tesseract 风格" | 分阶段堆栈：检测 → OCR → 布局 → 规则；确定性但脆弱 |
| 免 OCR | "Donut 风格" | 跳过显式 OCR、直接将图像映射为输出的 Transformer；单模型 |
| 布局感知 | "LayoutLM" | 输入包含逐 token 的 bbox 坐标；跨模态统一掩码 |
| VLM 原生 | "前沿 VLM" | 将页面图像直接喂给 Claude/GPT/Qwen VLM，高分辨率；无流水线 |
| DocVQA | "文档基准" | 文档 VQA 标准；引用最多的评分 |
| 标记输出 | "LaTeX / MD" | 结构化输出格式而非自由文本；支持下游自动化 |

## 延伸阅读

- [Li 等 — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher 等 — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang 等 — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim 等 — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang 等 — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
