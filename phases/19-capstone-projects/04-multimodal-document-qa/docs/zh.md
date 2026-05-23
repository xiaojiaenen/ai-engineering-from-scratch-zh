# 毕业项目 04 —— 多模态文档问答（视觉优先 PDF、表格、图表）

> 2026 年文档问答前沿已从 OCR 转文本方式转向视觉优先的后期交互。ColPali、ColQwen2.5 和 ColQwen3-omni 将每个 PDF 页面视为图像，使用多向量后期交互进行嵌入，并让查询直接关注图像块。在财务 10-K 报告、科学论文和手写笔记上，这种模式以较大优势超越了 OCR 优先方式。请在 10k 页面上端到端构建该流程，并发布与 OCR 转文本方法的对比结果。

**类型：** 毕业项目
**语言：** Python（流程）、TypeScript（查看器 UI）
**先决条件：** 第 4 阶段（计算机视觉）、第 5 阶段（NLP）、第 7 阶段（Transformer）、第 11 阶段（LLM 工程）、第 12 阶段（多模态）、第 17 阶段（基础设施）
**应用阶段：** P4 · P5 · P7 · P11 · P12 · P17
**时间：** 30 小时

## 问题描述

企业保存着被 OCR 流程破坏的 PDF 文件：扫描的 10-K 报告包含旋转的表格、富含方程的科学论文、只有作为图像才有意义的图表、手写批注。将这些视为文本优先意味着丢失一半的信号。2026 年的解决方案是对原始页面图像进行后期交互多向量检索。ColPali（Illuin Tech）引入了该方法；ColQwen2.5-v0.2 和 ColQwen3-omni 提升了准确率。在 ViDoRe v3 上，视觉优先检索的得分显著高于 OCR 转文本方式——并且在图表、表格和手写内容上差距更大。

代价是存储和延迟。一个 ColQwen 嵌入约有 2048 个图像块向量/页面，而不是单个 1024 维向量。原始存储空间急剧膨胀。DocPruner（2026）在可忽略的准确率损失下带来 50% 的压缩。你将索引 10k 页面，测量 ViDoRe v3 的 nDCG@5，在 2 秒内提供答案，并与 OCR 转文本基线进行直接比较。

## 核心概念

后期交互意味着每个查询 token 与每个图像块 token 进行评分，并将每个查询 token 的最大评分求和。你无需单个池化向量即可获得细粒度匹配。多向量索引（Vespa、Qdrant 多向量或 AstraDB）存储每图像块的嵌入，并在检索时运行 MaxSim。

回答者是一个视觉语言模型，接收查询和前 k 个检索到的页面图像，并生成带有证据区域（边界框或页面引用）的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年的前沿选择。对于方程和科学记法，可选的文本通道会接入 OCR 回退（Nougat，dots.ocr）。

评估是一个二维矩阵。一个轴：内容类型（纯文本段落、密集表格、条形图/折线图、手写笔记、方程）。另一个轴：检索方法（视觉优先后期交互 vs OCR 转文本 vs 混合）。每个单元格获得 nDCG@5 和答案准确率。报告即为交付物。

## 架构

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## 技术栈

-   页面渲染：PyMuPDF (fitz)，180 DPI，纵向归一化
-   后期交互模型：ColQwen2.5-v0.2 或 ColQwen3-omni（Hugging Face 上的 vidore 团队）
-   索引：带多向量字段的 Vespa，或多向量 Qdrant，或带 MaxSim 的 AstraDB
-   剪枝：DocPruner 2026 策略（保留高方差图像块，在 <0.5% 准确率损失下压缩 50%）
-   OCR 回退（方程/密集表格）：dots.ocr 或 Nougat
-   VLM 回答者：自托管 Qwen3-VL-30B 或托管 Gemini 2.5 Pro；InternVL3 作为回退
-   评估：ViDoRe v3 基准，M3DocVQA 用于多页推理
-   查看器 UI：Next.js 15，带画布叠加层用于证据区域

## 构建步骤

1.  **数据摄入。** 遍历包含 10-K 报告、科学论文和扫描文档的 10k PDF 页面语料库。将每页渲染为 1536x2048 的 PNG。持久化 `{doc_id, page_num, image_path}`。

2.  **嵌入。** 对每个页面图像运行 ColQwen2.5-v0.2。输出形状约为 2048 个 128 维的图像块嵌入。应用 DocPruner 保留信号最强的一半。写入 Vespa 多向量字段或多向量 Qdrant。

3.  **查询。** 对于每个传入的查询，使用查询塔（token 级嵌入）进行嵌入。对索引运行 MaxSim：对于每个查询 token，在页面图像块嵌入上取最大点积，然后求和。返回前 k 个页面。

4.  **合成。** 使用查询和前 5 个页面图像调用 Qwen3-VL-30B。提示：“仅使用提供的页面回答。通过（doc_id, page）引用每个声明，并命名区域（图表、表格、段落）。”

5.  **证据区域。** 后处理答案以提取引用的区域。如果 VLM 输出边界框（Qwen3-VL 可以），则在查看器中将其渲染为叠加层。

6.  **OCR 回退。** 对于被识别为方程密集的页面（基于图像方差的启发式方法），运行 Nougat 或 dots.ocr，并将 OCR 文本作为额外通道与图像一起传递。

7.  **评估。** 运行 ViDoRe v3（检索 nDCG@5）和 M3DocVQA（多页问答准确率）。同时在相同语料库和相同合成器上运行 OCR 转文本流程。生成内容类型 × 方法的矩阵。

8.  **用户界面。** 首先构建 Streamlit 原型；然后构建带逐页证据区域叠加的 Next.js 15 生产查看器。

## 使用示例

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## 交付说明

`outputs/skill-doc-qa.md` 描述了交付物：一个针对特定语料库调整的视觉优先多模态文档问答系统，并在 ViDoRe v3 上与 OCR 转文本基线进行评估。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 准确率 | 与 OCR 文本基线和已发布排行榜的基准数字比较 |
| 20 | 证据区域定位 | 实际包含答案跨度的被引用区域的比例 |
| 20 | 存储与延迟工程 | DocPruner 压缩比，索引 p95，答案 p95 |
| 20 | 多页推理 | 在手工标注的 100 题多页集合上的准确率 |
| 15 | 源检查用户体验 | 查看器清晰度，叠加保真度，并排比较工具 |
| **100** | | |

## 练习

1.  在相同语料库上测量 ColQwen2.5-v0.2 与 ColQwen3-omni。哪个页面一个模型正确而另一个遗漏？向索引添加“内容类别”标签以按类型路由。

2.  激进剪枝嵌入（75%，90%）。找到压缩悬崖：即 ViDoRe nDCG@5 低于 OCR 基线的点。

3.  构建混合方案：并行运行 OCR 转文本和 ColQwen，使用 RRF 融合，用交叉编码器重排。混合方案是否优于单独使用？在哪里帮助最大？

4.  将 Qwen3-VL-30B 替换为更小的 VLM（Qwen2.5-VL-7B）。测量每美元准确率曲线。

5.  添加手写笔记支持。渲染手写语料库，用 ColQwen 嵌入，测量检索。与手写 OCR 流程进行比较。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|------------------------|
| 后期交互 | "ColPali 风格检索" | 查询 token 与页面图像块独立评分；MaxSim 聚合 |
| 多向量 | "每图像块嵌入" | 每个文档有许多向量，而不是一个池化向量 |
| MaxSim | "后期交互评分" | 对于每个查询 token，在文档向量上取最大相似度；求和 |
| DocPruner | "图像块压缩" | 2026 年剪枝，在可忽略的准确率损失下保留 50% 的图像块 |
| ViDoRe v3 | "文档检索基准" | 2026 年衡量视觉文档检索的标准 |
| 证据区域 | "被引用的边界框" | 源页面上定位答案跨度的边界框 |
| OCR 回退 | "方程通道" | 与视觉并行用于方程或表格密集页面的文本流程 |

## 延伸阅读

-   [ColPali (Illuin Tech) 仓库](https://github.com/illuin-tech/colpali) —— 后期交互文档检索参考
-   [ColPali 论文 (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) —— 基础方法论文
-   [ColQwen 系列 (Hugging Face)](https://huggingface.co/vidore) —— 生产就绪检查点
-   [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) —— 多页多模态 RAG 基线
-   [Vespa 多向量教程](https://docs.vespa.ai/en/colpali.html) —— 参考服务栈
-   [Qdrant 多向量支持](https://qdrant.tech/documentation/concepts/vectors/#multivectors) —— 备选索引
-   [AstraDB 多向量](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) —— 备选托管索引
-   [Nougat OCR](https://github.com/facebookresearch/nougat) —— 支持方程的 OCR 回退