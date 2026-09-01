# Capstone 04 — 多模态文档问答（视觉优先的 PDF、表格、图表）

> 2026 年的文档问答前沿已经从"先 OCR 后文本"转向视觉优先的延迟交互（late interaction）。ColPali、ColQwen2.5 和 ColQwen3-omni 将每个 PDF 页面视为图像，通过多向量延迟交互进行嵌入，并让查询直接关注各个 patch。在财务 10-K 文件、科学论文和手写笔记上，该模式显著优于先 OCR 的方法。在 1 万页数据上构建端到端流水线，并发布与"先 OCR 后文本"的对比结果。

**类型：** Capstone
**语言：** Python（流水线）、TypeScript（查看器 UI）
**前置要求：** Phase 4（计算机视觉）、Phase 5（NLP）、Phase 7（transformers）、Phase 11（LLM 工程）、Phase 12（多模态）、Phase 17（基础设施）
**涉及阶段：** P4 · P5 · P7 · P11 · P12 · P17
**时间：** 30 小时

## 问题

企业坐拥大量 PDF，但传统 OCR 流水线会破坏这些文档：包含旋转表格的扫描版 10-K 文件、充满公式的科学论文、必须以图像形式才能理解的图表、手写批注。将其视为纯文本处理意味着丢失一半的信号。2026 年的答案是：对原始页面图像进行延迟交互多向量检索。ColPali（Illuin Tech）率先提出该方法；ColQwen2.5-v0.2 和 ColQwen3-omni 进一步提升了准确率。在 ViDoRe v3 上，视觉优先检索的得分显著高于"先 OCR 后文本"——且在图表、表格和手写字迹上的差距更大。

代价是存储和延迟。一个 ColQwen 嵌入每页约 2048 个 patch 向量，而非单一 1024 维向量。原始存储量急剧膨胀。DocPruner（2026）可在无显著准确率损失的情况下实现 50% 的剪枝压缩。你将索引 1 万页，测量 ViDoRe v3 nDCG@5，在 2 秒内返回答案，并与"先 OCR 后文本"的基线直接对比。

## 概念

延迟交互（late interaction）意味着每个查询 token 对每个 patch token 单独打分，然后将每个查询 token 的最大得分求和。你获得了细粒度的匹配能力，无需依赖单一聚合向量。多向量索引（Vespa、Qdrant 多向量或 AstraDB）存储每个 patch 的嵌入，并在检索时执行 MaxSim。

回答器是一个视觉语言模型（VLM），它将查询与 top-k 检索到的页面图像一起输入，生成带有证据区域（边界框或页面引用）的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年的前沿选择。针对公式和科学符号，OCR 回退（Nougat、dots.ocr）作为可选文本通道插入。

评估采用二维矩阵。一个维度是内容类型（纯文本段落、密集表格、柱状/折线图表、手写笔记、公式）；另一个维度是检索方式（视觉优先延迟交互 vs 先 OCR 后文本 vs 混合）。每个单元格记录 nDCG@5 和答案准确率。报告是交付物。

## 架构

```
PDFs -> 页面渲染器 (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 嵌入（每页多向量，约 2048 个 patch）
           |
           +------> DocPruner 50% 压缩
           |
           v
   多向量索引（Vespa 或 Qdrant 多向量）
           |
query ----+----> 检索 top-k 页面（MaxSim）
           |
           v
  VLM 回答器：Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    输入：查询 + top-k 页面图像 + 可选 OCR 文本
           |
           v
  带引用页码和证据区域的答案
           |
           v
  Streamlit / Next.js 查看器：源页面高亮显示边界框
```

## 技术栈

- 页面渲染：PyMuPDF（fitz），180 DPI，纵向标准化
- 延迟交互模型：ColQwen2.5-v0.2 或 ColQwen3-omni（Hugging Face 上 vidore 团队发布）
- 索引：Vespa（多向量字段）或 Qdrant 多向量或带 MaxSim 的 AstraDB
- 剪枝：DocPruner 2026 策略（保留高方差 patch，50% 压缩率，< 0.5% 准确率损失）
- OCR 回退（公式 / 密集表格）：dots.ocr 或 Nougat
- VLM 回答器：自托管 Qwen3-VL-30B 或云端 Gemini 2.5 Pro；InternVL3 作为备用
- 评估：ViDoRe v3 基准、M3DocVQA 多页推理
- 查看器 UI：Next.js 15，带画布叠加层显示证据区域

```figure
ce-late-interaction
```

## 构建步骤

1. **入库。** 遍历由 10-K 文件、科学论文和扫描文档组成的 1 万页 PDF 语料库。将每页渲染为 1536×2048 的 PNG。持久化存储 `{doc_id, page_num, image_path}`。

2. **嵌入。** 对每页图像运行 ColQwen2.5-v0.2。输出形状约为 2048 个 patch 嵌入，维度 128。应用 DocPruner 保留信号最高的半数。写入 Vespa 多向量字段或 Qdrant 多向量。

3. **查询。** 对每个传入查询，用查询塔（token 级嵌入）进行编码。与索引运行 MaxSim：对每个查询 token，取与页面 patch 嵌入的最大点积，然后求和。返回 top-k 页面。

4. **综合。** 用 Qwen3-VL-30B 调用查询和 top-5 页面图像。提示词："仅使用提供的页面回答问题。用 (doc_id, page) 引用每一项声明，并注明区域（图表、表格、段落）。"

5. **证据区域。** 后处理答案以提取被引用的区域。如果 VLM 输出了边界框（Qwen3-VL 支持），则在查看器中将其渲染为叠加层。

6. **OCR 回退。** 对识别为公式密集（基于图像方差的启发式方法）的页面，运行 Nougat 或 dots.ocr，并将 OCR 文本作为额外通道与图像一并传入。

7. **评估。** 运行 ViDoRe v3（检索 nDCG@5）和 M3DocVQA（多页 QA 准确率）。在同一语料库上对"先 OCR 后文本"流水线运行相同综合器。生成"内容类型 × 检索方式"矩阵。

8. **UI。** 先用 Streamlit 快速原型；再用 Next.js 15 生产级查看器，逐页叠加证据区域。

## 使用方式

```
$ doc-qa ask "EMEA 业务部门 2024 年运营利润率变化是多少？"
[retrieve]   top-5 页面，耗时 320ms（ColQwen2.5, MaxSim, Vespa）
[synth]      qwen3-vl-30b，1.4s，引用 (form-10k-2024, p. 88) + (..., p. 92)
答案：
  EMEA 运营利润率从 18.2% 降至 16.8%，下降 140bp。
  引用来源：10-K-2024.pdf p.88（表 4，分部门运营利润率）
            10-K-2024.pdf p.92（管理层讨论与分析，运营表现）
[viewer]     打开查看器，p.88 表 4 上叠加高亮边界框
```

## 交付物

`outputs/skill-doc-qa.md` 描述交付物：一个视觉优先的多模态文档问答系统，针对特定语料库进行调优，并在 ViDoRe v3 上与"先 OCR 后文本"基线进行对比评估。

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 准确率 | 基准数字与 OCR-文本基线及已发布排行榜对比 |
| 20 | 证据区域定位准确度 | 实际包含答案片段的引用区域占比 |
| 20 | 存储与延迟工程 | DocPruner 压缩比、索引 p95、回答 p95 |
| 20 | 多页推理 | 在手标 100 题多页集上的准确率 |
| 15 | 源码检查 UX | 查看器清晰度、叠加层保真度、并排对比工具 |
| **100** | | |

## 练习

1. 在同一语料库上对比 ColQwen2.5-v0.2 与 ColQwen3-omni。哪些页面一种方法答对而另一种遗漏？为索引添加"内容类别"标签以实现按类型路由。

2. 激进剪枝嵌入（75%、90%）。找到压缩拐点：ViDoRe nDCG@5 降至 OCR 基线以下的临界点。

3. 构建混合方案：并行运行"先 OCR 后文本"和 ColQwen，用 RRF 融合，再用 cross-encoder 重排序。混合方案是否优于各自单独运行？在哪些方面收益最大？

4. 将 Qwen3-VL-30B 替换为更小的 VLM（Qwen2.5-VL-7B）。测量准确率-成本曲线。

5. 增加手写笔记支持。渲染手写字迹语料库，用 ColQwen 嵌入，测量检索效果。与手写 OCR 流水线进行对比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 延迟交互（Late interaction） | "ColPali 式检索" | 查询 token 独立地对页面 patch 打分；MaxSim 进行聚合 |
| 多向量（Multi-vector） | "每 patch 嵌入" | 每个文档包含多个向量，而非一个聚合向量 |
| MaxSim | "延迟交互打分" | 对每个查询 token，在文档向量上取最大相似度；然后求和 |
| DocPruner | "Patch 压缩" | 2026 年的剪枝方法，保留 50% 的 patch，准确率损失可忽略 |
| ViDoRe v3 | "文档检索基准" | 2026 年衡量视觉文档检索的标准基准 |
| 证据区域（Evidence region） | "引用边界框" | 源页面上的 bbox，用于定位答案片段 |
| OCR 回退（OCR fallback） | "公式通道" | 在视觉之外并行使用的文本流水线，用于公式或表格密集的页面 |

## 延伸阅读

- [ColPali (Illuin Tech) 仓库](https://github.com/illuin-tech/colpali) — 参考性延迟交互文档检索实现
- [ColPali 论文 (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — 基础方法论文
- [Hugging Face 上的 ColQwen 系列](https://huggingface.co/vidore) — 可直接投入生产的 checkpoint
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) — 多页多模态 RAG 基线
- [Vespa 多向量教程](https://docs.vespa.ai/en/colpali.html) — 参考服务栈
- [Qdrant 多向量支持](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — 替代索引方案
- [AstraDB 多向量](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — 托管式替代索引
- [Nougat OCR](https://github.com/facebookresearch/nougat) — 支持公式的 OCR 回退方案
