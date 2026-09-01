# ColPali 与视觉原生文档 RAG

> 传统 RAG 将 PDF 解析为文本，切分为 chunk，嵌入 chunk，存储向量。每一步都在损失信号：OCR 丢弃图表数据，切块破坏表格行，文本嵌入忽略图像。ColPali（Faysse 等人，2024年7月）提出了一个更简单的问题：为何要提取文本？直接通过 PaliGemma 嵌入页面图像，使用 ColBERT 风格的晚期交互进行检索，保留文档携带的所有布局、图像、字体和格式信号。已发表的基准测试显示：在视觉上丰富的文档上，端到端准确率比文本 RAG 高 20-40%。ColQwen2、ColSmol 和 VisRAG 扩展了这一模式。本课研读视觉原生 RAG 论文，并构建一个小型的 ColPali 风格索引器。

**类型：** 构建实践
**语言：** Python（标准库、多向量索引器 + MaxSim 评分器）
**前置知识：** Phase 11（LLM 工程 — RAG 基础）、Phase 12 · 05（LLaVA）
**时间：** 约 180 分钟

## 学习目标

- 解释双编码器检索（每个文档一个向量）与晚期交互检索（每个文档多个向量）之间的区别。
- 描述 ColBERT 的 MaxSim 操作，以及 ColPali 如何将其从文本 token 推广到图像 patch。
- 构建一个小型的 ColPali 风格索引器：页面 → patch 嵌入 → 对查询词嵌入执行 MaxSim → top-k 页面。
- 在发票/财务报告用例中，比较 ColPali + Qwen2.5-VL 生成器与文本 RAG + GPT-4 的表现。

## 问题所在

PDF 上的文本 RAG 丢弃了文档的大部分信息。财务报告的第三季度营收增长通常在图表中；医疗报告的发现内容位于带标注的图像中；法律合同中的签名区块是布局事实而非文本事实。

文本 RAG 管道：

1. PDF → 通过 OCR / pdftotext 转为文本。
2. 文本 → 切分为 300-500 token 的 chunk。
3. Chunk → 双编码器嵌入（一个向量）。
4. 用户查询 → 嵌入 → 余弦相似度 → top-k chunks。
5. Chunks + 查询 → LLM。

五个有损步骤。图表未捕获。表格被切块打碎。多列布局被压平。图像注解消失。

ColPali 的解决方案：跳过 OCR，直接嵌入页面图像。使用 ColBERT 风格的晚期交互进行检索，使模型在查询时能够关注细粒度的 patch。

## 概念解析

### ColBERT（2020）

ColBERT（Khattab & Zaharia，arXiv:2004.12832）是一种文本检索方法。它不对每个文档生成单一向量，而是为每个 token 生成一个向量。在查询时：

- 查询 token 获得各自的嵌入（N_q 个向量）。
- 文档 token 获得嵌入（N_d 个向量，通常已缓存）。
- 分数 = 查询 token 上的求和，每个 token 取文档 token 的最大余弦相似度：Σ_i max_j cos(q_i, d_j)。

这就是 MaxSim 操作。每个查询 token "挑选"其最佳匹配的文档 token。最终分数为求和结果。

优点：召回率高，能处理词级语义。缺点：每个文档 N_d 个向量，存储开销大。

### ColPali

ColPali（Faysse 等人，arXiv:2407.01449）将 ColBERT 模式应用于图像。

- 每个页面由 PaliGemma（ViT + 语言模型）编码为 patch 嵌入：每页 N_p 个向量。
- 每个用户查询（文本）被编码为查询 token 嵌入：N_q 个向量。
- 分数 = Σ_i max_j cos(q_i, p_j)，即查询文本 token 与页面图像 patch 之间的 MaxSim。
- 按总分检索 top-k 页面。

在文档摄入阶段：用 PaliGemma 编码每一页，存储所有 patch 嵌入。在查询阶段：嵌入查询 token，与所有已索引页面的嵌入计算 MaxSim，返回 top-k 页面。

优点：在视觉上丰富的文档上，端到端性能比文本 RAG 提升 20-40%。每个 patch 向量捕获局部布局和内容的细节。

缺点：每页 N_p 个 patch × 4 字节浮点数 × D 维向量 = 存储增长迅速。可通过 PQ / OPQ 量化缓解。

### ColQwen2 与 ColSmol

ColQwen2（illuin-tech，2024-2025）用 Qwen2-VL 替换 PaliGemma 作为基础编码器。更好的基础编码器带来更好的检索效果。

ColSmol 是面向本地/边缘场景的小型化变体。约 1B 参数的 ColSmol 检索器可在消费级 GPU 上运行。

### VisRAG

VisRAG（Yu 等人，arXiv:2410.10594）是另一种变体：不在 patch 上做 MaxSim，而是先用 VLM 将每页汇聚为单个向量，再进行双编码器检索。索引更快、存储更小，但召回率较弱。

质量与成本的权衡：ColPali 追求质量，VisRAG 追求规模。

### M3DocRAG

M3DocRAG（Cho 等人，arXiv:2411.04952）将多模态检索扩展到多页面、多文档推理。跨文档检索页面，为 VLM 组合多页面上下文。

### ViDoRe — 基准测试

ColPali 的配套基准测试：Visual Document Retrieval Evaluation（视觉文档检索评估）。任务涵盖财务报告、科学论文、行政文档、医疗记录、手册等。评估指标：nDCG@5。

ColPali-v1 在 ViDoRe 上得分约 80% nDCG@5；同一批文档上的文本 RAG 得分约 50-60%。

### 端到端 RAG 管道

对于视觉原生 RAG：

1. 摄入：PDF → 页面图像 → PaliGemma 编码 → 存储所有 patch 嵌入。
2. 查询：用户文本 → 查询 token 嵌入 → 对所有已索引页面执行 MaxSim → top-k 页面。
3. 生成：top-k 页面图像 + 查询 → VLM（Qwen2.5-VL 或 Claude）→ 生成答案。

全程无需 OCR。图像、图表、字体、布局全部流入最终答案。

### 存储估算

一个 50 页的财务报告，每页 729 个 patch，128 维嵌入：

- ColPali：50 * 729 * 128 * 4 字节 ≈ 18 MB 原始大小，PQ 压缩后约 4 MB。
- 文本 RAG：50 chunks * 768 维 * 4 字节 ≈ 150 kB。

ColPali 每文档的存储约为文本 RAG 的 30 倍。在大规模场景下，OPQ/PQ 可将其降至约 5-10 倍，通常可接受。

### 文本 RAG 仍占优势的场景

- 纯文本文档，无布局信号（如维基百科文章、聊天日志）。文本 RAG 更简单且存储成本更低。
- 数百万页的档案库，存储成本占主导。
- 严格的合规要求需要在检索旁保留可提取的 OCR 文本。

对于 2026 年的其他所有场景——财务报告、科学论文、法律合同、医疗记录、UX 文档——视觉原生 RAG 胜出。

```figure
mm-maxsim
```

## 动手实践

`code/main.py`：

- 小型 patch 编码器：将"页面"（特征向量小网格）映射为 patch 嵌入数组。
- MaxSim 评分器：计算查询 token 嵌入集与页面 patch 集之间的 ColBERT 风格分数。
- 索引 5 个玩具页面，执行 3 个查询，返回带分数的 top-k 结果。

## 交付成果

本课产出 `outputs/skill-vision-rag-designer.md`。给定一个文档 RAG 项目，能够选择 ColPali / ColQwen2 / VisRAG / 文本 RAG 并评估存储规模。

## 练习

1. 一个 200 页的年度报告，每页 729 个 patch，128 维嵌入，4 字节浮点数。计算原始存储和 PQ 压缩（8 倍）存储。

2. MaxSim 定义为 Σ_i max_j cos(q_i, p_j)。这个求和捕捉了简单均值相似度无法捕捉的什么信息？

3. ColPali 以 patch 集形式索引页面。如果我们改为以词级别索引（如 ColBERT 所做的那样），会有什么变化？权衡如何？

4. 为一个 1M 页语料库设计端到端管道，查询延迟预算为 500ms。选择 ColQwen2 或 VisRAG 并给出理由。

5. 阅读 M3DocRAG（arXiv:2411.04952）。描述多页面注意力模式，以及它与单页面 ColPali 检索的区别。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|------------|----------|
| Late interaction（晚期交互） | "ColBERT-style" | 使用逐 token 或逐 patch 嵌入 + MaxSim 进行检索，而非单一文档向量 |
| MaxSim | "Max-over-patches" | 对每个查询 token，选取相似度最高的文档 token； across 查询求和 |
| Bi-encoder（双编码器） | "Single-vector" | 每个文档一个向量；速度快但粒度损失 |
| Multi-vector（多向量） | "Many-vectors-per-doc" | 每个文档/页面存储 N_p 个向量；存储成本增长但召回率提升 |
| Patch embedding（patch 嵌入） | "Page feature" | VLM 编码器产生的每个图像 patch 对应一个向量，按页面缓存 |
| ViDoRe | "Vision doc bench" | ColPali 的视觉文档检索基准测试套件 |
| PQ quantization（乘积量化） | "Product quantization" | 在缩小存储约 8 倍的同时保持向量相似度的压缩方法 |

## 延伸阅读

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
