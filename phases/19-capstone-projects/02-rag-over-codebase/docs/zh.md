# 综合项目 02 — 代码库检索增强生成（跨仓库语义搜索）

> 到2026年，每个严肃的工程组织都会运行一套能理解语义而非仅仅匹配字符串的内部代码搜索系统。Sourcegraph Amp、Cursor的代码库问答、Augment的企业知识图谱、Aider的仓库映射、Pinterest的内部MCP——它们形态相同。整合多个仓库，用tree-sitter解析，嵌入函数与类级别的代码块，执行混合搜索、重排序，并提供带引文的回答。本综合项目要求你构建一个能处理跨10个仓库共200万行代码，并能在每次git推送时支持增量重新索引的系统。

**类型：** 综合项目
**编程语言：** Python（数据摄取），TypeScript（API + 用户界面）
**前置要求：** 阶段5（自然语言处理基础），阶段7（Transformer），阶段11（大语言模型工程），阶段13（工具使用），阶段17（基础设施）
**涉及阶段：** P5 · P7 · P11 · P13 · P17
**预计时长：** 30小时

## 问题

到2026年，每个前沿的编程智能体都会配备一个代码库检索层，因为仅靠上下文窗口无法解决跨仓库的问题。Claude的100万token上下文有帮助，但它无法消除对排序检索的需求。对原始代码块进行朴素的余弦相似度搜索，会在生成代码、单体仓库重复代码以及很少被导入的符号的长尾分布上产生错误结果。生产级的解决方案是：结合密集检索与BM25稀疏检索的混合搜索，作用于AST感知的代码块，并配备一个重排序器，辅以符号引用图。

你必须通过索引一个真实的代码库集群——而非单个教程仓库——并测量MRR@10（前10名平均倒数排名）、引文可信度和增量更新时效性来学习这一点。失败模式源于基础设施：一个拥有10万文件的单体仓库、一次改动了一半文件的推送、以及一个需要跨越四个仓库才能正确回答的查询。

## 概念

一个AST感知的数据摄取管道使用tree-sitter解析每个文件，提取函数和类节点，并在节点边界而非固定token窗口处进行分块。每个代码块获得三种表示：一个密集嵌入向量（使用Voyage-code-3或nomic-embed-code模型），稀疏的BM25词条，以及一个简短的自然语言摘要。摘要增加了第三种可检索的模态——当用户询问“X是如何被授权的”，摘要中会提到“authz”，即使代码中只有`check_permission`。

检索是混合式的。查询会同时触发密集检索和BM25搜索，合并前k个结果，并将并集交给交叉编码器重排序器（使用Cohere rerank-3或自托管的bge-reranker-v2-gemma-2b模型）。重排序后的列表将被送入一个长上下文合成器（使用带提示缓存的Claude Sonnet 4.7，或自托管的Llama 3.3 70B），并指示其引用每个断言的文件和行范围。没有引用的回答将被后置过滤器拒绝。

增量更新时效性是基础设施问题。Git推送会触发差异分析：哪些文件变更了，哪些符号变更了。只有受影响的代码块需要重新嵌入。受影响的跨文件符号边（导入、方法调用）会被重新计算。索引保持一致性，而无需在每次提交时重新处理200万行代码。

## 架构

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 技术栈

- **解析：** 使用tree-sitter支持17种语言语法（Python, TypeScript, Rust, Go, Java, C++等）
- **密集嵌入：** Voyage-code-3（托管）或nomic-embed-code-v1.5（自托管），备选bge-code-v1
- **稀疏索引：** Tantivy（Rust）实现BM25F，对符号名称和正文设置不同字段权重
- **向量数据库：** Qdrant 1.12（支持混合搜索），或对于向量规模低于5000万的团队使用pgvector + pgvectorscale
- **代码块摘要模型：** Claude Haiku 4.5或Gemini 2.5 Flash，启用提示缓存
- **重排序器：** Cohere rerank-3或自托管的bge-reranker-v2-gemma-2b
- **编排：** LlamaIndex Workflows用于数据摄取，LangGraph用于查询智能体
- **合成器：** Claude Sonnet 4.7（100万上下文窗口），启用提示缓存
- **符号图谱：** Neo4j（托管）或kuzu（嵌入式）用于存储导入和调用边
- **可观测性：** Langfuse记录每次检索和合成步骤的span

## 构建步骤

1. **数据摄取遍历器。** 在每次推送钩子上迭代git历史。收集变更的文件。对每个文件，使用tree-sitter解析，提取函数和类节点及其完整源代码范围。生成代码块记录`{repo, path, start_line, end_line, symbol, body}`。
2. **代码块摘要器。** 将代码块批量发送给Haiku 4.5，在系统前置提示上启用提示缓存。提示词：“用一句话总结这个函数，说明其公开契约和副作用。”将摘要与代码块一起存储。
3. **嵌入池。** 两个并行队列：密集嵌入（Voyage-code-3，批处理大小128）和摘要嵌入（相同模型，但应用于摘要字符串）。将向量写入Qdrant，携带载荷`{repo, path, start_line, end_line, symbol, kind}`。
4. **BM25索引。** 字段加权的Tantivy索引：符号名称权重4，符号正文权重1，摘要权重2。支持“查找名为X的函数”和“查找执行X操作的函数”两类查询。
5. **符号图谱。** 对于每个代码块，记录边：导入（此文件使用了来自仓库Z的符号Y）、调用（此函数调用了类C上的方法M）、继承关系。存储在kuzu中。在查询时用于跨仓库边界的检索扩展。
6. **查询智能体。** LangGraph包含三个节点。`retrieve`并行执行密集和BM25检索，按（仓库，路径，符号）去重。`rerank`对前50个结果运行交叉编码器并保留前10个。`synth`调用Claude Sonnet 4.7，将重排序后的代码块放入上下文，缓存系统提示，并要求提供file:line引用。
7. **引用强制执行。** 解析模型输出；任何没有`(repo/path:start-end)`锚点的断言都会被标记要求重新回答或丢弃。仅返回带引用的回答给用户。
8. **增量重新索引。** 在每个webhook上，计算符号级别的差异。仅对文本变更的代码块重新嵌入。对导入变更的代码块重新计算符号边。衡量：在200万行代码库上，一个50文件的推送应在60秒内完成重新索引。
9. **评估。** 标记100个跨仓库问题及其标准答案（含文件和行号）。测量MRR@10、nDCG@10、引文可信度（带有可验证锚点的断言比例）以及p50/p99延迟。

## 使用示例

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 交付

交付的技能`outputs/skill-codebase-rag.md`。给定一个代码库仓库集合，它应能搭建数据摄取管道、混合索引和查询智能体，并为任何跨仓库问题返回带引用的回答。评分标准：

| 权重 | 评估标准 | 衡量方式 |
|:-:|---|---|
| 25 | 检索质量 | 在100个问题的留出集上测量MRR@10和nDCG@10 |
| 20 | 引文可信度 | 回答中带有可验证file:line锚点的断言比例 |
| 20 | 延迟与扩展性 | 在索引规模下，10k QPS时的p95查询延迟 |
| 20 | 增量索引正确性 | 从git推送到50文件提交可被搜索的耗时 |
| 15 | 用户体验与回答格式 | 引文可点击性、代码片段预览、后续提问便利性 |
| **100** | | |

## 练习

1. 将Voyage-code-3替换为自托管的nomic-embed-code。测量MRR@10的变化。报告在启用重排序后差距是否缩小。
2. 在代码库中注入20%的生成代码（LLM生成的样板代码）并重新评估。观察检索质量下降。在载荷中添加“generated”标志并降低这些结果的权重。
3. 在你的代码库规模下，对比基准测试Qdrant混合搜索与pgvector + pgvectorscale。报告批量大小为1时的p99延迟。
4. 添加基于采样的漂移检查：每周重新运行100个问题的评估。当MRR@10下降超过5%时发出警报。
5. 扩展到跨语言符号解析：一个Python函数通过gRPC调用Go服务。使用符号图谱将它们关联起来。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| AST感知分块 | “函数级别分割” | 在tree-sitter节点边界而非固定token窗口处切割代码 |
| 混合搜索 | “密集+稀疏” | 并行运行BM25和向量搜索，合并前k个结果，再重排序 |
| 交叉编码器重排序 | “第二阶段排序” | 对每个（查询，候选）对一起评分的模型，比余弦相似度更准确 |
| 提示缓存 | “缓存的系统提示” | 2026年Claude/OpenAI的特性，对重复前缀token的折扣高达90% |
| 符号图谱 | “代码图” | 跨文件和仓库的导入、调用、继承关系边 |
| 引文可信度 | “基于事实的回答率” | 用户可以通过点击锚点并阅读所引用代码段来验证的断言比例 |
| 增量重新索引 | “推送至可搜索时间” | 从git推送到变更符号可被查询的实际挂钟时间 |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) — 生产级跨仓库代码智能
- [Sourcegraph Cody RAG架构](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本综合项目的参考深度解析
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tree-sitter排名的仓库视图
- [Augment Code企业知识图谱](https://www.augmentcode.com) — 商用符号图谱RAG
- [Qdrant混合搜索文档](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 参考实现
- [Voyage AI代码嵌入](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3详情
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — 交叉编码器参考
- [Pinterest MCP内部搜索](https://medium.com/pinterest-engineering) — 内部平台参考