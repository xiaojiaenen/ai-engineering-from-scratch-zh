# Capstone 02 — 基于代码库的 RAG（跨仓库语义搜索）

> 到 2026 年，每家严肃的工程组织都在运行内部代码搜索，它理解语义而非仅仅匹配字符串。Sourcegraph Amp、Cursor 的代码库问答、Augment 的企业图、Aider 的 repomap、Pinterest 的内部 MCP——都是同一套模式。摄入多个仓库，用 tree-sitter 解析，嵌入函数和类级别的代码块，混合搜索，重排序，带引用回答。这个 Capstone 要求你构建一个能处理 10 个仓库共 200 万行代码，并在每次 git push 时都能增量重新索引的系统。

**类型：** Capstone 项目
**语言：** Python（摄入），TypeScript（API + UI）
**前置条件：** 第 5 阶段（NLP 基础）、第 7 阶段（Transformer）、第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 17 阶段（基础设施）
**涉及阶段：** P5 · P7 · P11 · P13 · P17
**时间：** 30 小时

## 问题

到 2026 年，每个前沿编程代理都会配备代码库检索层，因为上下文窗口本身无法解决跨仓库的问题。Claude 的 100 万 token 上下文有帮助；但它不能消除对排序检索的需求。对原始代码块进行朴素余弦搜索会导致结果被污染，出现在生成式代码、单体仓库重复代码以及极少导入符号的长尾上。生产环境的解决方案是对 AST 感知的代码块进行混合（密集 + BM25）搜索，并配合重排序器，背后由符号引用图支撑。

你通过学习对一个真实仓库舰队进行索引来理解这一点——而不是单个教程仓库——并测量 MRR@10、引用忠实度和增量新鲜度。失败模式是基础设施层面的：一个 10 万文件的单体仓库、一次修改了一半文件的推送、一个需要跨四个仓库才能正确回答的查询。

## 概念

AST 感知的摄入管道使用 tree-sitter 解析每个文件，提取函数和类节点，在节点边界处切分代码块而非固定 token 窗口。每个代码块拥有三种表示：密集嵌入（Voyage-code-3 或 nomic-embed-code）、稀疏 BM25 术语和简短自然语言摘要。摘要增加了第三种可检索的模态——用户询问"X 如何被授权"，摘要会提到"authz"，即使代码中只有 `check_permission`。

检索是混合式的。查询同时触发密集搜索和 BM25 搜索，合并 top-k，并将并集交给交叉编码器重排序器（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。重排序后的列表连同要求按文件和行范围引用每个声明的指令，一起送入长上下文合成器（Claude Sonnet 4.7 配合 prompt caching，或 Llama 3.3 70B 自托管）。没有引用的答案会被后置过滤器拒绝。

增量新鲜度是基础设施问题。git push 触发差异计算：哪些文件变了、哪些符号变了。仅重新嵌入受影响的代码块。受影响的跨文件符号边（导入、方法调用）被重新计算。索引保持一致，无需在每个提交时重新处理 200 万行代码。

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

- 解析：tree-sitter，含 17 种语言语法（Python、TS、Rust、Go、Java、C++ 等）
- 密集嵌入：Voyage-code-3（托管）或 nomic-embed-code-v1.5（自托管），bge-code-v1 作为后备
- 稀疏索引：Tantivy（Rust），带 BM25F，字段加权区分符号名与正文
- 向量数据库：Qdrant 1.12 支持混合搜索，或 pgvector + pgvectorscale 适合 5000 万向量以下的团队
- 代码块摘要模型：Claude Haiku 4.5 或 Gemini 2.5 Flash，支持 prompt 缓存
- 重排序器：Cohere rerank-3 或 bge-reranker-v2-gemma-2b 自托管
- 编排：LlamaIndex Workflows 用于摄入，LangGraph 用于查询代理
- 合成器：Claude Sonnet 4.7（100 万上下文）支持 prompt 缓存
- 符号图：Neo4j（托管）或 kuzu（嵌入式）用于导入和调用边
- 可观测性：Langfuse 每步检索和合成均有 span 记录

```figure
ce-hybrid-retrieval
```

## 构建步骤

1. **摄入遍历器。** 每次推送钩子遍历时迭代 git 历史。收集变更文件。对每个文件，使用 tree-sitter 解析，提取函数和类节点及其完整源码跨度。输出代码块记录 `{repo, path, start_line, end_line, symbol, body}`。

2. **代码块摘要器。** 将代码块批量发送至 Haiku 4.5，对系统提示元进行缓存。提示词："用一句话总结此函数，命名其公开契约和副作用。" 将摘要与代码块一同存储。

3. **嵌入池。** 两个并行队列：密集（Voyage-code-3，批大小 128）和摘要（同一模型，但作用于摘要字符串）。将向量写入 Qdrant，附带负载 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 索引。** 字段加权的 Tantivy 索引：符号名权重 4，符号正文权重 1，摘要权重 2。支持"查找名为 X 的函数"和"查找执行 X 操作的函数"两类查询。

5. **符号图。** 对每个代码块，记录边：导入（此文件从仓库 Z 使用了符号 Y）、调用（此函数调用类 C 上的方法 M）、继承。存储在 kuzu 中。在查询时用于跨仓库边界扩展检索。

6. **查询代理。** 含三个节点的 LangGraph。`retrieve` 并行触发密集搜索和 BM25 搜索，按 (repo, path, symbol) 去重。`rerank` 对 top-50 运行交叉编码器，保留 top-10。`synth` 调用 Claude Sonnet 4.7，上下文包含重排序后的代码块，缓存系统提示，要求提供 file:line 引用。

7. **引用强制。** 解析模型输出；任何缺少 `(repo/path:start-end)` 锚点的声明都被标记为需重新询问或直接丢弃。仅返回有引用的答案给用户。

8. **增量重新索引。** 每次 webhook 时，计算符号级差异。仅重新嵌入文本发生变化的代码块。对导入发生变化的代码块重新计算符号边。指标：一个 50 文件的推送在 60 秒内完成重新索引，针对 200 万行代码的舰队。

9. **评估。** 标注 100 个跨仓库问题及金标准 file:line 答案。测量 MRR@10、nDCG@10、引用忠实度（可验证锚点的声明比例）以及 p50/p99 延迟。

## 使用示例

```
$ code-rag ask "S3 分块上传中止如何接入我们的重试预算？"
[retrieve]  12 个密集代码块 + 7 个 BM25 代码块，去重后 16 个唯一块
[rerank]    保留 top-5（cohere rerank-3）
[synth]     claude-sonnet-4.7，缓存命中率 68%，2.1 秒
answer:
  分块中止由 services/uploader/retry.go:122-148 中的 `AbortMultipartOnFail` 触发，
  其递减 config/budgets.yaml:34-51 中定义的每桶重试预算 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 交付物

技能产出 `outputs/skill-codebase-rag.md`。给定一个仓库语料库，它启动摄入管道、混合索引和查询代理，并对任何跨仓库问题返回带引用的答案。评分标准：

| 权重 | 标准 | 测量方式 |
|:-:|---|---|
| 25 | 检索质量 | 在 100 题保留集中测量 MRR@10 和 nDCG@10 |
| 20 | 引用忠实度 | 答案声明中有可验证 file:line 锚点的比例 |
| 20 | 延迟与规模 | 在索引语料库规模下，10k QPS 的 p95 查询延迟 |
| 20 | 增量索引正确性 | 从 git push 到可搜索的 50 文件提交耗时 |
| 15 | 用户体验与答案格式 | 引用的可点击性、代码片段预览、后续交互支持 |
| **100** | | |

## 练习

1. 将 Voyage-code-3 替换为自托管的 nomic-embed-code。测量 MRR@10 的变化。报告启用重排序后差距是否缩小。

2. 向语料库注入 20% 生成代码（LLM 生成的样板代码）并重新评估。观察检索污染。在负载中添加"generated"标志并对这些结果降权。

3. 在你的语料库规模下，基准测试 Qdrant 混合搜索与 pgvector + pgvectorscale。报告批大小 1 时的 p99 延迟。

4. 添加基于采样的漂移检测：每周重新运行 100 题评估。当 MRR@10 下降超过 5% 时发出告警。

5. 扩展到跨语言符号解析：一个通过 gRPC 调用 Go 服务的 Python 函数。使用符号图将它们链接起来。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| AST 感知切分 | "函数级别切分" | 在 tree-sitter 节点边界处切分代码，而非固定 token 窗口 |
| 混合搜索 | "密集 + 稀疏" | 并行运行 BM25 和向量搜索，合并 top-k，重排序 |
| 交叉编码器重排序 | "第二阶梯度" | 对每个 (查询, 候选) 对联合打分，比余弦更准确 |
| Prompt 缓存 | "缓存的系统提示" | 2026 年 Claude / OpenAI 特性，对重复前缀 token 折扣高达 90% |
| 符号图 | "代码图" | 跨文件和仓库的导入、调用、继承边 |
| 引用忠实度 | "接地答案率" | 用户通过点击锚点并阅读引用段落即可验证的声明比例 |
| 增量重新索引 | "推送至可搜索时间" | 从 git push 到变更符号可被查询的墙钟时间 |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) — 生产级跨仓库代码智能
- [Sourcegraph Cody RAG 架构](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本 Capstone 的参考深入解读
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tree-sitter 排名仓库视图
- [Augment Code 企业图](https://www.augmentcode.com) — 商业符号图 RAG
- [Qdrant 混合搜索文档](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 参考实现
- [Voyage AI 代码嵌入](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 详情
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — 交叉编码器参考
- [Pinterest MCP 内部搜索](https://medium.com/pinterest-engineering) — 内部平台参考
