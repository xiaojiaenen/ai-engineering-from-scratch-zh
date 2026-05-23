# 毕业项目 08 — 面向受监管垂直领域的生产级 RAG 聊天机器人

> Harvey、Glean、Mendable 和 LlamaCloud 在 2026 年都运行着相同的生产模式。使用 docling 或 Unstructured 和 ColPali 处理视觉内容进行摄入。混合检索。使用 bge-reranker-v2-gemma 进行重排序。使用 Claude Sonnet 4.7 进行合成，并利用提示缓存实现 60-80% 的命中率。使用 Llama Guard 4 和 NeMo Guardrails 进行防护。使用 Langfuse 和 Phoenix 进行监控。使用 RAGAS 在 200 个问题的黄金数据集上进行评估。在一个受监管领域（法律、临床、保险）构建一个系统，毕业项目的要求是通过黄金数据集、红队测试和漂移仪表盘。

**类型：** 毕业项目
**语言：** Python（管道 + API），TypeScript（聊天界面）
**先修课程：** 阶段 5（自然语言处理），阶段 7（Transformer），阶段 11（大语言模型工程），阶段 12（多模态），阶段 17（基础设施），阶段 18（安全）
**涉及阶段：** P5 · P7 · P11 · P12 · P17 · P18
**时间：** 30 小时

## 问题

受监管领域的 RAG（法律合同、临床试验方案、保险策略）是 2026 年最常见的生产模式，因为其投资回报率显而易见，且风险具体。Harvey（Allen & Overy）将其应用于法律领域。Mendable 提供了面向开发者文档的版本。Glean 涵盖企业搜索。其模式是：高保真摄入、混合检索加重排序、带引用强制执行和提示缓存的合成、多层安全防护，以及持续监控漂移。

难点不在于模型本身。而在于管辖区感知的合规性（HIPAA、GDPR、SOC2）、引用级别的可审计性、成本控制（当命中率高时，提示缓存可带来 60-90% 的折扣）、通过 RAGAS 忠实度进行幻觉检测，以及当源文档更新但索引未及时跟上时的漂移检测。本毕业项目要求你将所有这些功能部署到一个包含 200 个问题的黄金数据集上，并附带一个红队测试套件。

## 概念

管道包含两个部分。**摄入**：docling 或 Unstructured 解析结构化文档；ColPali 处理视觉丰富的文档；文本块附带摘要、标签和基于角色的访问标签。向量存入 pgvector + pgvectorscale（少于 5000 万向量）或 Qdrant Cloud；稀疏索引 BM25 与之并行运行。**对话**：LangGraph 处理记忆和多轮对话；每个查询执行混合检索，使用 bge-reranker-v2-gemma-2b 重排序，使用 Claude Sonnet 4.7（启用提示缓存）合成，输出经过 Llama Guard 4 和 NeMo Guardrails 过滤，并发出以引用为锚点的响应。

评估栈有四层。**黄金数据集**（200 个带引用的标签化问答对）用于评估正确性。**红队测试**（越狱尝试、PII 提取尝试、领域外问题）用于评估安全性。**RAGAS** 用于自动评估每轮对话的忠实度/答案相关性/上下文精度。**漂移仪表盘**（Arize Phoenix）每周监控检索质量和幻觉分数。

提示缓存是成本杠杆。Claude 4.5+ 和 GPT-5+ 支持缓存系统提示和检索到的上下文。当命中率在 60-80% 时，每查询成本降低 3-5 倍。管道设计必须确保前缀稳定（系统提示 + 重排序后的上下文优先），以实现高缓存命中率。

## 架构

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## 技术栈

- 摄入：Unstructured.io 或 docling 用于结构化文档；ColPali 用于视觉丰富的 PDF
- 向量数据库：少于 5000 万向量时使用 pgvector + pgvectorscale；否则使用 Qdrant Cloud
- 稀疏索引：带字段权重的 Tantivy BM25
- 编排：LlamaIndex Workflows（摄入）+ LangGraph（对话）
- 重排序器：bge-reranker-v2-gemma-2b 自托管或 Voyage rerank-2 托管
- 大语言模型：带提示缓存的 Claude Sonnet 4.7；备选方案为自托管的 Llama 3.3 70B
- 评估：在线版 RAGAS 0.2，用于幻觉和越狱测试套件的 DeepEval
- 可观测性：带注释队列的自托管 Langfuse；用于监控漂移的 Arize Phoenix
- 防护措施：Llama Guard 4 输入/输出分类器，NeMo Guardrails v0.12 策略，Presidio PII 清洗
- 合规性：文本块上的基于角色的访问标签；用于 GDPR/HIPAA 的管辖区标签

## 构建步骤

1. **摄入。** 使用 Unstructured 或 docling 解析你的语料库（构建严肃项目需要 1000-10000 份文档）。对于扫描件/视觉密集页面，通过 ColPali 处理。生成带有摘要、角色标签、管辖区标签的文本块。

2. **索引。** 将密集嵌入（Voyage-3 或 Nomic-embed-v2）存入 pgvector + pgvectorscale。通过 Tantivy 创建 BM25 辅助索引。将角色和管辖区过滤器作为元数据。

3. **混合检索。** 首先按角色+管辖区过滤；然后并行执行密集检索 + BM25 检索；使用倒数排名融合进行合并；取前 20 个结果发送给重排序器；取前 5 个结果用于合成。

4. **带提示缓存的合成。** 将系统提示 + 静态策略放入缓存头部；将重排序后的上下文作为缓存扩展；用户问题作为未缓存的后缀。目标是在稳态下实现 60-80% 的缓存命中率。

5. **防护措施。** 在输入端使用 Llama Guard 4；NeMo Guardrails 的规则栏阻止领域外问题或策略禁止的话题；Presidio 清除输出中意外的 PII；引用强制执行后过滤。

6. **黄金数据集。** 200 个由领域专家标注的问答对，包含（答案、引用）。根据精确引用匹配、答案正确性、忠实度（RAGAS）对智能体进行评分。

7. **红队测试。** 50 个对抗性提示：越狱尝试（PAIR, TAP）、PII 窃取尝试、领域外问题、跨管辖区泄露。按通过/失败和严重性进行评分。

8. **漂移仪表盘。** Arize Phoenix 每周跟踪检索质量（nDCG、引用忠实度）。当下降 5% 时发出警报。

9. **成本报告。** Langfuse：提示缓存命中率、每查询 token 数、按阶段分解的每查询成本。

## 使用指南

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 交付物

`outputs/skill-production-rag.md` 描述了交付物。一个带有合规标签的受监管领域聊天机器人，通过评分标准进行评估，并通过实时漂移监控进行观测。

| 权重 | 标准 | 如何测量 |
|:-:|---|---|
| 25 | RAGAS 忠实度 + 答案相关性 | 在黄金数据集（200 个问答对）上的在线得分 |
| 20 | 引用正确性 | 具有可验证源锚点的答案比例 |
| 20 | 防护措施覆盖率 | Llama Guard 4 通过率 + 越狱测试套件结果 |
| 20 | 成本/延迟工程 | 提示缓存命中率、p95 延迟、每查询成本 |
| 15 | 漂移监控仪表盘 | 带每周检索质量趋势的 Phoenix 实时仪表盘 |
| **100** | | |

## 练习

1.  在另一个管辖区（例如，在 GDPR 的基础上增加 HIPAA）构建第二部分语料库。在包含 20 个问题的跨管辖区探测中，演示角色+管辖区过滤如何防止交叉泄露。

2.  测量一周生产流量下的提示缓存命中率。识别哪些查询破坏了缓存前缀。并重新组织结构。

3.  使用 10k token 的摘要缓冲区添加多轮记忆。测量随着对话增长，忠实度是否下降。

4.  将 Claude Sonnet 4.7 替换为自托管的 Llama 3.3 70B。测量每查询成本和忠实度的差异。

5.  添加一个“不确定”模式：如果顶部重排序分数低于阈值，智能体则回答“我没有确凿的引用”，而不是直接回答。测量错误自信的减少程度。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| 提示缓存 | "缓存的系统 + 上下文" | Claude/OpenAI 功能：缓存的前缀 token 在命中时享受 60-90% 的折扣 |
| RAGAS | "RAG 评估器" | 对忠实度、答案相关性、上下文精度的自动化评分 |
| 黄金数据集 | "标签化评估" | 200+ 个带引用的专家标注问答对；即基本事实 |
| 管辖区标签 | "合规标签" | 附加到文本块的 GDPR/HIPAA/SOC2 范围；由检索过滤器强制执行 |
| 引用忠实度 | "基于事实的回答率" | 有可检索源片段支持的声明比例 |
| 漂移 | "检索质量衰减" | nDCG 或引用分数的每周变化；警报阈值为 5% |
| 红队 | "对抗性评估" | 发布前的越狱、PII 提取、领域外探测测试 |

## 扩展阅读

- [Harvey AI](https://www.harvey.ai) — 法律生产栈参考
- [Glean 企业搜索](https://www.glean.com) — 企业级 RAG 参考
- [Mendable 文档](https://mendable.ai) — 开发者文档 RAG 参考
- [LlamaCloud 解析 + 索引](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — 托管摄入服务
- [Anthropic 提示缓存](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 成本杠杆参考
- [RAGAS 0.2 文档](https://docs.ragas.io/) — 标准 RAG 评估框架
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 漂移可观测性参考
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 年安全分类器
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 策略规则框架