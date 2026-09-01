# 毕业设计项目 08 — 面向受监管领域的生产级 RAG 聊天机器人

> Harvey、Glean、Mendable 和 LlamaCloud 在 2026 年均采用相同的生产形态：使用 docling 或 Unstructured 配合 ColPali 进行视觉文档摄取，混合检索，用 bge-reranker-v2-gemma 重排序，以 Claude Sonnet 4.7 进行综合生成（提示缓存命中率达 60-80%），用 Llama Guard 4 和 NeMo Guardrails 进行安全守卫，用 Langfuse 和 Phoenix 进行监控，基于 200 题黄金集用 RAGAS 进行评估。在一个受监管领域（法律、临床、保险）构建该系统，毕业项目的考核标准是：通过黄金集测试、红队测试以及漂移仪表盘监控。

**类型：** 毕业设计项目
**语言：** Python（管道 + API）、TypeScript（聊天界面）
**先修课程：** 第 5 阶段（NLP）、第 7 阶段（transformers）、第 11 阶段（LLM 工程）、第 12 阶段（多模态）、第 17 阶段（基础设施）、第 18 阶段（安全）
**涉及阶段：** P5 · P7 · P11 · P12 · P17 · P18
**预计时长：** 30 小时

## 问题背景

受监管领域的 RAG（法律合同、临床试验方案、保险保单）是 2026 年交付量最大的生产形态，因为投资回报率清晰且利害关系具体。Harvey（Allen & Overy 律所）为其构建了法律专用版本。Mendable 专注于开发者文档场景。Glean 覆盖企业搜索。该模式的核心流程是：高保真摄取、混合检索并配合重排序、带引用强制与提示缓存的综合生成、多层安全守卫，以及持续的漂移监控。

真正的难点不在模型本身。难点在于：管辖权感知合规（HIPAA、GDPR、SOC2）、引用级别的审计追溯、成本控制（提示缓存在高命中率下可节省 60-90% 费用）、通过 RAGAS 忠实度指标检测幻觉，以及源文档更新后索引未能同步时发生的漂移检测。本毕业设计项目要求你在一个 200 题的黄金集上完成全套交付，并配套红队测试套件。

## 概念设计

管道分为两大部分。**摄取侧**：docling 或 Unstructured 解析结构化文档；ColPali 处理视觉密集型文档；切片后生成摘要、标签和基于角色的访问权限标记。向量存入 pgvector + pgvectorscale（50M 向量以内）或 Qdrant Cloud；稀疏 BM25 同步运行。**对话侧**：LangGraph 管理记忆与多轮对话；每次查询执行混合检索，用 bge-reranker-v2-gemma-2b 重排序，用 Claude Sonnet 4.7（提示缓存）进行综合生成，输出经过 Llama Guard 4 和 NeMo Guardrails 守卫，最终生成带引用的响应。

评估栈包含四层。**黄金集**（200 条带标注的 Q/A 及引用）用于准确性评估。**红队**（越狱攻击、PII 提取尝试、超域问题）用于安全性评估。**RAGAS** 用于逐轮自动计算忠实度 / 答案相关性 / 上下文精确度。**漂移仪表盘**（Arize Phoenix）每周监控检索质量与幻觉得分。

提示缓存是成本控制的关键杠杆。Claude 4.5+ 与 GPT-5+ 支持缓存系统提示与检索上下文。在 60-80% 命中率下，单次查询成本可降低 3-5 倍。因此管道设计需保证稳定的前缀结构（系统提示 + 重排序上下文优先缓存），以实现高缓存命中率。

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

- 摄取：Unstructured.io 或 docling 解析结构化文档；ColPali 处理视觉密集型 PDF
- 向量数据库：50M 向量以内使用 pgvector + pgvectorscale；超出则使用 Qdrant Cloud
- 稀疏检索：Tantivy BM25，支持字段权重
- 编排：LlamaIndex Workflows（摄取）+ LangGraph（对话）
- 重排序器：bge-reranker-v2-gemma-2b 自托管 或 Voyage rerank-2 托管服务
- LLM：Claude Sonnet 4.7（支持提示缓存）；备用 Llama 3.3 70B 自托管
- 评估：RAGAS 0.2 在线评估，DeepEval 用于幻觉与越狱测试套件
- 可观测性：Langfuse 自托管（含标注队列）；Arize Phoenix 用于漂移监控
- 安全护栏：Llama Guard 4 输入/输出分类器、NeMo Guardrails v0.12 策略、Presidio PII 脱敏
- 合规：切片级基于角色的访问标签；管辖权标签用于 GDPR/HIPAA 控制

```figure
canary-rollout
```

## 实施步骤

1. **摄取**。使用 Unstructured 或 docling 解析你的语料库（扎实的项目建议 1000-10000 份文档）。针对扫描件/视觉密集页面，路由至 ColPali。输出带摘要、角色标签、管辖权标签的切片。

2. **索引**。将密集嵌入（Voyage-3 或 Nomic-embed-v2）写入 pgvector + pgvectorscale。通过 Tantivy 构建 BM25 旁路索引。将角色与管辖权过滤条件作为元数据载荷。

3. **混合检索**。先按角色 + 管辖权过滤；再并行执行密集检索与 BM25；使用 reciprocal rank fusion（RRF）合并；取 top-20 送入重排序器；取 top-5 用于综合生成。

4. **带提示缓存的综合生成**。系统提示与静态策略放入缓存头；重排序上下文作为缓存扩展；用户问题作为非缓存后缀。目标稳态命中率 60-80%。

5. **安全护栏**。Llama Guard 4 作用于输入；NeMo Guardrails 规则拦截超域问题或策略禁止话题；Presidio 对输出中的意外 PII 进行脱敏；引用强制作为后置过滤器。

6. **黄金集**。由领域专家标注 200 条 Q/A 对（含答案与引用）。对代理按精确引用匹配、答案正确性、忠实度（RAGAS）进行评分。

7. **红队**。准备 50 条对抗提示：越狱（PAIR、TAP 方法）、PII 外泄尝试、超域问题、跨管辖权泄露。按通过/失败与严重程度评分。

8. **漂移仪表盘**。Arize Phoenix 每周跟踪检索质量（nDCG、引用忠实度）。下降超过 5% 触发告警。

9. **成本报告**。通过 Langfuse 统计：提示缓存命中率、每查询 token 数、各阶段 $/query 成本分解。

## 使用示例

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

## 交付要求

`outputs/skill-production-rag.md` 描述交付物：部署了合规标签的受监管领域聊天机器人，通过评分量规，并接入实时漂移监控。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | RAGAS 忠实度 + 答案相关性 | 黄金集（200 条 Q/A）在线得分 |
| 20 | 引用正确性 | 答案中具备可验证来源锚点的比例 |
| 20 | 护栏覆盖率 | Llama Guard 4 通过率 + 越狱测试套件结果 |
| 20 | 成本 / 延迟工程 | 提示缓存命中率、p95 延迟、$每查询 |
| 15 | 漂移监控仪表盘 | Phoenix 实时仪表盘，含每周检索质量趋势 |
| **100** | | |

## 练习

1. 构建另一个管辖权下的语料切片（例如在 GDPR 之外增加 HIPAA）。演示角色 + 管辖权过滤如何在 20 题跨管辖权探测中阻止信息泄露。

2. 在一周的生产流量中测量提示缓存命中率。识别哪些查询会破坏缓存前缀，并重构管道。

3. 添加带 1万 token 摘要缓冲区的多轮记忆。测量随着对话推进，忠实度是否下降。

4. 将 Claude Sonnet 4.7 替换为自托管的 Llama 3.3 70B。测量 $每查询与忠实度的变化。

5. 添加“不确定”模式：若重排序得分低于阈值，代理回答“我缺乏可信引用”而非强行作答。测量虚假置信度的降低幅度。

## 关键术语

| 术语 | 人们常说的说法 | 实际含义 |
|------|----------------|----------|
| 提示缓存（Prompt caching） | "缓存系统提示与上下文" | Claude/OpenAI 特性：命中时缓存的前缀 token 享受 60-90% 折扣 |
| RAGAS | "RAG 评估器" | 自动评估忠实度、答案相关性、上下文精确度 |
| 黄金集（Golden set） | "带标注的评估集" | 由专家标注的 200+ 条 Q/A，含引用；作为地面真值 |
| 管辖权标签（Jurisdiction tag） | "合规标签" | 附加在切片上的 GDPR/HIPAA/SOC2 范围标识；由检索过滤器强制执行 |
| 引用忠实度（Citation faithfulness） | "有据可依的回答比例" | 由可检索来源片段支撑的声明占比 |
| 漂移（Drift） | "检索质量衰减" | nDCG 或引用分数的周度变化；告警阈值为 5% |
| 红队（Red team） | "对抗性评估" | 发布前的越狱、PII 提取、超域探测 |

## 延伸阅读

- [Harvey AI](https://www.harvey.ai) — 法律生产栈参考
- [Glean enterprise search](https://www.glean.com) — 企业级 RAG 参考
- [Mendable documentation](https://mendable.ai) — 开发者文档 RAG 参考
- [LlamaCloud Parse + Index](https://docs.cloud.llamaindex.ai/llamaparse/getting_started) — 托管式摄取
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 成本杠杆参考
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — 标准 RAG 评估框架
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 漂移可观测性参考
- [Llama Guard 4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) — 2026 安全分类器
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 策略护栏框架
