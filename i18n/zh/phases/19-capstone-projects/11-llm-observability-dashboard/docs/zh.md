# Capstone 11 — LLM 可观测性与评估仪表板

> Langfuse 转为开放核心模式。Arize Phoenix 发布了 2026 GenAI 语义约定映射。Helicone 和 Braintrust 都在按用户成本分摊上加大投入。Traceloop 的 OpenLLMetry 成为事实上的 SDK 自动插桩方案。生产形态为：ClickHouse 存 traces、Postgres 存元数据、Next.js 做 UI，以及一批评估作业（DeepEval、RAGAS、LLM-judge）在采样的 traces 上运行。构建一套自托管方案，从至少四个 SDK 系列接入数据，并展示在五分钟内检测到注入回归的能力。

**类型：** Capstone
**语言：** TypeScript（UI）、Python / TypeScript（接入 + 评估）、SQL（ClickHouse）
**前置要求：** Phase 11（LLM 工程）、Phase 13（工具）、Phase 17（基础设施）、Phase 18（安全）
**涉及阶段：** P11 · P13 · P17 · P18
**时间：** 25 小时

## 问题

2026 年每个在生产流量上跑 LLM 的 AI 团队都会在模型之外维护一个可观测层：成本分摊、幻觉检测、漂移监控、越狱信号、SLO 仪表盘、PII 泄漏告警。开源参考 —— Langfuse、Phoenix、OpenLLMetry —— 都收敛到 OpenTelemetry GenAI 语义约定作为采集 schema。你现在可以用一个 SDK 为 OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM 插桩并生成兼容的 spans。

你需要构建一个自托管仪表板，能从至少四个 SDK 系列接入数据、在采样的 traces 上运行一批评估作业、检测漂移并告警。衡量标准是：给定一个刻意注入的回归（某个提示开始产生 PII），仪表板在五分钟内抓到它并发出告警。

## 概念

采集使用 OTLP HTTP。SDK 产出符合 GenAI 语义约定的 spans：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions`。Spans 落入 ClickHouse 做列式分析；元数据（用户、会话、应用）落入 Postgres。

评估作为批处理作业在采样的 traces 上运行。DeepEval 打分忠实度、毒性和答案相关性。当 trace 携带检索上下文时 RAGAS 打分检索指标。自定义 LLM-judge 执行领域特定检查（PII 泄漏、偏离策略的响应）。评估结果写回到同一个 ClickHouse，以 eval spans 的形式链接到父 trace。

漂移检测通过 embedding 空间的分布随时间变化（prompt embeddings 的 PSI 或 KL 散度）加上评估分数趋势来工作。告警接入 Prometheus Alertmanager，然后推到 Slack / PagerDuty。UI 由 Next.js 15 + Recharts 构建。

## 架构

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector（采集、采样、扇出）
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   （spans）       （metadata）  （raw events）
       |
       +---> eval jobs（DeepEval、RAGAS、LLM-judge）
       |     采样或全量 trace
       |     将 eval spans 写回
       |
       +---> drift detector（PSI / KL on prompt embeddings）
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard（Recharts）
```

## 技术栈

- 采集：OpenTelemetry SDKs + GenAI 语义约定；OTLP HTTP 传输
- Collector：带 tail-sampling 处理器的 OpenTelemetry Collector（用于成本控制）
- 存储：ClickHouse 存 spans、Postgres 存元数据、S3 存原始事件归档
- 评估：DeepEval、RAGAS 0.2、Arize Phoenix evaluator pack、自定义 LLM-judge
- 漂移检测：每周对聚合的 prompt embeddings（sentence-transformers）计算 PSI / KL
- 告警：Prometheus Alertmanager → Slack / PagerDuty
- UI：Next.js 15 App Router + Recharts + server actions
- 开箱支持的 SDK：OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM

```figure
ce-otel-drift
```

## 构建步骤

1. **Collector 配置。** OpenTelemetry Collector 配备 OTLP HTTP receiver、一个 tail-sampler（保留 100% 出错 trace 和 10% 成功 trace），以及向 ClickHouse 和 S3 的 exporters。

2. **ClickHouse schema。** `spans` 表列镜像 GenAI 语义约定：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，以及用于长 payload 的 JSON bag。为 `user_id` 和 `app_id` 建立二级索引。

3. **SDK 覆盖测试。** 编写小型客户端应用，分别使用每个 SDK（OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM）配合 OpenLLMetry 自动插桩。验证每个都能产出规范的 GenAI spans 并落入 ClickHouse。

4. **评估作业。** 定时作业读取最近 15 分钟采样 trace 并运行 DeepEval 的忠实度、毒性和答案相关性评估。输出为链接到父 trace 的 eval spans。

5. **自定义 LLM-judge。** PII 泄漏 judge：给定响应，调用 guard LLM 打分 PII 泄漏概率。高分响应进入人工分类队列。

6. **漂移检测。** 每周作业计算本周聚合 prompt embeddings 与过去 4 周基线之间的 PSI。若 PSI 超过阈值则告警。

7. **仪表板。** Next.js 15，页面包括：概览（spans/sec、cost/user、p95 latency）、traces（搜索 + 瀑布图）、evals（忠实度趋势、毒性）、drift（PSI 随时间）、alerts。

8. **告警链路。** Prometheus exporter 读取 eval 分数聚合和延迟百分位；Alertmanager 将警告路由到 Slack，严重告警路由到 PagerDuty。

9. **回归探测。** 注入一个 bug：被评估的聊天机器人 1% 的概率开始泄漏虚假 SSN。测量 MTTR：从 bug 部署到 Slack 告警的时间。

## 使用方式

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 交付物

`outputs/skill-llm-observability.md` 是交付物。给定一个 LLM 应用，仪表板应能采集其 traces、运行评估、在漂移时告警，并在 Next.js 中展示 cost/user 明细。

| 权重 | 准则 | 衡量方式 |
|:-:|---|---|
| 25 | Trace schema 覆盖率 | 产出规范 GenAI spans 的 SDK 系列数量（目标：6+） |
| 20 | 评估准确性 | DeepEval / RAGAS 分数 vs 人工标注集 |
| 20 | 仪表板 UX | 注入回归的 MTTR（目标：5 分钟内） |
| 20 | 成本 / 规模 | 以 1k spans/sec 持续采集且无积压 |
| 15 | 告警 + 漂移检测 | Prometheus/Alertmanager 链路端到端验证 |
| **100** | | |

## 练习

1. 为 Haystack 框架添加自定义插桩。验证规范的 spans 落入 ClickHouse 并带有准确的 `gen_ai.*` 属性。

2. 在同一批 traces 上将 DeepEval 替换为 Phoenix evaluators。测量两个评估引擎之间的分数漂移。

3. 强化漂移检测：按 app-id 分别计算 PSI 而非全局计算。展示各 app 的漂移轨迹。

4. 新增"用户影响"页面：展示 cost-per-user 和 failure-rate-per-user，附带迷你图（sparklines）。

5. 构建一条尾采样策略：保留所有 toxicity > 0.5 的 trace，再加上 10% 的分层采样其余 trace。测量引入的采样偏差。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| GenAI semconv | "OTel LLM attributes" | 2025 OpenTelemetry 规范中 LLM span 属性的约定（system、model、tokens） |
| Tail sampling | "Post-trace sample" | Collector 在 trace 完成后决定是否保留或丢弃（可 peek 错误） |
| PSI | "Population stability index" | 漂移指标，比较两个分布；> 0.2 通常表示有意义的漂移 |
| LLM-judge | "Eval as model" | 一个 LLM 根据评分标准给另一个 LLM 的输出打分（忠实度、毒性、PII） |
| Tail-sampling policy | "Keep-rule" | 决定哪些 trace 持久化、哪些丢弃的规则；错误 + 采样率 |
| Eval span | "Linked eval trace" | 携带评估分数的子 span，链接到原始 LLM 调用 span |
| Cost per user | "Unit economics" | 在时间窗口内归因到 user_id 的美元成本；关键产品指标 |

## 延伸阅读

- [Langfuse](https://github.com/langfuse/langfuse) — 参考开放核心可观测平台
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 具有强大漂移支持的替代参考
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — 自动插桩 SDK 系列
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 采集 schema
- [Helicone](https://www.helicone.ai) — 托管可观测性替代方案
- [Braintrust](https://www.braintrust.dev) — 评估优先平台替代方案
- [ClickHouse 文档](https://clickhouse.com/docs) — 列式 span 存储
- [DeepEval](https://github.com/confident-ai/deepeval) — 评估库
