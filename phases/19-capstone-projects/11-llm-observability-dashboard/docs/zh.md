# 顶点项目 11 — LLM 可观测性与评估仪表板

> Langfuse 转向开源核心模式。Arize Phoenix 发布了 2026 GenAI 语义映射规范。Helicone 和 Braintrust 都加倍投入了基于用户的成本归因。Traceloop 的 OpenLLMetry 成为了事实上的 SDK 插桩标准。生产环境的技术栈通常采用 ClickHouse 存储追踪数据，Postgres 存储元数据，Next.js 构建 UI，并运行着一系列评估任务（DeepEval、RAGAS、LLM-judge）对采样追踪数据进行处理。你需要构建一个自托管的系统，能够从至少四个 SDK 系列中摄取数据，并演示在五分钟内捕获一个注入的回归问题。

**类型：** 顶点项目
**语言：** TypeScript (UI)，Python / TypeScript (数据摄取 + 评估)，SQL (ClickHouse)
**前置条件：** 阶段 11 (LLM 工程)，阶段 13 (工具)，阶段 17 (基础设施)，阶段 18 (安全)
**涉及阶段：** P11 · P13 · P17 · P18
**预计时间：** 25 小时

## 问题陈述

在 2026 年，每个运行生产流量的 AI 团队都会在模型旁边部署一个可观测性层。成本归因。幻觉检测。漂移监控。越狱信号。SLO 仪表板。PII 泄露警报。开源参考——Langfuse、Phoenix、OpenLLMetry——已统一采用 OpenTelemetry GenAI 语义约定作为摄取 schema。你现在可以用一个 SDK 插桩 OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM，并发送兼容的 span。

你将构建一个自托管的仪表板，它能够从至少四个 SDK 系列中摄取数据，在采样追踪数据上运行一组小的评估任务，检测漂移，并发出警报。衡量标准：给定一个故意注入的回归（一个开始产生 PII 的提示词），该仪表板应在五分钟内捕获它并触发警报。

## 核心概念

数据摄取通过 OTLP HTTP 进行。SDK 产生遵循 GenAI 语义约定的 span：`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.response.id`, `llm.prompts`, `llm.completions`。span 存入 ClickHouse 以进行列式分析；元数据（用户、会话、应用）存入 Postgres。

评估任务作为批量作业在采样追踪数据上运行。DeepEval 评估忠实度、毒性和答案相关性。RAGAS 在追踪数据包含检索上下文时评估检索指标。自定义的 LLM-judge 运行特定于领域的检查（PII 泄露、偏离策略的响应）。评估运行结果作为 eval span 写回到同一个 ClickHouse，并与父追踪关联。

漂移检测随时间监控嵌入空间的分布（基于提示词嵌入的 PSI 或 KL 散度），以及评估分数的趋势。警报推送到 Prometheus Alertmanager，然后发送到 Slack / PagerDuty。UI 采用 Next.js 15 和 Recharts 构建。

## 架构图

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## 技术栈

- **数据摄取：** OpenTelemetry SDKs + GenAI 语义约定；OTLP HTTP 传输
- **收集器：** OpenTelemetry Collector，带有尾部采样处理器（用于成本控制）
- **存储：** ClickHouse 存储 span，Postgres 存储元数据，S3 存储原始事件归档
- **评估：** DeepEval, RAGAS 0.2, Arize Phoenix 评估器套件，自定义 LLM-judge
- **漂移检测：** 每周对汇聚的提示词嵌入（使用 sentence-transformers）计算 PSI / KL
- **警报：** Prometheus Alertmanager -> Slack / PagerDuty
- **UI：** Next.js 15 App Router + Recharts + 服务器动作
- **开箱即用支持的 SDK：** OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM

## 构建步骤

1.  **收集器配置。** 使用 OTLP HTTP 接收器、一个尾部采样器（保留 100% 错误追踪和 10% 成功追踪）以及到 ClickHouse 和 S3 的导出器来配置 OpenTelemetry Collector。

2.  **ClickHouse schema。** 创建表 `spans`，列与 GenAI 语义约定镜像：`gen_ai_system`, `gen_ai_request_model`, `input_tokens`, `output_tokens`, `latency_ms`, `prompt_hash`, `trace_id`, `parent_span_id`，外加一个用于存储长载荷的 JSON 包。为 `user_id` 和 `app_id` 添加二级索引。

3.  **SDK 覆盖测试。** 使用每个 SDK（OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM）编写一个小的客户端应用，集成 OpenLLMetry 自动插桩。验证每个 SDK 生成的标准 GenAI span 都能正确存入 ClickHouse。

4.  **评估任务。** 一个定时任务读取过去 15 分钟的采样追踪数据，并运行 DeepEval 的忠实度、毒性和答案相关性评估。输出是与父追踪关联的 eval span。

5.  **自定义 LLM-judge。** 一个 PII 泄露 judge：给定一个响应，调用一个防护 LLM 来评估 PII 泄露的可能性。高分响应进入分诊队列。

6.  **漂移检测。** 每周任务计算本周汇聚的提示词嵌入与过去 4 周基线之间的 PSI。如果 PSI 超过阈值，则触发警报。

7.  **仪表板。** 使用 Next.js 15 构建，包含以下页面：总览（spans/sec、cost/user、p95 延迟）、追踪（搜索 + 瀑布图）、评估（忠实度趋势、毒性）、漂移（PSI 随时间变化）、警报。

8.  **警报链路。** Prometheus 导出器读取评估分数聚合和延迟百分位数；Alertmanager 将警告路由到 Slack，将严重违规路由到 PagerDuty。

9.  **回归探测。** 注入一个 bug：被评估的聊天机器人开始以 1% 的概率泄露假的社会安全号码。衡量平均修复时间（MTTR）：从 bug 部署到 Slack 告警的时间。

## 使用示例

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 交付物

`outputs/skill-llm-observability.md` 是交付物。给定一个 LLM 应用，该仪表板能够摄取其追踪数据，运行评估，在漂移时发出警报，并在 Next.js 中展示基于用户的成本细分。

| 权重 | 评估标准 | 如何衡量 |
|:-:|---|---|
| 25 | 追踪 schema 覆盖率 | 产生标准 GenAI span 的 SDK 系列数量（目标：6+） |
| 20 | 评估正确性 | DeepEval / RAGAS 分数与人工标注集的对比 |
| 20 | 仪表板用户体验 | 针对注入回归的平均修复时间（MTTR）（目标：低于 5 分钟） |
| 20 | 成本/可扩展性 | 在无积压的情况下，维持 1k spans/sec 的持续摄取能力 |
| 15 | 警报 + 漂移检测 | Prometheus/Alertmanager 链路端到端验证 |
| **100** | | |

## 练习

1.  为 Haystack 框架添加自定义插桩。验证生成的标准 span 在存入 ClickHouse 时具有忠实的 `gen_ai.*` 属性。

2.  在相同的追踪数据上，用 Phoenix 评估器替换 DeepEval。衡量两个评估引擎之间的分数漂移。

3.  优化漂移检测器：按 `app_id` 而非全局计算 PSI。展示每个应用的漂移轨迹。

4.  添加一个“用户影响”页面：每用户成本和每用户失败率，辅以迷你折线图。

5.  构建一个尾部采样策略：保留所有毒性 > 0.5 的追踪数据，并对剩余部分进行 10% 的分层采样。衡量引入的采样偏差。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| GenAI 语义约定 | "OTel LLM 属性" | 2025 年 OpenTelemetry 关于 LLM span 属性（系统、模型、token）的规范 |
| 尾部采样 | "追踪后采样" | 收集器在追踪完成后决定保留或丢弃该追踪（可查看错误） |
| PSI | "群体稳定性指数" | 比较两个分布的漂移度量；通常 > 0.2 表示显著漂移 |
| LLM-judge | "模型作为评估" | 一个 LLM 根据评分标准（忠实度、毒性、PII）对另一个 LLM 的输出进行评分 |
| 尾部采样策略 | "保留规则" | 决定哪些追踪被持久化或丢弃的规则；错误追踪 + 采样率 |
| Eval span | "关联的评估追踪" | 携带评估分数并与原始 LLM 调用 span 关联的子 span |
| 每用户成本 | "单位经济学" | 在一个时间窗口内归因于 user_id 的美元成本；关键的产品指标 |

## 延伸阅读

- [Langfuse](https://github.com/langfuse/langfuse) — 参考性开源核心可观测性平台
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 具有强大漂移支持的替代参考
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — 自动插桩 SDK 家族
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 摄取 schema
- [Helicone](https://www.helicone.ai) — 替代性的托管可观测性方案
- [Braintrust](https://www.braintrust.dev) — 替代性的评估优先平台
- [ClickHouse 文档](https://clickhouse.com/docs) — 列式 span 存储
- [DeepEval](https://github.com/confident-ai/deepeval) — 评估库