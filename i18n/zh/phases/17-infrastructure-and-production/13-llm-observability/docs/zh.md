# LLM 可观测性栈选择

> 2026 年的可观测性市场分为两大类：开发平台（LangSmith、Langfuse、Comet Opik）将监控与评估、提示词管理、会话回放打包在一起；网关/遥测工具（Helicone、SigNoz、OpenLLMetry、Phoenix）则专注于遥测数据。Langfuse 的开源核心采用 MIT 许可证，在开源生态方面平衡较好（免费云服务额度为每月 5 万条事件）。Phoenix 是 OpenTelemetry 原生工具，采用 Elastic License 2.0 许可证——非常适合漂移/RAG 可视化，但不是一个持久化的生产后端。Arize AX 使用零拷贝 Iceberg/Parquet 集成，声称比单体可观测性方案便宜 100 倍。LangSmith 在 LangChain/LangGraph 场景下领先，定价 $39/用户/月，仅企业版支持自托管。Helicone 基于代理模式，15-30 分钟即可部署，免费额度为每月 10 万请求，但在智能体追踪方面深度不足。典型的生产模式是：网关（Helicone/Portkey）+ 评估平台（Phoenix/TruLens），通过 OpenTelemetry 串联。

**类型：** 学习
**语言：** Python（标准库、玩具级 trace 采样模拟器）
**前置要求：** 阶段 17·08（推理指标）、阶段 14（智能体工程）
**预计时间：** 约 60 分钟

## 学习目标

- 区分开发平台（集成型：评估 + 提示词 + 会话）与网关/遥测工具（仅追踪 + 指标）。
- 将六种主流工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）映射到各自的许可证、定价和最佳适用场景。
- 解释利用 OpenTelemetry 将网关工具与独立评估平台组合在一起的模式。
- 说出 2026 年的成本差异点（Arize AX 的零拷贝方案 vs 单体采集），并说明约 100 倍这一数量级。

## 问题所在

你上线了一个 LLM 功能。它能工作。但你完全看不到提示词失败、工具循环、延迟回退、成本飙升或提示词缓存命中率的情况。你去 Google 搜索"LLM 可观测性"，结果出来八个工具，都宣称能解决同一个问题，只是定价策略各不相同。

但它们解决的并非同一个问题。LangSmith 回答的是"这个 LangGraph 运行为什么失败了？"；Phoenix 回答的是"我的 RAG 管道是否发生了漂移？"；Helicone 回答的是"哪个应用在烧 token？"；Langfuse 回答的是"我能自托管整个方案吗？"不同的工具，不同的受众。

选择时涉及四个维度：技术栈（LangChain？原始 SDK？多厂商混合？）、许可证偏好（仅接受 MIT？Elastic 也可以？商业许可没问题？）、预算（免费额度？$100/月？$1000/月？）和自托管需求（必须？有就好？永远不需要？）。

## 核心概念

### 两大类别

**开发平台**将可观测性与评估、提示词管理、数据集版本控制、会话回放等功能打包。你可以运行实验，查看哪个提示词效果最好，并用新提示词对历史获胜候选进行数据集回归测试。代表工具：LangSmith、Langfuse、Comet Opik。

**网关/遥测工具**负责接入推理调用——记录提示词、响应、token 数、延迟、模型、成本等。代表工具：Helicone、SigNoz、OpenLLMetry、Phoenix。轻量级，可通过 OpenTelemetry 与独立的评估工具组合使用。

### Langfuse —— 开源平衡之选

- 核心采用 Apache / MIT 许可证；可通过 Docker 自托管。
- 云服务免费额度：每月 5 万条事件。付费版：$29/月/团队。
- 提供评估、提示词管理、追踪、数据集。对四大开发平台功能均有合理覆盖。
- 最佳场景：你想要 LangSmith 级别的特性，但必须自托管或坚持使用 OSS 许可证。

### Phoenix (Arize) —— 遥测优先，OpenTelemetry 原生

- 采用 Elastic License 2.0；自托管非常简单。
- 在 RAG 和漂移可视化方面表现出色。Embedding 空间散点图作为一等公民直接提供。
- 并非设计为持久化生产后端——主要用于开发时观测。
- 最佳场景：RAG 管道开发、漂移调试，搭配独立的网关用于生产环境。

### Arize AX —— 规模优先方案

- 商业产品。通过 Iceberg/Parquet 实现零拷贝数据湖集成。
- 声称在大规模下比单体可观测性（Datadog 级别）便宜约 100 倍。原理是：你将追踪数据存储在自有 S3 上的 Parquet 文件中；Arize 直接读取。
- 最佳场景：日追踪量 >1000 万，已有数据湖基础设施，希望在不承受 Datadog 价格的前提下获得 LLM 专属仪表盘。

### LangSmith —— 为 LangChain/LangGraph 而生

- 商业产品，$39/用户/月。仅企业版支持自托管。
- 针对 LangChain 和 LangGraph 技术栈提供业界最佳体验。如果你不在上述任一技术栈上，吸引力会下降。
- 最佳场景：团队已锁定 LangChain，愿意为此付费。

### Helicone —— 基于代理的最低可行方案

- 只需将 `OPENAI_API_BASE` 指向 Helicone 代理，15-30 分钟即可完成部署。
- MIT 许可证；免费额度每月 10 万请求，付费版 $20/月起。
- 内置故障转移、缓存、速率限制——同时扮演网关角色。
- 在智能体/多步追踪方面的深度有限。
- 最佳场景：快速启动，单一技术栈应用，需要一个网关 + 可观测性一体化方案。

### Opik (Comet) —— 开源开发平台

- Apache 2.0，完全开源。
- 功能集与 Langfuse 类似，带有 Comet 的基因。
- 最佳场景：ML 团队已在 Comet 生态中，希望在同一界面内获得 LLM 可观测性。

### SigNoz —— 以 OpenTelemetry 为核心的完整 APM

- Apache 2.0。通过 OpenTelemetry 同时处理通用 APM 和 LLM 可观测性。
- 最佳场景：跨服务和 LLM 调用的统一可观测性。

### 粘合层：OpenTelemetry + GenAI 语义约定

OpenTelemetry 于 2025 年底发布了 GenAI 语义约定（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。支持 OTel 消费的工具之间可以互操作。当前涌现的生产模式如下：

1. 从每次 LLM 调用中按照 GenAI 约定输出 OTel 遥测数据。
2. 日常流量路由到网关（Helicone / Portkey）。
3. 双路投递到评估平台（Phoenix / Langfuse）用于回归分析。
4. 归档到数据湖（Iceberg），通过 Arize AX 或 DuckDB 进行长期分析。

### 陷阱：在错误的层级进行接入

在智能体框架内部进行接入（例如添加 LangSmith 追踪）会将你绑定到该框架。在 HTTP/OpenAI-SDK 层（通过 OpenLLMetry 或你的网关）进行接入则具备可移植性。

### 采样——你不可能保留所有数据

当日请求量超过 100 万时，全量保留追踪的开销会超过 LLM 调用本身的成本。按规则采样：100% 错误、100% 高成本、5% 成功。聚合数据始终保留，原始数据仅保留长尾部分。

### 需要记住的数字

- Langfuse 免费云服务：每月 5 万条事件。
- LangSmith：$39/用户/月。
- Helicone 免费额度：每月 10 万请求。
- Arize AX 声称：大规模下比单体方案便宜约 100 倍。
- OpenTelemetry GenAI 约定：2025 年发布，2026 年广泛采用。

```figure
i4-otel-glue
```

## 动手实践

`code/main.py` 模拟了跨越不同保留策略（100% 全量接入、采样、采样+错误）的一天 100 万条追踪的数据，报告各策略下的存储成本和丢失的数据量。

## 交付成果

本课将产出 `outputs/skill-observability-stack.md`。根据技术栈、规模、预算和许可证偏好，选择合适的工具（组合）。

## 练习题

1. 你的团队使用 LangChain，希望采用开源自托管的可观测性方案。选择 Langfuse 或 Opik 并说明理由。
2. 在日追踪量 500 万且 Datadog 报价为 $150K/月的情况下，计算 Arize AX 的盈亏平衡点。
3. 设计一组 OpenTelemetry GenAI 属性，作为你组织应强制要求在每次 LLM 调用中添加的规范。
4. 论证 Phoenix 单独是否足以支撑生产环境。在哪些情况下它不够用？
5. Helicone 引入 20ms 代理开销。在 P99 TTFT 为 300ms 的情况下，这是否可接受？如果 SLA 要求是 100ms 呢？

## 关键术语

| 术语 | 人们常说的说法 | 实际含义 |
|------|----------------|----------|
| OpenLLMetry | "面向 LLM 的 OTel" | 面向 LLM 的开源 OpenTelemetry 接入工具 |
| GenAI 约定 | "OTel 属性" | LLM 调用的标准 OTel 属性命名规范 |
| LangSmith | "LangChain 可观测性" | 与 LangChain 生态捆绑的商业平台 |
| Langfuse | "开源版 LangSmith" | MIT 许可证的 OSS，功能集与之类似 |
| Phoenix | "Arize 开发工具" | OpenTelemetry 原生的开发/评估平台 |
| Arize AX | "大规模可观测性" | 商业产品，基于零拷贝 Iceberg/Parquet 的可观测性方案 |
| Helicone | "代理式可观测性" | HTTP 代理，收集 LLM 遥测数据 + 网关功能 |
| Opik | "Comet LLM" | 来自 Comet 的 Apache 2.0 开源开发平台 |
| 会话回放 | "追踪重放" | 重放包含工具调用的完整智能体会话 |
| 评估 | "离线测试" | 在标注数据集上运行候选模型/提示词 |

## 延伸阅读

- [SigNoz — 2026 年最佳 LLM 可观测性工具](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX 替代方案分析](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — 配置 Langfuse、LangSmith、Helicone、Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix 文档](https://docs.arize.com/phoenix)
- [Helicone 文档](https://docs.helicone.ai/)
