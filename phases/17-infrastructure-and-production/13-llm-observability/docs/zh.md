# LLM 可观测性技术栈选择

> 2026年的可观测性市场可分为两大类。开发平台（如 LangSmith, Langfuse, Comet Opik）将监控与评估、提示词管理、会话回放等功能捆绑在一起。网关/遥测工具（如 Helicone, SigNoz, OpenLLMetry, Phoenix）专注于遥测数据。Langfuse 的核心采用 MIT 许可，在开源与商业服务间取得了很好的平衡（免费云服务每月提供5万事件）。Phoenix 基于 Elastic License 2.0，原生支持 OpenTelemetry —— 非常擅长漂移/RAG 可视化，但不适合作为持久化的生产后端。Arize AX 使用零拷贝的 Iceberg/Parquet 集成，声称比一体化可观测性方案成本低100倍。LangSmith 对 LangChain/LangGraph 支持最好，价格为每位用户每月39美元，仅企业版可自托管。Helicone 基于代理模式，15-30分钟即可完成设置，免费提供每月10万次请求，但对智能体链路追踪的深度支持较弱。常见的生产部署模式是：网关（Helicone/Portkey） + 评估平台（Phoenix/TruLens），通过 OpenTelemetry 粘合。

**类型：** 学习
**语言：** Python（标准库，用于模拟链路采样）
**前置知识：** Phase 17 · 08 (推理指标), Phase 14 (智能体工程)
**时间：** 约60分钟

## 学习目标

- 区分开发平台（集成：评估 + 提示词 + 会话）与网关/遥测工具（仅链路追踪 + 指标）。
- 将六种主要工具（Langfuse, LangSmith, Phoenix, Arize AX, Helicone, Opik）对应到其许可证、定价和最佳应用场景。
- 解释 OpenTelemetry 粘合模式如何允许您将网关工具与独立的评估平台结合使用。
- 说明2026年的成本差异因素（Arize AX 的零拷贝方法与一体化数据摄入的对比）并说出大致的100倍成本差距。

## 问题所在

您已经上线了一个 LLM 功能。它能正常工作。但您对提示词失败、工具循环调用、延迟回归、成本激增或提示词缓存命中率等情况缺乏可见性。您搜索“LLM 可观测性”，会得到八个工具，它们都声称能以三个不同的价格点解决同一个问题。

它们解决的并非同一个问题。LangSmith 解答“为什么这次 LangGraph 运行失败了？” Phoenix 解答“我的 RAG 管道是否正在发生漂移？” Helicone 解答“哪个应用在消耗 token？” Langfuse 解答“我能完全自托管这套东西吗？” 不同的工具，面向不同的用户。

选择涉及四个维度：技术栈（LangChain？原生 SDK？多厂商？）、许可证容忍度（只接受 MIT？可以接受 Elastic？商业许可也行？）、预算（免费额度？每月100美元？每月1000美元？）、以及自托管需求（必须？最好有？不需要？）。

## 核心概念

### 两大类别

**开发平台** 将可观测性与评估、提示词管理、数据集版本控制、会话回放等功能捆绑在一起。您可以运行实验，查看哪个提示词有效，将新提示词与旧的最佳版本进行数据集回归测试。代表工具有 LangSmith, Langfuse, Comet Opik。

**网关/遥测工具** 为推理调用提供埋点——记录提示词、响应、token数、延迟、模型和成本。代表工具有 Helicone, SigNoz, OpenLLMetry, Phoenix。功能更精简。可以通过 OpenTelemetry 与独立的评估工具结合使用。

### Langfuse — 开源与商业的平衡

- 核心采用 Apache / MIT 许可；可通过 Docker 自托管。
- 云免费版：每月5万事件。付费版：团队每月29美元。
- 提供评估、提示词管理、链路追踪、数据集功能。对开发平台的四项核心功能覆盖合理。
- 最佳场景：您想要 LangSmith 级别的功能，但必须自托管或坚持使用开源许可证。

### Phoenix (Arize) — 遥测优先，原生 OpenTelemetry

- 采用 Elastic License 2.0；自托管非常简单。
- 擅长 RAG 和漂移可视化。嵌入空间散点图作为一等公民功能提供。
- 并非设计为持久化的生产后端 —— 主要用于开发阶段的可观测性。
- 最佳场景：RAG 管道开发、漂移调试，可与独立的网关配合用于生产环境。

### Arize AX — 规模化之选

- 商业许可。通过 Iceberg/Parquet 集成实现零拷贝数据湖接入。
- 声称在规模化场景下比一体化可观测性方案（如 Datadog 级）成本低约100倍。原理：您将链路追踪数据存储在自己的 S3 Parquet 文件中；Arize 直接读取。
- 最佳场景：每日链路追踪数据超过1000万条、已有数据湖、希望获得 LLM 专用仪表板且不想承受 Datadog 的定价。

### LangSmith — LangChain/LangGraph 优先

- 商业许可，每位用户每月39美元。仅企业版可自托管。
- 对 LangChain 和 LangGraph 技术栈的支持属于最佳级别。如果您不使用这两者，其吸引力就会下降。
- 最佳场景：团队致力于使用 LangChain，且愿意付费。

### Helicone — 基于代理的极简可行方案

- 通过将您的 `OPENAI_API_BASE` 替换为 Helicone 代理，可在15-30分钟内完成设置。
- 采用 MIT 许可；免费提供每月10万次请求，付费版每月20美元起。
- 包含故障转移、缓存、速率限制功能——也充当网关。
- 对智能体/多步骤链路追踪的支持深度有限。
- 最佳场景：快速启动、单栈应用、需要网关与可观测性一体化方案。

### Opik (Comet) — 开源开发平台

- Apache 2.0 许可，完全开源。
- 功能集与 Langfuse 类似，带有 Comet 的基因。
- 最佳场景：已使用 Comet 的 ML 团队，希望在同一个界面中进行 LLM 可观测性分析。

### SigNoz — 原生 OpenTelemetry 的全栈 APM

- Apache 2.0 许可。通过 OpenTelemetry 处理通用 APM 及 LLM 调用。
- 最佳场景：跨服务与 LLM 调用的统一可观测性。

### 粘合剂：OpenTelemetry + GenAI 语义约定

OpenTelemetry 于2025年底发布了 GenAI 语义约定（`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`）。支持 OTel 的工具可以互操作。出现的生产模式是：

1. 从每次 LLM 调用中发出带有 GenAI 约定的 OTel 数据。
2. 将其路由到网关（Helicone / Portkey）用于日常监控。
3. 双写至评估平台（Phoenix / Langfuse）用于回归测试。
4. 归档至数据湖（Iceberg）以供通过 Arize AX 或 DuckDB 进行长期分析。

### 陷阱：在错误层级进行埋点

在智能体框架内部进行埋点（例如，添加 LangSmith 追踪）会使您与该框架耦合。在 HTTP/OpenAI-SDK 层进行埋点（通过 OpenLLMetry 或您的网关）则更具可移植性。

### 采样 — 无法保存一切

当每日请求数超过100万时，完整链路追踪的存储成本会超过 LLM 调用本身的成本。根据规则进行采样：100% 保存错误，100% 保存高成本调用，5% 保存成功调用。始终保存聚合数据；仅对长尾数据保留原始数据。

### 您应该记住的数字

- Langfuse 免费云版：每月5万事件。
- LangSmith：每位用户每月39美元。
- Helicone 免费版：每月10万次请求。
- Arize AX 声称：在规模化场景下比一体化方案便宜约100倍。
- OpenTelemetry GenAI 语义约定：2025年发布，2026年广泛采用。

## 动手使用

`code/main.py` 模拟了在不同保留策略下（100%摄入、采样、采样+保存错误）单日产生100万条链路追踪的情况。报告了每种策略下的存储成本以及丢失的内容。

## 实践部署

本课程产出 `outputs/skill-observability-stack.md`。给定技术栈、规模、预算、许可证要求，为您推荐合适的工具。

## 练习

1. 您的团队使用 LangChain，希望获得可自托管的开源可观测性方案。请在 Langfuse 和 Opik 中选择一个并说明理由。
2. 假设每日有500万条链路追踪，Datadog 的报价是每月15万美元，计算采用 Arize AX 的盈亏平衡点。
3. 设计一套您的组织指南应在每次 LLM 调用中强制要求的 OpenTelemetry GenAI 属性集。
4. 论证仅 Phoenix 是否足以满足生产需求。在哪些情况下它会不足？
5. Helicone 的代理增加了20毫秒的延迟。当 TTFT P99 为300毫秒时，这个延迟是否可接受？如果 SLA 要求是100毫秒呢？

## 关键术语

| 术语 | 人们怎么说 | 其实际含义 |
|------|------------|------------|
| OpenLLMetry | “用于 LLM 的 OTel” | 开源的 LLM OpenTelemetry 埋点工具 |
| GenAI 约定 | “OTel 属性” | 用于 LLM 调用的标准 OTel 属性名 |
| LangSmith | “LangChain 的可观测性” | 与 LangChain 生态系统捆绑的商业平台 |
| Langfuse | “开源版 LangSmith” | 功能集相似的 MIT 许可开源工具 |
| Phoenix | “Arize 的开发工具” | 原生支持 OpenTelemetry 的开发/评估平台 |
| Arize AX | “规模化可观测性” | 商业零拷贝 Iceberg/Parquet 可观测性方案 |
| Helicone | “代理式可观测性” | 收集 LLM 遥测数据并提供网关功能的 HTTP 代理 |
| Opik | “Comet 的 LLM 工具” | 来自 Comet 的 Apache 2.0 开源开发平台 |
| 会话回放 | “链路重放” | 重现一个完整的、包含工具调用的智能体会话 |
| 评估 | “离线测试” | 在带标签的数据集上运行候选模型/提示词 |

## 延伸阅读

- [SigNoz — 2026年顶级 LLM 可观测性工具](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX 替代方案分析](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — 配置 Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix 文档](https://docs.arize.com/phoenix)
- [Helicone 文档](https://docs.helicone.ai/)