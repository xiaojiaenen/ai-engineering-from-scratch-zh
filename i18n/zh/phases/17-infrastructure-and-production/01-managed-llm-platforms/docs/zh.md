# 托管式大语言模型平台 — Bedrock、Vertex AI 与 Azure OpenAI

> 三大云厂商，三种截然不同的策略。AWS Bedrock 是模型市场——Claude、Llama、Titan、Stability、Cohere 均通过单一 API 接入。Azure OpenAI 是独家 OpenAI 合作，并提供预置吞吐量单元（PTUs）用于独占容量。Vertex AI 以 Gemini 为核心，拥有最强的长上下文与多模态能力。2026 年 Artificial Analysis 测得在 Llama 3.1 405B 等效模型上，Azure OpenAI 的中位延迟约 50 ms，Bedrock 约 75 ms——PTU 解释了这一差距，因为独占容量优于共享按需模式。决策规则不是“哪个最快”，而是“哪个模型目录和 FinOps 成本核算体系契合我的产品”。本节教你如何基于明确写下的权衡做选择，而不是凭感觉。

**类型：** 学习
**语言：** Python（标准库，用于估算成本与延迟的示例程序）
**先修知识：** 第 11 阶段（LLM 工程）、第 13 阶段（工具与协议）
**预计耗时：** 约 60 分钟

## 学习目标

- 说出三种平台策略（市场型、独占型、Gemini 优先型）并匹配到对应的产品使用场景。
- 解释 Azure OpenAI 的预置吞吐量单元（PTUs）能带来什么价值，以及为什么在 405B 规模下 Bedrock 的按需模式通常慢约 25 ms。
- 绘制各平台的 FinOps 成本归因体系图（Bedrock 应用推理配置文件 vs Vertex 按团队划分项目 vs Azure 作用域与 PTU 预留）。
- 写下“最低双提供商”策略并解释为何在 2026 年单一厂商锁定是代价高昂的错误。

## 问题背景

你为你的产品选定了 Claude 3.7 Sonnet。现在需要把它投入服务。你可以直接调用 Anthropic API，也可以通过 AWS Bedrock 调用，或者走网关。直接 API 最简单；Bedrock 增加了 BAAs（商业伙伴协议）、VPC 终端节点、IAM 权限和 CloudWatch 成本归因。网关则带来故障转移、统一计费以及跨提供商的速率限制。

更深层的问题是模型目录。如果你的产品同时需要 Claude、Llama 和 Gemini，除非同时使用 Bedrock、Vertex AI 和 Azure OpenAI，否则你无法从单一来源购买它们。三大云厂商并不等价——它们在模型层归属上各自做出了不同的押注。

本节将梳理这三种押注、延迟差距、FinOps 差距以及锁定风险。

## 核心概念

### 三种策略

**AWS Bedrock** —— 市场型。Claude（Anthropic）、Llama（Meta）、Titan（AWS 自有）、Stability（图像）、Cohere（Embedding）、Mistral，以及图像和 Embedding 子目录。单一 API、单一 IAM 体系、单一 CloudWatch 导出。Bedrock 的策略是：客户更需要选择权，而不是单一模型。

**Azure OpenAI** —— 独占合作。你可以获得 GPT-4 / 4o / 5 / o 系列、DALL·E、Whisper，以及在 Azure 数据中心对 OpenAI 模型进行微调。在“Azure OpenAI Service”目录中没有非 OpenAI 模型——这些会进入 Azure AI Foundry（独立产品）。Azure 的策略是：OpenAI 仍是前沿，客户希望对这种特定合作关系施加企业级管控。

**Vertex AI** —— Gemini 优先，其余其次。Gemini 1.5 / 2.0 / 2.5 Flash 与 Pro，以及 Model Garden（第三方模型）。Vertex 的策略是多模态长上下文——100 万 token 的 Gemini 上下文是差异化优势。

### 规模化的延迟差距

Artificial Analysis 持续运行基准测试。在等效的 Llama 3.1 405B 部署（共享按需模式）上，Azure OpenAI 的首 Token 延迟中位数约为 50 ms；Bedrock 约为 75 ms。差距不是 AWS 的失败，而是容量模型不同。Azure 出售 PTU（预置吞吐量单元），为你的租户预留 GPU 容量。Bedrock 的等效方案（预置吞吐量）也存在，但单价起步约 $21/小时，大多数客户仍使用共享按需模式。

按需共享容量需要与所有其他客户的流量竞争。独占容量则不会。如果你的产品 SLA 要求 P99 TTFT < 100 ms，你要么在 Azure 购买 PTU，要么购买 Bedrock 预置吞吐量，要么接受默认的波动。

### 预置吞吐量的经济性

Azure PTU：预留的一块推理计算资源。对于可预测的负载，相比按需模式最高可节省约 70%。费用按小时固定收取，与流量无关——即使空闲也要为预留付费。盈亏平衡点通常在 40%-60% 的持续利用率。

Bedrock 预置吞吐量：根据模型和地区，每小时约 $21-$50。逻辑相似——盈亏平衡点约为峰值利用率的一半。需要月度承诺。

Vertex 的预留容量按 Gemini SKU 出售；价格因模型和地区而异，公开信息较少。

### FinOps 体系——真正的差异点

**Bedrock 应用推理配置文件** 是市场中体系最清晰的一种。为配置文件打上 `team`、`product`、`feature` 标签；所有模型调用都通过它路由；CloudWatch 无需后处理即可按配置文件拆分成本。该功能于 2025 年新增，仍是粒度最细的云厂商原生方案。

**Vertex** 的归因方式是按团队划分 GCP 项目，并在所有资源上打标签，配合 BigQuery Billing Export + DataStudio 进行汇总。工作量更大，但 BigQuery 允许你对成本数据执行任意 SQL。

**Azure** 依赖订阅/资源组作用域加标签，PTU 预留则作为一等成本对象。标签继承自资源组而非请求，因此要实现请求级归因需要借助 Application Insights 自定义指标或通过网关对请求打标。

总结规律：Bedrock 原生最清晰，Vertex 通过 BigQuery 最灵活，Azure 若不主动做度量则最不明确。

### 锁定是 2026 年的主要风险

当单一模型占据主导时，押注单一云厂商尚可接受。但在 2026 年，前沿模型每月都在变化——一个季度是 Claude 3.7，下一个是 Gemini 2.5，再下一个是 GPT-5。锁定单一平台会让你与三分之二的边缘前沿失之交臂。

目前主流团队采用的模式是：对任何关键 LLM 调用至少保留两家提供商。Bedrock 与 Azure OpenAI 是最常见的组合——一家用 Claude，另一家用 GPT，通过同一网关实现故障转移。成本增量可忽略不计，因为网关会路由最优路径；而在出现故障时（如 Azure OpenAI 2025 年 1 月事件、AWS us-east-1 中断）可用性提升则是决定性的。

### 数据驻留、BAAs 与受监管行业

- **Bedrock：** 大部分地区的 BAAs；VPC 终端节点；护栏。是金融科技领域的常见默认选择。
- **Azure OpenAI：** HIPAA、SOC 2、ISO 27001；欧盟数据驻留；是企业合规场景的默认选项。
- **Vertex：** HIPAA、GDPR、按地区的数据驻留；Google Cloud 的合规体系。

三者均能满足基础合规检查。差异在于数据保留策略、日志处理方式，以及滥用监控是否读取你的流量（多数默认开启；企业客户可选择不开启）。

### 需要记住的数字

- Azure OpenAI 在 Llama 3.1 405B 等效模型上的中位 TTFT（含 PTU）：约 50 ms
- Bedrock 按需模式的中位 TTFT：约 75 ms
- Bedrock 预置吞吐量：每单元 $21-$50/小时
- Azure PTU 盈亏平衡点：持续利用率约 40%-60%
- PTU 在高利用率下相比按需模式的节省：最高可达 70%

```figure
i4-platform-lanes
```

## 实践

`code/main.py` 在合成负载下对比三个平台——模拟按需模式与 PTU 的经济性、TTFT 方差以及成本归因精度。运行它以了解 PTU 何时划算，以及市场型的模型广度何时能抵消 TTFT 的差距。

## 产出

本节将生成 `outputs/skill-managed-platform-picker.md`。给定负载特征（所需模型、TTFT SLA、日吞吐量和合规要求），它会推荐主用平台、备用平台以及 FinOps 度量方案。

## 练习题

1. 运行 `code/main.py`。Azure PTU 在持续利用率为多少时，能在 70B 级模型上超越按需模式？计算盈亏平衡点并与官方宣称的 40%-60% 区间进行对比。
2. 你的产品需要 Claude 3.7 Sonnet 与 GPT-4o。设计双提供商部署方案——哪个模型放在哪家云厂商，前置网关是什么，故障转移策略如何制定？
3. 一位受监管的医疗客户要求提供 BAAs、美东数据驻留，以及 P99 TTFT < 100 ms。选择一个平台并用三个具体特性加以论证。
4. 你发现本月 Bedrock 账单增长了 4 倍，但流量并无变化。如果没有应用推理配置文件，你将如何定位问题根源？如果配置了文件，定位需要多长时间？
5. 阅读 Azure OpenAI 与 Bedrock 的定价页面。对于每月 1 亿 token 的 Claude 负载，哪个更便宜——Anthropic 直连 API、Bedrock 按需模式，还是 Bedrock 预置吞吐量？

## 关键术语

| 术语 | 一般说法 | 实际含义 |
|------|---------|---------|
| Bedrock | "AWS 的 LLM 服务" | 覆盖 Claude、Llama、Titan、Mistral、Cohere 的模型市场 |
| Azure OpenAI | "Azure 上的 ChatGPT" | 在 Azure 数据中心独占运行 OpenAI 模型，并提供企业级管控 |
| Vertex AI | "Google 的 LLM 平台" | Gemini 优先的平台，Model Garden 用于引入第三方模型 |
| PTU | "独占容量" | 预置吞吐量单元——预留的推理 GPU，按小时计费 |
| Application Inference Profile | "Bedrock 标签" | 带标签的按产品维度成本/用量配置文件，原生支持 CloudWatch |
| Model Garden | "Vertex 目录" | Vertex AI 的第三方模型专区，与 Gemini 体系分离 |
| 双提供商最低策略 | "LLM 冗余" | 要求每条关键 LLM 调用路径至少运行在 ≥2 家云厂商上的策略 |
| BAA | "HIPAA 文书" | 商业伙伴协议；处理 PHI 时必须签署；三家云厂商均提供 |
| 滥用监控 | "日志审计方" | 提供商侧对提示词/输出进行的安全扫描；企业客户可选择关闭 |

## 延伸阅读

- [AWS Bedrock 定价](https://aws.amazon.com/bedrock/pricing/)——权威价目表与预置吞吐量定价。
- [Azure OpenAI Service 定价](https://azure.microsoft.com/en-us/pricing/details/azure-openai/)——PTU 经济性与价目表。
- [Vertex AI 生成式 AI 定价](https://cloud.google.com/vertex-ai/generative-ai/pricing)——Gemini 各档位与 Model Garden 附加费。
- [Artificial Analysis LLM 排行榜](https://artificialanalysis.ai/)——跨提供商的持续延迟与吞吐量基准测试。
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO 指南 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/)——企业级决策框架。
- [Finout — Bedrock vs Vertex vs Azure FinOps 对比](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend)——成本归因机制的逐项对照。
