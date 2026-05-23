# 托管式LLM平台 — Bedrock、Vertex AI、Azure OpenAI

> 三大超大规模云服务商，三种截然不同的策略。AWS Bedrock是模型市场——Claude、Llama、Titan、Stability、Cohere等模型通过统一API提供。Azure OpenAI是OpenAI独家合作加上预配置吞吐量单元（PTU）提供专属容量。Vertex AI以Gemini为核心，在长上下文和多模态方面表现最佳。2026年，Artificial Analysis测评显示，在Llama 3.1 405B等效模型上，Azure OpenAI中位延迟约50毫秒，Bedrock约75毫秒——PTU解释了差距，因为专属容量优于按需共享容量。决策规则并非“哪个最快”，而是“哪个模型目录和FinOps表面更匹配我的产品”。本课程教你权衡利弊后做出选择，而非凭感觉。

**类型：** 学习  
**语言：** Python（标准库，简易成本-延迟比较器）  
**先决条件：** 第11阶段（LLM工程）、第13阶段（工具与协议）  
**时间：** 约60分钟

## 学习目标

- 命名三种平台策略（市场模式 vs 独家模式 vs Gemini优先）并将其与产品用例匹配。
- 解释Azure OpenAI中预配置吞吐量单元（PTU）的作用，以及为何在405B规模下，按需的Bedrock通常慢约25毫秒。
- 绘制每个平台的FinOps归因表面示意图（Bedrock应用推理配置文件 vs Vertex的团队-项目模式 vs Azure作用域+PTU预留）。
- 撰写“双供应商最低要求”策略，并解释为何单一供应商锁定是2026年的昂贵错误。

## 问题背景

你为产品选择了Claude 3.7 Sonnet。现在你需要提供它。你可以直接调用Anthropic API，或者通过AWS Bedrock调用，也可以通过网关调用。直接API最简单；Bedrock增加了BAA、VPC端点、IAM和CloudWatch归因。网关则增加了跨供应商的故障转移、统一计费和速率限制。

更深层的问题是目录。如果你需要在同一产品中同时使用Claude、Llama和Gemini，你无法从单一地方购买所有模型，除非那个地方是Bedrock加上Vertex加上Azure OpenAI的组合。超大规模云服务商并非可互换的——它们对谁拥有模型层做了不同的押注。

本课程将映射这三种押注、延迟差距、FinOps差距以及锁定风险。

## 核心概念

### 三种策略

**AWS Bedrock** — 模型市场。提供Claude（Anthropic）、Llama（Meta）、Titan（AWS自有）、Stability（图像）、Cohere（嵌入）、Mistral，以及图像和嵌入子目录。一个API、一个IAM表面、一个CloudWatch导出。Bedrock的押注是，客户更看重选择权而非单一模型。

**Azure OpenAI** — 独家合作。你可以在Azure数据中心获得GPT-4/4o/5/o系列、DALL·E、Whisper以及OpenAI模型的微调服务。“Azure OpenAI服务”目录中不包含非OpenAI模型——那些模型属于Azure AI Foundry（独立产品）。Azure的押注是OpenAI保持前沿，客户希望在该特定关系上拥有企业级控制。

**Vertex AI** — 以Gemini为核心，其他为辅。Gemini 1.5/2.0/2.5 Flash和Pro，加上Model Garden（第三方模型）。Vertex的押注是多模态长上下文——Gemini的100万token上下文是其差异化优势。

### 规模下的延迟差距

Artificial Analysis进行持续基准测试。在等效的Llama 3.1 405B部署（共享按需）上，Azure OpenAI的中位首token延迟约为50毫秒；Bedrock约为75毫秒。差距并非AWS的失败——而是容量模型的差异。Azure销售PTU（预配置吞吐量单元），为你的租户预留GPU容量。Bedrock的等效产品（预配置吞吐量）也存在，但起价约为每单元每小时21美元，且大多数客户仍使用共享按需。

按需共享容量与其他所有客户的流量竞争。专属容量则不然。如果你的产品SLA要求P99下首token时间（TTFT）< 100毫秒，你要么在Azure购买PTU，要么购买Bedrock预配置吞吐量，要么接受默认的波动性。

### 预配置吞吐量的经济学

Azure PTU：预留的推理计算块。对于可预测的工作负载，相比按需最高可节省约70%。无论流量如何，按固定每小时费用支付——即使闲置也要为预留付费。盈亏平衡点通常在持续利用率的40-60%左右。

Bedrock预配置吞吐量：每小时21-50美元，取决于模型和区域。逻辑相似——盈亏平衡点大约在峰值利用率的一半左右。需要按月承诺。

Vertex的预配置容量按Gemini SKU销售；定价因模型和区域而异，公开宣传较少。

### FinOps表面 — 真正的差异化因素

**Bedrock应用推理配置文件**是市场中最清晰的归因方式。用`team`、`product`、`feature`标记配置文件；将所有模型调用通过它路由；CloudWatch无需后处理即可按配置文件分解成本。2025年新增，至今仍是超大规模云中最精细的原生功能。

**Vertex**的归因采用团队-项目模式加全面标签。你将每个团队建模为一个GCP项目，在每个资源上打标签，并使用BigQuery计费导出+DataStudio进行汇总。工作量更大，但BigQuery允许你对成本数据使用任意SQL。

**Azure**依赖订阅/资源组作用域加标签，PTU预留作为一等成本对象。标签从资源组继承，而非请求，因此按请求归因需要应用洞察自定义指标或可标记请求头的网关。

模式：Bedrock原生最清晰，Vertex通过BigQuery最灵活，Azure除非进行额外监测否则最不透明。

### 锁定是2026年的风险

当一个模型占主导时，单一超大规模云承诺是可行的。在2026年，前沿每月都在移动——某个季度是Claude 3.7，下个季度是Gemini 2.5，再下个季度是GPT-5。锁定在一个平台就等于错失三分之二的前沿技术。

成功团队采用的模式：任何产品关键的LLM调用都遵循双供应商最低要求。Bedrock加Azure OpenAI是常见组合——从一个获取Claude，从另一个获取GPT，两者间进行故障转移，使用同一个网关。成本增加可忽略不计，因为网关会选择最优路由；在中断期间（如2025年1月Azure OpenAI事件、AWS us-east-1中断）的可用性提升是决定性的。

### 数据驻留、BAA与受监管行业

Bedrock：大多数区域提供BAA；VPC端点；防护栏。金融科技公司的常见默认选择。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；欧盟数据驻留；企业受监管默认选项。
Vertex：HIPAA、GDPR；按区域的数据驻留；谷歌云的合规性套件。

三者都满足基本要求。差异在于数据保留政策、日志处理方式，以及滥用监控是否读取你的流量（默认大多数启用；企业可选择退出）。

### 你应该记住的数字

- Azure OpenAI在Llama 3.1 405B等效模型上的中位TTFT（带PTU）：约50毫秒。
- Bedrock按需中位TTFT：约75毫秒。
- Bedrock预配置吞吐量：每单元每小时21-50美元。
- Azure PTU盈亏平衡点：持续利用率约40-60%。
- PTU相比高利用率按需的节省：最高70%。

## 实践应用

`code/main.py`在合成工作负载上比较三个平台——它建模按需与PTU经济学、TTFT波动以及成本归因保真度。运行它以查看PTU何时划算，以及市场模型广度何时超过TTFT差距。

## 部署产出

本课程生成`outputs/skill-managed-platform-picker.md`。给定工作负载配置（所需模型、TTFT SLA、每日流量、合规性要求），它会推荐主平台、备用平台以及FinOps监测计划。

## 练习题

1.  运行`code/main.py`。对于70B级别模型，Azure PTU在持续利用率达到多少时优于按需？计算盈亏平衡点并与宣传的40-60%范围进行比较。
2.  你的产品需要Claude 3.7 Sonnet和GPT-4o。设计一个双供应商部署——哪个模型用哪个超大规模云，前面放什么网关，故障转移策略是什么？
3.  一家受监管的医疗保健客户要求BAA、美国东部数据驻留和低于100毫秒的P99 TTFT。选择一个平台并用三个具体特性证明。
4.  你发现本月的Bedrock账单上涨了4倍，但流量没有变化。如果没有应用推理配置文件，你如何找到罪魁祸首？有配置文件呢？需要多长时间？
5.  阅读Azure OpenAI和Bedrock定价页面。对于每月1亿token的Claude工作负载，哪个更便宜——直接Anthropic API、Bedrock按需还是Bedrock预配置吞吐量？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Bedrock | "AWS的LLM服务" | 跨Claude、Llama、Titan、Mistral、Cohere的模型市场 |
| Azure OpenAI | "Azure版的ChatGPT" | 在Azure数据中心运行的OpenAI独家模型，带企业控制 |
| Vertex AI | "谷歌的LLM" | 以Gemini为核心的平台，含第三方模型的Model Garden |
| PTU | "专属容量" | 预配置吞吐量单元——预留的推理GPU，按小时计费 |
| 应用推理配置文件 | "Bedrock打标签" | 带标签的按产品成本/使用情况配置文件，CloudWatch原生支持 |
| Model Garden | "Vertex目录" | Vertex AI的第三方模型区，与Gemini分开 |
| 双供应商最低要求 | "LLM冗余" | 跨≥2个超大规模云运行每条关键LLM路径的策略 |
| BAA | "HIPAA文书" | 商业伙伴协议；处理PHI所需；三者都提供 |
| 滥用监控 | "日志观察者" | 供应商端对提示/输出的安全扫描；企业可选择退出 |

## 延伸阅读

- [AWS Bedrock定价](https://aws.amazon.com/bedrock/pricing/) — 权威费率表及预配置吞吐量定价。
- [Azure OpenAI服务定价](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — PTU经济学与费率表。
- [Vertex AI生成式AI定价](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini层级与Model Garden附加费。
- [Artificial Analysis LLM排行榜](https://artificialanalysis.ai/) — 跨供应商的持续延迟和吞吐量基准测试。
- [AI Journal — AWS Bedrock vs Azure OpenAI CTO指南2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — 企业决策框架。
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — 归因机制并排比较。