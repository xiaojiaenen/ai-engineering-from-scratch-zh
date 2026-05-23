# 推理平台经济学 — Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026年的推理市场已不再是单纯的GPU时间租赁。它分化为定制硅片（Groq、Cerebras、SambaNova）、GPU平台（Baseten、Together、Fireworks、Modal）和API优先的市场（Replicate、DeepInfra）。Fireworks于2026年5月1日将每GPU每小时价格提高了1美元，而每日10万亿+token处理量和40亿美元的估值表明，其量驱动模型是行得通的。Baseten于2026年1月以50亿美元估值完成了3亿美元的E轮融资。竞争定位规则很简单：Fireworks优化延迟，Together优化模型目录广度，Baseten优化企业级打磨，Modal优化Python原生开发体验，Replicate优化多模态覆盖，Anyscale优化分布式Python。本课将提供一个你可以交给创始人的评估矩阵。

**类型：** 学习
**语言：** Python（标准库，玩具级单次调用经济学比较器）
**前置课程：** 阶段 17 · 01（托管式大语言模型平台），阶段 17 · 04（vLLM 服务内部原理）
**时间：** ~60 分钟

## 学习目标

- 列出三个市场细分领域（定制硅片、GPU平台、API优先），并将每个供应商映射到对应细分领域。
- 解释为什么"按token计费"的API定价模式趋向于服务引擎的成本曲线，而非硬件成本曲线。
- 计算至少三个供应商的有效单次请求成本，并解释在何种情况下按分钟计费（Baseten, Modal）优于按token计费。
- 识别对于给定工作负载（无服务器突发、稳定高吞吐、微调变体、多模态），哪个平台是首选。

## 问题所在

你评估了托管式超大规模云平台。你决定需要一个更专注、更快的提供商——Fireworks追求延迟，Together追求模型广度，Baseten追求定制化微调模型。现在你有六个实际选择，但定价页面无法直接对齐。Fireworks显示每百万token价格；Baseten显示每分钟价格；Modal显示每秒价格；Replicate显示每次预测价格。如果不建模工作负载，你无法进行直接比较。

更糟的是，每个定价页面背后的商业模式不同。Fireworks在共享GPU上运行其自定义引擎（FireAttention）；其按token费率反映了它们的利用率曲线。Baseten提供Truss框架和专用GPU；按分钟费率体现了独占性。Modal是真正的Python无服务器——按秒计费，亚秒级冷启动。相同的输出（一个大语言模型响应），三种不同的成本函数。

本课将为这六个平台建模，并告诉你何时选择哪一个最合适。

## 核心概念

### 三个细分市场

**定制硅片** — Groq（LPU）、Cerebras（WSE）、SambaNova（RDU）。通常在相同模型上，解码速度比基于GPU的集群快5-10倍。单token价格较高（Groq在2025年末对Llama-70B的定价约为0.99美元/百万token），但对于延迟敏感的应用场景无可匹敌。Groq是语音助手和实时翻译的生产级首选。

**GPU平台** — Baseten、Together、Fireworks、Modal、Anyscale。运行在NVIDIA（2026年的H100、H200、B200）或有时AMD的硬件上。是介于"原始GPU租赁"（RunPod、Lambda）和"超大规模云托管服务"（Bedrock）之间的经济层。

**API优先的市场** — Replicate、DeepInfra、OpenRouter、Fal。模型目录广泛，按预测或按秒计费，强调首次调用速度。

### Fireworks — 延迟优化型GPU平台

- 自定义FireAttention引擎；在同等配置下，宣传比vLLM延迟低4倍。
- 针对非交互式工作负载提供批处理层，价格约为无服务器价格的50%。
- 微调模型的服务价格与基础模型相同——这相对于其他对你的LoRA收取溢价的提供商是一个真正的差异化优势。
- 2026年中：自2026年5月1日起，按需GPU租赁价格实际提高1美元/小时。大规模用量可协商定价。
- 财务信号：40亿美元估值，日处理10万亿+token。

### Together — 广度优化型

- 拥有200+模型，包括在上游发布后几天内就上线的开源模型。
- 在同等大语言模型上，比Replicate便宜50-70%——"AI原生云"的定位在于规模和目录。
- 单一API提供推理、微调和训练。

### Baseten — 企业级打磨优化型

- Truss框架：将模型、依赖项、密钥、服务配置打包在一个清单中。
- GPU范围从T4到B200。按分钟计费，冷启动缓解措施合理。
- SOC 2 Type II，符合HIPAA要求。金融科技和医疗领域的常见选择。
- 50亿美元估值，2026年1月E轮融资（CapitalG、IVP、NVIDIA投资3亿美元）。

### Modal — Python原生优化型

- 纯Python的基础设施即代码。用 `@modal.function(gpu="A100")` 装饰一个函数，一条命令即可部署。
- 按秒计费。冷启动时间2-4秒（带预热）；小模型<1秒。
- 2025年B轮融资8700万美元，估值11亿美元。在独立调查中获得最强开发者体验评分。

### Replicate — 多模态广度型

- 按预测计费。图像、视频和音频模型的默认平台。
- 集成生态系统（Zapier、Vercel、CMS插件）。
- 在大语言模型按token计费方面竞争力较弱，但在多模态多样性上胜出。

### Anyscale — Ray原生型

- 基于Ray构建；RayTurbo是Anyscale的专有推理引擎（与vLLM竞争）。
- 最适合分布式Python工作负载，其中推理步骤是更大计算图中的一个节点。
- 托管式Ray集群；与Ray AIR和Ray Serve紧密集成。

### 按token计费 vs 按分钟计费 — 各自胜出的场景

当工作负载对延迟不敏感且具有突发性时，按token计费是合理的——你只需为实际使用付费。当利用率高且可预测时，按分钟计费更划算——一旦你能让GPU饱和，就能胜过按token计费。

粗略规则：对于专用GPU持续利用率高于约30%的工作负载，按分钟计费（Baseten、Modal）开始胜过按token计费（Fireworks、Together）。低于此比例，按token计费胜出，因为它避免了为空闲时间付费。

### 自定义引擎是真正的护城河

每个在vLLM和SGLang之上的平台都声称拥有自定义引擎。FireAttention、RayTurbo、Baseten的推理栈。自定义引擎的声明带有营销色彩——诚实的框架是，vLLM + SGLang代表了约80%的生产级开源推理，平台层的差异化在于开发体验、归因分析和服务等级协议。

### 你应该记住的数字

- Fireworks GPU租赁：自2026年5月1日起，价格有效上涨1美元/小时。
- Fireworks宣称：在同等配置下，延迟比vLLM低4倍。
- Together：在大语言模型上比Replicate便宜50-70%。
- Baseten估值：50亿美元（2026年1月E轮，3亿美元融资）。
- Modal估值：11亿美元（2025年B轮融资）。
- 按分钟计费在持续利用率超过约30%时胜过按token计费。

## 动手使用

`code/main.py` 在一个跨定价模型的合成工作负载上比较这六个供应商。报告每天成本和有效的每百万token成本。运行它以找到按token计费和按分钟计费之间的盈亏平衡点。

## 交付使用

本课产出 `outputs/skill-inference-platform-picker.md`。根据工作负载特征、SLA和预算，选择首选推理平台并指出备选方案。

## 练习

1. 运行 `code/main.py`。在一台H100上运行70B模型时，Baseten（按分钟计费）需要达到多少持续利用率才能胜过Fireworks（按token计费）？自行推导交叉点，并与经验法则进行比较。
2. 你的产品提供图像生成、聊天和语音转文字服务。为每种模态选择平台，并指出统一它们的网关模式。
3. Fireworks对你主要模型的价格上涨了1美元/小时。如果40%的流量转向批处理层（打5折），请建模其混合成本影响。
4. 一个受监管的客户要求SOC 2 Type II + HIPAA + 专用GPU。哪三个平台可行？哪个在财务运营方面胜出？
5. 比较在Fireworks无服务器、Together按需、Baseten专用和Replicate API上运行Llama 3.1 70B的每1000次预测成本。在每天10次预测时，哪个最便宜？在每天10,000次预测时呢？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|----------|
| 定制硅片 | "非GPU芯片" | Groq LPU、Cerebras WSE、SambaNova RDU — 针对解码优化 |
| FireAttention | "Fireworks引擎" | 自定义注意力机制内核；宣传比vLLM延迟低4倍 |
| Truss | "Baseten的格式" | 模型打包清单；包含依赖项、密钥和服务配置 |
| 按token计费 | "API定价" | 按消耗的token收费；不为空闲时间付费 |
| 按分钟计费 | "专用定价" | 按GPU实际运行时间收费；在高利用率时胜出 |
| 按预测计费 | "Replicate定价" | 按模型调用次数收费；常见于图像/视频 |
| RayTurbo | "Anyscale引擎" | Ray上的专有推理引擎；在Ray集群上与vLLM竞争 |
| 批处理层 | "打5折" | 降低费率的非交互式队列；Fireworks、OpenAI常用 |
| 按基础价格微调 | "Fireworks的LoRA" | 按基础模型的价格为LoRA服务的请求收费（差异化优势） |

## 扩展阅读

- [Fireworks 定价](https://fireworks.ai/pricing) — 按token费率、批处理层、GPU租赁。
- [Baseten 定价](https://www.baseten.co/pricing/) — 按分钟费率、承诺容量、企业层级。
- [Modal 定价](https://modal.com/pricing) — 按秒GPU费率和免费额度。
- [Together AI 定价](https://www.together.ai/pricing) — 模型目录和按token费率。
- [Anyscale 定价](https://www.anyscale.com/pricing) — RayTurbo和托管Ray定价。
- [Northflank — Fireworks AI 替代方案](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 比较评估。
- [Infrabase — 2026年AI推理API提供商](https://infrabase.ai/blog/ai-inference-api-providers-compared) — 供应商格局。