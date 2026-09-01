# 推理平台经济学 — Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026 年的推理市场已不再是 GPU 时间租赁。它分裂为三个细分市场：定制芯片（Groq、Cerebras、SambaNova）、GPU 平台（Baseten、Together、Fireworks、Modal）和 API 优先的市场（Replicate、DeepInfra）。Fireworks 在 2026 年 5 月 1 日将价格提高了 $1/小时每 GPU，而 10T+ token/天的日处理量加上 40 亿美元估值告诉你，以体积为驱动的模式是可行的。Baseten 在 2026 年 1 月完成了 3 亿美元的 E 轮融资，估值达到 50 亿美元。竞争定位规则很简单：Fireworks 优化延迟，Together 优化目录广度，Baseten 优化企业级打磨，Modal 优化 Python 原生开发者体验，Replicate 优化多模态覆盖，Anyscale 优化分布式 Python。本课为你提供一份可以交给创始人的矩阵表。

**类型：** 学习
**语言：** Python (stdlib，玩具级别的每次调用经济学比较器)
**前置条件：** 阶段 17 · 01（托管 LLM 平台），阶段 17 · 04（推理引擎内部原理）
**时间：** 约 60 分钟

## 学习目标

- 说出三个市场细分（定制芯片、GPU 平台、API 优先），并将每个供应商映射到对应细分。
- 解释为什么"按 token 计费"的 API 定价模式会趋近于推理引擎的成本曲线，而非硬件的成本曲线。
- 计算至少三个供应商的单次请求有效成本，并解释什么时候按分钟计费（Baseten、Modal）优于按 token 计费。
- 识别哪个平台是给定工作负载的默认选择（无服务器突发流量、持续高吞吐量、微调变体、多模态）。

## 问题陈述

你评估了托管型云服务商平台，决定需要一个更窄、更快的提供商——Fireworks 用于延迟敏感场景，Together 用于模型广度，Baseten 用于微调自定义模型。现在你有六个真实选择，但定价页面格式不统一。Fireworks 显示 $/M tokens；Baseten 显示 $/分钟；Modal 显示 $/秒；Replicate 显示 $/prediction。没有工作负载建模就无法直接比较。

更糟的是，每个定价页面背后的商业模式各不相同。Fireworks 在共享 GPU 上运行自己的定制引擎（FireAttention）；按 token 费率反映其利用率曲线。Baseten 提供 Truss + 专用 GPU；按分钟计费反映独占性。Modal 是真正的 Python 无服务器平台——按秒计费，冷启动低于 1 秒。同样的输出（LLM 响应），三种不同的成本函数。

本课对这六个平台进行建模，并说明每个何时胜出。

## 核心概念

### 三个市场细分

**定制芯片** — Groq（LPU）、Cerebras（WSE）、SambaNova（RDU）。通常比同等模型下的 GPU 集群快 5-10 倍解码。按 token 价格更高（Groq 在 2025 年末 Llama-70B 约为 ~$0.99/M tokens），但对于延迟敏感用例无可匹敌。Groq 是语音代理和实时翻译的生产首选。

**GPU 平台** — Baseten、Together、Fireworks、Modal、Anyscale。运行在 NVIDIA（H100、H200、2026 年的 B200）或有时 AMD 上。处于"裸 GPU 租赁"（RunPod、Lambda）和"超大规模云托管服务"（Bedrock）之间的经济层。

**API 优先市场** — Replicate、DeepInfra、OpenRouter、Fal。广泛的目录，按预测或按秒付费，强调首次调用时间。

### Fireworks — 延迟优化 GPU 平台

- FireAttention 引擎（定制）； marketed 为等效配置下比 vLLM 延迟低 4 倍。
- 批量层级价格约为无服务器价格的 50%，适用于非交互式工作负载。
- 微调模型以与基础模型相同的费率提供服务——这是 versus 对 LoRA 收取溢价的提供商的真正差异化优势。
- 2026 年中：按需 GPU 租赁价格提高 $1/小时，2026 年 5 月 1 日生效。大规模可协商 volumetric pricing。
- 财务信号：40 亿美元估值，日处理 10T+ tokens。

### Together — 广度优化

- 200+ 模型，包括上游发布后几天内开源的模型。
- 等效 LLM 模型比 Replicate 便宜 50-70%——"AI Native Cloud"定位是体积和目录。
- 推理 + 微调 + 训练在一个 API 中。

### Baseten — 企业级打磨优化

- Truss 框架：模型打包，含依赖项、密钥、推理配置，在一个清单文件中。
- GPU 范围从 T4 到 B200。按分钟计费，有合理的冷启动缓解措施。
- SOC 2 Type II、HIPAA 就绪。常见于金融科技和医疗保健选择。
- 50 亿美元估值，2026 年 1 月 E 轮（3 亿美元，来自 CapitalG、IVP、NVIDIA）。

### Modal — Python 原生优化

- 纯 Python 的基础设施即代码。用 `@modal.function(gpu="A100")` 装饰一个函数并用一条命令部署。
- 按秒计费。冷启动 2-4 秒（预预热）；小模型 <1 秒。
- 2025 年 8700 万美元 B 轮，估值 11 亿美元。独立调查中开发者体验评分最强。

### Replicate — 多模态广度

- 按预测付费。图像、视频和音频模型的默认平台。
- 集成生态（Zapier、Vercel、CMS 插件）。
- LLM 按 token 费率竞争力较弱，但在多模态多样性上胜出。

### Anyscale — Ray 原生

- 基于 Ray 构建；RayTurbo 是 Anyscale 的专有推理引擎（与 vLLM 竞争）。
- 最适合分布式 Python 工作负载，其中推理步骤只是更大图中的一個节点。
- 托管 Ray 集群；与 Ray AIR 和 Ray Serve 紧密集成。

### 按 token 计费和按分钟计费——何时各自胜出

当工作负载对延迟不敏感且是突发的时，按 token 计费合理——你只为使用的付费。当利用率高且可预测时，按分钟计费合理——一旦饱和 GPU，按分钟计费会优于按 token 计费。

粗略规则：对于超过专用 GPU 约 30% 持续利用率的工作负载，按分钟计费（Baseten、Modal）开始优于按 token 计费（Fireworks、Together）。低于这个阈值，按 token 计费胜出，因为你避免为空闲付费。

### 定制引擎才是真正的护城河

每个平台都声称拥有超越 vLLM 和 SGLang 的定制引擎。FireAttention、RayTurbo、Baseten 的推理栈。定制引擎的声明带有营销色彩——诚实的表述是，vLLM + SGLang 代表约 80% 的生产开源推理，平台层的差异化在于 DX、归因和 SLA。

### 你应该记住的数字

- Fireworks GPU 租赁：2026 年 5 月 1 日起提高 $1/小时。
- Fireworks 声称：等效配置下延迟比 vLLM 低 4 倍。
- Together：LLM 比 Replicate 便宜 50-70%。
- Baseten 估值：50 亿美元（E 轮，2026 年 1 月，3 亿美元轮）。
- Modal 估值：11 亿美元（B 轮，2025 年）。
- 超过约 30% 持续利用率时按分钟计费优于按 token 计费。

```figure
cost-per-token
```

## 使用它

`code/main.py` 在合成工作负载上比较六个供应商，跨越不同定价模型。报告每日成本和有效 $/M tokens。运行它以找到按 token 计费和按分钟计费之间的盈亏平衡点。

## 交付物

本课生成 `outputs/skill-inference-platform-picker.md`。给定工作负载配置文件、SLA 和预算，选择主要推理平台并列出备选。

## 练习

1. 运行 `code/main.py`。对于 70B 模型在一个 H100 上，Baseten（按分钟计费）在什么持续利用率下胜过 Fireworks（按 token 计费）？自己推导临界点并与经验法则比较。
2. 你的产品提供图像生成、聊天和语音转文本。为每种模态选择平台，并命名统一它们的网关模式。
3. Fireworks 对主要模型提价 $1/小时。如果 40% 流量移到批量层级（50% 折扣），建模混合成本影响。
4. 受监管客户需要 SOC 2 Type II + HIPAA + 专用 GPU。哪三个平台可行，哪个在 FinOps 上胜出？
5. 比较 Fireworks serverless、Together on-demand、Baseten dedicated、Replicate API 上 Llama 3.1 70B 每 1,000 次预测的成本。每天 10 次预测时哪个最便宜？每天 10,000 次时呢？

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|---------|---------|
| Custom silicon | "非 GPU 芯片" | Groq LPU、Cerebras WSE、SambaNova RDU——针对解码优化 |
| FireAttention | "Fireworks 引擎" | 定制注意力核；宣传比 vLLM 延迟低 4 倍 |
| Truss | "Baseten 的格式" | 模型打包清单；依赖项 + 密钥 + 推理配置 |
| Per-token | "API 定价" | 按消耗 token 计费；不为空闲付费 |
| Per-minute | "专属定价" | 按墙钟 GPU 时间计费；高利用率时胜出 |
| Per-prediction | "Replicate 定价" | 按模型调用计费；常见于图像/视频 |
| RayTurbo | "Anyscale 引擎" | Ray 上的专有推理；在 Ray 集群上与 vLLM 竞争 |
| Batch tier | "5 折" | 非交互式队列，降价；Fireworks、OpenAI 常见 |
| Fine-tuned at base rate | "Fireworks LoRA" | 按基础模型费率收取 LoRA 服务请求（差异化） |

## 延伸阅读

- [Fireworks 定价](https://fireworks.ai/pricing) — 按 token 费率、批量层级、GPU 租赁。
- [Baseten 定价](https://www.baseten.co/pricing/) — 按分钟费率、承诺容量、企业层级。
- [Modal 定价](https://modal.com/pricing) — 按秒 GPU 费率和免费层级。
- [Together AI 定价](https://www.together.ai/pricing) — 模型目录和按 token 费率。
- [Anyscale 定价](https://www.anyscale.com/pricing) — RayTurbo 和托管 Ray 定价。
- [Northflank — Fireworks AI 替代方案](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 比较评估。
- [Infrabase — 2026 AI 推理 API 提供商](https://infrabase.ai/blog/ai-inference-api-providers-compared) — 供应商格局。
