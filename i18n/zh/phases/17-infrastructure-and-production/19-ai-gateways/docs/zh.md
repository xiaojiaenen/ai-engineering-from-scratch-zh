# AI 网关 — LiteLLM、Portkey、Kong AI 网关、Bifrost

> 网关位于你的应用和模型提供商之间。核心功能包括：提供商路由、故障转移、重试、限速、密钥引用、可观测性和护栏。2026 年市场格局：**LiteLLM** 是 MIT 开源协议，支持 100+ 提供商，OpenAI 兼容，但在约 2000 RPS 时性能崩溃（8 GB 内存，发布基准测试中存在级联故障）；适合 Python 场景、<500 RPS、开发/原型阶段。**Portkey** 定位为控制面板（护栏、PII 脱敏、越狱检测、审计追踪），2026 年 3 月转为 Apache 2.0 开源，每次请求延迟开销 20-40 ms，生产级方案 $49/月。**Kong AI 网关** 基于 Kong Gateway 构建——Kong 自身在同一 12 CPU 上的基准测试显示：比 Portkey 快 228%，比 LiteLLM 快 859%；定价为 $100/模型/月（Plus 层最多 5 个）；适合已在 Kong 生态内的企业用户。**Bifrost**（Maxim AI）— 支持可配置退避的自动重试，OpenAI 返回 429 时可故障转移到 Anthropic。**Cloudflare / Vercel AI 网关** — 托管式、零运维、基础重试。数据驻留要求是选择自托管的驱动力；Portkey 和 Kong 处于中间位置，提供开源版本 + 可选托管服务。

**类型：** 学习
**语言：** Python（标准库、玩具网关路由模拟器）
**前置条件：** 第 17 阶段 · 01（托管 LLM 平台）、第 17 阶段 · 16（模型路由）
**时间：** 约 60 分钟

## 学习目标

- 列举六大核心网关功能（路由、故障转移、重试、限速、密钥、可观测性、护栏）。
- 将四个 2026 年网关（LiteLLM、Portkey、Kong AI、Bifrost）映射到规模上限和使用场景。
- 引用 Kong 基准测试结果（比 Portkey 快 228%，比 LiteLLM 快 859%）并解释其对 >500 RPS 场景的意义。
- 根据数据驻留和运维预算选择自托管或托管方案。

## 问题所在

你的产品同时调用 OpenAI、Anthropic 和一个自托管的 Llama。每个提供商都有不同的 SDK、错误模型、速率限制和认证方式。你希望实现故障转移（如果 OpenAI 返回 429，就尝试 Anthropic）、统一的凭据存储、统一的可观测性，以及按租户的速率限制。

在应用层重新发明这些，会让每个服务都与每个提供商紧耦合。网关层将其整合到一个进程中，提供一个统一的 API（通常是 OpenAI 兼容的），然后分发到各个提供商。

## 概念解析

### 六大核心功能

1. **提供商路由** — OpenAI、Anthropic、Gemini、自托管等，通过一个 API 统一接入。
2. **故障转移** — 遇到 429、5xx 或质量失败时，重试其他提供商。
3. **重试** — 指数退避，有界尝试次数。
4. **速率限制** — 按租户、按密钥、按模型分别限速。
5. **密钥引用** — 运行时从密钥库获取凭据（绝不硬编码在应用中）。
6. **可观测性** — OTel + GenAI 属性（第 17 阶段 · 13）+ 成本归因。
7. **护栏** — PII 脱敏、越狱检测、允许的主题过滤器。

### LiteLLM — MIT 开源，Python

- 支持 100+ 提供商，OpenAI 兼容，路由器配置，故障转移，基础可观测性。
- 在 Kong 的基准测试中约 2000 RPS 时崩溃；8 GB 内存占用，持续负载下出现级联故障。
- 最佳适用：Python 应用、<500 RPS、开发/测试网关、实验性路由。
- 成本：开源版免费；云免费版存在。

### Portkey — 控制面板定位

- 2026 年 3 月起采用 Apache 2.0 开源协议。护栏、PII 脱敏、越狱检测、审计追踪。
- 每次请求延迟开销 20-40 ms。
- 生产级 $49/月，包含保留策略和 SLA。
- 最佳适用：需要护栏和可观测性捆绑的受监管行业。

### Kong AI 网关 — 面向大规模场景

- 基于 Kong Gateway（成熟 API 网关产品，lua + OpenResty）。
- Kong 自身在 12 CPU 等价环境下的基准测试：比 Portkey 快 228%，比 LiteLLM 快 859%。
- 定价：$100/模型/月，Plus 层最多 5 个。
- 最佳适用：已在 Kong 生态内；>1000 RPS；愿意授权使用。

### Bifrost（Maxim AI）

- 支持可配置退避的自动重试。
- 遇到 OpenAI 429 时故障转移到 Anthropic 是一个经典方案。
- 较新入局者；商业化产品。

### Cloudflare AI 网关 / Vercel AI 网关

- 托管式，零运维。基础重试和可观测性。
- 最佳适用：部署在 Cloudflare/Vercel 上的边缘 JavaScript 应用。
- 在护栏和限速方面相比 Kong/Portkey 功能有限。

### 自托管 vs 托管

数据驻留是强制因素。医疗和金融行业默认自托管（LiteLLM 或 Portkey 开源版或 Kong）。消费类产品默认托管（Cloudflare AI 网关）或中等方案（Portkey 托管）。混合模式：受监管租户自托管，其他租户使用托管。

### 延迟预算

- LiteLLM：典型开销 5-15 ms。
- Portkey：开销 20-40 ms。
- Kong：开销 3-8 ms。
- Cloudflare/Vercel：边缘优势，开销 1-3 ms。

网关延迟直接累加到 TTFT（首 token 延迟）。若 TTFT P99 < 100 ms SLA，选择 Kong 或 Cloudflare。若 P99 < 500 ms，任意均可。

### 限速语义很重要

简单令牌桶算法适用于中等规模场景。多租户需要滑动窗口 + 突发 allowance + 按租户分层。LiteLLM 提供令牌桶；Kong 提供滑动窗口；Portkey 提供分层限速。

### 网关 + 可观测性 + 路由组合

第 17 阶段 · 13（可观测性）+ 16（模型路由）+ 19（网关）在生产环境中是同一层。选择能覆盖三者的单一工具，或仔细串联各组件：2026 年大多数部署将 Helicone（可观测性）或 Portkey（护栏）与 Kong（规模）组合，实现职责分离。

### 需要记住的数字

- LiteLLM：约 2000 RPS 时崩溃，8 GB 内存。
- Portkey：开销 20-40 ms；2026 年 3 月起采用 Apache 2.0。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong 定价：$100/模型/月，Plus 层最多 5 个。
- Cloudflare/Vercel：边缘开销 1-3 ms。

```figure
mx-gateway-fallback
```

## 实践使用

`code/main.py` 模拟在 429/5xx 注入下跨 3 个提供商的网关路由故障转移。报告延迟、重试率和故障转移命中率。

## 交付成果

本课产出 `outputs/skill-gateway-picker.md`。根据规模、运维模式、合规要求和延迟预算，选择网关。

## 练习

1. 运行 `code/main.py`。配置从 OpenAI→Anthropic→自托管的故障转移。在 5% 提供商错误率下，预期命中率是多少？
2. 你的 SLA 是 TTFT P99 < 200 ms（基线 300 ms）。哪些网关能在预算范围内？
3. 一个医疗健康客户需要自托管 + PII 脱敏 + 审计。选择 Portkey 开源版还是 Kong？
4. 对比 LiteLLM 和 Kong：团队应在多少 RPS 上限时迁移？
5. 为一个多租户 SaaS 设计限速策略：免费层、试用层、付费层。用令牌桶还是滑动窗口？

## 关键术语

| 术语 | 人们常说的说法 | 实际含义 |
|------|----------------|----------|
| 网关 | "API 代理" | 位于应用和提供商之间的进程 |
| LiteLLM | "MIT 那个" | Python 开源，100+ 提供商，2K RPS 时崩溃 |
| Portkey | "护栏网关" | 控制面板 + 可观测性，Apache 2.0 |
| Kong AI 网关 | "大规模那个" | 基于 Kong Gateway 构建，基准测试领先 |
| Bifrost | "Maxim 的网关" | 重试 + Anthropic 故障转移方案 |
| Cloudflare AI 网关 | "边缘托管" | 边缘部署的托管网关，零运维 |
| PII 脱敏 | "数据清洗" | 发送前用正则 + NER 进行掩码 |
| 越狱检测 | "提示注入防护" | 对用户输入进行分类器检测 |
| 审计追踪 | "合规日志" | 每条 LLM 调用的不可变记录 |
| 令牌桶 | "简单限速" | 基于填充的限速器 |
| 滑动窗口 | "精确限速" | 时间窗口限速器；公平性更好 |

## 延伸阅读

- [Kong AI 网关基准测试](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI 网关 2026 年对比](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — 2026 年顶级 LLM 网关工具](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI 网关文档](https://docs.konghq.com/gateway/latest/ai-gateway/)
