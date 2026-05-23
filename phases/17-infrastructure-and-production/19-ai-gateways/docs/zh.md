# AI 网关 —— LiteLLM、Portkey、Kong AI Gateway、Bifrost

> 网关位于应用与模型提供商之间。核心功能包括提供商路由、故障转移、重试、速率限制、密钥引用、可观测性、安全护栏。2026年市场格局：**LiteLLM** 采用MIT开源协议，支持100+提供商，兼容OpenAI，但在约2000 RPS时性能下降（8 GB内存，公开基准测试中出现级联故障）；最适合Python应用、低于500 RPS、开发/原型场景。**Portkey** 定位为控制平面（安全护栏、PII脱敏、越狱检测、审计追踪），2026年3月转为Apache 2.0开源，单请求延迟开销20-40毫秒，生产层49美元/月。**Kong AI Gateway** 基于Kong网关构建——Kong在相同12核CPU上的基准测试显示：比Portkey快228%，比LiteLLM快859%；定价100美元/模型/月（Plus层上限5个）；适合已使用Kong的企业。**Bifrost**（Maxim AI）——自动重试与可配置退避策略，当OpenAI返回429时自动切换至Anthropic。**Cloudflare / Vercel AI 网关** ——托管式、免运维、基础重试功能。数据驻留要求决定是否自托管；Portkey和Kong处于中间地带，兼具开源和可选托管服务。

**类型：** 学习  
**语言：** Python（标准库、简易网关路由模拟器）  
**前置课程：** 阶段17 · 01（托管式LLM平台），阶段17 · 16（模型路由）  
**时间：** 约60分钟

## 学习目标

- 列举六项核心网关功能（路由、故障转移、重试、速率限制、密钥管理、可观测性、安全护栏）。
- 将四款2026年主流网关（LiteLLM、Portkey、Kong AI、Bifrost）映射到扩展上限与应用场景。
- 引述Kong基准测试（比Portkey快228%，比LiteLLM快859%），并解释其对超过500 RPS场景的重要性。
- 根据数据驻留要求与运维预算，选择自托管或托管方案。

## 问题背景

您的产品调用OpenAI、Anthropic和自托管Llama。每个提供商使用不同的SDK、错误模型、速率限制和认证机制。您需要实现故障转移（若OpenAI返回429则尝试Anthropic）、统一的凭证存储、可观测性，以及基于租户的速率限制。

在应用层重复构建此类功能会导致每个服务都与所有提供商紧密耦合。网关层将其整合到单个进程中，通过统一API（通常兼容OpenAI）将请求分发给各提供商。

## 核心概念

### 六项核心功能

1. **提供商路由** —— 将OpenAI、Anthropic、Gemini、自托管模型等统一在一个API后端。
2. **故障转移** —— 遇到429、5xx或质量问题时，自动切换至其他提供商重试。
3. **重试** —— 指数退避策略，限定尝试次数。
4. **速率限制** —— 基于租户、密钥或模型的独立限制。
5. **密钥引用** —— 运行时从密钥库提取凭证（绝不存储于应用中）。
6. **可观测性** —— OpenTelemetry + GenAI属性（阶段17 · 13） + 成本归因。
7. **安全护栏** —— PII脱敏、越狱检测、主题过滤。

### LiteLLM —— MIT开源，Python实现

- 支持100+提供商，兼容OpenAI，提供路由配置、故障转移、基础可观测性。
- 在Kong基准测试中约2000 RPS时性能下降；内存占用8 GB，持续负载下出现级联故障。
- 最佳适用场景：Python应用、低于500 RPS、开发/测试网关、实验性路由。
- 成本：开源版本免费；存在云服务免费层。

### Portkey —— 控制平面定位

- 2026年3月起采用Apache 2.0开源协议。提供安全护栏、PII脱敏、越狱检测、审计追踪。
- 单请求延迟开销20-40毫秒。
- 生产层49美元/月（含数据保留与SLA）。
- 最佳适用场景：需要集成安全护栏与可观测性的受监管行业。

### Kong AI Gateway —— 规模化解决方案

- 基于Kong网关构建（成熟的API网关产品，采用lua+OpenResty）。
- Kong在12核等效环境的基准测试：比Portkey快228%，比LiteLLM快859%。
- 定价：100美元/模型/月，Plus层上限5个模型。
- 最佳适用场景：已使用Kong；超过1000 RPS；愿意购买许可证。

### Bifrost（Maxim AI）

- 支持可配置退避策略的自动重试。
- 当OpenAI返回429时切换至Anthropic是其经典应用场景。
- 新兴竞争者；商业化产品。

### Cloudflare AI Gateway / Vercel AI Gateway

- 托管式、免运维。提供基础重试与可观测性。
- 最佳适用场景：Cloudflare/Vercel上的边缘JavaScript应用。
- 相比Kong/Portkey，在安全护栏和速率限制方面功能有限。

### 自托管 vs 托管

数据驻留要求是决定因素。医疗和金融领域默认选择自托管（LiteLLM或Portkey开源版或Kong）。消费产品通常选择托管方案（如Cloudflare AI Gateway）或中间方案（Portkey托管版）。混合模式：受监管租户使用自托管，其他租户使用托管服务。

### 延迟预算

- LiteLLM：典型开销5-15毫秒。
- Portkey：开销20-40毫秒。
- Kong：开销3-8毫秒。
- Cloudflare/Vercel：边缘优势下开销1-3毫秒。

网关延迟直接增加首字节时间（TTFT）。若要求TTFT P99 < 100毫秒SLA，选择Kong或Cloudflare。若P99 < 500毫秒，任何方案均可满足。

### 速率限制机制的重要性

简单令牌桶算法适用于中等规模。多租户场景需要滑动窗口 + 突发许可 + 分层租户策略。LiteLLM采用令牌桶；Kong采用滑动窗口；Portkey采用分层策略。

### 网关 + 可观测性 + 路由的整合

阶段17 · 13（可观测性） + 16（模型路由） + 19（网关）在生产环境中属于同一层级。选择覆盖三者的工具，或谨慎组合使用：2026年大多数部署采用Helicone（可观测性）或Portkey（安全护栏）与Kong（扩展性）分工协作。

### 关键数据记忆点

- LiteLLM：约2000 RPS时性能下降，内存占用8 GB。
- Portkey：20-40毫秒开销；2026年3月起采用Apache 2.0协议。
- Kong：比Portkey快228%，比LiteLLM快859%。
- Kong定价：100美元/模型/月，Plus层上限5个。
- Cloudflare/Vercel：边缘开销1-3毫秒。

## 实践应用

`code/main.py` 模拟在429/5xx注入条件下跨3个提供商的网关路由与故障转移。报告延迟、重试率和故障转移命中率。

## 部署建议

本课程产出 `outputs/skill-gateway-picker.md`。根据扩展需求、运维策略、合规要求和延迟预算，选择合适的网关方案。

## 练习

1. 运行 `code/main.py`。配置从OpenAI→Anthropic→自托管模型的故障转移。当提供商错误率为5%时，预期命中率是多少？
2. 您的SLA要求基线300毫秒下TTFT P99 < 200毫秒。哪些网关能保持在延迟预算内？
3. 某医疗客户要求自托管 + PII脱敏 + 审计功能。选择Portkey开源版还是Kong。
4. 比较LiteLLM与Kong：当达到多少RPS上限时团队应该迁移？
5. 为多租户SaaS设计速率限制策略：免费层、试用层、付费层。选择令牌桶还是滑动窗口？

## 核心术语

| 术语 | 通俗说法 | 准确含义 |
|------|---------|---------|
| 网关 | “API代理” | 位于应用与提供商之间的处理进程 |
| LiteLLM | “MIT那个” | Python开源方案，支持100+提供商，2K RPS时性能下降 |
| Portkey | “安全护栏网关” | 控制平面 + 可观测性，Apache 2.0协议 |
| Kong AI Gateway | “扩展型网关” | 基于Kong网关构建，基准测试领先 |
| Bifrost | “Maxim的网关” | 自动重试 + Anthropic故障转移方案 |
| Cloudflare AI Gateway | “边缘托管网关” | 边缘部署的托管网关，免运维 |
| PII脱敏 | “数据清洗” | 发送给模型前使用正则+NER进行脱敏 |
| 越狱检测 | “提示词注入防护” | 对用户输入的分类器检测 |
| 审计追踪 | “合规日志” | 每次LLM调用的不可篡改记录 |
| 令牌桶 | “简易速率限制” | 基于补充的速率限制器 |
| 滑动窗口 | “精确速率限制” | 基于时间窗口的速率限制器；公平性更佳 |

## 扩展阅读

- [Kong AI Gateway基准测试](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry —— 2026年AI网关对比](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy —— 2026年顶级LLM网关工具](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway文档](https://docs.konghq.com/gateway/latest/ai-gateway/)