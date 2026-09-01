# 提示词缓存与上下文缓存

> 你的系统提示词为 4,000 tokens，你的 RAG 上下文为 20,000 tokens。你每次请求都发送两者，并且两者都要付费。提示词缓存让提供商在服务端保持该前缀的"预热"状态，复用时只按正常费率的 10% 计费。正确使用可将推理成本降低 50–90%，首 token 延迟降低 40–85%。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 11 · 01（提示词工程）、Phase 11 · 05（上下文工程）、Phase 11 · 11（缓存与成本）
**时间：** 约 60 分钟

## 问题所在

一个编码代理在对话的每一步都向 Claude 发送相同的 15,000-token 系统提示词。20 轮对话，按 $3/百万输入 token 计费，仅输入成本就达 $0.90——还没算用户实际消息。若扩大到每天 10,000 次对话，这笔永不改变的文字费用可达 $9,000/天。

你不能在不损害质量的前提下缩短提示词，也无法避免在每轮发送它——模型在每轮都需要它。唯一的办法是停止为提供商已经见过的缓存前缀支付全价。

这个办法就是提示词缓存。Anthropic 在 2024 年 8 月推出了它（2025 年增加了 1 小时超长 TTL 变体），OpenAI 在同年稍后自动实现了它，Google 随 Gemini 1.5 推出了显式的上下文缓存，三家如今都在其前沿模型上提供了这一一等公民级特性。

## 概念

![提示词缓存：写一次，廉价读](../assets/prompt-caching.svg)

**机制。** 当一个请求的前缀与近期某个请求的前缀匹配时，提供商直接复用上次运行的 KV 缓存，而非重新编码这些 token。你第一次支付一小笔写入溢价，之后每次读取享受大幅折扣。

**2026 年的三种提供商实现。**

| 提供商 | API 风格 | 命中折扣 | 写入溢价 | 默认 TTL | 最小可缓存量 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 内容块上使用显式 `cache_control` 标记 | 输入打 1 折（90% off） | 额外 25% | 5 分钟（可扩展至 1 小时） | 1,024 tokens（Sonnet/Opus），2,048（Haiku） |
| OpenAI | 自动前缀检测 | 输入打 5 折 | 无 | 最高 1 小时（尽力而为） | 1,024 tokens |
| Google（Gemini） | 显式 `CachedContent` API | 按存储计费；读取约为正常的 25% | 每 token·小时存储费 | 用户设定（默认 1 小时） | 4,096 tokens（Flash），32,768（Pro） |

**不变量。** 三者都只缓存前缀。如果两次请求之间有任何 token 不同，从第一个不同 token 开始之后的内容全是缓存未命中。将*稳定的*部分放在顶部，*动态的*部分放在底部。

### 缓存友好布局

```
[系统提示词]             <-- 缓存这部分
[工具定义]               <-- 缓存这部分
[少样本示例]             <-- 缓存这部分
[检索文档]               <-- 若会被复用则缓存，否则不缓存
[对话历史]               <-- 最多缓存到上一轮
[当前用户消息]           <-- 永远不缓存（每次不同）
```

违反顺序——把用户消息放在系统提示词上方、在少样本示例之间交错动态检索结果——缓存永远不会命中。

### 盈亏平衡计算

Anthropic 的 25% 写入溢价意味着缓存块至少需要被读取两次才能节省成本。1 次写入 + 1 次读取平均每次请求成本为 0.675x（节省 32%）；1 次写入 + 10 次读取平均为 0.205x（节省 80%）。经验法则：将预期在 TTL 内至少复用 3 次的任何内容放入缓存。

```figure
prompt-cache-hit
```

## 构建它

### 步骤 1：使用显式标记的 Anthropic 提示词缓存

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 标记告知 Anthropic 将该内容块存储 5 分钟。在该窗口内复用会命中；过期后复用最重新写入。

**响应中的使用字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # 按 1.25x 计费
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # 按 0.1x 计费
```

在 CI 中检查这两个字段——如果 `cache_read_input_tokens` 在多次请求中始终为零，说明你的缓存键在漂移。

### 步骤 2：1 小时超长 TTL

对于长时间运行的批处理任务，5 分钟的默认 TTL 会在任务之间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL 的写入溢价为 2x（比基线多 50% 而非 25%），但只要任何批处理任务复用该前缀超过 5 次就能快速回本。

### 步骤 3：OpenAI 自动缓存

OpenAI 不给你任何可配置的选项。任何超过 1,024 token 且与近期请求匹配的前缀都会自动获得 50% 折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # 长且稳定
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # 享受折扣的部分
```

同样的缓存友好布局规则适用。有两件事会破坏 OpenAI 的缓存但不会破坏 Anthropic 的：修改 `user` 字段（作为缓存键的一部分）和重新排列工具。

### 步骤 4：Gemini 显式上下文缓存

Gemini 将缓存视为一个你创建并命名的头等公民对象：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini 为缓存存活期间的每个 token·小时收取存储费，读取时按约 25% 的正常输入费率计费。当你连续多天在多轮会话中复用同一个巨大提示词时，这是正确的模式。

### 步骤 5：在生产环境中测量命中率

参见 `code/main.py` 中模拟的三家提供商会计程序，它跟踪写入/读取/未命中计数并计算每 1,000 次请求的混合成本。以目标命中率为部署门禁——大多数生产环境的 Anthropic 设置在预热后应能达到 >80% 的读取占比。

## 2026 年仍然常见的陷阱

- **顶部动态时间戳。** `"Current time: 2026-04-22 15:30:02"` 放在系统提示词顶部。每次请求都会未命中。将时间戳移到缓存断点下方。
- **工具重排序。** 以稳定顺序序列化工具——部署间的字典重排会破坏所有命中。
- **近似的自由文本。** "You are helpful." vs "You are a helpful assistant."——相差一个字节就等于完全未命中。
- **块太小。** Anthropic 强制要求 1,024 token 下限（Haiku 为 2,048）。更小的块会静默地无法缓存。
- **盲目标看板。** 将"输入 token"拆分为已缓存与未缓存。否则流量下降会被误认为是缓存胜利。

## 使用它

2026 年缓存栈：

| 场景 | 选择 |
|-----------|------|
| 具有稳定 10k+ 系统提示词的代理，多轮对话 | Anthropic `cache_control` + 5 分钟 TTL |
| 批处理任务复用前缀超过 30 分钟 | Anthropic + `ttl: "1h"` |
| GPT-5 上的无服务器端点，无自定义基础设施 | OpenAI 自动缓存（保持前缀稳定且足够长即可） |
| 跨多天的巨型代码/文档语料复用 | Gemini 显式 `CachedContent` |
| 跨提供商回退 | 在各提供商间保持相同的可缓存前缀布局，确保任一命中都能工作 |

与语义缓存（Phase 11 · 11）结合用于用户消息层：提示词缓存处理*token 级完全一致*的复用，语义缓存处理*语义级一致*的复用。

## 交付它

保存 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: 设计缓存友好的提示词布局并选择合适的提供商缓存模式。
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

给定一个提示词（系统 + 工具 + 少样本 + 检索 + 历史 + 用户）和一个使用画像（每小时请求数、所需 TTL、提供商），输出：

1. 布局。重排序各节并标记单一缓存断点；说明哪些部分稳定、哪些部分动态。
2. 提供商模式。Anthropic cache_control、OpenAI 自动、或 Gemini CachedContent。从 TTL 和复用模式论证。
3. 盈亏平衡。预期 TTL 内每次写入的读取次数；与无缓存的净成本对比及数学推导。
4. 验证方案。CI 断言：第二次相同请求时 cache_read_input_tokens > 0；看板按已缓存与未缓存 token 拆分。
5. 故障模式。列出此设置中缓存最可能未命中的三个原因（动态时间戳、工具重排、近似文本）及各自的预防措施。

拒绝交付将动态字段放置在断点上方的缓存方案。拒绝在无法通过 2x 写入溢价回本的复用次数下启用 1 小时 TTL。
```

## 练习

1. **简单。** 对一条 5,000-token 系统提示词与 Claude 的 10 轮对话，分别用和不用 `cache_control` 运行。报告每次的输入 token 账单。
2. **中等。** 编写一个测试工具，给定提示词模板和请求日志，计算各提供商（Anthropic 5 分钟、Anthropic 1 小时、OpenAI 自动、Gemini 显式）的预期命中率和美元节省额。
3. **困难。** 构建一个布局优化器：给定一个提示词和一组标记了 `stable=True/False` 的字段，重写提示词，在不丢失信息的前提下将单一缓存断点置于最缓存友好的位置。在真实 Anthropic 端点上验证。

## 关键术语

| 术语 | 人们通常怎么说 | 实际含义 |
|------|----------------|-----------------------|
| 提示词缓存 | "让长提示词变便宜" | 复用服务端 KV 缓存匹配前缀；重复输入 token 享受 50-90% 折扣。 |
| `cache_control` | "Anthropic 的标记" | 内容块属性，声明"到此为止的内容均可缓存"；值为 `{"type": "ephemeral"}`。 |
| 缓存写入 | "支付溢价" | 首次填充缓存的请求；Anthropic 按约 1.25x 输入费率计费，OpenAI 免费。 |
| 缓存读取 | "折扣" | 后续匹配前缀的请求；Anthropic 按 10%、OpenAI 按 50%、Gemini 约 25% 计费。 |
| TTL | "存活多久" | 缓存保持"预热"的秒数；Anthropic 默认 5 分钟（可扩展至 1 小时），OpenAI 尽力而为最高 1 小时，Gemini 用户设定。 |
| 超长 TTL | "1 小时 Anthropic 缓存" | `{"type": "ephemeral", "ttl": "1h"}`；写入溢价为 2x，但批处理复用场景值得。 |
| 前缀匹配 | "为什么我的缓存未命中" | 只有从开头到断点的所有 token 都字节级相同时缓存才会命中。 |
| 上下文缓存（Gemini） | "显式的那个" | Google 的命名、按存储计费的缓存对象；最适合多天的巨型语料复用。 |

## 延伸阅读

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`、1 小时 TTL、盈亏平衡表。
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — 自动前缀匹配。
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API 与存储定价。
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — 原始发布帖含延迟数据。
- Phase 11 · 05（上下文工程）—— 在哪里切分提示词才能让缓存落地。
- Phase 11 · 11（缓存与成本）—— 将提示词缓存与用户消息上的语义缓存搭配使用。
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — 提示词缓存面向用户暴露的 KV 缓存内存模型；解释了为什么缓存前缀的重新读取成本约为重新计算的 1/10。
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — Prefill 是提示词缓存所跳过的阶段；本文解释了为何缓存命中时 TTFT 大幅下降而 TPOT 不受影响。
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — 提示词缓存与投机解码、Flash Attention、MQA/GQA 并列，是弯曲推理成本曲线的三大杠杆之一；阅读此文了解其余三项。
