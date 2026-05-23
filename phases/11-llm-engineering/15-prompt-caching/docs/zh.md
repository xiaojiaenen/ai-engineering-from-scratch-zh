# 提示缓存与上下文缓存

> 您的系统提示词包含4,000个token。您的RAG上下文包含20,000个token。每次请求您都需要发送两者，并且每次都需为此付费。提示缓存允许提供商在其端保持该前缀的“温热”状态，并在重用时向您收取正常费率的10%。若使用得当，它可将推理成本降低50–90%，并将首个token的延迟减少40–85%。

**类型：** 构建
**语言：** Python
**前置要求：** 阶段 11 · 01（提示工程），阶段 11 · 05（上下文工程），阶段 11 · 11（缓存与成本）
**时间：** 约60分钟

## 问题描述

一个编程智能体在与Claude进行的对话的每个轮次中都发送相同的15,000个token的系统提示词。按照每百万输入token 3美元的价格计算，仅输入成本在二十轮后就高达0.90美元——这还不包括用户实际发送的任何消息。如果每天有10,000次这样的对话，那么账单每天就会高达9,000美元，而这些文字却从未改变。

您不能在不损害质量的前提下缩减提示词。您也不能避免发送它——模型在每个轮次中都需要它。唯一的办法是停止为提供商已经见过的前缀支付全价。

这个办法就是提示缓存。Anthropic于2024年8月推出了此功能（2025年增加了1小时的延长TTL变体），OpenAI在同年底将其自动化，Google则在发布Gemini 1.5的同时推出了显式上下文缓存，如今这三家都将其作为其前沿模型的首要特性提供。

## 核心概念

![提示缓存：一次写入，廉价读取](../assets/prompt-caching.svg)

**工作原理。** 当一个请求的前缀与最近某个请求的前缀匹配时，提供商将从之前的运行结果中提供KV缓存，而不是重新编码这些token。您第一次支付少量写入溢价，之后每次都会获得大幅读取折扣。

**2026年的三种提供商风格。**

| 提供商 | API风格 | 命中折扣 | 写入溢价 | 默认TTL | 最小可缓存长度 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 在内容块上使用显式 `cache_control` 标记 | 输入费用的90%折扣 | 25% 附加费 | 5分钟（可延长至1小时） | 1,024 个 token（Sonnet/Opus），2,048（Haiku） |
| OpenAI | 自动前缀检测 | 输入费用的50%折扣 | 无 | 最长1小时（尽力而为） | 1,024 个 token |
| Google (Gemini) | 显式 `CachedContent` API | 按存储计费；读取费用约为正常的25% | 按 token·小时 收取存储费 | 用户设定（默认1小时） | 4,096 个 token（Flash），32,768（Pro） |

**不变规则。** 这三家都只缓存前缀。如果请求之间有任何token不同，那么从第一个不同token之后的所有内容都会是缓存未命中。将*稳定*的部分放在顶部，*可变*的部分放在底部。

### 缓存友好的布局

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

违反此顺序——将用户消息置于系统提示词之上，或在few-shot示例之间穿插动态检索——缓存将永远无法命中。

### 收支平衡计算

Anthropic的25%写入溢价意味着一个缓存块至少需要被读取两次才能净省成本。1次写入 + 1次读取，平均每次请求成本为0.675倍（节省32%）；1次写入 + 10次读取，平均每次请求成本为0.205倍（节省80%）。经验法则：将任何预计在TTL内重用至少3次的内容进行缓存。

## 实践构建

### 步骤1：使用显式标记的Anthropic提示缓存

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

`cache_control` 标记告知Anthropic存储该块5分钟。在此时间窗口内重用会命中缓存；过期后重用将重新写入。

**响应中的用量字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

在CI中检查这两个字段——如果跨请求的 `cache_read_input_tokens` 一直为零，说明您的缓存键发生了漂移。

### 步骤2：一小时延长TTL

对于长时间运行的批处理作业，默认的5分钟TTL会在作业之间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1小时TTL的写入溢价是两倍（基线的50%而非25%），但对于任何重用该前缀超过5次的批处理作业，很快就能收回成本。

### 步骤3：OpenAI自动缓存

OpenAI不需要您进行任何配置。任何超过1,024个token且与最近请求匹配的前缀都会自动获得50%的折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

同样的缓存友好布局规则适用。有两件事会破坏OpenAI的缓存而不会破坏Anthropic的：更改 `user` 字段（用作缓存键组成部分）和重新排序工具。

### 步骤4：Gemini显式上下文缓存

Gemini将缓存视为您创建和命名的首要对象：

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

Gemini在缓存存在期间按 token·小时 收取存储费，读取费用约为正常输入费率的25%。当您在多天内跨多个会话重用同一个巨大的提示词时，这是正确的选择。

### 步骤5：在生产中测量命中率

参见 `code/main.py`，一个模拟三家提供商成本核算的脚本，用于跟踪写入/读取/未命中计数并计算每1,000次请求的混合成本。根据目标命中率控制部署——大多数生产环境的Anthropic配置在预热后，读取比例应超过80%。

## 2026年仍在出现的陷阱

- **顶部的动态时间戳。** 在系统提示词顶部使用 `"Current time: 2026-04-22 15:30:02"`。每个请求都会未命中。将时间戳移到缓存断点下方。
- **工具重排序。** 以稳定的顺序序列化工具——部署之间的字典重排会破坏每次命中。
- **近似的自由文本。** “你是乐于助人的。” vs “你是一个乐于助人的助手。”——一字之差 = 完全未命中。
- **过小的块。** Anthropic强制要求至少1,024个token（Haiku为2,048）。更小的块会静默地不被缓存。
- **盲目的成本仪表盘。** 将“输入token”拆分为缓存和未缓存。否则，流量下降看起来像是缓存的成功。

## 使用场景

2026年缓存技术栈：

| 场景 | 选择 |
|-----------|------|
| 具有稳定10k+系统提示词的多轮次智能体 | Anthropic `cache_control`，使用5分钟TTL |
| 批处理作业重用前缀超过30分钟 | Anthropic，使用 `ttl: "1h"` |
| 在GPT-5上的无服务器端点，无自定义基础设施 | OpenAI自动缓存（只需确保您的前缀稳定且足够长） |
| 大型代码/语料库的多天重用 | Gemini显式 `CachedContent` |
| 跨提供商回退 | 跨提供商保持相同的可缓存前缀布局，这样任何命中都有效 |

结合语义缓存（阶段 11 · 11）用于用户消息层：提示缓存处理*token完全相同*的重用，语义缓存处理*含义相同*的重用。

## 部署实施

保存 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## 练习

1.  **简单。** 使用一个5,000个token的系统提示词与Claude进行10轮对话。分别在不使用 `cache_control` 和使用的情况下运行。报告每次的输入token费用。
2.  **中等。** 编写一个测试工具，给定一个提示词模板和请求日志，计算每种提供商（Anthropic 5分钟，Anthropic 1小时，OpenAI自动，Gemini显式）的预期命中率和美元节省额。
3.  **困难。** 构建一个布局优化器：给定一个提示词和一个标记为 `stable=True/False` 的字段列表，重写提示词，在不丢失信息的前提下，将单个缓存断点放在最缓存友好的位置。在真实的Anthropic端点上进行验证。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|-----------------|-----------------------|
| 提示缓存 | “让长提示词变得便宜” | 为匹配的前缀重用提供商端的KV缓存；对重复的输入token提供50-90%的折扣。 |
| `cache_control` | “Anthropic的标记” | 内容块属性，声明“这里之前的所有内容都是可缓存的”；`{"type": "ephemeral"}`。 |
| 缓存写入 | “支付溢价” | 填充缓存的第一个请求；在Anthropic上按约1.25倍输入费率计费，在OpenAI上免费。 |
| 缓存读取 | “折扣” | 后续匹配前缀的请求；在Anthropic上按10%计费，OpenAI 50%，Gemini 约25%。 |
| TTL | “它存活多久” | 缓存保持温热的秒数；Anthropic默认5分钟（可延长至1小时），OpenAI尽力而为最长1小时，Gemini用户设定。 |
| 延长TTL | “1小时Anthropic缓存” | `{"type": "ephemeral", "ttl": "1h"}`；写入溢价为2倍，但对于批处理重用是值得的。 |
| 前缀匹配 | “为什么我的缓存未命中” | 只有当从开始到断点的每个字节都完全相同时，缓存才会命中。 |
| 上下文缓存 (Gemini) | “那个显式的” | Google的命名、按存储计费的缓存对象；最适合大型语料库的多天重用。 |

## 扩展阅读

- [Anthropic — 提示缓存](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`，1小时TTL，收支平衡表。
- [OpenAI — 提示缓存](https://platform.openai.com/docs/guides/prompt-caching) — 自动前缀匹配。
- [Google — 上下文缓存](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API和存储定价。
- [Anthropic工程博客 — 长上下文工作负载的提示缓存](https://www.anthropic.com/news/prompt-caching) — 包含延迟数字的原始发布文章。
- 阶段 11 · 05（上下文工程） — 如何切分提示词以便缓存能够命中。
- 阶段 11 · 11（缓存与成本） — 将提示缓存与针对用户消息的语义缓存结合使用。
- [Pope 等人, "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — 提示缓存向用户暴露的KV缓存内存模型；解释了为什么缓存的前比重算便宜约10倍。
- [Agrawal 等人, "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — 预填充是提示缓存所规避的阶段；这篇论文解释了为什么在缓存命中时TTFT会显著下降，而TPOT不受影响。
- [Leviathan 等人, "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — 提示缓存与推测解码、Flash Attention以及MQA/GQA一起，作为改变推理成本曲线的杠杆；阅读此文献以了解其他三种技术。