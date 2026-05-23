# 流式语音对语音 — Moshi、Hibiki 与全双工对话

> 2024-2026 年重新定义了语音 AI。Moshi 发布了一个单一模型，能够以 200 毫秒延迟同时听和说。Hibiki 逐块进行语音对语音翻译。两者都抛弃了 ASR → LLM → TTS 流水线，转而采用基于 Mimi 编解码器 token 的统一全双工架构。这是新的参考设计。

**类型:** 学习
**语言:** Python
**先决条件:** 阶段 6 · 13（神经音频编解码器），阶段 6 · 11（实时音频），阶段 7 · 05（完整 Transformer）
**时间:** 约 75 分钟

## 问题所在

每个由课程 11 和 12 构建的语音智能体都有一个大约 300-500 毫秒的基本延迟下限：VAD 触发，STT 处理，LLM 推理，TTS 生成。每个阶段都有其自身的最小延迟。你可以进行调优和并行化，但流水线的结构限制了你。

Moshi（Kyutai, 2024-2026）提出了一个不同的问题：如果没有流水线会怎样？如果一个模型直接接收音频输入并持续地输出音频，将文本作为中间的“内心独白”而不是必需阶段，会怎样？

答案是**全双工语音对语音**。理论延迟 160 毫秒（80 毫秒 Mimi 帧 + 80 毫秒声学延迟）。在单张 L4 GPU 上的实践延迟为 200 毫秒。这比同类最佳流水线语音智能体的延迟少了一半。

## 核心概念

![Moshi 架构：两条并行的 Mimi 流 + 内心独白文本](../assets/moshi-hibiki.svg)

### Moshi 架构

**输入。** 两条 Mimi 编解码器流，均为 12.5 Hz × 8 个码本：

- 流 1：用户音频（Mimi 编码，持续到达）
- 流 2：Moshi 自身的音频（由 Moshi 生成）

**Transformer。** 一个 70 亿参数的时序 Transformer 处理这两条流和一条文本“内心独白”流。在每个 80 毫秒步骤中，它：

1. 消费最新的用户 Mimi token（8 个码本）。
2. 消费最近的 Moshi Mimi token（8 个码本，按生成顺序）。
3. 生成下一个 Moshi 文本 token（内心独白）。
4. 生成下一个 Moshi Mimi token（通过一个小型深度 Transformer 生成 8 个码本）。

这三条流——用户音频、Moshi 音频、Moshi 文本——并行运行。Moshi 可以在说话的同时听到用户；当用户打断时可以中断自己；可以在不中断主要发言的情况下进行反馈性插话（“嗯”）。

**深度 Transformer。** 在一个帧内，8 个码本不是并行预测的——它们之间存在码本依赖性。一个小型 2 层“深度 Transformer”在 80 毫秒内按顺序预测它们。这是自回归编解码器语言模型的标准因式分解方法（VALL-E、VibeVoice 也使用）。

### 为什么内心独白文本有帮助

没有显式文本，模型必须在声学流中隐式地建模语言。Moshi 的洞见是：强制它在生成音频的同时输出文本 token。文本流本质上是 Moshi 所说内容的转录。这提高了语义连贯性，使得更换语言模型头更加容易，并且免费为你提供了转录文本。

### Hibiki：流式语音对语音翻译

相同的架构，在翻译对上训练。源语言音频输入，目标语言音频输出，持续不断。Hibiki-Zero（2026 年 2 月）消除了对词级对齐训练数据的需求——使用句子级数据 + GRPO 强化学习进行延迟优化。

最初支持四种语言对；可以通过大约 1000 小时数据适应新语言。

### 更广泛的 Kyutai 技术栈（2026）

- **Moshi** — 全双工对话（首先支持法语，英语支持良好）
- **Hibiki / Hibiki-Zero** — 同声语音翻译
- **Kyutai STT** — 流式 ASR（500 毫秒或 2.5 秒前瞻）
- **Kyutai Pocket TTS** — 1 亿参数的 TTS，可在 CPU 上运行（2026 年 1 月）
- **Unmute** — 在公共服务器上组合这些功能的完整流水线

在 L40S GPU 上的吞吐量：以 3 倍实时速度并发 64 个会话。

### Sesame CSM — 近亲

Sesame CSM（2025）使用类似的思想——一个带有 Mimi 编解码器头的 Llama-3 骨干网络。但 CSM 是单向的（接收上下文 + 文本，生成语音），而不是全双工。它是市场上最好的“语音存在感” TTS；与 Moshi 的全双工能力并不完全相同。

### 2026 性能数据

| 模型 | 延迟 | 用例 | 许可 |
|-------|---------|----------|---------|
| Moshi | 200 ms (L4) | 全双工英语 / 法语对话 | CC-BY 4.0 |
| Hibiki | 12.5 Hz 帧率 | 法语 ↔ 英语流式翻译 | CC-BY 4.0 |
| Hibiki-Zero | 同上 | 5 种语言对，无需对齐数据 | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | 上下文条件 TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | 封闭，OpenAI API | 商业 |
| Gemini 2.5 Live | ~350 ms | 封闭，Google API | 商业 |

## 动手构建

### 步骤 1：接口

Moshi 暴露一个 WebSocket 服务器，它接收 80 毫秒块的 Mimi 编码音频，并返回 80 毫秒块的 Mimi 编码音频。双向进行。持续不断。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### 步骤 2：全双工循环

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

两个方向同时运行。Python asyncio 或 Rust futures 是标准的传输方式。

### 步骤 3：训练目标（概念性）

对于每个 80 毫秒帧 `t`：

- 输入：`user_mimi[0..t]`, `moshi_mimi[0..t-1]`, `moshi_text[0..t-1]`
- 预测：`moshi_text[t]`，然后 `moshi_mimi[t, codebook_0..7]`

文本在音频之前预测（内心独白）；音频在深度 Transformer 内按码本顺序预测。

### 步骤 4：Moshi 的优势与不足

Moshi 优势：

- 在廉价硬件上实现低于 250 毫秒的端到端延迟。
- 自然的反馈性插话和打断。
- 无需流水线粘合代码。

Moshi 不足：

- 工具调用（未针对此训练；你需要单独的 LLM 路径）。
- 长推理（Moshi 是一个 8B 参数级对话模型，不是 Claude/GPT-4）。
- 小众主题的事实准确性。
- 大多数生产型企业用例（2026 年仍使用流水线）。

## 使用场景

| 场景 | 选择 |
|-----------|------|
| 最低延迟语音伴侣 | Moshi |
| 实时翻译通话 | Hibiki |
| 语音演示 / 研究 | Moshi, CSM |
| 带工具的企业级智能体 | 流水线（课程 12），而非 Moshi |
| 上下文中的自定义语音 TTS | Sesame CSM |
| 语音对语音，任意语言 | GPT-4o Realtime 或 Gemini 2.5 Live（商业） |

## 注意事项

- **工具调用受限。** Moshi 是一个对话模型，而不是智能体框架。结合流水线来使用工具。
- **特定语音条件。** Moshi 使用一个单一的训练人格；克隆是单独的训练运行。
- **语言覆盖。** 法语 + 英语支持很好；其他语言有限。Hibiki-Zero 有所帮助，但仍需训练数据。
- **资源成本。** 一个完整的 Moshi 会话占用一个 GPU 槽；不是廉价的共享租户部署模式。

## 交付

保存为 `outputs/skill-duplex-pipeline.md`。针对语音智能体工作负载，选择流水线 vs 全双工架构，并说明理由。

## 练习

1.  **简单。** 运行 `code/main.py`。它象征性地模拟了双流 + 内心独白架构。
2.  **中等。** 从 HuggingFace 拉取 Moshi，运行服务器，测试一次对话。测量从用户发言结束到 Moshi 响应开始的挂钟延迟。
3.  **困难。** 取你的课程 12 流水线智能体，与 Moshi 在 20 个匹配测试语句上比较 P50 延迟。撰写报告说明何时流水线在架构上仍然胜出。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 全双工 (Full-duplex) | 同时听和说 | 两个音频流在同一个模型上同时活动。 |
| 内心独白 (Inner monologue) | 模型的文本流 | Moshi 在生成音频输出的同时发出文本 token。 |
| 深度 Transformer (Depth transformer) | 码本间预测器 | 在一个 80 毫秒帧内预测 8 个码本的小型 Transformer。 |
| Mimi | Kyutai 的编解码器 | 12.5 Hz × 8 个码本；语义+声学；为 Moshi 提供动力。 |
| 流式 S2S (Streaming S2S) | 实时音频 → 音频 | 逐块翻译/对话，无流水线阶段。 |
| 反馈性插话 (Back-channeling) | “嗯”之类的反应 | Moshi 可以发出微小的确认，而不打断其发言轮次。 |

## 扩展阅读

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — 论文。
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — 无需对齐数据的流式翻译。
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM 规格。
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — 安装 + 服务器。
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — 封闭的商业同行。
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — 底层的 STT/TTS 框架。