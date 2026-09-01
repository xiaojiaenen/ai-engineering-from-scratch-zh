# 流式语音到语音对话 — Moshi、Hibiki 与全双工对话

> 2024-2026 年重新定义了语音 AI。Moshi 交付了一个单次模型，可在 200 ms 延迟下同时听和说。Hibiki 逐块完成语音到语音翻译。两者都放弃了 ASR → LLM → TTS 流水线，转向基于 Mimi 编解码 token 的统一全双工架构。这是新的参考设计。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 6 · 13（神经音频编解码），Phase 6 · 11（实时音频），Phase 7 · 05（完整 Transformer）
**时间：** ~75 分钟

## 问题所在

所有基于第 11 + 12 课构建的语音代理都有一个基础延迟下限，约 300–500 ms：VAD 触发、STT 处理、LLM 推理、TTS 生成。每个阶段都有其自身的最小延迟。你可以调优和并行化，但流水线结构限制了你的上限。

Moshi（Kyutai，2024–2026）提出了一个不同的问题：如果没有流水线呢？如果有一个模型直接接收音频并持续输出音频，而文本只是中间"内心独白"而非必需环节呢？

答案就是**全双工语音到语音**。理论延迟 160 ms（80 ms Mimi 帧 + 80 ms 声学延迟）。在单张 L4 GPU 上的实际延迟为 200 ms。这仅为业界最佳流水线语音代理延迟的一半。

## 核心概念

![Moshi 架构：两条并行 Mimi 流 + 内心独白文本](../assets/moshi-hibiki.svg)

### Moshi 架构

**输入。** 两条 Mimi 编解码流，均为 12.5 Hz × 8 codebooks：

- 流 1：用户音频（Mimi 编码，持续到达）
- 流 2：Moshi 自身的音频（由 Moshi 生成）

**Transformer。** 一个 7B 参数的时域 Transformer 同时处理两条音频流和一条文本"内心独白"流。在每个 80 ms 步骤中，它：

1. 消费最新的用户 Mimi token（8 个 codebook）。
2. 消费最新的 Moshi Mimi token（8 个 codebook，按生成顺序）。
3. 生成下一个 Moshi 文本 token（内心独白）。
4. 生成下一个 Moshi Mimi token（8 个 codebook，通过小型深度 Transformer）。

三条流——用户音频、Moshi 音频、Moshi 文本——并行运行。Moshi 可以在说话的同时听用户讲话；可以在用户打断时中断自己；可以在不中断主话语的情况下进行背景回应（"嗯"）。

**深度 Transformer。** 在一帧内部，8 个 codebook 并非并行预测——它们之间存在 inter-codebook 依赖关系。一个小型 2 层"深度 Transformer"在 80 ms 内按顺序预测它们。这是 AR 编解码语言模型的标准因式分解方法（VALL-E、VibeVoice 也采用此方法）。

### 为什么内心独白文本有帮助

若无显式文本，模型必须在隐式层面通过声学流来建模语言。Moshi 的洞察是：强制模型与音频同步输出文本 token。文本流本质上就是 Moshi 所说内容的 transcript。这提升了语义连贯性，便于替换语言模型头部，并免费获得 transcript。

### Hibiki：流式语音到语音翻译

相同架构，在翻译对上训练。源语言音频输入，目标语言音频输出，持续进行。Hibiki-Zero（2026 年 2 月）消除了对词级对齐训练数据的需求——使用句子级数据 + GRPO 强化学习进行延迟优化。

初始支持四种语言对；适配新语言约需 1000 小时数据。

### Kyutai 完整技术栈（2026）

- **Moshi** — 全双工对话（法语优先，英语支持良好）
- **Hibiki / Hibiki-Zero** — 同声传译
- **Kyutai STT** — 流式 ASR（500 ms 或 2.5 s 前瞻）
- **Kyutai Pocket TTS** — 100M 参数 TTS，可在 CPU 上运行（2026 年 1 月）
- **Unmute** — 在上述公开服务器上组合这些组件的完整流水线

L40S GPU 上的吞吐量：64 路并发会话，达 3× 实时速度。

### Sesame CSM — 近亲

Sesame CSM（2025）采用了类似思路——以 Llama-3 为骨干，搭配 Mimi 编解码头部。但 CSM 是单向的（接收上下文 + 文本，产出语音），而非全双工。它是市场上最佳的"语音存在感"TTS；与 Moshi 的全双工能力并非同一类别。

### 2026 年性能数据

| 模型 | 延迟 | 用途 | 许可 |
|------|------|------|------|
| Moshi | 200 ms（L4） | 全双工英语/法语对话 | CC-BY 4.0 |
| Hibiki | 12.5 Hz 帧率 | 法语 ↔ 英语流式翻译 | CC-BY 4.0 |
| Hibiki-Zero | 相同 | 5 种语言对，无需对齐数据 | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | 上下文条件 TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | 闭源，OpenAI API | 商业 |
| Gemini 2.5 Live | ~350 ms | 闭源，Google API | 商业 |

```figure
sp-fullduplex
```

## 动手实现

### 步骤 1：接口

Moshi 暴露了一个 WebSocket 服务器，接收 80 ms 的 Mimi 编码音频块，并返回 80 ms 的 Mimi 编码音频块。双向如此，持续运行。

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

两个方向同时运行。Python asyncio 或 Rust futures 是标准传输方式。

### 步骤 3：训练目标（概念性）

对于每个 80 ms 帧 `t`：

- 输入：`user_mimi[0..t]`、`moshi_mimi[0..t-1]`、`moshi_text[0..t-1]`
- 预测：`moshi_text[t]`，然后 `moshi_mimi[t, codebook_0..7]`

文本先于音频预测（内心独白）；音频在深度 Transformer 内按 codebook 顺序预测。

### 步骤 4：Moshi 的优势与局限

Moshi 擅长的场景：

- 在廉价硬件上实现 250 ms 以内的端到端延迟。
- 自然的背景回应和打断处理。
- 无需流水线胶水代码。

Moshi 不擅长的场景：

- 工具调用（未为此训练；需要单独的 LLM 路径）。
- 长程推理（Moshi 是一个约 8B 参数的对话模型，而非 Claude/GPT-4）。
- 小众话题的事实准确性。
- 大多数企业级生产场景（2026 年仍使用流水线）。

## 如何使用

| 场景 | 选择 |
|------|------|
| 最低延迟语音助手 | Moshi |
| 实时翻译通话 | Hibiki |
| 语音演示 / 研究 | Moshi、CSM |
| 带工具的企业代理 | 流水线（第 12 课），非 Moshi |
| 上下文中定制音色 TTS | Sesame CSM |
| 任意语言语音到语音 | GPT-4o Realtime 或 Gemini 2.5 Live（商业） |

## 常见陷阱

- **工具调用能力有限。** Moshi 是对话模型，而非代理框架。需结合流水线实现工具调用。
- **特定音色条件。** Moshi 使用单一训练 persona；音色克隆是单独的训练任务。
- **语言覆盖范围。** 法语 + 英语表现优秀；其他语言有限。Hibiki-Zero 有所帮助，但你仍需提供训练数据。
- **资源成本。** 完整 Moshi 会话占用一个 GPU 插槽；不适合廉价的共享租户部署模式。

## 交付物

保存为 `outputs/skill-duplex-pipeline.md`。为你的语音代理工作负载选择流水线架构还是全双工架构，并给出理由。

## 练习

1. **简单。** 运行 `code/main.py`。它以符号方式模拟双流 + 内心独白架构。
2. **中等。** 从 HuggingFace 拉取 Moshi，运行服务器，测试一轮对话。测量从用户讲话结束到 Moshi 开始回应的墙钟延迟。
3. **困难。** 取你的第 12 课流水线代理，在 20 条配对测试语句上比较 P50 延迟与 Moshi 的表现。总结流水线在哪些情况下架构上仍然占优。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 全双工 | 边听边说 | 同一模型上同时活跃两条音频流。 |
| 内心独白 | 模型的文本流 | Moshi 在与音频输出同步时输出文本 token。 |
| 深度 Transformer | inter-codebook 预测器 | 小型 Transformer，在一个 80 ms 帧内按序预测 8 个 codebook。 |
| Mimi | Kyutai 的编解码器 | 12.5 Hz × 8 codebook；语义 + 声学；驱动 Moshi。 |
| 流式 S2S | 实时音频→音频 | 逐块翻译/对话，无流水线环节。 |
| 背景回应 | "嗯"类反应 | Moshi 可发出简短确认而不打断自身话轮。 |

## 延伸阅读

- [Défossez 等（2024）。Moshi — 语音-文本基础模型](https://arxiv.org/html/2410.00037v2) — 论文原文。
- [Kyutai Labs（2026）。Hibiki-Zero](https://arxiv.org/abs/2602.12345) — 无需对齐数据的流式翻译。
- [Sesame（2025）。跨越语音恐怖谷](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM 规范。
- [Kyutai — Moshi 仓库](https://github.com/kyutai-labs/moshi) — 安装与服务器部署。
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — 闭源商业竞品。
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — 底层 STT/TTS 框架。
