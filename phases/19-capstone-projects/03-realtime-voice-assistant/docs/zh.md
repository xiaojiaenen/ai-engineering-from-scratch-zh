# Capstone 03 — 实时语音助手（ASR 到 LLM 再到 TTS）

> 一个感觉自然的语音智能体，其端到端延迟应低于 800 毫秒，能感知用户何时停止说话，能处理打断，且能在不中断音频的情况下调用工具。Retell、Vapi、LiveKit Agents 和 Pipecat 在 2026 年都已达到这一标准。它们都采用相同的架构：流式 ASR、轮次检测器、流式 LLM 和流式 TTS，全部通过 WebRTC 连接，并在每个环节都设定了严格的延迟预算。构建这样一个系统，测量 WER、MOS 和误切率，并在丢包环境下运行测试。

**类型：** 综合项目
**语言：** Python（智能体 + 管道），TypeScript（网页客户端）
**先决条件：** 第 6 阶段（语音与音频），第 7 阶段（转换器），第 11 阶段（LLM 工程），第 13 阶段（工具），第 14 阶段（智能体），第 17 阶段（基础设施）
**涵盖阶段：** P6 · P7 · P11 · P13 · P14 · P17
**时间：** 30 小时

## 问题

语音是 2025-2026 年 AI 用户体验领域发展最快的方向。其技术门槛每个季度都在降低。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70 都使得低于 800 毫秒的首次音频输出成为可能。标准不仅仅是延迟，还包括交互体验：不打断用户、不被用户打断、从半句话的打断中恢复、在对话中途调用工具时不中断音频、在不稳定的移动网络下保持可用。

仅靠拼接三个 REST 调用是达不到要求的。其架构是端到端的流水线式流式处理。构建它后，失败模式会显现：为电话音频调优的 VAD 会因背景电视声音而触发；等待一个永远不会出现的标点符号的轮次检测器；在输出前缓冲了 400 毫秒的 TTS。本综合项目的任务是在负载下逐一解决这些问题，并发布一份延迟与质量报告。

## 概念

该流水线包含五个流式处理阶段：**音频输入**（来自浏览器或 PSTN 的 WebRTC 流）、**ASR**（来自 Deepgram Nova-3 或 faster-whisper 的流式部分转录）、**轮次检测**（VAD 加上一个读取部分转录以寻找完成信号的小型轮次检测模型）、**LLM**（在判断轮次完成后立即流式输出 token）、**TTS**（在首个 LLM token 生成后约 200 毫秒内流式输出音频）。

三个跨切面关注点：**打断处理**：当用户在智能体说话时开始发言，TTS 会立即取消，ASR 立即开始工作。**工具使用**：对话中途的函数调用（天气、日历）必须在不影响音频的情况下通过辅助通道运行；如果延迟超过 300 毫秒，智能体会预填充一个确认 token（“稍等...”）。**背压**：在丢包情况下，部分转录会被暂存，VAD 提高语音门限阈值，智能体避免在未收到确认的消息上发言。

衡量标准是定量的。在 Hamming VAD 基准测试（15 dB SNR）下，WER 低于 8%。100 次测量通话的首次音频输出 p50 延迟低于 800 毫秒。误切率低于 3%。TTS 的 MOS 高于 4.2。在单台 g5.xlarge 实例上支持 50 路并发通话。这些数字是交付成果。

## 架构

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## 技术栈

- **传输层：** LiveKit Agents 1.0 (WebRTC) 加上 Twilio PSTN 网关；Pipecat 0.0.70 作为备选框架
- **ASR：** Deepgram Nova-3（流式，首个部分转录延迟低于 300 毫秒）或自托管的 faster-whisper Whisper-v3-turbo
- **VAD：** Silero VAD v5 加上 LiveKit 轮次检测器（读取部分转录的小型转换器模型）
- **LLM：** OpenAI GPT-4o-realtime 用于紧密集成，Gemini 2.5 Flash Live，或级联的 Claude Haiku 4.5（流式补全，独立音频路径）
- **TTS：** Cartesia Sonic-2（首字节延迟最低），ElevenLabs Flash v3，或用于自托管的开源 Orpheus
- **工具：** FastMCP 辅助通道用于天气/日历/预订；如果工具调用耗时超过 300 毫秒，智能体会预先输出填充词
- **可观测性：** OpenTelemetry 语音跨度，带音频回放的 Langfuse 语音追踪
- **部署：** 用于自托管 Whisper + Orpheus 的单台 g5.xlarge 实例（24GB VRAM）；托管 API 用于最低延迟

## 构建步骤

1.  **WebRTC 会话。** 搭建一个 LiveKit 房间和一个流式传输麦克风音频的网页客户端。在服务器端，附加一个加入房间的智能体工作进程。

2.  **ASR 流式处理。** 将 20ms PCM 帧发送到 Deepgram Nova-3（或 GPU 上的 faster-whisper）。订阅部分和最终转录结果。记录每个部分转录的延迟。

3.  **VAD 和轮次检测器。** 在帧流上运行 Silero VAD v5。在语音结束事件触发时，使用最新的部分转录调用 LiveKit 轮次检测器。只有当 VAD 报告静音 500 毫秒且轮次检测器完成度评分大于 0.6 时，才确认“轮次完成”。

4.  **LLM 流式处理。** 在轮次完成时，使用当前对话历史加上最终转录开始 LLM 调用。流式输出 token。在首个 token 生成时，将其传递给 TTS。

5.  **TTS 流式处理。** Cartesia Sonic-2 流式返回音频块。首个音频块必须在首个 LLM token 生成后 200 毫秒内离开服务器。将音频块发送到 LiveKit 房间；客户端通过 WebRTC 抖动缓冲区播放。

6.  **打断处理。** 当 VAD 在 TTS 播放时检测到新的用户语音，立即取消 TTS 流，丢弃剩余的 LLM 输出，并重新启动 ASR。发布一个 `tts_canceled` 跨度。

7.  **工具辅助通道。** 将天气和日历注册为函数调用工具。当工具被调用时，立即并发执行；如果在 300 毫秒内未完成，让 LLM 输出“稍等，让我查一下”作为填充词；待工具返回后继续。

8.  **评估工具。** 录制 100 次通话。计算 WER（基于保留的转录文本）、误切率（在用户说话中途取消 TTS）、首次音频输出 p50 延迟、TTS MOS（人工或 NISQA 评估）以及抖动丢包测试（丢弃 3% 的数据包）。

9.  **负载测试。** 在单台 g5.xlarge 实例上使用合成呼叫方驱动 50 路并发通话。测量持续的首次音频输出 p95 延迟。

## 使用方法

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## 交付

`outputs/skill-voice-agent.md` 是交付成果。给定一个领域（客户支持、日程安排或信息亭），它将部署一个针对该领域调优了 ASR/VAD/LLM/TTS 管道以满足衡量标准的 LiveKit 智能体。评分标准：

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 端到端延迟 | 100 次录制通话的首次音频输出 p50 延迟低于 800 毫秒 |
| 20 | 轮次转换质量 | Hamming VAD 基准测试下的误切率低于 3% |
| 20 | 工具使用正确性 | 对话中途的工具调用能返回正确数据且不中断音频 |
| 20 | 丢包下的可靠性 | 注入 3% 数据包丢失时的 WER 和轮次转换稳定性 |
| 15 | 评估工具完整性 | 使用公开配置可复现的测量结果 |
| **100** | | |

## 练习

1.  在 g5.xlarge 实例上用 faster-whisper v3 turbo 替换 Deepgram Nova-3。测量延迟和 WER 的差距。识别出 CPU 与 GPU 决策在哪些地方起作用。

2.  添加一个打断仲裁策略：当用户在工具调用期间打断时，智能体该怎么做？比较三种策略（硬取消、完成工具调用后停止、排队等待下一轮）。

3.  运行对抗性轮次检测器测试：在用户句子中间给出长时间停顿。调优 VAD 静音阈值和轮次检测器评分阈值，以在不超过 900 毫秒延迟的前提下获得最低误切率。

4.  通过 Twilio 将同一个智能体部署到 PSTN 上。比较 PSTN 和 WebRTC 的首次音频输出延迟。解释抖动缓冲区和编解码器的差异。

5.  为非英语语言（日语、西班牙语）添加语音活动检测。测量 Silero VAD v5 对比语言特定微调模型的误触发率。

## 关键术语

| 术语 | 人们如何称呼它 | 它的确切含义 |
|------|-----------------|------------------------|
| 轮次检测 | “话语结束” | 一种分类器，基于 VAD 静音和部分转录，判断用户是否已说完 |
| 打断处理 | “打断处理” | 当 VAD 检测到新的用户语音时，中途取消 TTS 播放 |
| 首次音频输出 | “延迟” | 从用户停止说话到第一个音频数据包离开服务器的时间 |
| VAD | “语音门限” | 将音频帧分类为语音或静音的模型；Silero VAD v5 是 2026 年的默认选择 |
| 抖动缓冲区 | “音频平滑” | 客户端缓冲区，短暂持有数据包以吸收网络抖动 |
| 填充词 | “确认 token” | 智能体在工具响应慢时输出的短语，以避免沉默 |
| MOS | “平均意见分” | 语音质量感知评分；NISQA 是其自动化代理指标 |

## 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 参考 WebRTC 智能体框架
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 备选的 Python 优先流式智能体框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 集成语音模型的参考
- [Deepgram Nova-3 文档](https://developers.deepgram.com/docs) — 流式 ASR 参考
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD 参考模型
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 低延迟 TTS 参考
- [Retell AI 架构](https://docs.retellai.com) — 生产环境语音智能体架构
- [Vapi.ai 生产技术栈](https://docs.vapi.ai) — 备选生产环境参考