# Capstone 03 — 实时语音助手（ASR → LLM → TTS）

> 一个体验到位的语音代理需要端到端延迟低于 800ms，知道用户何时停止说话，能处理插话（barge-in），且能调用工具而不停滞音频。Ret、Vapi、LiveKit Agents 和 Pipecat 在 2026 年均达到了这一标准。它们的做法相同：流式 ASR、话轮检测器、流式 LLM 和流式 TTS，全部通过 WebRTC 连接，每一跳都有激进的延迟预算。构建一个，测量 WER 和 MOS 以及误截止率，并在丢包条件下运行它。

**类型：** Capstone
**语言：** Python（代理 + 流水线）、TypeScript（Web 客户端）
**前置知识：** 第 6 阶段（语音与音频）、第 7 阶段（Transformer）、第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 14 阶段（代理）、第 17 阶段（基础设施）
**涉及阶段：** P6 · P7 · P11 · P13 · P14 · P17
**预计时间：** 30 小时

## 问题

语音是 2025-2026 年发展最快的 AI UX 类别。技术上限每个季度都在下降。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70 都让首音频出延迟低于 800ms 成为可能。标准不仅仅是延迟本身，而是交互体验：不打断用户、不被误打断、能从句子中途的插话中恢复、在对话中途调用工具且不停滞音频、在波动较大的移动网络下稳定运行。

单纯拼接三次 REST 调用无法实现。架构必须是端到端的管道流式处理。构建它之后，各种失败模式就会显现：为电话音频调优的 VAD 却对背景电视声触发、等待标点符号的 turn-detector 却等到天荒地老、TTS 在发出第一帧前先缓冲 400ms。这个 Capstone 的目标是在负载压力下逐个修复这些问题，并发布延迟与质量报告。

## 概念

流水线包含五个流式阶段：**音频输入**（来自浏览器或 PSTN 的 WebRTC）、**ASR**（来自 Deepgram Nova-3 或 faster-whisper 的流式部分转录）、**话轮检测**（VAD + 一个小型话轮检测模型，从部分转录中读取完成线索）、**LLM**（一旦判定话轮完成即开始流式输出 token）、**TTS**（首个 LLM token 出来后约 200ms 内开始流式输出音频）。

三个横切关注点。**插话（Barge-in）**：当用户开始说话时代理正在发声，TTS 立即取消，ASR 立刻拾取。**工具调用**：对话中途的函数调用（天气、日历）必须在旁路通道上运行，不能停滞音频；若延迟超过 300ms，代理会预填充一个确认 token（"请稍等…"）。**背压（Backpressure）**：在丢包情况下，部分转录会被缓存，VAD 提高语音门限，代理避免在未确认的消息上叠加声音。

衡量指标是量化的。在 15 dB SNR 的 Hamming VAD 基准测试上 WER 低于 8%。100 次实测通话中首音频出 p50 低于 800ms。误截止率低于 3%。TTS 的 MOS 高于 4.2。单台 g5.xlarge 支持 50 路并发通话。这些数字就是交付物。

## 架构

```
浏览器 / Twilio PSTN
        |
        v
   WebRTC / SIP 边缘
        |
        v
  LiveKit Agents 1.0（或 Pipecat 0.0.70）
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
        LLM（流式）
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS 流式
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     音频返回主叫方
            |
            v
   OpenTelemetry voice traces → Langfuse
```

## 技术栈

- 传输层：LiveKit Agents 1.0（WebRTC）加 Twilio PSTN 网关；Pipecat 0.0.70 作为备选框架
- ASR：Deepgram Nova-3（流式，首部分转录低于 300ms）或自托管 faster-whisper Whisper-v3-turbo
- VAD：Silero VAD v5 加 LiveKit turn-detector（小型 Transformer，读取部分转录）
- LLM：OpenAI GPT-4o-realtime（深度集成）、Gemini 2.5 Flash Live，或级联 Claude Haiku 4.5（流式补全，独立音频路径）
- TTS：Cartesia Sonic-2（首字节最低延迟）、ElevenLabs Flash v3，或自托管开源 Orpheus
- 工具：FastMCP 旁路通道，支持天气/日历/预约；若工具耗时超过 300ms，代理提前发出填充音
- 可观测性：OpenTelemetry voice spans，Langfuse voice traces 带音频回放
- 部署：单台 g5.xlarge（24GB VRAM）用于自托管 Whisper + Orpheus；使用托管 API 以获得最低延迟

```figure
ce-voice-latency
```

## 构建步骤

1. **WebRTC 会话。** 搭建一个 LiveKit 房间和一个流式传输麦克风音频的 Web 客户端。在服务器上，附加一个加入房间的 agent worker。

2. **ASR 流式处理。** 将 20ms PCM 帧送入 Deepgram Nova-3（或 GPU 上的 faster-whisper）。订阅部分转录和最终转录，记录每次部分转录的延迟。

3. **VAD 和话轮检测器。** 在帧流上运行 Silero VAD v5。在语音结束事件时，将最新的部分转录送入 LiveKit turn-detector。只有当 VAD 检测到 500ms 静音且 turn-detector 完成度评分 > 0.6 时，才提交"话轮完成"。

4. **LLM 流式输出。** 话轮完成后，以当前对话历史和最终转录发起 LLM 调用，流式输出 token。在第一个 token 时，立即移交给 TTS。

5. **TTS 流式输出。** Cartesia Sonic-2 流式返回音频块。第一块必须在首个 LLM token 后 200ms 内离开服务器。将块发送到 LiveKit 房间；客户端通过 WebRTC 抖动缓冲区播放。

6. **插话（Barge-in）。** 当 VAD 检测到 TTS 播放期间用户再次说话时，立即取消 TTS 流，丢弃剩余 LLM 输出，并重新启用 ASR。发布一个 `tts_canceled` span。

7. **工具旁路通道。** 将天气和日历注册为函数调用工具。调用时并行触发；若在 300ms 内未返回，让 LLM 先输出"请稍等，我查一下…"作为填充；工具返回后继续。

8. **评估工具。** 录制 100 次通话。计算 WER（对照保留的转录文本）、误截止率（TTS 在用户句子中途被取消）、首音频出 p50、TTS MOS（人工或 NISQA）、以及抖动丢包测试（丢弃 3% 的包）。

9. **负载测试。** 用合成呼叫方驱动单台 g5.xlarge 上的 50 路并发通话。测量持续状态下的首音频出 p95。

## 使用示例

```
主叫方："what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
话轮延迟：1040ms 用户停话 → 音频输出
```

## 交付物

`outputs/skill-voice-agent.md` 是交付物。针对指定领域（客户支持、日程安排或自助终端），启动一个 LiveKit agent，将 ASR/VAD/LLM/TTS 流水线调优至达标水平。评分标准：

| 权重 | 标准 | 如何衡量 |
|:---:|---|---|
| 25 | 端到端延迟 | 100 次录制通话中首音频出 p50 低于 800ms |
| 20 | 话轮切换质量 | 在 Hamming VAD 基准测试上误截止率低于 3% |
| 20 | 工具调用正确性 | 对话中途的工具调用返回正确数据且不停滞音频 |
| 20 | 丢包下的可靠性 | 注入 3% 丢包时 WER 和话轮切换稳定性 |
| 15 | 评估工具完整性 | 可复现的测量结果附带公开配置 |
| **100** | | |

## 练习

1. 将 Deepgram Nova-3 替换为 g5.xlarge 上的 faster-whisper v3 turbo。测量延迟和 WER 差距。识别 CPU 与 GPU 决策的关键影响点。

2. 增加插话仲裁策略：当用户在工具调用过程中插话时，代理应如何处理？比较三种策略（硬取消、完成工具再停止、排队下一个话轮）。

3. 运行对抗性话轮检测测试：给用户在句子中途设置长停顿。调优 VAD 静音阈值和 turn-detector 评分阈值，使误截止率最低，同时不超过 900ms。

4. 通过 Twilio 将同一代理部署到 PSTN。比较 PSTN 与 WebRTC 的首音频出延迟，解释抖动缓冲区和编解码器差异。

5. 为非英语语言（日语、西班牙语）添加语音活动检测。测量 Silero VAD v5 的误触发率与特定语言微调模型的对比。

## 关键术语

| 术语 | 人们常说的说法 | 实际含义 |
|------|----------------|----------|
| 话轮检测 | "话语结束" | 给定 VAD 静音和部分转录后，判断用户已说完话的分类器 |
| 插话（Barge-in） | "中断处理" | VAD 检测到用户新语音时，取消正在播放的 TTS |
| 首音频出 | "延迟" | 从用户停止说话到第一个音频包离开服务器的时间 |
| VAD | "语音门" | 将音频帧分类为语音或静音的模型；Silero VAD v5 是 2026 年的默认选择 |
| 抖动缓冲 | "音频平滑" | 客户端缓冲，短暂存储数据包以吸收网络波动 |
| 填充音（Filler） | "确认 token" | 工具响应慢时，代理发出的短短语以避免沉默 |
| MOS | "平均意见得分" | 感知语音质量评分；NISQA 是其自动化代理指标 |

## 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 参考 WebRTC 代理框架
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 备选 Python 优先的流式代理框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 集成语音模型的参考文档
- [Deepgram Nova-3 文档](https://developers.deepgram.com/docs) — 流式 ASR 参考
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD 参考模型
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 低延迟 TTS 参考
- [Retell AI 架构](https://docs.retellai.com) — 生产级语音代理架构
- [Vapi.ai 生产栈](https://docs.vapi.ai) — 另一套生产参考方案
