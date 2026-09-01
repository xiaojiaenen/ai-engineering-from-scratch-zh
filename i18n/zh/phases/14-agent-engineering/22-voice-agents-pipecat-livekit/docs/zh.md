# 语音代理：Pipecat 与 LiveKit

> 语音代理是 2026 年的一类一等生产级类别。Pipecat 提供基于 Python 的帧级流水线（VAD → STT → LLM → TTS → 传输）。LiveKit Agents 通过 WebRTC 将 AI 模型连接到用户。生产级延迟目标为端到端 450–600ms（高端架构）。

**类型：** 学习
**语言：** Python（标准库）
**前置知识：** Phase 14 · 01（代理循环）、Phase 14 · 12（工作流模式）
**耗时：** 约 60 分钟

## 学习目标

- 描述 Pipecat 的帧级流水线：DOWNSTREAM（源→汇）和 UPSTREAM（控制）。
- 说出语音流水线的规范阶段及 Pipecat 支持的传输协议。
- 解释 LiveKit Agents 的两种语音代理类（MultimodalAgent、VoicePipelineAgent）及各自的适用场景。
- 总结 2026 年生产级延迟预期及其如何驱动架构选择。

## 问题

语音代理不是在文本循环上硬挂 TTS 那么简单。延迟预算极其严苛（约 600ms），部分音频是默认行为，轮次检测本身是一个模型，传输协议从电话 SIP 到 WebRTC 不等。你要么自己搭建帧级流水线（Pipecat），要么依赖平台（LiveKit）。

## 概念

### Pipecat（pipecat-ai/pipecat）

- Python 帧级流水线框架。
- `Frame` → `FrameProcessor` 链。
- 两个流向：
  - **DOWNSTREAM** — 源 → 汇（音频输入，TTS 输出）。
  - **UPSTREAM** — 反馈与控制（取消、指标、插话）。
- `PipelineTask` 通过事件（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）管理生命周期，并支持观察者用于指标/追踪/RTVI。

典型流水线：

```
VAD（Silero）→ STT → LLM（上下文交替用户/助手）→ TTS → 传输
```

传输：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 提供结构化对话（状态机）。Pipecat Cloud 是托管运行时。

### LiveKit Agents（livekit/agents）

- 通过 WebRTC 将 AI 模型连接到用户。
- 核心概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两种语音代理类：
  - **MultimodalAgent** — 通过 OpenAI Realtime 或直接音频处理。
  - **VoicePipelineAgent** — STT → LLM → TTS 级联；提供文本级控制。
- 基于 Transformer 模型的语义轮次检测。
- 原生 MCP 集成。
- 通过 SIP 支持电话。
- 通过 LiveKit Inference 支持 50+ 模型且无需 API 密钥；通过插件再支持 200+。

### 商业平台

Vapi（优化后的高端堆栈约 450–600ms）和 Retell（180 次测试通话端到端约 600ms）构建在这些基础之上。当你需要一个托管语音堆栈而不想维护 WebRTC 团队时，选择平台。

### 此模式的常见错误

- **缺少插话处理。** 用户打断时，代理仍在说话。Pipecat 中需要 UPSTREAM cancel 帧，LiveKit 中需等效处理。
- **忽略 STT 置信度。** 低置信度转录被当作真理喂给 LLM。应设置置信度门控或请求确认。
- **TTS 句子中途截断。** 当流水线中途取消时，TTS 需要知道或主动截断音频。
- **忽略延迟预算。** 每个组件增加 50–200ms。在发布前对你的链路求和。

### 典型 2026 年延迟

- VAD：20–60ms
- STT 部分输出：100–250ms
- LLM 首 token：150–400ms
- TTS 首音频：100–200ms
- 传输往返：30–80ms

端到端 450–600ms 属高端水平。800–1200ms 较常见。超过 1500ms 会感觉卡顿。

```figure
voice-pipeline
```

## 动手构建

`code/main.py` 是一个基于帧的玩具流水线，包含：

- `Frame` 类型（音频、转录、文本、tts_audio、控制）。
- 带 `process(frame)` 的 `Processor` 接口。
- 以脚本化处理器形式实现的五阶段流水线（VAD → STT → LLM → TTS → 传输）。
- 一个 UPSTREAM cancel 帧用于演示插话。

运行它：

```
python3 code/main.py
```

轨迹显示正常流程和一次插话取消，可在 TTS 句中途停止。

## 如何使用

- **Pipecat** — 完全控制：自定义处理器、Python 优先、可插拔提供商。
- **LiveKit Agents** — WebRTC 优先部署和电话支持。
- **Vapi / Retell** — 托管语音代理，无需 WebRTC 团队。
- **OpenAI Realtime / Gemini Live** — 直接音频输入/输出（MultimodalAgent）。

## 上线部署

`outputs/skill-voice-pipeline.md` 搭建了一个 Pipecat 风格的语音流水线，包含 VAD + STT + LLM + TTS + 传输以及插话处理。

## 练习

1. 为你的玩具流水线添加指标观察者：统计每秒每阶段的帧数。延迟在何处累积？
2. 实现置信度门控 STT：低于阈值时，请求"能否重复一遍？"
3. 添加语义轮次检测：简单规则——若转录以"?"结尾，则视为一轮结束。
4. 阅读 Pipecat 的传输文档。将 stdlib 传输替换为 SmallWebRTCTransport 配置（stub）。
5. 在同一查询上对比 OpenAI Realtime 与 STT+LLM+TTS 级联。文本级控制带来了多少延迟成本？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Frame | "事件" | 流水线中的类型化数据单元（音频、转录、文本、控制） |
| Processor | "流水线阶段" | 带 process(frame) 的处理程序 |
| DOWNSTREAM | "正向流" | 源到汇：音频输入，语音输出 |
| UPSTREAM | "反馈流" | 控制：取消、指标、插话 |
| VAD | "语音活动检测" | 检测用户何时在说话 |
| 语义轮次检测 | "智能轮次结束" | 基于模型的判断用户已结束发言 |
| MultimodalAgent | "直接音频代理" | 音频进，音频出；中间无文本 |
| VoicePipelineAgent | "级联代理" | STT + LLM + TTS；文本级控制 |

## 延伸阅读

- [Pipecat 文档](https://docs.pipecat.ai/getting-started/introduction) — 帧级流水线、处理器、传输
- [LiveKit Agents 文档](https://docs.livekit.io/agents/) — WebRTC + 语音原语
- [Vapi](https://vapi.ai/) — 托管语音平台
- [Retell AI](https://www.retellai.com/) — 托管语音，经延迟基准测试
