# 语音智能体：Pipecat 与 LiveKit

> 语音智能体在2026年已成为主流的生产应用类别。Pipecat 为您提供一个基于 Python 的帧处理流水线（VAD → STT → LLM → TTS → 传输）。LiveKit Agents 通过 WebRTC 将 AI 模型与用户连接。对于优化良好的技术栈，生产环境的端到端延迟目标为 450-600 毫秒。

**类型：** 学习  
**语言：** Python（标准库）  
**前置要求：** 第14阶段 · 01（智能体循环），第14阶段 · 12（工作流模式）  
**时间：** 约60分钟

## 学习目标

- 描述 Pipecat 的基于帧的流水线：**下行流**（source→sink）和**上行流**（控制）。
- 列出标准语音流水线的各个阶段以及 Pipecat 支持的传输方式。
- 解释 LiveKit Agents 的两种语音智能体类（MultimodalAgent, VoicePipelineAgent）及其各自的适用场景。
- 总结 2026 年的生产延迟预期，以及它们如何驱动架构决策。

## 问题所在

语音智能体不仅仅是接了 TTS 的文本循环。延迟预算极其苛刻（约 600 毫秒），部分音频是常态，轮次检测依赖模型，传输方式从电话 SIP 到 WebRTC 不一而足。要么构建一个基于帧的流水线（Pipecat），要么依托一个平台（LiveKit）。

## 核心概念

### Pipecat (pipecat-ai/pipecat)

- Python 基于帧的流水线框架。
- `Frame` → `FrameProcessor` 链。
- 两种流向：
  - **下行流** — 源 → 接收器（音频输入，TTS 输出）。
  - **上行流** — 反馈与控制（取消、指标、打断）。
- `PipelineTask` 通过事件（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）和用于指标/追踪/RTVI 的观察者管理生命周期。

典型流水线：

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

传输方式：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 增加了结构化对话（状态机）。Pipecat Cloud 是托管运行时。

### LiveKit Agents (livekit/agents)

- 通过 WebRTC 将 AI 模型与用户连接。
- 核心概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两种语音智能体类：
  - **MultimodalAgent** — 通过 OpenAI Realtime 或等效服务直接传输音频。
  - **VoicePipelineAgent** — STT → LLM → TTS 级联；提供文本层面的控制。
- 通过 Transformer 模型实现语义轮次检测。
- 原生集成 MCP。
- 通过 SIP 支持电话服务。
- 通过 LiveKit Inference 免密钥使用 50+ 模型；通过插件可再扩展 200+ 模型。

### 商业平台

Vapi（在优化良好的高级技术栈上约 450-600 毫秒）和 Retell（跨 180 次测试通话的端到端约 600 毫秒）都建立在这些基础之上。如果您希望拥有一个无需 WebRTC 团队的托管语音栈，可以选择平台。

### 此模式易出错之处

- **未处理打断机制。** 用户中断；智能体继续发言。需要在 Pipecat 中通过上行流取消帧实现，LiveKit 中有等效机制。
- **忽略 STT 置信度。** 低置信度的转录文本被当作金科玉律送入 LLM。应基于置信度进行门控或请求确认。
- **TTS 在句子中间被截断。** 当流水线在语音中途取消时，TTS 需要知道并停止音频输出。
- **忽略延迟预算。** 每个组件都会增加 50-200 毫秒延迟。上线前务必核算整条链路的延迟总和。

### 2026 年典型延迟

- VAD：20–60 毫秒
- STT 部分结果：100–250 毫秒
- LLM 首个 token：150–400 毫秒
- TTS 首段音频：100–200 毫秒
- 传输往返时延：30–80 毫秒

端到端 450-600 毫秒属于高级水平。800-1200 毫秒很常见。超过 1500 毫秒会感觉体验很差。

## 动手构建

`code/main.py` 是一个基于帧的玩具流水线，包含：

- `Frame` 类型（音频、转录、文本、tts_audio、控制）。
- `Processor` 接口及 `process(frame)`。
- 一个五阶段流水线（VAD → STT → LLM → TTS → 传输）作为脚本化处理器。
- 一个上行流取消帧，用于演示打断机制。

运行它：

```
python3 code/main.py
```

追踪信息显示了正常流程以及一个在 TTS 语音中途停止它的打断取消操作。

## 如何使用

- 使用 **Pipecat** 实现完全控制 — 自定义处理器、Python 优先、可插拔的提供者。
- 使用 **LiveKit Agents** 进行以 WebRTC 优先的部署和电话集成。
- 使用 **Vapi / Retell** 获取无需 WebRTC 团队的托管语音智能体。
- 使用 **OpenAI Realtime / Gemini Live** 实现直接的音频输入/输出（MultimodalAgent）。

## 部署上线

`outputs/skill-voice-pipeline.md` 会搭建一个 Pipecat 风格的语音流水线，包含 VAD + STT + LLM + TTS + 传输，并配备打断处理机制。

## 练习

1.  为您的玩具流水线添加一个指标观察者：统计每个阶段每秒处理的帧数。延迟主要累积在哪里？
2.  实现基于置信度的 STT 门控：低于阈值时，请求用户“您能再说一遍吗？”
3.  添加语义轮次检测：一个简单规则 — 如果转录文本以“？”结尾，则判定为轮次结束。
4.  阅读 Pipecat 的传输文档。将标准库传输替换为 SmallWebRTCTransport 配置（存根）。
5.  针对相同查询，测量 OpenAI Realtime 与 STT+LLM+TTS 级联方案的延迟差异。文本层面的控制会带来多少延迟成本？

## 核心术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 帧 | "事件" | 流水线中的类型化数据单元（音频、转录、文本、控制） |
| 处理器 | "流水线阶段" | 带有 process(frame) 方法的处理器 |
| 下行流 | "前向流" | 源到接收器：音频输入，语音输出 |
| 上行流 | "反馈流" | 控制：取消、指标、打断 |
| VAD | "语音活动检测" | 检测用户何时在说话 |
| 语义轮次检测 | "智能的轮次结束判断" | 基于模型的决策，判断用户是否说完 |
| MultimodalAgent | "直接音频智能体" | 音频输入，音频输出；中间无文本 |
| VoicePipelineAgent | "级联智能体" | STT + LLM + TTS；提供文本层面的控制 |

## 扩展阅读

- [Pipecat 文档](https://docs.pipecat.ai/getting-started/introduction) — 基于帧的流水线、处理器、传输
- [LiveKit Agents 文档](https://docs.livekit.io/agents/) — WebRTC + 语音基础组件
- [Vapi](https://vapi.ai/) — 托管语音平台
- [Retell AI](https://www.retellai.com/) — 托管语音，提供延迟基准测试