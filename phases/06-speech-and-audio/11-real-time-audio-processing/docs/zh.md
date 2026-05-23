# 实时音频处理

> 批处理流水线处理文件。实时流水线必须在下一个20毫秒音频到达前处理完当前20毫秒。每个对话式AI、广播工作室和电话机器人都生死攸关地依赖于这个延迟预算。

**类型：** 构建
**语言：** Python, Rust
**先决条件：** 阶段6·02（频谱图），阶段6·04（ASR），阶段6·07（TTS）
**时间：** 约75分钟

## 问题所在

你需要一个感觉活生生的语音助手。人类对话的轮流响应延迟约为230毫秒（从静默到回应）。超过500毫秒感觉机械；超过1500毫秒感觉坏了。2026年一个完整的**听→理解→响应→说**循环的预算如下：

| 阶段 | 预算 |
|-------|--------|
| 麦克风 → 缓冲区 | 20 毫秒 |
| VAD | 10 毫秒 |
| ASR（流式） | 150 毫秒 |
| LLM（首个token） | 100 毫秒 |
| TTS（首个分块） | 100 毫秒 |
| 渲染 → 扬声器 | 20 毫秒 |
| **总计** | **约400 毫秒** |

Moshi（Kyutai, 2024）测得200毫秒全双工延迟。GPT-4o实时版（2024）约320毫秒。2022年的级联流水线交付延迟为2500毫秒。10倍的改进得益于三项技术：(1) 全面流式处理，(2) 带部分结果的异步流水线，(3) 可中断生成。

## 核心概念

![带环形缓冲区、VAD门控、中断的流式音频流水线](../assets/real-time.svg)

**帧 / 分块 / 窗口。** 实时音频以固定大小的块流动。常见选择：20毫秒（16kHz下320个采样点）。所有下游环节必须跟上这个节拍。

**环形缓冲区。** 固定大小的循环缓冲区。生产者线程写入新帧，消费者线程读取。防止在热路径上进行内存分配。大小 ≈ 最大延迟 × 采样率；2秒16kHz的环形缓冲区 = 32,000个采样点。

**VAD（语音活动检测）。** 当无人说话时，门控下游工作。Silero VAD 4.0（2024）在CPU上每30毫秒帧运行耗时<1毫秒。`webrtcvad` 是较旧的替代方案。

**流式ASR。** 在音频到达时生成部分转录文本的模型。Parakeet-CTC-0.6B在流式模式下（NeMo, 2024）以320毫秒延迟实现2-5%的词错误率。Whisper-Streaming（Macháček等人，2023）对Whisper进行分块以实现近流式，延迟约2秒。

**中断。** 当用户在助手说话时发言，你必须 (a) 检测到用户插入，(b) 停止TTS，(c) 丢弃剩余的LLM输出。所有这些必须在100毫秒内完成，否则用户会觉得助手是聋的。

**WebRTC Opus传输。** 20毫秒帧，48kHz，自适应比特率8-128kbps。浏览器和移动端的标准。LiveKit、Daily.co、Pion是2026年构建语音应用的技术栈。

**抖动缓冲区。** 网络包可能乱序/延迟到达。抖动缓冲区负责重新排序和平滑；太小 → 可听见的间断，太大 → 延迟。典型值60-80毫秒。

### 常见陷阱

- **线程争用。** Python的GIL + 沉重模型可能会饿死音频线程。使用C回调音频库（sounddevice, PortAudio），并让Python远离热路径。
- **采样率转换延迟。** 流水线内部的重采样会增加5-20毫秒延迟。要么预先重采样，要么使用零延迟重采样器（PolyPhase，`soxr_hq`）。
- **TTS预热。** 即使像Kokoro这样快速的TTS，首次请求也有100-200毫秒预热时间。缓存模型，并在第一个真实对话前用虚拟运行预热。
- **回声消除。** 没有AEC，TTS输出会重新进入麦克风，并触发ASR识别机器人自己的声音。WebRTC AEC3是开源的默认方案。

## 动手构建

### 步骤1：环形缓冲区

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量决定最大缓冲延迟。16kHz下32,000个采样点 = 2秒。

### 步骤2：VAD门控

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

在生产环境中替换为Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 步骤3：流式ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 步骤4：中断处理器

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

依赖于异步I/O和可取消的TTS流式传输。在音频轨道上执行WebRTC peerconnection.stop() 是规范做法。

## 使用指南

2026年技术栈：

| 层级 | 选择 |
|-------|------|
| 传输层 | LiveKit (WebRTC) 或 Pion (Go) |
| VAD | Silero VAD 4.0 |
| 流式ASR | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| LLM首个token | Groq, Cerebras, vLLM-streaming |
| 流式TTS | Kokoro 或 ElevenLabs Turbo v2.5 |
| 回声消除 | WebRTC AEC3 |
| 端到端原生 | OpenAI Realtime API 或 Moshi |

## 陷阱

- **为了安全缓冲500毫秒。** 缓冲区*就是*你的延迟下限。缩小它。
- **未固定线程优先级。** 音频回调运行在优先级低于UI的线程上 = 负载高时出现杂音。
- **TTS分块太小。** 小于200毫秒的分块会使声码器伪影可闻。320毫秒的分块是最佳平衡点。
- **没有抖动缓冲区。** 真实网络有抖动；没有平滑处理会导致爆音。
- **单次错误处理。** 音频流水线必须防崩溃。一个异常就会终止整个会话。

## 部署上线

保存为 `outputs/skill-realtime-designer.md`。设计一个实时音频流水线，为每个阶段设定具体的延迟预算。

## 练习

1. **简单。** 运行 `code/main.py`。模拟一个环形缓冲区 + 能量VAD；为一个10秒的模拟流打印各阶段延迟。
2. **中等。** 使用 `sounddevice`，构建一个透传循环，以20毫秒帧处理你的麦克风，并在每帧打印VAD状态。
3. **困难。** 使用 `aiortc` 构建一个完整的双工回声测试：浏览器 → WebRTC → Python → WebRTC → 浏览器。使用1kHz脉冲测量端到端延迟。

## 关键术语

| 术语 | 人们怎么说 | 它实际是什么意思 |
|------|-----------------|-----------------------|
| 环形缓冲区 | 循环队列 | 用于音频帧的固定大小、无锁（或单生产者单消费者锁）的FIFO。 |
| VAD | 静默门控 | 区分语音与非语音的模型或启发式方法。 |
| 流式ASR | 实时语音转文本 | 在音频到达时生成部分文本；有界的前瞻窗口。 |
| 抖动缓冲区 | 网络平滑器 | 重新排列乱序包的队列；典型值60-80毫秒。 |
| AEC | 回声消除 | 减去扬声器到麦克风的反馈路径。 |
| 插入 (Barge-in) | 用户中断 | 系统在TTS播放中途检测到用户语音；必须取消播放。 |
| 全双工 | 双向同时通信 | 用户和机器人可以同时说话；Moshi是全双工的。 |

## 延伸阅读

- [Macháček等人（2023）。Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 分块近流式Whisper。
- [Kyutai（2024）。Moshi](https://kyutai.org/Moshi.pdf) — 200毫秒延迟的全双工。
- [LiveKit Agents框架（2024）](https://docs.livekit.io/agents/) — 生产级音频代理编排。
- [Silero VAD仓库](https://github.com/snakers4/silero-vad) — 亚1毫秒VAD，Apache 2.0协议。
- [WebRTC AEC3论文](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 开源环境下的回声消除。