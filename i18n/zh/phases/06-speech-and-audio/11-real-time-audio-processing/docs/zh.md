# 实时音频处理

> 批处理管道处理一个文件。实时管道在每个 20 毫秒到来之前处理下一个 20 毫秒的音频。每个对话式 AI、广播工作室和电话机器人的生死都取决于这个延迟预算。

**类型：** Build
**语言：** Python
**前置知识：** Phase 6 · 02 (频谱图), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**时间：** 约 75 分钟

## 问题

你想要一个感觉有生命力的语音助手。人类对话的轮流发言延迟约为 230 毫秒（从静音到响应）。超过 500 毫秒会感觉像机器人；超过 1500 毫秒则感觉已损坏。2026 年完整 **听 → 理解 → 响应 → 说话** 循环的预算为：

| 阶段 | 预算 |
|-------|--------|
| 麦克风 → 缓冲区 | 20 毫秒 |
| VAD（语音活动检测） | 10 毫秒 |
| ASR（流式） | 150 毫秒 |
| LLM（首 token） | 100 毫秒 |
| TTS（首个片段） | 100 毫秒 |
| 渲染 → 扬声器 | 20 毫秒 |
| **总计** | **约 400 毫秒** |

Moshi（Kyutai，2024）实现了 200 毫秒全双工。GPT-4o-realtime（2024）约 320 毫秒。2022 年的级联管道交付延迟高达 2500 毫秒。10 倍的改进来自三种技术：（1）全链路流式传输，（2）带部分结果的异步流水线，（3）可中断生成。

## 概念

![带有环形缓冲区、VAD 门控、中断的流式音频管道](../assets/real-time.svg)

**帧 / 块 / 窗口。** 实时音频以固定大小的块流动。常见选择：20 毫秒（16 kHz 下为 320 个采样点）。所有下游模块必须跟上这个节拍。

**环形缓冲区。** 固定大小的循环缓冲区。生产者线程写入新帧，消费者线程读取。避免在热路径中分配内存。大小 ≈ 最大延迟 × 采样率；2 秒的 16 kHz 环形缓冲区 = 32,000 个采样点。

**VAD（语音活动检测）。** 当无人说话时阻断下游工作。Silero VAD 4.0（2024）在 CPU 上运行单个 30 毫秒帧不到 1 毫秒。`webrtcvad` 是较老的替代方案。

**流式 ASR。** 随着音频到达而发出部分转写的模型。Parakeet-CTC-0.6B 以流式模式运行（NeMo，2024）在 320 毫秒延迟下可实现 2–5% 的 WER。Whisper-Streaming（Macháček 等，2023）分块处理 Whisper 以实现近似流式传输，延迟约 2 秒。

**中断。** 当用户说话而助手正在说话时，你必须（a）检测到打断，（b）停止 TTS，（c）丢弃剩余的 LLM 输出。全部在 100 毫秒内完成，否则用户会感觉助手听不见。

**WebRTC Opus 传输。** 20 毫秒帧，48 kHz，自适应码率 8–128 kbps。浏览器和移动设备的标准。LiveKit、Daily.co、Pion 是 2026 年构建语音应用的堆栈。

**抖动缓冲区。** 网络数据包乱序 / 延迟到达。抖动缓冲区重新排序并平滑；太小会导致可听见的缺口，太大会增加延迟。典型值为 60–80 毫秒。

### 常见陷阱

- **线程争用。** Python 的 GIL + 重量级模型可能使音频线程饿死。使用 C 回调音频库（sounddevice、PortAudio），并将 Python 排除在热路径之外。
- **采样率转换延迟。** 在管道内进行重采样会增加 5–20 毫秒延迟。要么预先重采样，要么使用零延迟重采样器（PolyPhase、`soxr_hq`）。
- **TTS 预热。** 即使是 Kokoro 这样的快速 TTS，首次请求也有 100–200 毫秒的预热时间。缓存模型并在第一个真实对话前用虚拟输入预热它。
- **回声消除。** 没有 AEC，TTS 输出会重新进入麦克风并以机器人自己的声音触发 ASR。WebRTC AEC3 是开源默认方案。

```figure
nyquist-aliasing
```

## 构建

### 步骤 1：环形缓冲区

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

容量决定最大缓冲延迟。16 kHz 下 32,000 个采样点 = 2 秒。

### 步骤 2：VAD 门控

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

生产环境中替换为 Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 步骤 3：流式 ASR

```python
# 通过 NeMo 使用 Parakeet-CTC-0.6B 流式传输
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 毫秒，look_ahead_ms=80 毫秒
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 步骤 4：中断处理器

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # 打断
        # 然后传入流式 ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

关键在于异步 I/O 和可取消的 TTS 流式传输。WebRTC peerconnection.stop() 在音频轨道上是标准做法。

## 使用

2026 年技术栈：

| 层级 | 选择 |
|-------|------|
| 传输层 | LiveKit (WebRTC) 或 Pion (Go) |
| VAD | Silero VAD 4.0 |
| 流式 ASR | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| LLM 首 token | Groq、Cerebras、vLLM-streaming |
| 流式 TTS | Kokoro 或 ElevenLabs Turbo v2.5 |
| 回声消除 | WebRTC AEC3 |
| 端到端原生 | OpenAI Realtime API 或 Moshi |

## 陷阱

- **为安全起见缓冲 500 毫秒。** 缓冲区 *就是* 你的延迟下限。缩小它。
- **未固定线程优先级。** 音频回调线程优先级低于 UI 线程 = 高负载时出现卡顿。
- **TTS 片段过小。** 小于 200 毫秒的片段会使声码器伪影可闻。320 毫秒的片段是最佳选择。
- **没有抖动缓冲区。** 真实网络存在抖动；没有平滑处理会出现爆音。
- **单次错误处理。** 音频管道必须抗崩溃。一个异常即可终止会话。

## 交付

保存为 `outputs/skill-realtime-designer.md`。设计一个实时音频管道，并为每个阶段制定具体延迟预算。

## 练习

1. **简单。** 运行 `code/main.py`。模拟环形缓冲区 + 能量 VAD；打印伪造 10 秒流的各阶段延迟。
2. **中等。** 使用 `sounddevice`，构建一个直通循环，以 20 毫秒帧处理你的麦克风，并在每帧打印 VAD 状态。
3. **困难。** 使用 `aiortc` 构建全双工回声测试：浏览器 → WebRTC → Python → WebRTC → 浏览器。使用 1 kHz 脉冲测量端到端延迟。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|-----------------|-----------------------|
| 环形缓冲区 | 循环队列 | 用于音频帧的固定大小、无锁（或 SPSC 加锁）FIFO。 |
| VAD | 静音门控 | 通过模型或启发式方法标记语音与非语音。 |
| 流式 ASR | 实时语音识别 | 随着音频到达发出部分文本；有界前瞻。 |
| 抖动缓冲区 | 网络平滑器 | 对乱序数据包重新排序的队列；典型值 60–80 毫秒。 |
| AEC | 回声消除 | 减去扬声器到麦克风的反馈路径。 |
| Barge-in | 用户打断 | 系统在 TTS 播放中途检测到用户说话；必须取消播放。 |
| 全双工 | 同时双向 | 用户和机器人可以同时说话；Moshi 支持全双工。 |

## 延伸阅读

- [Macháček 等 (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 分块近流式 Whisper。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — 全双工 200 毫秒延迟。
- [LiveKit Agents 框架 (2024)](https://docs.livekit.io/agents/) — 生产级音频智能体编排。
- [Silero VAD 仓库](https://github.com/snakers4/silero-vad) — 亚毫秒级 VAD，Apache 2.0 许可。
- [WebRTC AEC3 论文](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 开源下的回声消除。
