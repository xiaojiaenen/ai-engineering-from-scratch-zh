# 语音活动检测与轮次转换 — Silero、Cobra 与 Flush 技巧

> 每个语音代理的成败取决于两个关键决策：用户当前是否正在说话，以及用户是否已说完。VAD（语音活动检测）回答第一个问题。轮次检测（VAD + 静默后延时 + 语义端点模型）回答第二个问题。任何一个判断出错，你的助手要么会打断用户，要么会说个不停。

**类型：** 构建指南
**编程语言：** Python
**前置要求：** 第 6 阶段 · 第 11 节（实时音频），第 6 阶段 · 第 12 节（语音助手）
**预计时间：** 约 45 分钟

## 问题描述

语音代理在处理每一个 20 毫秒音频片段时，需要做出三个截然不同的决策：

1.  **这一帧是语音吗？** — VAD。逐帧二值判断。
2.  **用户是否开始了新的讲话？** — 起始检测。
3.  **用户是否说完了？** — 端点检测（轮次结束）。

朴素的解决方案（能量阈值）在任何噪音环境下都会失效——交通声、键盘声、人群嘈杂声。2026 年的解决方案是：Silero VAD（开源、深度学习） + 轮次检测模型（语义端点检测） + 经 VAD 校准的静默后延时。

## 核心概念

![VAD 级联：能量 → Silero → 轮次检测器 → Flush 技巧](../assets/vad-turn-taking.svg)

### 三级 VAD 级联系统

**第一级：能量门。** 最廉价。在 -40 dBFS 设置 RMS 阈值。能过滤明显的静音，但任何超过阈值的噪音都会触发。

**第二级：Silero VAD** (2020-2026, MIT 许可)。100 万参数。在 6000 多种语言上训练。在单 CPU 线程上，处理每个 30 毫秒片段约需 1 毫秒。在 5% 误报率下真阳性率达 87.7%。开源默认选择。

**第三级：语义轮次检测器。** LiveKit 的轮次检测模型（2024-2026 年）或你自定义的小型分类器。能区分“句中停顿”和“说完话了”。利用语言上下文（语调 + 最近词语），而不仅仅是静默时长。

### 关键参数及其默认值

- **阈值。** Silero 输出概率值；大于 0.5（默认）或大于 0.3（高敏感度）则判定为语音。阈值越低 = 越少截断首个词，但误报越多。
- **最短语音时长。** 忽略短于 250 毫秒的语音——通常是咳嗽声或椅子噪音。
- **静默后延时（端点检测）。** VAD 返回 0 后，等待 500-800 毫秒再宣布轮次结束。太短 → 打断用户。太长 → 响应迟缓。
- **预录缓冲区。** 在 VAD 触发前保留 300-500 毫秒的音频。防止“嘿”字被截断。

### Flush 技巧（Kyutai，2025 年）

流式 STT 模型有前瞻延迟（Kyutai STT-1B 为 500 毫秒，STT-2.6B 为 2.5 秒）。通常，你需要在语音结束后等待那么长时间才能获得转录文本。Flush 技巧：当 VAD 检测到语音结束时，**向 STT 发送一个 Flush 信号**，强制其立即输出。STT 以约 4 倍实时速度处理，因此 500 毫秒的缓冲区在约 125 毫秒内即可处理完毕。

端到端：125 毫秒 VAD + Flush STT = 对话级延迟。

### 2026 年 VAD 对比

| VAD 模型 | 5% 误报率下的真阳性率 | 延迟 | 许可协议 |
|---------|-------------------|------|----------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 毫秒 | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 毫秒 | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 毫秒 | 商业 |
| pyannote segmentation | 95% | ~10 毫秒 | MIT 类 |

Silero 是合理的默认选择。Cobra 则是合规性/准确性的升级。在 2026 年的生产环境中，纯能量 VAD 已无立足之地。

## 动手构建

### 步骤 1：能量门

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 步骤 2：在 Python 中使用 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 步骤 3：轮次结束状态机

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 步骤 4：Flush 技巧骨架代码

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT 服务（Kyutai、Deepgram、AssemblyAI）必须支持 Flush 功能才能实现此技巧。Whisper 流式处理不支持——它是基于块的，总是需要等待数据块。

## 使用建议

| 场景 | VAD 选择 |
|------|----------|
| 开放、快速、通用 | Silero VAD |
| 商业呼叫中心 | Cobra VAD |
| 设备端（手机） | Silero VAD ONNX |
| 研究/说话人分离 | pyannote segmentation |
| 零依赖后备方案 | WebRTC VAD (旧版) |
| 需要轮次结束质量 | Silero + LiveKit 轮次检测器叠加使用 |

经验法则：除非实在没有其他选择，否则永远不要只使用纯能量 VAD。

## 常见陷阱

- **固定阈值。** 在安静环境有效，在嘈杂环境失效。要么在设备端进行校准，要么换用 Silero。
- **静默后延时太短。** 助手在用户说话中途打断。对于对话式语音，500-800 毫秒是最佳区间。
- **静默后延时太长。** 感觉响应迟缓。与目标用户进行 A/B 测试。
- **没有预录缓冲区。** 丢失了用户音频的前 200-300 毫秒。始终保留一个滚动预录缓冲区。
- **忽略语义端点检测。** “嗯，让我想想……”中包含长时间的停顿。用户讨厌在思考中途被打断。使用 LiveKit 的轮次检测器或类似方案。

## 交付使用

将此文档保存为 `outputs/skill-vad-tuner.md`。根据工作负载选择 VAD 模型、阈值、静默后延时、预录缓冲区大小以及轮次检测策略。

## 练习

1.  **简单。** 运行 `code/main.py`。它模拟了一个语音 + 静音 + 语音 + 咳嗽声的序列，并测试三级 VAD 系统。
2.  **中等。** 安装 `silero-vad`，处理一段 5 分钟的录音，调整阈值以最小化首个词截断和误报。报告精确率/召回率。
3.  **困难。** 构建一个迷你轮次检测器：Silero VAD + 基于最近 10 个词嵌入的 3 层 MLP（使用 sentence-transformers）。在一个手动标注的轮次结束数据集上进行训练。将 F1 分数提升 10%，超过仅使用 Silero 的效果。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| VAD | 语音检测器 | 逐帧二值判断：这是语音吗？ |
| 轮次检测 | 端点检测 | VAD + 静默后延时 + 语义端点。 |
| 静默后延时 | 说完后等待 | 宣布轮次结束前需要等待的时间；500-800 毫秒。 |
| 预录缓冲 | 语音前缓冲 | VAD 触发前保留的 300-500 毫秒音频。 |
| Flush 技巧 | Kyutai 的 hack | VAD → Flush STT → 125 毫秒，而非 500 毫秒延迟。 |
| 语义端点 | “他们是想停顿吗？” | 查看词语而非仅仅静默时长的 ML 分类器。 |
| 5% 误报率下的真阳性率 | ROC 曲线点 | VAD 标准基准测试；Silero 为 87.7%，WebRTC 为 50%。 |

## 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) — 参考级开源 VAD。
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 商业准确率领先者。
- [Kyutai — Unmute + Flush 技巧](https://kyutai.org/stt) — 低于 200 毫秒延迟的工程技巧。
- [LiveKit — 轮次检测](https://docs.livekit.io/agents/logic/turns/) — 生产环境中的语义端点检测。
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — 旧版基准线。
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — 说话人分离级别的分割。