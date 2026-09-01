# 语音活动检测与话轮控制 — Silero、Cobra 与 Flush 技巧

> 每一个语音助手都取决于两个决策：用户现在是否在说话？他们是否说完了？VAD 回答第一个问题。话轮检测（VAD + 静音延后 + 语义端点模型）回答第二个问题。任何一个出错，你的助手要么打断用户，要么喋喋不休。

**类型：** 构建
**语言：** Python
**前置要求：** 第 6 阶段 · 11（实时音频），第 6 阶段 · 12（语音助手）
**时间：** 约 45 分钟

## 问题所在

语音助手在每个 20 ms 的音频块上做出三个独立决策：

1. **这一帧是语音吗？** — VAD。逐帧的二元判断。
2. **用户是否开始了新的语句？** — 起始检测。
3. **用户是否说完了？** — 终止检测（话轮结束）。

朴素的答案（能量阈值）在任何噪声下都会失效 — 交通声、键盘声、人群嘈杂声。2026 年的答案：Silero VAD（开源、深度学习）+ 话轮检测模型（语义端点）+ 经 VAD 校准的静音延后。

## 概念解析

![VAD 级联：能量 → Silero → 话轮检测器 → flush 技巧](../assets/vad-turn-taking.svg)

### 三层 VAD 级联

**第一层：能量门控。** 最便宜。阈值设为 -40 dBFS 的 RMS。过滤明显的静音，但会在超过阈值的任何噪声时触发。

**第二层：Silero VAD**（2020-2026，MIT 协议）。1M 参数。在 6000+ 种语言上训练。在单个 CPU 线程上运行约 30 ms 的块仅需 ~1 ms。在 5% FPR 下达到 87.7% TPR。开源默认选择。

**第三层：语义话轮检测器。** LiveKit 的话轮检测模型（2024-2026）或你自己的小型分类器。区分"句子中间的停顿"和"说完了"。使用语言上下文（语调 + 最近的词汇），而不仅仅是静音。

### 关键参数及其默认值

- **阈值。** Silero 输出概率；在 > 0.5（默认）或 > 0.3（敏感）时分类为语音。较低的阈值 = 更少的首字裁剪，更多的误报。
- **最小语音时长。** 拒绝短于 250 ms 的语音 — 通常是咳嗽或椅子噪音。
- **静音延后（终止检测）。** VAD 返回 0 后，等待 500-800 ms 再宣告话轮结束。太短 → 打断用户。太长 → 感觉迟钝。
- **前滚缓冲区。** 保留 VAD 触发前 300-500 ms 的音频。防止"嘿"被截断。

### Flush 技巧（Kyutai 2025）

流式 STT 模型具有前瞻延迟（Kyutai STT-1B 为 500 ms，STT-2.6B 为 2.5 s）。通常你需要在语音结束后等待那么长时间才能获得转录。Flush 技巧：当 VAD 触发语音结束信号时，**向 STT 发送 flush 信号**强制立即输出。STT 以约 4 倍实时速度处理，因此 500 ms 缓冲区只需约 125 ms 即可完成。

端到端：125 ms VAD + flush STT = 对话式延迟。

### 2026 年 VAD 对比

| VAD | 5% FPR 下的 TPR | 延迟 | 许可证 |
|-----|--------------|---------|---------|
| WebRTC VAD（Google，2013） | 50.0% | 30 ms | BSD |
| Silero VAD（2020-2026） | 87.7% | ~1 ms | MIT |
| Cobra VAD（Picovoice） | 98.9% | ~1 ms | 商业 |
| pyannote segmentation | 95% | ~10 ms | MIT 类 |

Silero 是正确的默认选择。Cobra 是合规性 / 精度的升级。仅靠能量的 VAD 在 2026 年的生产环境中没有立足之地。

```figure
sp-vad-cascade
```

## 构建它

### 步骤 1：能量门控

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 步骤 2：Python 中的 Silero VAD

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

### 步骤 3：话轮结束状态机

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

### 步骤 4：flush 技巧骨架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT（Kyutai、Deepgram、AssemblyAI）必须支持 flush 才能正常工作。Whisper 流式处理不支持 — 它是基于块的，总是等待块完成。

## 使用它

| 场景 | VAD 选择 |
|-----------|-----------|
| 开源、快速、通用场景 | Silero VAD |
| 商业呼叫中心 | Cobra VAD |
| 设备端（手机） | Silero VAD ONNX |
| 研究 / 说话人分离 | pyannote segmentation |
| 零依赖回退方案 | WebRTC VAD（遗留） |
| 需要高质量话轮终止 | Silero + LiveKit 话轮检测器分层 |

经验法则：除非真的没有其他选择，否则不要发布仅靠能量的 VAD。

## 陷阱

- **固定阈值。** 在安静环境下有效，在嘈杂环境下失效。要么在设备上校准，要么切换到 Silero。
- **静音延后太短。** 助手在句子中间打断用户。500-800 ms 是对话式语音的最佳区间。
- **静音延后太长。** 感觉迟钝。与目标用户进行 A/B 测试。
- **没有前滚缓冲区。** 用户音频的前 200-300 ms 丢失。始终保留滚动前滚缓冲区。
- **忽略语义端点检测。** "嗯，让我想想..." 包含长停顿。用户讨厌被中途打断。使用 LiveKit 的话轮检测器或类似方案。

## 交付

保存为 `outputs/skill-vad-tuner.md`。为特定工作负载选择 VAD 模型、阈值、延后时间、前滚缓冲区和话轮检测策略。

## 练习

1. **简单。** 运行 `code/main.py`。它模拟了语音 + 静音 + 语音 + 咳嗽的序列，并测试三个 VAD 层级。
2. **中等。** 安装 `silero-vad`，处理 5 分钟录音，调整阈值以最小化首字裁剪和误触发。报告精确率和召回率。
3. **困难。** 构建一个迷你话轮检测器：Silero VAD + 基于最近 10 个词嵌入的 3 层 MLP（使用 sentence-transformers）。在手标注的话轮结束数据集上训练。以 10% F1 超过仅用 Silero 的方案。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| VAD | 语音检测器 | 逐帧二元判断：这是语音吗？ |
| 话轮检测 | 终止检测 | VAD + 静音延后 + 语义端点。 |
| 静音延后 | 说话后的等待 | 宣告话轮结束前等待的时间；500-800 ms。 |
| 前滚 | 语音前缓冲区 | 在 VAD 触发前保留 300-500 ms 音频。 |
| Flush 技巧 | Kyutai 黑魔法 | VAD → flush-STT → 125 ms 而非 500 ms 延迟。 |
| 语义端点 | "他们是想停下来吗？" | ML 分类器，看的是词汇而不仅仅是静音。 |
| 5% FPR 下的 TPR | ROC 点 | 标准 VAD 基准测试；Silero 为 87.7%，WebRTC 为 50%。 |

## 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) — 参考开源 VAD。
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 商业精度领先者。
- [Kyutai — Unmute + flush 技巧](https://kyutai.org/stt) — 亚 200 ms 工程技巧。
- [LiveKit — 话轮检测](https://docs.livekit.io/agents/logic/turns/) — 生产环境中的语义端点检测。
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — 遗留基线。
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — 说话人分离级分段。
