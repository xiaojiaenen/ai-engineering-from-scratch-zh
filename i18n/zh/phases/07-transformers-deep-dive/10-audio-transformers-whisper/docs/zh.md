# Audio Transformers — Whisper 架构

> 音频是随时间变化的频率图像。Whisper 是一个 ViT，能够吞入 mel 频谱图并回吐文本。

**类型：** 学习
**语言：** Python
**前置知识：** Phase 7 · 05（完整 Transformer）、Phase 7 · 08（Encoder-Decoder）、Phase 7 · 09（ViT）
**时间：** 约 45 分钟

## 问题背景

在 Whisper（OpenAI，Radford 等人 2022）出现之前，最先进的自动语音识别（ASR）方案是 wav2vec 2.0 和 HuBERT——即自监督特征提取器加上微调的分类头。质量高，但需要昂贵的数据管线，且在领域间泛化能力差。多语言语音识别需要按语言族分别训练模型。

Whisper 做了三个关键押注：

1. **海量训练数据。** 从互联网抓取的 68 万小时弱标注音频，覆盖 97 种语言。没有干净的学术语料库，没有音素标签。
2. **多任务单模型。** 一个解码器通过任务 token 联合训练转录、翻译、语音活动检测、语言识别和时间戳预测。
3. **标准 encoder-decoder transformer。** Encoder 消费 log-mel 频谱图，Decoder 自回归生成文本 token。无需 vocoder、CTC 或 HMM。

结果：Whisper large-v3 在口音、噪音和零干净标注数据的语言上都具有鲁棒性。它是 2026 年所有开源语音助手及大多数商业产品的默认语音前端。

## 概念

![Whisper 流程：音频 → mel → encoder → decoder → 文本](../assets/whisper.svg)

### 步骤 1 — 重采样 + 分帧

音频采样率为 16 kHz。裁剪/填充至 30 秒。计算 log-mel 频谱图：80 个 mel 频带，10 ms 步幅 → 约 3,000 帧 × 80 个特征。这就是 Whisper 所见的"输入图像"。

### 步骤 2 — 卷积前端

两个 kernel=3、stride=2 的 Conv1D 层将 3,000 帧压缩至 1,500 帧。在不大幅增加参数的情况下将序列长度减半。

### 步骤 3 — Encoder

一个 24 层（large 版本）transformer encoder，处理 1,500 个时间步。使用正弦位置编码、自注意力和 GELU FFN。输出 1,500 × 1,280 的隐状态。

### 步骤 4 — Decoder

一个 24 层 transformer decoder。它从 BPE 词表中自回归地生成 token，该词表是 GPT-2 词表的超集，并加入了少量音频专用的特殊 token。

### 步骤 5 — 任务 Token

解码器提示以控制 token 开头，告诉模型执行什么任务：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型是在这个约定下训练的。你通过前缀控制任务。这是 2026 年相当于指令调优（instruction-tuning）的技术，但应用于语音领域。

### 步骤 6 — 输出

使用宽度为 5 的 beam search 和 log-prob 阈值。当缺少 `<|notimestamps|>` token 时，每 0.02 秒音频预测一次时间戳。

### Whisper 模型规格

| 模型 | 参数 | 层数 | d_model | Head 数 | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB（4层 decoder） |

Large-v3-turbo（2024 年）将 decoder 从 32 层砍到 4 层。解码速度提升 8×，WER 仅回退不到 1 个百分点。正是这次解码速度的释放，让 Whisper-turbo 成为 2026 年实时语音代理的默认选择。

### Whisper 不做的事情

- 不做说话人分离（diarization，判断谁在说话）。可搭配 pyannote 使用。
- 原生不支持实时流式处理——30 秒窗口是固定的。现代封装（`faster-whisper`、`WhisperX`）通过 VAD + 重叠窗口实现流式。
- 超出 30 秒的长文本需要外部分块处理。实践中效果很好，因为人类语音在转录时很少需要长距离上下文。

### 2026 年生态

| 任务 | 模型 | 备注 |
|------|-------|-------|
| 英语 ASR | Whisper-turbo、Moonshine | Moonshine 在边缘设备上快 4 倍 |
| 多语言 ASR | Whisper-large-v3 | 支持 97 种语言 |
| 流式 ASR | faster-whisper + VAD | 可实现 150 ms 延迟目标 |
| TTS | Piper、XTTS-v2、Kokoro | Encoder-decoder 模式，但受 Whisper 启发 |
| 音频 + 语言 | AudioLM、SeamlessM4T | 文本 token 和音频 token 统一在一个 transformer 中 |

```figure
n5-mel-decode
```

## 动手构建

见 `code/main.py`。我们不训练 Whisper——我们构建 log-mel 频谱图管线 + 任务 token 提示格式化器。这些才是生产中真正会触碰的部分。

### 步骤 1：合成音频

生成一个 1 秒长、440 Hz 的正弦波，采样率 16 kHz。共 16,000 个采样点。

### 步骤 2：Log-mel 频谱图（简化版）

完整的 mel 频谱图需要 FFT。我们用一个简化的分帧 + 每帧能量版本来展示管线，无需 `librosa`：

```python
def frame_signal(x, frame_size=400, hop=160):
    # 将信号按固定帧长和步幅切分为重叠帧
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

帧长 = 25 ms，步幅 = 10 ms。与 Whisper 的窗口设置一致。每帧能量在此替代 mel 频带来进行教学演示。

### 步骤 3：填充至 30 秒

Whisper 始终处理 30 秒的片段。将频谱图填充（或裁剪）到 3,000 帧。

### 步骤 4：构建提示 Token

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    # 构建任务控制前缀
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

整个任务控制界面就是这样：一个 4 个 token 的前缀。

## 使用方式

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快、兼容 OpenAI 的用法：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026 年何时选用 Whisper：**

- 单一模型的多语言 ASR。
- 对嘈杂、多样化的音频进行鲁棒转录。
- 研究 / 快速原型 ASR——最快的起步方案。

**何时选用其他方案：**

- 边缘设备上超低延迟流式处理——同质量下 Moonshine 比 Whisper 快 4 倍。
- 需要 <200 ms 延迟的实时对话 AI——专用流式 ASR。
- 说话人分离——Whisper 不做此功能，需搭配 pyannote。

## 交付输出

见 `outputs/skill-asr-configurator.md`。该 skill 会为新语音应用选择合适的 ASR 模型、解码参数和预处理管线。

## 练习题

1. **简单。** 运行 `code/main.py`。确认 1 秒 16 kHz 信号以 10 ms 步幅分帧后约为 ~100 帧。30 秒时约为 ~3,000 帧。
2. **中等。** 使用 `numpy.fft` 构建完整的 log-mel 频谱图。验证 80 个 mel 频带与 `librosa.feature.melspectrogram(n_mels=80)` 在数值误差范围内一致。
3. **困难。** 实现流式推理：将音频切成 10 秒窗口，重叠 2 秒，对每个窗口运行 Whisper 并合并转录结果。在 5 分钟播客样本上测量词错误率（WER），并与单次透传对比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Mel 频谱图 | "音频图像" | 二维表示：一轴为频率频带，另一轴为时间帧；每个单元格为对数缩放的能量值。 |
| Log-mel | "Whisper 所见之物" | 经过对数变换的 mel 频谱图；近似人类对响度的感知。 |
| Frame（帧） | "一个时间切片" | 25 ms 的采样窗口；以 10 ms 步幅重叠。 |
| Task Token（任务 Token） | "语音的提示前缀" | 解码器提示中的特殊 token，如 `<\|transcribe\|>` / `<\|translate\|>`。 |
| 语音活动检测（VAD） | "找到语音" | 在 ASR 前过滤静音的门控；能大幅降低成本。 |
| CTC | "连接时序分类" | 经典的无对齐 ASR 训练损失；Whisper 不使用它。 |
| Whisper-turbo | "小解码器 + 全编码器" | large-v3 encoder + 4 层 decoder；解码速度快 8 倍。 |
| Faster-whisper | "生产级封装" | CTranslate2 重新实现；int8 量化；比 OpenAI 参考实现快 4 倍。 |

## 延伸阅读

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper 论文。
- [OpenAI Whisper 仓库](https://github.com/openai/whisper) — 参考代码 + 模型权重。阅读 `whisper/model.py` 了解 Conv1D 前端 + encoder + decoder 的完整实现，约 400 行。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — 步骤 5–6 中描述的 beam search + 任务 token 逻辑都在此处；500 行，完全可读。
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — Whisper 的前身；在某些场景下仍是 SOTA 特征。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 生产级封装，比参考实现快 4 倍。
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024 年面向边缘设备的 ASR，Whisper 风格但更轻量。
- [HuggingFace 博客 — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — 标准微调配方，包含 mel 频谱图预处理器和 token 时间戳处理。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完整实现（encoder、decoder、cross-attention、生成），与本课架构图一致。
