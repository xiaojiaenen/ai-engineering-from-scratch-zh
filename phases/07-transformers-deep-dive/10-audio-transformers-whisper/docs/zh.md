# 音频 Transformer — Whisper 架构

> 音频是频率随时间变化的图像。Whisper 是一个视觉 Transformer，它吞食梅尔频谱图并说话输出。

**类型：** 学习
**语言：** Python
**前置要求：** 阶段 7 · 05（完整 Transformer），阶段 7 · 08（编码器-解码器），阶段 7 · 09（ViT）
**时间：** ~45 分钟

## 问题

在 Whisper（OpenAI，Radford 等，2022）出现之前，最先进的自动语音识别（ASR）意味着 wav2vec 2.0 和 HuBERT —— 自监督特征提取器加上一个微调头。质量高，但数据管道昂贵，领域脆弱。多语言语音识别需要为每个语系使用单独的模型。

Whisper 做了三个赌注：

1. **在一切数据上训练。** 从互联网抓取的 680,000 小时弱标注音频，覆盖 97 种语言。没有干净的学术语料库。没有音素标签。
2. **多任务单模型。** 一个解码器通过任务 token 在转录、翻译、语音活动检测、语言标识和时间戳任务上联合训练。
3. **标准的编码器-解码器 Transformer。** 编码器消耗对数梅尔频谱图。解码器自回归地生成文本 token。没有声码器，没有 CTC，没有 HMM。

结果：Whisper large-v3 在各种口音、噪声以及完全没有干净标注数据的语言中都表现出色。它是 2026 年所有开源语音助手以及大多数商业语音助手的默认语音前端。

## 概念

![Whisper 管道：音频 → 梅尔频谱图 → 编码器 → 解码器 → 文本](../assets/whisper.svg)

### 步骤 1 — 重采样 + 加窗

音频采样率为 16 kHz。裁剪/填充到 30 秒。计算对数梅尔频谱图：80 个梅尔频段，10 毫秒步长 → ~3,000 帧 × 80 个特征。这是 Whisper 看到的“输入图像”。

### 步骤 2 — 卷积主干

两个核大小为 3、步长为 2 的 Conv1D 层将 3,000 帧减少到 1,500 帧。在增加少量参数的同时将序列长度减半。

### 步骤 3 — 编码器

一个 24 层（对于 large 模型）的 Transformer 编码器，处理 1,500 个时间步。使用正弦位置编码、自注意力机制、GELU FFN。生成 1,500 × 1,280 的隐藏状态。

### 步骤 4 — 解码器

一个 24 层的 Transformer 解码器。它自回归地生成 token，词汇表基于 BPE，是 GPT-2 词汇表的超集，并添加了一些音频相关的特殊 token。

### 步骤 5 — 任务 token

解码器的提示词以控制 token 开头，告诉模型要做什么：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或者

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型基于这个约定进行训练。你通过前缀来控制任务。这是 2026 年的指令微调，但应用于语音领域。

### 步骤 6 — 输出

使用对数概率阈值的束搜索（宽度 5）。当 `<|notimestamps|>` token 不存在时，每 0.02 秒的音频预测一个时间戳。

### Whisper 模型尺寸

| 模型 | 参数量 | 层数 | d_model | 注意力头数 | 显存占用 (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB（4 层解码器）|

Large-v3-turbo（2024）将解码器从 32 层减少到 4 层。解码速度提升 8 倍，词错误率（WER）下降不到 1 个百分点。正是这种解码速度的解锁，使得 Whisper-turbo 成为 2026 年实时语音代理的默认选择。

### Whisper 不做的事情

- 不做说话人分离（谁在说话）。需要配合 pyannote 来实现。
- 原生不支持实时流式传输 —— 30 秒的窗口是固定的。现代封装（`faster-whisper`, `WhisperX`）通过 VAD + 重叠来添加流式功能。
- 不支持超过 30 秒的长上下文，除非进行外部分块。在实践中效果很好，因为人类语音的转录很少需要长程上下文。

### 2026 年的格局

| 任务 | 模型 | 备注 |
|------|-------|------|
| 英语 ASR | Whisper-turbo, Moonshine | Moonshine 在边缘设备上速度快 4 倍 |
| 多语言 ASR | Whisper-large-v3 | 97 种语言 |
| 流式 ASR | faster-whisper + VAD | 可达到 150 毫秒延迟目标 |
| 文本转语音 (TTS) | Piper, XTTS-v2, Kokoro | 编码器-解码器模式，但形状类似 Whisper |
| 音频 + 语言 | AudioLM, SeamlessM4T | 在一个 Transformer 中同时处理文本 token 和音频 token |

## 构建

参见 `code/main.py`。我们不训练 Whisper —— 我们构建对数梅尔频谱图管道 + 任务 token 提示格式化器。这些是你在实际生产中会接触到的部分。

### 步骤 1：合成音频

生成一个 1 秒、440 Hz、采样率为 16 kHz 的正弦波。共 16,000 个采样点。

### 步骤 2：对数梅尔频谱图（简化版）

完整的梅尔频谱图需要 FFT。我们采用一个简化的分帧 + 逐帧能量版本来展示流程，无需 `librosa`：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

帧长 = 25 毫秒，帧移 = 10 毫秒。与 Whisper 的加窗参数匹配。为了教学目的，用逐帧能量代替梅尔频段。

### 步骤 3：填充到 30 秒

Whisper 总是处理 30 秒的块。将频谱图填充（或裁剪）到 3,000 帧。

### 步骤 4：构建提示 token

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是整个任务控制接口。一个 4 个 token 的前缀。

## 使用

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快、兼容 OpenAI 的版本：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026 年何时选择 Whisper：**

- 使用一个模型进行多语言 ASR。
- 对嘈杂、多样化的音频进行稳健的转录。
- 研究/原型 ASR —— 最快的起点。

**何时选择其他方案：**

- 在边缘设备上进行超低延迟流式传输 —— Moonshine 在匹配质量下击败 Whisper。
- 需要 <200 毫秒延迟的实时对话式 AI —— 使用专用的流式 ASR。
- 说话人分离 —— Whisper 不做这个；需要集成 pyannote。

## 部署

参见 `outputs/skill-asr-configurator.md`。此技能为一个新的语音应用选择 ASR 模型、解码参数和预处理管道。

## 练习

1. **简单。** 运行 `code/main.py`。确认在 16 kHz 采样率、10 毫秒帧移下，1 秒信号的帧数约为 100。对于 30 秒：约 3,000 帧。
2. **中等。** 使用 `numpy.fft` 构建完整的对数梅尔频谱图。验证 80 个梅尔频段在数值误差范围内与 `librosa.feature.melspectrogram(n_mels=80)` 匹配。
3. **困难。** 实现流式推理：将音频分块为 10 秒窗口，重叠 2 秒，在每个块上运行 Whisper，合并转录结果。在一个 5 分钟的播客样本上测量词错误率，并与单次处理进行比较。

## 关键术语

| 术语 | 常用说法 | 实际含义 |
|------|----------|----------|
| 梅尔频谱图 | “音频图像” | 二维表示：一个轴是频率频段，另一个轴是时间帧；每个单元格是经过对数缩放的能量。 |
| 对数梅尔 | “Whisper 看到的东西” | 经过对数处理的梅尔频谱图；近似于人类对响度的感知。 |
| 帧 | “一个时间切片” | 一个 25 毫秒的采样窗口；以 10 毫秒的步长重叠。 |
| 任务 token | “语音的提示前缀” | 解码器提示中的特殊 token，如 `<|transcribe|>` / `<|translate|>`。 |
| 语音活动检测 (VAD) | “找到语音” | 在 ASR 之前去除静音的门控机制；大幅降低成本。 |
| CTC | “连接主义时间分类” | 经典的 ASR 损失函数，用于无需对齐的训练；Whisper **不**使用它。 |
| Whisper-turbo | “小解码器，完整编码器” | large-v3 编码器 + 4 层解码器；解码速度快 8 倍。 |
| Faster-whisper | “生产环境封装” | CTranslate2 重实现；int8 量化；比 OpenAI 参考实现快 4 倍。 |

## 扩展阅读

- [Radford 等 (2022)。通过大规模弱监督实现稳健的语音识别](https://arxiv.org/abs/2212.04356) —— Whisper 论文。
- [OpenAI Whisper 代码库](https://github.com/openai/whisper) —— 参考代码 + 模型权重。阅读 `whisper/model.py` 以查看约 400 行内从上到下的 Conv1D 主干 + 编码器 + 解码器。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) —— 步骤 5-6 描述的束搜索 + 任务 token 逻辑在此；500 行，完全可读。
- [Baevski 等 (2020)。wav2vec 2.0：语音表示的自监督学习框架](https://arxiv.org/abs/2006.11477) —— 先驱；在某些设置下仍是 SOTA 特征提取器。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) —— 生产环境封装，比参考实现快 4 倍。
- [Jia 等 (2024)。Moonshine：用于实时转录和语音命令的语音识别](https://arxiv.org/abs/2410.15608) —— 2024 年边缘友好的 ASR，形状类似 Whisper 但更小。
- [HuggingFace 博客 — “使用 🤗 Transformers 为多语言 ASR 微调 Whisper”](https://huggingface.co/blog/fine-tune-whisper) —— 权威的微调教程，包括梅尔频谱图预处理器和 token-时间戳处理。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) —— 完整实现（编码器、解码器、交叉注意力、生成），与本课架构图一致。