# 文本转语音 (TTS) —— 从 Tacotron 到 F5 与 Kokoro

> 语音识别 (ASR) 将语音反转为文本；文本转语音 (TTS) 将文本反转为语音。2026 年的技术栈分为三部分：文本 → tokens，tokens → 梅尔频谱图，梅尔频谱图 → 波形。每个部分都有一个适合在笔记本电脑上运行的默认模型。

**类型：** 构建
**语言：** Python
**前置知识：** 阶段 6 · 02（频谱图与梅尔），阶段 5 · 09（序列到序列），阶段 7 · 05（完整 Transformer）
**时间：** 约 75 分钟

## 问题描述

你有一段字符串："Please remind me to water the plants at 6 pm."。你需要一段 3 秒的音频片段，它听起来自然，韵律（停顿、重音）正确，以正确的元音发音“plants”，并且在 CPU 上的实时语音助手场景中运行时间少于 300 毫秒。你还需要切换声音、处理混合语言输入（“remind me at 6 pm, daijoubu?”），并且在处理名字时不会出错。

现代 TTS 流水线如下所示：

1.  **文本前端。** 规范化文本（日期、数字、电子邮件），转换为音素或子词 tokens，预测韵律特征。
2.  **声学模型。** 文本 → 梅尔频谱图。Tacotron 2 (2017), FastSpeech 2 (2020), VITS (2021), F5-TTS (2024), Kokoro (2024)。
3.  **声码器。** 梅尔频谱图 → 波形。WaveNet (2016), WaveRNN, HiFi-GAN (2020), BigVGAN (2022), 2024 年后的神经编码声码器。

到了 2026 年，声学模型和声码器的界限随着端到端的扩散模型和流匹配模型而变得模糊。但对于调试而言，三个部分的思维模型仍然成立。

## 核心概念

![Tacotron, FastSpeech, VITS, F5/Kokoro 对比](../assets/tts.svg)

**Tacotron 2 (2017).** 序列到序列模型：字符嵌入 → 双向 LSTM 编码器 → 位置敏感注意力 → 自回归 LSTM 解码器生成梅尔频谱帧。速度慢（自回归），长文本时不稳定。仍被引用作为基线。

**FastSpeech 2 (2020).** 非自回归。时长预测器输出每个音素对应多少梅尔频谱帧。单次前向传播，比 Tacotron 快 10 倍。损失一些自然度（单调对齐），但易于部署。

**VITS (2021).** 联合训练编码器 + 基于流的时长预测器 + HiFi-GAN 声码器，通过变分推理端到端进行。高质量，单模型。2022–2024 年主导开源 TTS。变体：YourTTS（多说话人零样本），XTTS v2 (2024, Coqui)。

**F5-TTS (2024).** 基于流匹配的扩散 Transformer。自然的韵律，使用 5 秒参考音频进行零样本声音克隆。2026 年开源 TTS 排行榜的榜首。3.35 亿参数。

**Kokoro (2024).** 轻量（8200 万参数），可在 CPU 上运行，面向实时应用的最佳英语 TTS。封闭词汇表，仅限英语，Apache-2.0 许可。

**OpenAI TTS-1-HD, ElevenLabs v2.5, Google Chirp-3.** 商业领域最先进模型。ElevenLabs v2.5 的情绪标签（“[whispered]”, “[laughing]”）和角色声音在 2026 年主导有声书制作。

### 声码器演进

| 时代 | 声码器 | 延迟 | 质量 |
|------|--------|------|------|
| 2016 | WaveNet | 仅离线 | 发布时最优 |
| 2018 | WaveRNN | 约实时 | 良好 |
| 2020 | HiFi-GAN | 100 倍实时 | 接近人声 |
| 2022 | BigVGAN | 50 倍实时 | 跨说话人/语言泛化 |
| 2024 | SNAC, DAC (神经编码) | 与自回归模型集成 | 离散 tokens，比特高效 |

到 2026 年，大多数“TTS”模型都是从文本到波形的端到端模型；梅尔频谱图是一种内部表示。

### 评估指标

-   **MOS (平均意见分).** 1-5 分制，众包。仍是黄金标准；但速度极慢。
-   **CMOS (比较平均意见分).** A 与 B 偏好对比。每次标注的置信区间更窄。
-   **UTMOS, DNSMOS.** 无参考的神经 MOS 预测器。用于排行榜。
-   **CER (字符错误率)，通过 ASR 计算。** 通过 Whisper 运行 TTS 输出，根据输入文本计算 CER。可理解度的代理指标。
-   **SECS (说话人嵌入余弦相似度).** 声音克隆质量。

2026 年在 LibriTTS test-clean 上的数据：

| 模型 | UTMOS | CER (通过 Whisper) | 大小 |
|------|-------|---------------------|------|
| 真实语音 | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## 构建它

### 步骤 1：音素化输入

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

音素是通用的桥梁。避免将原始文本输入给任何低于 VITS 质量水平的模型。

### 步骤 2：运行 Kokoro (2026 年 CPU 默认)

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

离线运行，单个文件，8200 万参数。

### 步骤 3：使用声音克隆运行 F5-TTS

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

传递一段 5 秒的参考音频片段及其转录文本；F5 克隆其韵律和音色。

### 步骤 4：从零开始构建 HiFi-GAN 声码器

太大，无法放入教程脚本，但基本结构如下：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

训练：对抗性（短窗口上的判别器）+ 梅尔频谱图重建损失 + 特征匹配损失。已经商品化——使用来自 `hifi-gan` 仓库或 nvidia-NeMo 的预训练检查点。

### 步骤 5：完整流水线（伪代码）

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 使用它

2026 年的技术栈：

| 场景 | 选择 |
|------|------|
| 实时英语语音助手 | Kokoro (CPU) 或 XTTS v2 (GPU) |
| 基于 5 秒参考的声音克隆 | F5-TTS |
| 商业角色声音 | ElevenLabs v2.5 |
| 有声书朗读 | ElevenLabs v2.5 或 XTTS v2 + 微调 |
| 低资源语言 | 在 5–20 小时目标语言数据上训练 VITS |
| 富有表现力 / 情绪标签 | ElevenLabs v2.5 或 StyleTTS 2 微调 |

截至 2026 年的开源领先者：**F5-TTS 注重质量，Kokoro 注重效率**。除非你是历史研究者，否则不要使用 Tacotron。

## 陷阱

-   **没有文本规范化器。** “Dr. Smith” 读作“Doctor”还是“Drive”？“2026”读作“twenty twenty six”还是“two zero two six”？在音素化**之前**进行规范化。
-   **未登录词（OOV）专有名词。** “Ghumare” → “ghyu-mair”？为未知 tokens 提供备用的字素到音素转换模型。
-   **削波。** 声码器输出很少削波，但推理时梅尔频谱缩放不匹配可能导致输出超过 ±1.0。始终 `np.clip(wav, -1, 1)`。
-   **采样率不匹配。** Kokoro 输出 24 kHz；你的下游流水线期望 16 kHz → 需要重采样，否则会产生混叠。

## 部署它

保存为 `outputs/skill-tts-designer.md`。为给定的声音、延迟和语言目标设计一个 TTS 流水线。

## 练习

1.  **简单。** 运行 `code/main.py`。它从一个玩具词汇表构建音素词典，估计每个音素的时长，并打印一个假的“梅尔”计划。
2.  **中等。** 安装 Kokoro，使用声音 `af_bella` 和 `am_adam` 合成同一个句子。比较音频时长和主观质量。
3.  **困难。** 录制一段你自己的 5 秒参考音频。使用 F5-TTS 克隆它。报告参考和克隆输出之间的 SECS。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| 音素 (Phoneme) | 声音单元 | 抽象的声音类别；英语中有 39 个 (ARPABet)。 |
| 时长预测器 (Duration predictor) | 每个音素持续多长时间 | 非自回归模型输出；每个音素对应的整数帧数。 |
| 声码器 (Vocoder) | 梅尔频谱图 → 波形 | 将梅尔频谱图映射到原始波形采样点的神经网络。 |
| HiFi-GAN | 标准声码器 | 基于生成对抗网络；2020–2024 年占主导地位。 |
| MOS | 主观质量 | 1–5 分的平均意见分，来自人类评估者。 |
| SECS | 声音克隆指标 | 目标和输出说话人嵌入之间的余弦相似度。 |
| F5-TTS | 2024 年开源最优 | 流匹配扩散模型；零样本克隆。 |
| Kokoro | CPU 英语领先者 | 8200 万参数模型，Apache 2.0 许可。 |

## 延伸阅读

-   [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) —— 序列到序列基线。
-   [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) —— 端到端基于流。
-   [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) —— 当前开源最优。
-   [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) —— 直到 2026 年仍在部署的声码器。
-   [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) —— 2024 年对 CPU 友好的英语 TTS。