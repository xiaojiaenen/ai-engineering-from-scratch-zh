# 语音合成 (TTS) — 从 Tacotron 到 F5 和 Kokoro

> ASR 将语音转为文本；TTS 将文本转为语音。2026 年的技术栈分为三部分：文本 → token、token → mel、mel → 波形。每部分都有可运行在笔记本上的默认模型。

**类型：** 动手实践
**语言：** Python
**前置知识：** 第 6 阶段 · 02（频谱图与 Mel），第 5 阶段 · 09（Seq2Seq），第 7 阶段 · 05（完整 Transformer）
**时间：** 约 75 分钟

## 问题描述

你有一段文本："Please remind me to water the plants at 6 pm."你需要生成一段听起来自然、韵律正确（停顿、重音）、"plants"的元音发音准确，且能在 CPU 上 300 毫秒内运行的 3 秒音频，用于实时语音助手。你还需要同声线切换、处理中英混杂输入（"remind me at 6 pm, daijoubu?"），并且不能把人名念错。

现代 TTS 流水线如下：

1. **文本前端。** 归一化文本（日期、数字、邮箱），转换为音素或子词 token，预测韵律特征。
2. **声学模型。** 文本 → mel 频谱图。Tacotron 2（2017）、FastSpeech 2（2020）、VITS（2021）、F5-TTS（2024）、Kokoro（2024）。
3. **声码器。** mel → 波形。WaveNet（2016）、WaveRNN、HiFi-GAN（2020）、BigVGAN（2022）、2024 年后的神经编解码声码器。

2026 年，随着端到端扩散模型和流匹配模型的出现，声学模型与声码器的界限变得模糊。但三分法的思维模型仍然适用于调试。

## 核心概念

![Tacotron、FastSpeech、VITS、F5/Kokoro 并排对比](../assets/tts.svg)

**Tacotron 2（2017）。** Seq2seq：字符嵌入 → 双向 LSTM 编码器 → 位置敏感注意力机制 → 自回归 LSTM 解码器逐帧输出 mel。速度慢（自回归），长文本不稳定。仍常被用作基线引用。

**FastSpeech 2（2020）。** 非自回归。时长预测器输出每个音素应分配的 mel 帧数。单次前向传播，比 Tacotron 快 10 倍。损失部分自然度（单调对齐），但广泛部署。

**VITS（2021）。** 联合训练编码器 + 基于流的时长预测 + HiFi-GAN 声码器，端到端变分推断。质量高，单模型。2022–2024 年主流开源 TTS。变体包括：YourTTS（多说话人零样本）、XTTS v2（2024，Coqui）。

**F5-TTS（2024）。** 基于流匹配的扩散 Transformer。韵律自然，仅需 5 秒参考音频即可实现零样本声音克隆。位列 2026 年开源 TTS 排行榜榜首。3.35 亿参数。

**Kokoro（2024）。** 轻量级（8200 万），CPU 可运行，实时英文 TTS 质量领先。闭词表，仅支持英文，Apache 2.0 许可证。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3。** 商业级最优水平。ElevenLabs v2.5 的情绪标签（如 `"[whispered]"`、`"[laughing]"`）及角色声音在 2026 年有声书制作中占据主导地位。

### 声码器演进

| 时代 | 声码器 | 延迟 | 音质 |
|-----|---------|---------|---------|
| 2016 | WaveNet | 仅离线 | 发布时 SOTA |
| 2018 | WaveRNN | 接近实时 | 良好 |
| 2020 | HiFi-GAN | 100 倍实时 | 接近人声 |
| 2022 | BigVGAN | 50 倍实时 | 跨说话人/语言泛化 |
| 2024 | SNAC、DAC（神经编解码器）| 与 AR 模型集成 | 离散 token，比特效率高 |

到 2026 年，大多数"TTS"模型是端到端的文本到波形；mel 频谱图只是内部表示。

### 评估指标

- **MOS（平均意见得分）。** 1–5 分制，众包采集。仍是黄金标准；测试过程繁琐缓慢。
- **CMOS（比较 MOS）。** A vs B 偏好选择。每份标注的置信区间更窄。
- **UTMOS、DNSMOS。** 无参考的神经网络 MOS 预测器。用于排行榜评测。
- **通过 ASR 计算的 CER（字符错误率）。** 将 TTS 输出送入 Whisper，与输入文本计算 CER。作为可懂度的代理指标。
- **SECS（说话人嵌入余弦相似度）。** 声音克隆质量指标。

2026 年 LibriTTS test-clean 上的数据：

| 模型 | UTMOS | CER（经 Whisper）| 参数量 |
|-------|-------|-------------------|------|
| 真实音频 | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

```figure
sp-tts-stack
```

## 动手实践

### 步骤 1：对输入进行音素化

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

音素是通用桥梁。喂入原始文本到 VITS 级别以下的模型会导致质量下降。

### 步骤 2：运行 Kokoro（2026 年 CPU 默认方案）

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = 美式英语
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 张量, sr=24000
```

离线运行，单个文件，8200 万参数。

### 步骤 3：使用 F5-TTS 进行声音克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

传入 5 秒参考音频片段及其转录文本；F5 会克隆韵律和音色。

### 步骤 4：从零实现 HiFi-GAN 声码器

代码量太大，不适合放入教程脚本，但结构如下：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 个上采样块，总计 256 倍，从 mel 速率提升到音频速率
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> 波形
```

训练方式：对抗损失（短窗判别器）+ mel 频谱图重建损失 + 特征匹配损失。已被广泛商品化——建议使用 `hifi-gan` 仓库或 nvidia-NeMo 的预训练检查点。

### 步骤 5：完整流水线（伪代码）

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 使用指南

2026 年技术栈选型：

| 场景 | 推荐方案 |
|-----------|------|
| 实时英文语音助手 | Kokoro（CPU）或 XTTS v2（GPU）|
| 基于 5 秒参考的声音克隆 | F5-TTS |
| 商业角色声音 | ElevenLabs v2.5 |
| 有声书旁白 | ElevenLabs v2.5 或 XTTS v2 + 微调 |
| 低资源语言 | 在 5–20 小时目标语言数据上训练 VITS |
| 表现力 / 情绪标签 | ElevenLabs v2.5 或 StyleTTS 2 微调 |

2026 年开源领域领头羊：**F5-TTS 注重质量，Kokoro 注重效率**。除非你是历史研究者，否则不要再用 Tacotron。

## 常见陷阱

- **缺少文本归一化。** "Dr. Smith"读成"Doctor"还是"Drive"？"2026"读成"twenty twenty six"还是"two zero two six"？务必在音素化**之前**完成归一化。
- **未登录专有名词。** "Ghumare" → "ghyu-mair"？为未知 token 提供 fallback 的 Grapheme-to-Phoneme 模型。
- **削波失真。** 声码器输出很少削波，但推理时 mel 缩放不匹配可能超出 ±1.0 范围。务必 `np.clip(wav, -1, 1)`。
- **采样率不匹配。** Kokoro 输出 24 kHz；你的下游管道期望 16 kHz → 重采样或出现混叠。

## 交付成果

保存为 `outputs/skill-tts-designer.md`。针对给定的声音、延迟要求和语言目标设计 TTS 流水线。

## 练习题

1. **简单。** 运行 `code/main.py`。从玩具词表构建音素字典，估计每个音素的时长，并打印假"mel"调度表。
2. **中等。** 安装 Kokoro，以 `af_bella` 和 `am_adam` 两种声音合成同一段文本。对比音频时长和主观音质。
3. **困难。** 录制 5 秒个人参考音频。使用 F5-TTS 进行克隆。报告参考音频与克隆输出之间的 SECS。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| Phoneme（音素）| 声音单位 | 抽象声音类别；英文有 39 个（ARPABet）。 |
| Duration predictor（时长预测器）| 每个音素持续多久 | 非自回归模型输出；每个音素的整数帧数。 |
| Vocoder（声码器）| mel → 波形 | 将 mel 频谱映射到原始样本的神经网络。 |
| HiFi-GAN | 标准声码器 | 基于 GAN；2020–2024 年主流。 |
| MOS | 主观质量 | 由人工评分者给出的 1–5 分平均意见得分。 |
| SECS | 声音克隆指标 | 目标说话人与输出说话人嵌入之间的余弦相似度。 |
| F5-TTS | 2024 年开源 SOTA | 流匹配扩散；零样本克隆。 |
| Kokoro | CPU 英文 TTS 标杆 | 8200 万参数模型，Apache 2.0 许可。 |

## 延伸阅读

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) —— seq2seq 基线方案。
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) —— 端到端基于流的方案。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) —— 当前开源 SOTA。
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) —— 2026 年仍在广泛使用的声码器。
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) —— 2024 年面向 CPU 的英文 TTS 方案。
