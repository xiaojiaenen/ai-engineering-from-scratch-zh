# 神经音频编解码器 —— EnCodec、SNAC、Mimi、DAC 与语义-声学分离

> 2026 年的音频生成几乎全部基于 token。EnCodec、SNAC、Mimi 和 DAC 将连续波形转换为离散序列，供 Transformer 进行预测。语义与声学 token 的分离——将第一个码本就作为语义部分，其余作为声学部分——是自 Transformer 以来音频领域最重要的架构变革。

**类型：** 学习
**语言：** Python
**前置知识：** 阶段 6 · 02（频谱图）、阶段 10 · 11（量化）、阶段 5 · 19（子词分词）
**时间：** 约 60 分钟

## 问题背景

语言模型在离散 token 上工作，而音频是连续的。如果你想要一个类 LLM 的语音/音乐模型——MusicGen、Moshi、Sesame CSM、VibeVoice、Orpheus——你首先需要一个**神经音频编解码器**：一个学习到的编码器，将音频离散化为少量 token 词汇，并配合一个匹配的分路解码器重建波形。

已出现两大流派：

1. **重建优先的编解码器**——EnCodec、DAC。优化感知音频质量。token 是"声学"的——它们捕获一切，包括说话人身份、音色、背景噪声。
2. **语义优先的编解码器**——Mimi（Kyutai）、SpeechTokenizer。强制第一个码本编码语言/语音内容（通常通过对 WavLM 进行蒸馏）。后续的码本负责声学细节。

2024-2026 年的关键洞察：**纯重建编解码器在尝试从文本生成时会产生模糊的语音。** LLM 经过编解码 token 必须同时学习语言结构和声学结构，这在同一个码本中难以扩展。将它们分离——语义码本 0、声学码本 1-N——正是 Moshi 和 Sesame CSM 可行的原因。

## 核心概念

![四种编解码器概览：EnCodec、DAC、SNAC（多尺度）、Mimi（语义+声学）](../assets/codec-comparison.svg)

### 核心技巧：残差向量量化（RVQ）

与其使用一个大型码本（这需要数百万个码才能实现高质量），所有现代音频编解码器都使用 **RVQ**：一系列小码本的级联。第一个码本对编码器输出进行量化；第二个码本对残差进行量化；依此类推。每个码本包含 1024 个码。8 个码本 = 等效词表大小为 1024^8 = 10^24。

在推理时，解码器按帧累加所有选定的码以重建波形。

### 2026 年最关键的四种编解码器

**EnCodec（Meta，2022）。** 基线方案。编码器-解码器结构作用于波形，使用 RVQ 瓶颈。24 kHz，最多支持 32 个码本，默认 4 个码本 @ 1.5 kbps。采用 `1D 卷积 + Transformer + 1D 卷积` 架构。被 MusicGen 使用。

**DAC（Descript，2023）。** RVQ 配合 L2 归一化码本、周期性激活函数、改进的 loss。任何开源编解码器中重建保真度最高——有时使用 12 个码本就与原始语音难以区分。支持 44.1 kHz 全频段。

**SNAC（Hubert Siuzdak，2024）。** 多尺度 RVQ——粗粒度码本比细粒度码本具有更低的帧率。实质上对音频进行了分层建模：约 12 Hz 的粗粒度"草图"加上 50 Hz 的细节。被 Orpheus-3B 使用，因为其分层结构很好地适配了基于 LM 的生成。

**Mimi（Kyutai，2024）。** 2026 年的颠覆性方案。12.5 Hz 帧率（极低），8 个码本 @ 4.4 kbps。码本 0 **由 WavLM 蒸馏而来**——训练目标是预测 WavLM 的语音内容特征。码本 1-7 为声学残差。这种分离为 Moshi（第 15 课）和 Sesame CSM 提供了动力。

### 帧率对语言建模的影响

更低的帧率 = 更短的序列 = 更快的 LM。

| 编解码器 | 帧率 | 1 秒 = N 帧 | 适用场景 |
|---------|-----|------------|---------|
| EnCodec-24k | 75 Hz | 75 | 音乐、通用音频 |
| DAC-44.1k | 86 Hz | 86 | 高保真音乐 |
| SNAC-24k（粗粒度） | ~12 Hz | 12 | AR-LM 高效 |
| Mimi | 12.5 Hz | 12.5 | 流式语音 |

在 12.5 Hz 下，10 秒的语音仅有 125 个编解码帧——Transformer 可以轻松预测它们。

### 语义与声学 Token

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **语义 Token（Mimi 中的码本 0）。** 编码说了什么——音素、词汇、内容。通过辅助预测 loss 从 WavLM 蒸馏而来。
- **声学 Token（码本 1-7）。** 编码音色、说话人身份、韵律、背景噪声、细粒度细节。

AR LM 先预测语义 token（由文本条件驱动），然后预测声学 token（由语义 + 说话人参考条件驱动）。这种因式分解正是现代 TTS 能够实现零样本声音克隆的原因：语义模型处理内容，声学模型处理音色。

### 2026 年重建质量对比（每秒比特数，越低越好）

| 编解码器 | 比特率 | PESQ | ViSQOL |
|---------|-------|-----|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

传统的编解码器如 Opus 在每比特感知质量上仍然占优。神经编解码器在**离散 token**（Opus 不产生此类输出）和**生成模型质量**（LM 能对这些 token 做什么）方面胜出。

```figure
rvq-codec-cascade
```

## 构建实现

### 步骤 1：使用 EnCodec 编码

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

在 6 kbps 下 `n_codebooks=8`。每个 code 的范围是 0-1023（10 bit）。

### 步骤 2：解码并测量重建效果

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### 步骤 3：语义-声学分离（Mimi 风格）

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

语义码本 0 与 WavLM 对齐。你可以训练一个文本到语义的 Transformer——词汇表远小于直接到音频的方式。然后一个独立的声学到波形的解码器会在说话人参考条件的驱动下工作。

### 步骤 4：为什么 AR LM 在编解码 token 上有效

对于 Mimi 12.5 Hz × 8 码本下的 10 秒语音片段：

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 个 token 对 Transformer 来说是 trivial 的上下文。一个 256M 参数的 Transformer 可以在现代 GPU 上以毫秒级生成 10 秒的语音。

## 使用指南

将问题映射到编解码器：

| 任务 | 编解码器 |
|------|---------|
| 通用音乐生成 | EnCodec-24k |
| 最高保真重建 | DAC-44.1k |
| 语音的 AR LM（TTS） | SNAC 或 Mimi |
| 流式全双工语音 | Mimi（12.5 Hz） |
| 带文本的音效库 | EnCodec + T5 条件 |
| 细粒度音频编辑 | DAC + 修复 |

经验法则：**如果你在构建生成模型，从 Mimi 或 SNAC 开始。如果你在构建压缩管道，使用 Opus。**

## 常见陷阱

- **码本过多。** 增加码本会以线性方式提升保真度，但 LM 序列长度也会线性增长。建议在 8-12 个码本处停止。
- **帧率不匹配。** 在 12.5 Hz Mimi 上训练 LM，然后在 50 Hz EnCodec 上微调，会静默失败。
- **假设所有码本等价。** 在 Mimi 中，码本 0 携带内容信息；丢失它会破坏可懂度。丢失码本 7 几乎察觉不到。
- **仅用重建质量作为评估指标。** 一个编解码器可能具有出色的重建效果，但如果语义结构不佳，在基于 LM 的生成中仍然无用。

## 交付成果

保存为 `outputs/skill-codec-picker.md`。为给定的生成或压缩任务选择编解码器。

## 练习

1. **简单。** 运行 `code/main.py`。它实现了一个玩具标量+残差量化器，并测量随着码本增加的重建误差。
2. **中等。** 安装 `encodec`，在预留的语音片段上比较 1、4、8、32 个码本的效果。绘制 PESQ 或 MSE 随比特率的曲线。
3. **困难。** 加载 Mimi。编码一个片段。将码本 0 替换为随机整数后解码。然后将码本 7 同样替换。比较两种损坏——码本 0 损坏应破坏可懂度；码本 7 损坏应几乎无影响。

## 关键术语

| 术语 | 人们所说的 | 实际含义 |
|------|-----------|---------|
| RVQ | 残差量化 | 小码本的级联；每个码本量化前一个的残差。 |
| 帧率 | 编解码器速度 | 每秒的 token 帧数。越低 = 越快的 LM。 |
| 语义码本 | Mimi 中的码本 0 | 从 SSL 特征蒸馏的码本；编码内容。 |
| 声学码本 | 其他所有码本 | 音色、韵律、噪声、细粒度细节。 |
| PESQ / ViSQOL | 感知质量 | 与 MOS 相关的客观指标。 |
| EnCodec | Meta 编解码器 | RVQ 基线；被 MusicGen 使用。 |
| Mimi | Kyutai 编解码器 | 12.5 Hz 帧率；语义-声学分离；驱动 Moshi。 |

## 延伸阅读

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — RVQ 基线。
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — 最高保真度开源方案。
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — 多尺度 RVQ。
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — 语义-声学分离、WavLM 蒸馏。
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) — 两阶段语义/声学范式。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 原始的可流式 RVQ 编解码器。
