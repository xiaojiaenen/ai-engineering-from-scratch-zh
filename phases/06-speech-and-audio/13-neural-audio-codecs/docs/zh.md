# 神经音频编解码器 —— EnCodec、SNAC、Mimi、DAC 与语义-声学分离

> 2026 年的音频生成几乎完全基于 token。EnCodec、SNAC、Mimi 和 DAC 将连续波形转化为离散序列，使 Transformer 得以预测。语义与声学 token 的分离 —— 第一个码本作为语义，其余作为声学 —— 是自 Transformer 出现以来音频领域最重要的架构转变。

**类型：** 学习资料
**编程语言：** Python
**先修知识：** 阶段 6 · 02（频谱图），阶段 10 · 11（量化），阶段 5 · 19（子词分词）
**时间：** 约 60 分钟

## 问题所在

语言模型处理的是离散 token，而音频是连续的。如果你想为语音/音乐构建一个类 LLM 模型（如 MusicGen、Moshi、Sesame CSM、VibeVoice、Orpheus），你首先需要一个**神经音频编解码器**：一个学习型的编码器，将音频离散化为小词汇量的 token 序列；以及一个匹配的解码器，用以重建波形。

目前出现了两大类：

1.  **重建优先编解码器** —— EnCodec、DAC。优化感知音频质量。其 token 是“声学的”——它们捕获所有信息，包括说话者身份、音色、背景噪声。
2.  **语义优先编解码器** —— Mimi（来自 Kyutai）、SpeechTokenizer。强制第一个码本编码语言/语音内容（通常通过从 WavLM 蒸馏）。后续码本用于捕获声学细节。

2024-2026 年的洞察是：**一个纯粹的重建编解码器，在你尝试从文本生成时，会产生模糊的语音。** LLM 基于编解码器 token 进行工作时，需要在同一个码本中同时学习语言结构和声学结构，这无法良好扩展。将它们分离 —— 语义码本为 0，声学码本为 1-N —— 正是 Moshi 和 Sesame CSM 能够成功的关键。

## 核心概念

![四大编解码器概览：EnCodec、DAC、SNAC（多尺度）、Mimi（语义+声学）](../assets/codec-comparison.svg)

### 核心技巧：残差向量量化 (RVQ)

与使用一个巨大的码本（要获得好质量可能需要数百万个码字）不同，所有现代音频编解码器都使用 **RVQ**：一个级联的小型码本序列。第一个码本量化编码器的输出；第二个量化残差；依此类推。每个码本有 1024 个码字。8 个码本 = 有效词汇量为 1024^8 = 10^24。

在推理阶段，解码器对每帧所有选定的码字求和以进行重建。

### 2026 年最重要的四种编解码器

**EnCodec (Meta, 2022)。** 基线模型。在波形上进行编码器-解码器结构，带有 RVQ 瓶颈。24 kHz，最多支持 32 个码本，默认 4 个码本 @ 1.5 kbps。使用 `1D conv + transformer + 1D conv` 架构。被 MusicGen 使用。

**DAC (Descript, 2023)。** 采用 L2 归一化码本的 RVQ，带有周期性激活函数和改进的损失函数。在任何开放编解码器中重建保真度最高 —— 使用 12 个码本时，有时与原始语音无法区分。44.1 kHz 全频带。

**SNAC (Hubert Siuzdak, 2024)。** 多尺度 RVQ —— 粗糙码本的帧率低于精细码本。有效地在层级上建模音频：约 12 Hz 的粗略“草图”加上 50 Hz 的细节。被 Orpheus-3B 使用，因为其层级结构与基于 LM 的生成非常契合。

**Mimi (Kyutai, 2024)。** 2026 年的变革者。12.5 Hz 帧率（极低），8 个码本 @ 4.4 kbps。码本 0 是**从 WavLM 蒸馏得到** —— 训练用于预测 WavLM 的语音内容特征。码本 1-7 是声学残差。这种分离驱动了 Moshi（课程 15）和 Sesame CSM。

### 帧率对语言建模很重要

更低的帧率 = 更短的序列 = 更快的 LM。

| 编解码器           | 帧率      | 1 秒 = N 帧 | 适用于                 |
|-------------------|-----------|-------------|-----------------------|
| EnCodec-24k       | 75 Hz     | 75          | 音乐，通用音频         |
| DAC-44.1k         | 86 Hz     | 86          | 高保真音乐             |
| SNAC-24k（粗糙）   | ~12 Hz    | 12          | 自回归 LM 高效生成     |
| Mimi              | 12.5 Hz   | 12.5        | 流式语音               |

在 12.5 Hz 下，一段 10 秒的语音只有 125 个编解码器帧 —— Transformer 可以轻松预测它们。

### 语义 vs 声学 token

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **语义 token（Mimi 中的码本 0）。** 编码说了什么 —— 音素、单词、内容。通过辅助预测损失从 WavLM 蒸馏得到。
- **声学 token（码本 1-7）。** 编码音色、说话者身份、韵律、背景噪声、细节。

自回归 LM 首先预测语义 token（以文本为条件），然后预测声学 token（以语义 + 说话者参考为条件）。这种分解正是现代 TTS 能够零样本克隆声音的原因：语义模型处理内容；声学模型处理音色。

### 2026 年重建质量（比特/秒，比特率越低越好）

| 编解码器        | 比特率    | PESQ | ViSQOL |
|----------------|-----------|------|--------|
| Opus-20kbps    | 20 kbps   | 4.0  | 4.3    |
| EnCodec-6kbps  | 6 kbps    | 3.2  | 3.8    |
| DAC-6kbps      | 6 kbps    | 3.5  | 4.0    |
| SNAC-3kbps     | 3 kbps    | 3.3  | 3.8    |
| Mimi-4.4kbps   | 4.4 kbps  | 3.1  | 3.7    |

像 Opus 这样的传统编解码器，在感知质量上仍然在每比特上胜出。神经编解码器的优势在于**离散 token**（Opus 不产生）和**生成模型的质量**（即 LM 利用这些 token 能做什么）。

## 动手实现

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

`n_codebooks=8` 在 6 kbps 下。每个码字范围为 0-1023（10 位）。

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

语义码本 0 与 WavLM 对齐。你可以训练一个文本到语义的 Transformer —— 比直接到音频的 Transformer 词汇量小得多。然后，一个独立的声学到波形解码器以说话者参考为条件。

### 步骤 4：为什么基于编解码器 token 的自回归 LM 可行

对于一段 10 秒的语音片段，使用 Mimi 的 12.5 Hz × 8 个码本：

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 个 token 对于 Transformer 来说是一个微不足道的上下文长度。一个 2.56 亿参数的 Transformer 可以在现代 GPU 上几毫秒内生成 10 秒的语音。

## 应用指南

将任务映射到编解码器：

| 任务                     | 推荐编解码器             |
|--------------------------|-------------------------|
| 通用音乐生成             | EnCodec-24k             |
| 最高保真度重建           | DAC-44.1k               |
| 语音自回归 LM（TTS）      | SNAC 或 Mimi            |
| 流式全双工语音           | Mimi (12.5 Hz)          |
| 带文本的声音效果库       | EnCodec + T5 条件        |
| 细粒度音频编辑           | DAC + 修复               |

经验法则：**如果你正在构建生成模型，从 Mimi 或 SNAC 开始。如果你正在构建压缩流水线，使用 Opus。**

## 常见陷阱

- **码本过多。** 增加码本能线性提高保真度，但 LM 序列长度也会线性增加。在 8-12 个左右停止。
- **帧率不匹配。** 在 12.5 Hz 的 Mimi 上训练 LM，然后在 50 Hz 的 EnCodec 上进行微调，会静默失败。
- **假设所有码本都同等重要。** 在 Mimi 中，码本 0 携带内容；丢失它会破坏可懂度。丢失码本 7 则几乎无感。
- **仅将重建质量作为唯一指标。** 一个编解码器可能重建质量很好，但如果语义结构差，对于基于 LM 的生成来说可能毫无用处。

## 部署上线

保存为 `outputs/skill-codec-picker.md`。根据给定的生成或压缩任务选择一个编解码器。

## 练习

1.  **简单。** 运行 `code/main.py`。它实现了一个玩具标量+残差量化器，并在你添加码本时测量重建误差。
2.  **中等。** 安装 `encodec`，并在一个保留的语音片段上比较 1、4、8、32 个码本。绘制 PESQ 或 MSE 与比特率的关系图。
3.  **困难。** 加载 Mimi。编码一个片段。用随机整数替换码本 0；解码。然后用类似方法替换码本 7。比较这两种损坏情况 —— 码本 0 的损坏应该会破坏可懂度；码本 7 的损坏应该几乎不会改变任何东西。

## 关键术语

| 术语           | 人们怎么说         | 实际含义                                   |
|----------------|-------------------|-------------------------------------------|
| RVQ            | 残差量化           | 级联的小型码本；每个量化前一个的残差。          |
| 帧率           | 编解码器速度       | 每秒多少个 token 帧。帧率越低，LM 越快。      |
| 语义码本       | 码本 0 (Mimi)     | 从 SSL 特征蒸馏得到的码本；编码内容。          |
| 声学码本       | 其他所有           | 音色、韵律、噪声、细节。                      |
| PESQ / ViSQOL  | 感知质量           | 与 MOS 相关的客观指标。                      |
| EnCodec        | Meta 编解码器      | RVQ 基线；被 MusicGen 使用。                 |
| Mimi           | Kyutai 编解码器    | 12.5 Hz 帧率；语义-声学分离；驱动 Moshi。    |

## 延伸阅读

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) —— RVQ 基线。
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) —— 最高保真度开放模型。
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) —— 多尺度 RVQ。
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) —— 语义-声学分离，WavLM 蒸馏。
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) —— 两阶段语义/声学范式。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) —— 原始可流式 RVQ 编解码器。