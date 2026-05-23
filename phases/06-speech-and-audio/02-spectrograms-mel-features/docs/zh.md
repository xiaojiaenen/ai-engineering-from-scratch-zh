# 频谱图、梅尔刻度与音频特征

> 神经网络无法很好地处理原始波形数据。它们处理的是频谱图。它们处理梅尔频谱图的效果更好。2026年，每一个语音识别、语音合成和音频分类模型的成败都系于这单一的预处理选择上。

**类型:** 构建
**语言:** Python
**前置要求:** 第六阶段 · 01（音频基础）
**时间:** ~45 分钟

## 问题描述

取一个10秒、16千赫兹的音频片段。它包含160,000个浮点数，所有这些数值都包含在 `[-1, 1]` 中，几乎与"狗叫"或"猫"这样的标签完全无关。原始波形中包含信息，但其形式让模型难以提取。相隔100毫秒发出的两个完全相同的音素，其原始样本数据却完全不同。

频谱图解决了这个问题。它折叠了人类感知忽略的时间细节（微秒级抖动），并保留了感知所关注的结构（在约10-25毫秒的时间窗口内，哪些频率能量集中）。

梅尔频谱图更进一步。人类感知音高是呈对数关系的：100赫兹与200赫兹之间的"距离感"与1000赫兹与2000赫兹之间的"距离感"相似。梅尔刻度扭曲了频率轴以匹配这种感知。梅尔尺度的频谱图是2010年至2026年期间语音机器学习中最重要的单一特征。

## 核心概念

![波形 -> STFT -> 梅尔频谱图 -> MFCC 的特征阶梯](../assets/mel-features.svg)

**STFT (短时傅里叶变换)。** 将波形切分成重叠的帧（典型：25毫秒窗长，10毫秒帧移 = 在16千赫兹下为400个样本/160个样本）。将每帧乘以一个窗函数（默认汉宁窗；哈明窗有略微不同的权衡）。对每帧进行快速傅里叶变换。将幅度谱堆叠成一个形状为 `(n_frames, n_freq_bins)` 的矩阵。这就是你的频谱图。

**对数幅度。** 原始幅度跨越5-6个数量级。取 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 以压缩动态范围。所有生产级流水线都使用对数幅度，而非原始幅度。

**梅尔刻度。** 以赫兹为单位的频率 `f` 通过 `m = 2595 * log10(1 + f / 700)` 映射到梅尔值 `m`。该映射在1千赫兹以下近似线性，在1千赫兹以上近似对数关系。覆盖0-8千赫兹的80个梅尔频段是标准的语音识别输入。

**梅尔滤波器组。** 一组在梅尔刻度上等间距的三角形滤波器。每个滤波器是相邻快速傅里叶变换频段的加权和。将短时傅里叶变换幅度矩阵乘以滤波器组矩阵，通过一次矩阵乘法即可得到梅尔频谱图。

**对数梅尔频谱图。** `log(mel_spec + 1e-10)`。Whisper的输入。Parakeet的输入。SeamlessM4T的输入。2026年通用的音频前端。

**MFCCs (梅尔频率倒谱系数)。** 取对数梅尔频谱图，应用DCT (II型)，保留前13个系数。对特征进行去相关并进一步压缩。在约2015年之前，这是主流特征，直到卷积神经网络/转换器直接在原始对数梅尔频谱图上取得同等性能。目前仍用于说话人识别（x-vectors, ECAPA）。

**分辨率权衡。** 较大的FFT点数 = 更好的频率分辨率，但时间分辨率更差。25毫秒/10毫秒是音频机器学习的默认值；50毫秒/12.5毫秒用于音乐；5毫秒/2毫秒用于瞬态检测（鼓点、爆破音）。

## 动手构建

### 步骤 1: 对波形分帧

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

一个10秒、16千赫兹的音频片段，帧移 `frame_len=400, hop=160`，产生998帧。

### 步骤 2: 汉宁窗

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

在快速傅里叶变换之前进行逐元素相乘。消除由于在非零端点截断而引起的频谱泄漏。

### 步骤 3: STFT 幅度

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产环境使用 `torch.stft` 或 `librosa.stft`（基于FFT，向量化）。这里的循环是为了教学目的；它在 `code/main.py` 中运行于短片段上。

### 步骤 4: 梅尔滤波器组

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

覆盖0-8千赫兹的80个梅尔频段，滤波器数量 `n_fft=400`，生成一个 `(80, 201)` 矩阵。将形状为 `(n_frames, 201)` 的STFT幅度矩阵与该转置矩阵相乘，得到 `(n_frames, 80)` 梅尔频谱图。

### 步骤 5: 对数梅尔

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常用替代方案：`librosa.power_to_db`（基于参考归一化的分贝），`10 * log10(power + eps)`。Whisper使用更复杂的裁剪+归一化流程（参见Whisper的 `log_mel_spectrogram`）。

### 步骤 6: MFCCs

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每个对数梅尔帧应用DCT，保留前13个系数。这就是你的MFCC矩阵。第一个系数通常被丢弃（它编码了整体能量）。

## 实际应用

2026年的技术栈：

| 任务 | 特征 |
|------|------|
| 语音识别 (Whisper, Parakeet, SeamlessM4T) | 80维对数梅尔，10毫秒帧移，25毫秒窗长 |
| 语音合成声学模型 (VITS, F5-TTS, Kokoro) | 80维梅尔，5-12毫秒帧移以实现精细的时间控制 |
| 音频分类 (AST, PANNs, BEATs) | 128维对数梅尔，10毫秒帧移 |
| 说话人嵌入 (ECAPA-TDNN, WavLM) | 80维对数梅尔或原始波形自监督学习 |
| 音乐生成 (MusicGen, Stable Audio 2) | EnCodec离散token（非梅尔） |
| 关键词检测 | 40维MFCCs，适用于微型设备 |

经验法则：**如果你不是在做音乐相关的工作，就从80维对数梅尔开始。** 任何偏离都需要充分的理由。

## 2026年仍在出现的陷阱

- **梅尔数量不匹配。** 训练时用80个梅尔，推理时用128个梅尔。会导致静默失败。请在两端记录特征形状。
- **上游采样率不匹配。** 在22.05千赫兹下计算的梅尔特征与16千赫兹下计算的不同。请在特征提取之前就确定好采样率。
- **分贝 vs 对数。** Whisper期望的是对数梅尔，而不是分贝梅尔。一些HuggingFace流水线会自动检测；你的自定义代码不会。
- **归一化漂移。** 训练时使用逐话语归一化，推理时使用全局归一化。这是一个会使词错误率翻倍的生产环境bug。
- **填充泄漏。** 在音频片段末尾进行零填充会在尾部帧产生平坦的频谱。请使用对称填充或复制填充。

## 交付

保存为 `outputs/skill-feature-extractor.md`。该技能会根据目标模型选择特征类型、梅尔数量、帧长/帧移和归一化方法。

## 练习

1. **简单。** 运行 `code/main.py`。它会合成一个啁啾信号（频率从200赫兹扫到4000赫兹），并打印每帧中最大梅尔频段的索引。（可选）绘图并确认它与频率扫描相符。
2. **中等。** 在 `{40, 80, 128}` 中使用 `n_mels`，在 `{200, 400, 800}` 中使用 `frame_len`，重新运行。测量沿时间轴的尖锐峰带宽。哪种组合解析啁啾信号的效果最好？
3. **困难。** 实现 `power_to_db`，并比较一个小型CNN分类器在AudioMNIST数据集上使用 (a) 原始对数梅尔、(b) 带有 `ref=max` 的分贝梅尔、(c) MFCC-13 + delta + delta-delta 的语音识别准确率。报告Top-1准确率。

## 关键术语

| 术语 | 人们如何说 | 它的实际含义 |
|------|------------|--------------|
| 帧 (Frame) | 一个切片 | 送入一次FFT的25毫秒波形片段。 |
| 帧移 (Hop) | 步长 | 连续帧之间的样本间隔；10毫秒是语音识别的默认值。 |
| 窗函数 (Window) | 汉宁/哈明窗之类的东西 | 逐点乘法器，将帧的边缘衰减到零。 |
| STFT | 频谱图生成器 | 分帧 + 加窗后的FFT；生成 时间 × 频率 矩阵。 |
| 梅尔 (Mel) | 扭曲的频率 | 对数感知刻度；`m = 2595·log10(1 + f/700)`。 |
| 滤波器组 (Filterbank) | 那个矩阵 | 将STFT投影到梅尔频段的三角形滤波器。 |
| 对数梅尔 (Log-mel) | Whisper的输入 | `log(mel_spec + eps)`；2026年已标准化。 |
| MFCC | 老派特征 | 对数梅尔的DCT；13个系数，已去相关。 |

## 扩展阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCC论文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 原始的梅尔刻度论文。
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 阅读参考实现。
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) — `mfcc`、`melspectrogram` 和 帧移/窗函数的参考文档。
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — 用于Parakeet + Canary模型的生产规模流水线。