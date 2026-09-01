# 语谱图、梅尔刻度与音频特征

> 神经网络无法很好地直接消费原始波形。它们消费语谱图。它们更擅长消费梅尔语谱图。2026 年的每一个 ASR、TTS 和音频分类器都取决于这一个预处理选择。

**类型：** 构建
**语言：** Python
**前置知识：** Phase 6 · 01（音频基础）
**耗时：** 约 45 分钟

## 问题所在

取一段 10 秒、16 kHz 的音频片段。那是 160,000 个浮点数，全部位于 `[-1, 1]` 区间内，与"狗叫"或"单词 cat"这类标签几乎完全无关。原始波形包含信息，但处于模型难以直接提取的形式。两个相隔 100 ms 的相同音素，其原始采样值完全不同。

语谱图解决了这个问题。它将人类感知忽略的时间细节（微秒级抖动）压缩掉，同时保留了感知关注的结构（哪些频率随时间窗口约 10–25 ms 具有能量）。

梅尔语谱图进一步改进。人类对音高的感知是对数的：100 Hz 与 200 Hz 的听觉距离"等同于" 1000 Hz 与 2000 Hz。梅尔刻度对频率轴进行了非线性映射以匹配这一感知。梅尔语谱图是 2010 年至 2026 年间语音 ML 中最重要的特征。

## 概念解析

![Waveform to STFT to mel spectrogram to MFCC ladder](../assets/mel-features.svg)

**STFT（短时傅里叶变换）。** 将波形切分为重叠的帧（典型设置：25 ms 窗口，10 ms 步长 = 16 kHz 下 400 个采样点 / 160 个采样点）。将每帧乘以窗函数（Hann 是默认选项；Hamming 略有不同的权衡）。对每帧做 FFT。将幅度谱堆叠为形状为 `(n_frames, n_freq_bins)` 的矩阵。这就是你的语谱图。

**对数幅度。** 原始幅度跨越 5-6 个数量级。取 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 以压缩动态范围。所有生产管线都使用对数幅度，而非原始幅度。

**梅尔刻度。** 频率 `f`（Hz）通过 `m = 2595 * log10(1 + f / 700)` 映射为梅尔 `m`。该映射在 1 kHz 以下大致线性，在 1 kHz 以上大致对数。80 个梅尔频带覆盖 0–8 kHz 是标准 ASR 输入。

**梅尔滤波器组。** 一组在梅尔刻度上等间距排列的三角滤波器。每个滤波器是相邻 FFT 频bin 的加权和。将 STFT 幅度乘以滤波器组矩阵，一次矩阵乘法即可得到梅尔语谱图。

**对数梅尔语谱图。** `log(mel_spec + 1e-10)`。Whisper 的输入。Parakeet 的输入。SeamlessM4T 的输入。2026 年的通用音频前端。

**MFCC。** 对对数梅尔语谱图应用 DCT（II 型），保留前 13 个系数。去相关并进一步压缩。在约 2015 年 CNN/Transformer 在原始对数梅尔上赶上之前一直是主导特征。至今仍用于说话人识别（x-vector、ECAPA）。

**分辨率权衡。** 更大的 FFT = 更好的频率分辨率但更差的时间分辨率。25 ms / 10 ms 是音频 ML 的默认值；50 ms / 12.5 ms 用于音乐；5 ms / 2 ms 用于瞬态检测（鼓击、爆破音）。

```figure
spectrogram-window
```

## 动手实现

### 步骤 1：分帧波形

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

一段 10 秒、16 kHz 的片段，设置 `frame_len=400, hop=160`，将产生 998 帧。

### 步骤 2：Hann 窗

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

在 FFT 之前逐元素相乘。消除因在非零端点处截断而产生的频谱泄漏。

### 步骤 3：STFT 幅度

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

生产环境使用 `torch.stft` 或 `librosa.stft`（基于 FFT，向量化）。此处的循环是教学用途；它在 `code/main.py` 中对短片段运行。

### 步骤 4：梅尔滤波器组

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

使用 `n_fft=400` 时，80 个梅尔频带覆盖 0–8 kHz 会产生一个 `(80, 201)` 矩阵。将 `(n_frames, 201)` 的 STFT 幅度乘以转置矩阵，即可得到 `(n_frames, 80)` 的梅尔语谱图。

### 步骤 5：对数梅尔

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见替代方案：`librosa.power_to_db`（参考归一化的 dB）、`10 * log10(power + eps)`。Whisper 使用更复杂的 clip + normalize 流程（见 Whisper 的 `log_mel_spectrogram`）。

### 步骤 6：MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每个对数梅尔帧应用 DCT，保留前 13 个系数。这就是你的 MFCC 矩阵。第一个系数通常被丢弃（它编码整体能量）。

## 使用场景

2026 技术栈：

| 任务 | 特征 |
|------|----------|
| ASR（Whisper、Parakeet、SeamlessM4T） | 80 对数梅尔，10 ms 步长，25 ms 窗口 |
| TTS 声学模型（VITS、F5-TTS、Kokoro） | 80 个梅尔，5–12 ms 步长以实现精细时间控制 |
| 音频分类（AST、PANNs、BEATs） | 128 对数梅尔，10 ms 步长 |
| 说话人嵌入（ECAPA-TDNN、WavLM） | 80 对数梅尔或原始波形 SSL |
| 音乐（MusicGen、Stable Audio 2） | EnCodec 离散标记（非梅尔） |
| 关键词检测 | 40 个 MFCC（用于微型设备） |

经验法则：**如果不是处理音乐，从 80 对数梅尔开始。** 任何偏离都需要举证。

## 2026 年仍会踩的坑

- **梅尔数量不匹配。** 训练时用 80 个梅尔，推理时用 128 个梅尔。静默失败。在两端记录特征形状。
- **上游采样率不匹配。** 在 22.05 kHz 下计算的梅尔与 16 kHz 下看起来不同。在特征提取**之前**修复采样率。
- **dB 与对数。** Whisper 期望对数梅尔，而非 dB 梅尔。某些 HF 管线会自动检测；自定义代码不会。
- **归一化漂移。** 训练时逐 utterance 归一化，推理时全局归一化。导致 WER 翻倍的常见生产 bug。
- **填充泄漏。** 对片段末尾进行零填充会在尾部帧中产生平坦频谱。使用对称填充或复制填充。

## 交付

保存为 `outputs/skill-feature-extractor.md`。该技能根据目标模型选择特征类型、梅尔数量、帧长/步长和归一化方式。

## 练习

1. **简单。** 运行 `code/main.py`。它会合成一个线性调频信号（频率从 200 扫描到 4000 Hz），并打印每帧的最大值梅尔频bin。绘制图表（可选）并确认与扫描匹配。
2. **中等。** 使用 `n_mels` 在 `{40, 80, 128}` 和 `frame_len` 在 `{200, 400, 800}` 中重新运行。测量沿时间轴的锐峰带宽。哪种组合最能解析线性调频信号？
3. **困难。** 实现 `power_to_db` 并比较使用 (a) 原始对数梅尔、(b) 参考最大值 dB 梅尔、(c) MFCC-13 + delta + delta-delta 时 AudioMNIST 上小型 CNN 分类器的 ASR 准确率。报告 top-1 准确率。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| Frame | 一段切片 | 喂给单次 FFT 的 25 ms 波形块。 |
| Hop | 步长 | 连续帧之间的采样点数；10 ms 是 ASR 默认值。 |
| Window | Hann/Hamming 窗函数 | 逐点乘数，将帧边缘衰减至零。 |
| STFT | 语谱图生成器 | 分帧 + 加窗的 FFT；产生时间 × 频率矩阵。 |
| Mel | 非线性频率 | 对数感知尺度；`m = 2595·log10(1 + f/700)`。 |
| Filterbank | 滤波器组矩阵 | 将 STFT 投影到梅尔频带的三角滤波器。 |
| Log-mel | Whisper 的输入 | `log(mel_spec + eps)`；2026 年已标准化。 |
| MFCC | 传统特征 | 对数梅尔的 DCT；13 个系数，已去相关。 |

## 延伸阅读

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCC 论文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 原始梅尔刻度论文。
- [OpenAI — Whisper 源码，log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 阅读参考实现。
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) — `mfcc`、`melspectrogram` 及 hop/window 的参考文档。
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Parakeet 和 Canary 模型的生产级管线。
