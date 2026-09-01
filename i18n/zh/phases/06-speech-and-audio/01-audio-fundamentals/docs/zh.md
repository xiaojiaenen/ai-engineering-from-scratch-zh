# 音频基础 — 波形、采样、傅里叶变换

> 波形是原始信号。频谱图是表示形式。Mel 特征是对 ML 友好的形式。每个现代 ASR 和 TTS 流水线都沿着这个阶梯向上，而第一级台阶是理解采样和傅里叶变换。

**类型:** 学习
**语言:** Python
**前置知识:** 阶段 1 · 06（向量与矩阵），阶段 1 · 14（概率分布）
**时间:** 约 45 分钟

## 问题所在

麦克风产生的是压力-时间信号。你的神经网络消费的是张量。在两者之间存在一整套约定，违反它们就会产生无声的 bug：模型训练看起来正常，但 WER 翻倍；TTS 交付的音频带嘶嘶声；或者语音克隆系统记住了麦克风而不是说话人。

语音系统中的每一个 bug 都可以追溯到以下三个问题之一：

1. 数据以什么采样率录制，模型期望的是什么？
2. 信号是否混叠？
3. 你操作的是原始样本还是频率表示？

把这些弄对了，阶段 6 的其余部分就顺理成章。把这些弄错了，即便是 Whisper-Large-v4 也会产出垃圾。

## 核心概念

![波形、采样、DFT 和频率 bin 可视化](../assets/audio-fundamentals.svg)

**波形。** 一个取值范围在 `[-1.0, 1.0]` 的一维浮点数组。按样本编号索引。要转换为秒，除以采样率：`t = n / sr`。一段 16 kHz 下的 10 秒音频就是一个包含 160,000 个浮点的数组。

**采样率（sr）。** 每秒多少个样本。2026 年常见的采样率：

| 采样率 | 用途 |
|------|-----|
| 8 kHz | 电话、传统 VOIP。奈奎斯特在 4 kHz，会丢掉辅音。避免用于 ASR。 |
| 16 kHz | ASR 标准。Whisper、Parakeet、SeamlessM4T v2 都以 16 kHz 作为输入。 |
| 22.05 kHz | 旧版 TTS 声码器训练的采样率。 |
| 24 kHz | 现代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD 音频、音乐。 |
| 48 kHz | 电影、专业音频、高保真 TTS（VALL-E 2、NaturalSpeech 3）。 |

**奈奎斯特-香农。** 采样率为 `sr` 时，可以无歧义地表示高达 `sr/2` 的频率。`sr/2` 的边界就是 *奈奎斯特频率*。高于奈奎斯特频率的能量会发生*混叠*——折叠到较低的频率中——从而污染信号。下采样前务必做低通滤波。

**比特深度。** 16-bit PCM（有符号 int16，范围 ±32,767）是通用的交换格式。音乐用 24-bit，内部 DSP 用 32-bit float。`soundfile` 等库读取 int16，但以 float32 数组形式暴露值在 `[-1, 1]` 范围内。

**傅里叶变换。** 任何有限信号都是不同频率的正弦波之和。离散傅里叶变换（DFT）对 `N` 个样本计算 `N` 个复系数——每个频率 bin 一个。`bin k` 对应频率 `k · sr / N` Hz。模是此频率处的振幅，角度是相位。

**FFT。** 快速傅里叶变换：当 `N` 是 2 的幂时，DFT 的 `O(N log N)` 算法。每个音频库底层都在使用 FFT。在 16 kHz 下做一个 1024 样本的 FFT，得到 512 个可用频率 bin，覆盖 0–8 kHz，分辨率为 15.6 Hz。

**分帧 + 加窗。** 我们不会对整段音频做 FFT。而是将其切成重叠的*帧*（通常为 25 ms，步长 10 ms），将每帧乘以窗函数（Hann、Hamming）以消除边缘不连续，然后对每帧做 FFT。这就是短时傅里叶变换（STFT）。第 02 课将从这里继续。

```figure
mel-scale
```

## 动手实现

### 第 1 步：读取一段音频并绘制波形

`code/main.py` 仅使用标准库 `wave` 模块来保持演示无第三方依赖。在生产环境中你会使用 `soundfile` 或 `torchaudio.load`（两者都返回 `(waveform, sr)` 元组）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 第 2 步：从零开始合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

16 kHz 下 1 秒的 440 Hz 正弦波（标准 A 音）包含 16,000 个浮点。用 `wave.open(..., "wb")` 写入 16-bit PCM 编码。

### 第 3 步：手工计算 DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` —— 对于 `N=256` 验证正确性绰绰有余，但对真实音频毫无用处。实际代码会调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 第 4 步：找主导频率

幅度峰值的索引 `k_star` 对应频率 `k_star * sr / N`。在 440 Hz 正弦波上运行此代码，应在 bin `440 * N / sr` 处得到峰值。

### 第 5 步：演示混叠效应

以 10 kHz 采样率采样 7 kHz 正弦波（奈奎斯特 = 5 kHz）。7 kHz 音调高于奈奎斯特频率，折叠到 `10 − 7 = 3 kHz`。FFT 峰值出现在 3 kHz。这是经典的混叠演示，也是每个 DAC/ADC 都附带砖墙式低通滤波器的原因。

## 使用指南

你在 2026 年实际要用的工具栈：

| 任务 | 库 | 理由 |
|------|---------|-----|
| 读写 WAV/FLAC/OGG | `soundfile`（libsndfile 封装） | 最快、最稳定，返回 float32。 |
| 重采样 | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确的抗混叠。 |
| STFT / Mel | `torchaudio` 或 `librosa` | GPU 友好；PyTorch 生态。 |
| 实时流式 | `sounddevice` 或 `pyaudio` | 跨平台 PortAudio 绑定。 |
| 检查文件 | `ffprobe` 或 `soxi` | CLI，快速，报告 sr/声道/编码格式。 |

决策规则：**先匹配采样率，再匹配其他一切**。Whisper 期望 16 kHz 单声道 float32。给它传 44.1 kHz 立体声，你会得到看起来像模型 bug 的垃圾。

## 交付

保存为 `outputs/skill-audio-loader.md`。该 skill 帮助你确认音频输入是否与下游模型的期望一致，并在不一致时正确重采样。

## 练习

1. **简单。** 在 16 kHz 下合成 220 Hz + 440 Hz + 880 Hz 的 1 秒混音。运行 DFT。确认在预期 bin 处出现三个峰值。
2. **中等。** 以 48 kHz 录制 3 秒的人声 WAV。先用 `torchaudio.transforms.Resample`（带抗混叠）下采样到 16 kHz，再用朴素抽取（每第三个样本）下采样到 16 kHz。对两者做 FFT。混叠出现在哪里？
3. **困难。** 仅使用 `math` 和第 3 步的 DFT 从零构建 STFT。帧长 400，步长 160，Hann 窗。用 `matplotlib.pyplot.imshow` 绘制幅度。这就是第 02 课的频谱图。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|-----------------------|
| 采样率 | 每秒多少样本 | ADC 测量信号的频率（Hz）。 |
| 奈奎斯特 | 可表示的最高频率 | `sr/2`；高于此频率的能量会混叠回低频。 |
| 比特深度 | 每个样本的精度 | `int16` = 65,536 个层级；`float32` = 在 `[-1, 1]` 范围内 24-bit 精度。 |
| DFT | 序列的傅里叶变换 | `N` 个样本 → `N` 个复频率系数。 |
| FFT | 快速 DFT | `O(N log N)` 算法，要求 `N` = 2 的幂。 |
| Bin | 频率列 | `k · sr / N` Hz；分辨率 = `sr / N`。 |
| STFT | 频谱图的底层 | 随时间分帧 + 加窗的 FFT。 |
| 混叠 | 奇怪的频率鬼影 | 高于奈奎斯特频率的能量镜像折叠到低频 bin。 |

## 延伸阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) —— 采样定理背后的论文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) —— 免费、经典的 DSP 教材。
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) —— 附带代码的实用指南。
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) —— 解释为什么真实世界音频不是干净正弦波的参考书。
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) —— 10 分钟讲清频率 bin 直觉。
