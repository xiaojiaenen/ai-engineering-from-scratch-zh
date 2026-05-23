# 音频基础 — 波形、采样、傅里叶变换

> 波形是原始信号。频谱图是其表示形式。梅尔特征则是适合机器学习的形式。每一个现代语音识别和文本转语音的流程都遵循这条路径，而第一步就是理解采样和傅里叶变换。

**类型：** 学习
**语言：** Python
**先决条件：** 阶段1 · 06（向量与矩阵），阶段1 · 14（概率分布）
**时间：** ~45分钟

## 问题所在

麦克风产生压力-时间信号。你的神经网络消耗张量。介于两者之间是一套惯例，违反它们会产生隐蔽错误：模型训练得不错，但词错误率翻倍，或者文本转语音输出噪音，又或者语音克隆系统记住了麦克风的特性而非说话人的声音。

语音系统中的每个错误都可追溯到以下三个问题之一：

1.  数据录制时的采样率是多少？模型期望的采样率是多少？
2.  信号是否存在混叠？
3.  你是在原始样本上操作还是在频率表示上操作？

弄明白这些，阶段6的其余部分就易于处理了。搞错了，即便是 Whisper-Large-v4 也会输出垃圾。

## 核心概念

![波形、采样、DFT 和频率箱可视化](../assets/audio-fundamentals.svg)

**波形。** 一个一维浮点数数组，范围在 `[-1.0, 1.0]`。以样本编号索引。要转换为秒，需除以采样率：`t = n / sr`。一个在16 kHz下10秒长的片段，是一个包含160,000个浮点数的数组。

**采样率 (sr)。** 每秒的样本数。2026年的常见采样率：

| 采样率 | 用途 |
|------|-----|
| 8 kHz | 电话，传统网络电话。奈奎斯特频率在4 kHz，会损害辅音清晰度。避免用于语音识别。 |
| 16 kHz | 语音识别标准。Whisper、Parakeet、SeamlessM4T v2 均处理16 kHz输入。 |
| 22.05 kHz | 旧版模型的文本转语音声码器训练。 |
| 24 kHz | 现代文本转语音（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD音质，音乐。 |
| 48 kHz | 电影，专业音频，高保真文本转语音（VALL-E 2，NaturalSpeech 3）。 |

**奈奎斯特-香农定理。** 采样率为 `sr` 时，可以无歧义地表示最高至 `sr/2` Hz 的频率。这个 `sr/2` 边界称为*奈奎斯特频率*。高于奈奎斯特频率的能量会发生*混叠* — 折叠到较低频率 — 并破坏信号。在降采样前务必进行低通滤波。

**位深度。** 16位PCM（有符号int16，范围±32,767）是通用的交换格式。24位用于音乐，32位浮点用于内部数字信号处理。像 `soundfile` 这样的库读取int16数据，但会以 `[-1, 1]` 形式暴露float32数组。

**傅里叶变换。** 任何有限信号都是不同频率正弦波的叠加。离散傅里叶变换对于 `N` 个样本，计算 `N` 个复系数 — 每个频率箱对应一个。`bin k` 对应频率 `k · sr / N` Hz。幅度是该频率的振幅，角度是相位。

**FFT。** 快速傅里叶变换：当 `N` 是2的幂时，一种用于计算DFT的 `O(N log N)` 算法。每个音频库底层都使用FFT。在16 kHz下对1024个样本进行FFT，会得到512个可用频率箱，覆盖0–8 kHz，分辨率为15.6 Hz。

**分帧与加窗。** 我们不会对整个片段做FFT。我们将其切成重叠的*帧*（通常25毫秒，帧移10毫秒），将每帧乘以一个窗函数（如汉宁窗、汉明窗）以消除边缘不连续性，然后对每帧做FFT。这就是短时傅里叶变换。第02课将从这里继续。

## 动手实践

### 步骤1：读取音频片段并绘制波形

`code/main.py` 仅使用标准库 `wave` 模块，以保持演示无外部依赖。在生产环境中，你会使用 `soundfile` 或 `torchaudio.load`（两者都返回 `(waveform, sr)` 元组）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 步骤2：从基本原理合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

一个在16 kHz下持续1秒的440 Hz正弦波（音乐会标准音A）是16,000个浮点数。使用 `wave.open(..., "wb")` 以16位PCM编码写入。

### 步骤3：手动计算DFT

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

`O(N²)` — 对于 `N=256` 来说足够好，可以验证正确性，但对实际音频无用。实际代码调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 步骤4：找到主频

幅度峰值索引 `k_star` 对应频率 `k_star * sr / N`。在440 Hz正弦波上运行此计算，应会在频率箱 `440 * N / sr` 处返回峰值。

### 步骤5：演示混叠

以10 kHz采样7 kHz正弦波（奈奎斯特频率=5 kHz）。7 kHz音调高于奈奎斯特频率，会折叠到 `10 − 7 = 3 kHz`。FFT峰值出现在3 kHz处。这是经典的混叠演示，也是每个DAC/ADC都配备砖墙式低通滤波器的原因。

## 实际应用

你将在2026年实际使用的工具栈：

| 任务 | 库 | 原因 |
|------|---------|-----|
| 读写WAV/FLAC/OGG | `soundfile` (libsndfile 封装) | 最快，稳定，返回float32。 |
| 重采样 | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确的抗混叠滤波。 |
| STFT / 梅尔特征 | `torchaudio` 或 `librosa` | 友好GPU；PyTorch生态系统。 |
| 实时流处理 | `sounddevice` 或 `pyaudio` | 跨平台PortAudio绑定。 |
| 检查文件信息 | `ffprobe` 或 `soxi` | 命令行工具，快速，报告采样率/通道/编码。 |

决策规则：**首先匹配采样率，再考虑其他**。Whisper期望16 kHz单声道float32数据。传入44.1 kHz立体声数据，你会得到看起来像模型错误的垃圾输出。

## 交付使用

保存为 `outputs/skill-audio-loader.md`。此技能帮助你检查音频输入是否符合下游模型的预期，并在不匹配时进行正确的重采样。

## 练习

1.  **简单。** 在16 kHz下合成一段1秒长的混合音频，包含220 Hz + 440 Hz + 880 Hz三个频率。运行DFT。确认在预期的频率箱处出现三个峰值。
2.  **中等。** 用你的声音录制一段48 kHz、3秒长的WAV文件。使用 `torchaudio.transforms.Resample`（带抗混叠）将其降采样到16 kHz，然后使用朴素抽取法（每三个样本取一个）也降采样到16 kHz。对两者进行FFT。混叠出现在哪里？
3.  **困难。** 仅使用 `math` 和步骤3中的DFT，从零开始构建STFT。帧大小400，帧移160，汉宁窗。使用 `matplotlib.pyplot.imshow` 绘制幅度图。这就是第02课的频谱图。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|-----------------------|
| 采样率 | 每秒多少个样本 | ADC以该频率（Hz）测量信号。 |
| 奈奎斯特频率 | 你能表示的最大频率 | `sr/2`；高于此频率的能量会混叠回来。 |
| 位深度 | 每个样本的分辨率 | `int16` = 65,536个级别；`float32` = 在 `[-1, 1]` 中具有24位精度。 |
| 离散傅里叶变换 (DFT) | 序列的傅里叶变换 | `N` 个样本 → `N` 个复数频率系数。 |
| 快速傅里叶变换 (FFT) | 快速的DFT | `O(N log N)` 算法，要求 `N` = 2的幂。 |
| 频率箱 | 频率列 | `k · sr / N` Hz；分辨率 = `sr / N`。 |
| 短时傅里叶变换 (STFT) | 底层的频谱图 | 随时间进行的加窗、分帧FFT。 |
| 混叠 | 奇怪的频率幻影 | 高于奈奎斯特频率的能量镜像映射到较低的频率箱。 |

## 延伸阅读

- [香农 (1949)。《噪声下的通信》](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 采样定理背后的论文。
- [史密斯 — 《工程师与科学家的数字信号处理指南》](https://www.dspguide.com/ch8.htm) — 免费、权威的DSP教材。
- [librosa文档 — 音频入门](https://librosa.org/doc/latest/tutorial.html) — 带代码的实践教程。
- [海因里希·库特鲁夫 — 《室内声学》（第六版）](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — 解释现实世界音频并非纯正弦波的参考书。
- [史蒂夫·埃丁斯 — FFT解释笔记本](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 10分钟理清频率箱直觉。