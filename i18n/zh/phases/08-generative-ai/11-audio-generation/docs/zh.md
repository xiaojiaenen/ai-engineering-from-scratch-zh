# 音频生成

> 音频是 16-48 kHz 的 1D 信号。一段五秒的音频包含 80k-240k 个采样点。没有任何 transformer 会直接处理如此长的序列。2026 年所有主流音频生成模型的解决方案都是相同的：先使用神经编解码器（Encodec、SoundStream、DAC）将音频压缩为 50-75 Hz 的离散 token，再由 transformer 或扩散模型生成这些 token。

**类型：** Build
**语言：** Python
**前置知识：** Phase 6 · 02（音频特征）、Phase 6 · 04（ASR）、Phase 8 · 06（DDPM）
**预计时间：** 约 45 分钟

## 问题概述

三类音频生成任务：

1. **语音合成（Text-to-speech）。** 给定文本，生成语音。清晰语音属于窄带信号且具有强语音结构——用基于 token 的 transformer 可良好解决。代表工作：VALL-E（Microsoft）、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **音乐生成。** 给定提示（文本、旋律、和弦进行、风格），生成音乐。分布更宽泛。代表工作：MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **音效生成 / 声音设计。** 给定提示，生成环境音或拟音。代表工作：AudioGen、AudioLDM 2、Stable Audio Open。

这三类任务共享同一基础架构：神经音频编解码器 + token 自回归或扩散生成器。

## 核心概念

![音频生成：编解码 token + transformer 或扩散模型](../assets/audio-generation.svg)

### 神经音频编解码器

Encodec（Meta，2022）、SoundStream（Google，2021）、Descript Audio Codec（DAC，2023）。卷积编码器将波形压缩为逐时序向量；残差矢量量化（RVQ）将每个向量转换为 K 个码本索引的级联。解码器反向操作。以 24 kHz 音频、2 kbps 码率、使用 8 个 RVQ 码本、75 Hz 输出速率为例，生成速度为 600 tokens/秒。

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### 上层两种生成范式

**Token 自回归。** 将 RVQ token 展平为序列，运行仅解码器架构的 transformer。MusicGen 使用"延迟并行"机制，以每路偏移的方式并行输出 K 条码本流。VALL-E 从文本提示 + 3 秒语音样本生成语音 token。

**潜在扩散。** 将编解码 token 打包为连续潜在变量，或使用类别型扩散建模。Stable Audio 2.5 在连续音频潜在变量上使用流匹配。AudioLDM 2 采用文本→梅尔谱→音频的扩散流程。

2024-2026 年的趋势：流匹配在音乐领域占据优势（推理更快、音质更纯净），而 token 自回归仍在语音领域占主导，因为其天然具有因果性且易于流式输出。

## 生产格局

| 系统 | 任务 | 骨干模型 | 延迟 |
|------|------|----------|------|
| ElevenLabs V3 | TTS | Token-AR + 神经声码器 | 首 token ~300ms |
| OpenAI GPT-4o audio | 全双工语音 | 端到端多模态 AR | ~200ms |
| NaturalSpeech 3 | TTS | 潜在流匹配 | 非流式 |
| Stable Audio 2.5 | 音乐 / 音效 | DiT + 音频潜在流匹配 | 1 分钟片段约 10s |
| Suno v4 | 完整歌曲 | 未公开；疑似 token-AR | 每首歌约 30s |
| Udio v1.5 | 完整歌曲 | 未公开 | 每首歌约 30s |
| MusicGen 3.3B | 音乐 | Encodec 32kHz 上的 Token-AR | 实时 |
| AudioCraft 2 | 音乐 + 音效 | 流匹配 | 5 秒片段约 5s |
| Riffusion v2 | 音乐 | 谱图扩散 | 约 10s |

```figure
score-matching
```

## 动手实践

`code/main.py` 模拟了核心思路：在两类不同"风格"（风格 A 为交替的高低 token，风格 B 为单调递增序列）生成的合成"音频 token"序列上，训练一个小型 next-token transformer。以风格为条件并进行采样。

### 步骤 1：合成音频 token

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### 步骤 2：训练小型 token 预测器

一个受风格条件约束的双词模型（bigram）预测器。重点在于展示模式：编解码 token → 交叉熵训练 → 自回归采样。

### 步骤 3：条件采样

给定风格 token 和起始 token，从预测分布中采样下一个 token。重复 20-40 个 token。

## 常见陷阱

- **编解码器质量决定输出上限。** 若编解码器无法忠实表征某类声音，生成器质量再高也无济于事。DAC 是目前开源方案中的最佳选择。
- **RVQ 误差累积。** 每个 RVQ 层建模上一层的残差。第一层的误差会逐级传播。对高层采样时使用温度参数 0 可缓解此问题。
- **音乐结构。** 75 Hz 下 30 秒的 token 序列超过 20k 个 token，对 transformer 构成挑战。MusicGen 使用滑动窗口 + 提示续写；Stable Audio 使用短片段 + 交叉淡入淡出。
- **边界伪影。** 生成片段的交叉淡入淡出需要精细的重叠叠加处理。
- **对高质量数据的渴求。** 音乐生成器需要数万小时的授权音乐数据。Suno / Udio 的 RIAA 诉讼（2024 年）将此问题推向台前。
- **声音克隆伦理。** 仅需 3 秒样本加文本提示，VALL-E / XTTS / ElevenLabs 即可克隆人声。每个生产模型都需要滥用检测机制和禁用名单。

## 应用场景

| 任务 | 2026 年技术栈 |
|------|--------------|
| 商业 TTS | ElevenLabs、OpenAI TTS 或 Azure Neural |
| 声音克隆（已获授权） | XTTS v2（开源）或 ElevenLabs Pro |
| 背景音乐（快速生成） | Stable Audio 2.5 API、Suno 或 Udio |
| 带歌词的音乐 | Suno v4 或 Udio v1.5 |
| 音效 / 拟音 | AudioCraft 2、ElevenLabs SFX 或 Stable Audio Open |
| 实时语音代理 | GPT-4o realtime 或 Gemini Live |
| 开源权重音乐研究 | MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2 |
| 配音 / 翻译 | HeyGen、ElevenLabs Dubbing |

## 交付成果

保存 `outputs/skill-audio-brief.md`。技能输入一份音频需求简报（任务、时长、风格、音色、版权），输出：模型选型与托管方案、提示格式（风格标签、风格描述、结构标记）、编解码器 + 生成器 + 声码器链路、种子协议，以及评估方案（MOS / CLAP 分数 / TTS 的 CER / 用户 A/B 测试）。

## 练习题

1. **简单。** 运行 `code/main.py` 并显式设置风格。验证生成序列是否符合对应风格的模式。
2. **中等。** 添加延迟并行解码：模拟 2 条必须保持 1 步偏移的 token 流。训练联合预测器。
3. **困难。** 使用 HuggingFace transformers 本地运行 MusicGen-small。用三个不同提示各生成一段 10 秒音频，进行风格一致性的 A/B 测试。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Codec | "神经压缩" | 音频编/解码器；典型输出为 50-75 Hz 的 token。 |
| RVQ | "残差 VQ" | K 个量化器的级联；每一层建模上一级的残差。 |
| Token | "一个编解码符号" | 码本中的离散索引；通常为 1024 或 2048。 |
| Delayed parallel | "偏移码本" | 以交错偏移输出 K 条 token 流以降低序列长度。 |
| Flow matching | "2024 年的音频赢家" | 扩散的更直线路径替代方案；采样更快。 |
| Voice prompt | "3 秒样本" | 引导克隆音色的说话人嵌入或 token 前缀。 |
| Mel spectrogram | "可视化频谱" | 对数幅度感知频谱图；被许多 TTS 系统使用。 |
| Vocoder | "Mel 转波形" | 将 Mel 频谱图还原为音频的神经网络组件。 |

## 生产笔记：音频是流式问题

音频是唯一一种用户期望"边生成边播放"的输出模态，而非一次性渲染完毕。在生产环境中，这意味着 TPOT（每输出 token 耗时）至关重要——用户的收听速度就是目标吞吐量，而非阅读速度。对于 16 kHz 音频、以约 75 tokens/秒编解码（Encodec）的场景，服务器必须以 ≥75 tokens/秒的速度生成，才能保持流畅播放。

由此产生两个架构后果：

- **流匹配音频模型无法轻易流式输出。** Stable Audio 2.5 和 AudioCraft 2 一次性渲染固定长度的片段。要实现流式输出，需将片段分块并在边界处重叠——类似于滑动窗口扩散——相比编解码器 AR 模型会增加 100-300ms 的延迟开销。

若产品定位是"实时语音对话"或"实时音乐续写"，请选择编解码器 AR 方案；若产品定位是"提交后渲染 30 秒片段"，流匹配在质量和总延迟上更具优势。

## 延伸阅读

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) —— 编解码器标准。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) —— 首个广泛使用的神经音频编解码器。
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) —— DAC。
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) —— VALL-E。
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) —— MusicGen。
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) —— AudioLDM 2。
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) —— 2025 年基于流匹配的文生音乐模型。
