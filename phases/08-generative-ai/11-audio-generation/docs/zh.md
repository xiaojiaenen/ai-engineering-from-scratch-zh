# 音频生成

> 音频是一种采样率为16-48 kHz的一维信号。一个5秒的片段包含8-24万个采样点。没有Transformer能直接处理如此长的序列。截至2026年，所有生产级音频模型的解决方案都相同：使用神经编解码器（Encodec、SoundStream、DAC）将音频压缩为50-75 Hz的离散token，然后由Transformer或扩散模型生成这些token。

**类型：** 构建
**语言：** Python
**先修课程：** 第6阶段 · 02（音频特征），第6阶段 · 04（自动语音识别），第8阶段 · 06（去噪扩散概率模型）
**时长：** ~45分钟

## 问题定义

三项音频生成任务：

1.  **文本转语音。** 给定文本，生成语音。清晰的语音是窄带信号并具有强音素结构——非常适合用基于token的Transformer解决。代表系统：VALL-E（微软）、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2.  **音乐生成。** 给定提示（文本、旋律、和弦进行、风格），生成音乐。分布范围更广。代表系统：MusicGen（Meta）、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3.  **音效/声音设计。** 给定提示，生成环境音或拟音。代表系统：AudioGen、AudioLDM 2、Stable Audio Open。

这三者都运行在相同的底层架构上：神经音频编解码器 + token自回归模型或扩散生成器。

## 核心概念

![音频生成：编解码器token + Transformer或扩散模型](../assets/audio-generation.svg)

### 神经音频编解码器

Encodec（Meta，2022）、SoundStream（Google，2021）、Descript Audio Codec（DAC，2023）。卷积编码器将波形压缩为每个时间步的向量；残差向量量化（RVQ）将每个向量转换为K个码本索引的级联。解码器执行逆操作。以24 kHz采样率、2 kbps比特率、使用8个75 Hz的RVQ码本，相当于每秒600个token。

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### 两种顶层生成范式

**Token自回归。** 将RVQ token展平为序列，运行仅解码器的Transformer。MusicGen使用“延迟并行”方式并行输出K个码本流，每个流具有偏移量。VALL-E通过文本提示+3秒语音样本来生成语音token。

**潜在扩散。** 将编解码器token作为连续潜在变量打包，或用分类扩散建模。Stable Audio 2.5对连续音频潜在变量使用流匹配。AudioLDM 2使用文本到梅尔频谱到音频的扩散流程。

2024-2026年的趋势：流匹配在音乐生成中占据优势（推理更快，样本更干净），而token自回归在语音生成中仍占主导，因为它天然具有因果性且利于流式传输。

## 生产级应用现状

| 系统 | 任务 | 骨干架构 | 延迟 |
|--------|------|----------|---------|
| ElevenLabs V3 | 文本转语音 | Token自回归 + 神经声码器 | ~300ms首个token |
| OpenAI GPT-4o audio | 全双工对话 | 端到端多模态自回归模型 | ~200ms |
| NaturalSpeech 3 | 文本转语音 | 潜在流匹配 | 非流式 |
| Stable Audio 2.5 | 音乐/音效 | DiT + 音频潜在空间流匹配 | ~10秒生成1分钟片段 |
| Suno v4 | 完整歌曲 | 未公开；推测为token自回归 | ~30秒/首 |
| Udio v1.5 | 完整歌曲 | 未公开 | ~30秒/首 |
| MusicGen 3.3B | 音乐 | 基于Encodec 32kHz的token自回归 | 实时 |
| AudioCraft 2 | 音乐 + 音效 | 流匹配 | ~5秒生成5秒片段 |
| Riffusion v2 | 音乐 | 频谱图扩散 | ~10秒 |

## 动手构建

`code/main.py` 模拟了核心思想：在一个从两种不同“风格”（风格A为交替的低值和高值token，风格B为单调递增序列）生成的合成“音频token”序列上，训练一个微型next-token Transformer。基于风格条件并进行采样。

### 步骤1：生成合成音频token

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### 步骤2：训练一个微型token预测器

一个基于风格条件的二元语法风格预测器。重点在于模式：编解码器token → 交叉熵训练 → 自回归采样。

### 步骤3：条件采样

给定风格token和起始token，从预测分布中采样下一个token。持续采样20-40个token。

## 常见陷阱

- **编解码器质量决定输出质量上限。** 如果编解码器无法忠实表示某种声音，生成器再好也无济于事。DAC是当前开源最佳选择。
- **RVQ误差累积。** 每个RVQ层建模前一层的残差。第1层的误差会传播。对更高层进行温度为0的采样有助于缓解。
- **音乐结构。** 30秒的token在75 Hz下是2万多个token。对Transformer来说很困难。MusicGen使用滑动窗口+提示续写；Stable Audio使用较短片段+交叉淡入淡出。
- **边界伪影。** 在生成的片段之间进行交叉淡入淡出需要仔细的重叠相加处理。
- **对干净数据的需求。** 音乐生成器需要数万小时经授权的音乐。Suno/Udio与RIAA的诉讼（2024年）凸显了此问题。
- **语音克隆伦理。** 一个3秒的样本加一个文本提示，就足以让VALL-E / XTTS / ElevenLabs克隆语音。每个生产模型都需要滥用检测机制和退出名单。

## 应用场景

| 任务 | 2026技术栈 |
|------|------------|
| 商业文本转语音 | ElevenLabs, OpenAI TTS, 或 Azure Neural |
| 语音克隆（经同意验证） | XTTS v2（开源）或 ElevenLabs Pro |
| 背景音乐，快速生成 | Stable Audio 2.5 API, Suno, 或 Udio |
| 带歌词音乐 | Suno v4 或 Udio v1.5 |
| 音效/拟音 | AudioCraft 2, ElevenLabs SFX, 或 Stable Audio Open |
| 实时语音代理 | GPT-4o realtime 或 Gemini Live |
| 开源权重音乐研究 | MusicGen 3.3B, Stable Audio Open 1.0, AudioLDM 2 |
| 配音/翻译 | HeyGen, ElevenLabs Dubbing |

## 部署实践

保存 `outputs/skill-audio-brief.md`。该技能接收一个音频简报（任务、时长、风格、语音、许可）并输出：模型+托管方案、提示格式（流派标签、风格描述符、结构标记）、编解码器+生成器+声码器链、种子协议以及评估计划（MOS / CLAP分数 / TTS的字符错误率 / 用户A/B测试）。

## 练习

1.  **简单。** 运行 `code/main.py` 并显式设置风格。验证生成的序列是否符合该风格的模式。
2.  **中等。** 添加延迟并行解码：模拟2个token流，它们必须保持1步的偏移。训练一个联合预测器。
3.  **困难。** 使用HuggingFace transformers在本地运行MusicGen-small。用三个不同的提示生成10秒片段；进行A/B测试以评估风格一致性。

## 关键术语

| 术语 | 人们常说 | 其实际含义 |
|------|-----------------|-----------------------|
| 编解码器 | “神经压缩” | 音频的编码器/解码器；典型输出是50-75 Hz的token。 |
| RVQ | “残差VQ” | K个量化器的级联；每个量化器建模前一个的残差。 |
| Token | “一个编解码器符号” | 码本中的离散索引；通常为1024或2048。 |
| 延迟并行 | “偏移码本” | 以交错偏移输出K个token流以减少序列长度。 |
| 流匹配 | “2024年音频领域的胜利” | 比扩散路径更直的替代方案；采样更快。 |
| 语音提示 | “3秒样本” | 引导克隆语音的说话人嵌入或token前缀。 |
| 梅尔频谱图 | “那个可视化图” | 对数幅度感知频谱图；被许多TTS系统使用。 |
| 声码器 | “梅尔转波形” | 将梅尔频谱图转换回音频的神经组件。 |

## 生产须知：音频是流式处理问题

音频是唯一一种用户期望*在生成过程中*就到达的输出模态，而非一次性全部呈现。在生产环境中，这意味着TPOT（每个输出token的生成时间）很重要，因为用户的目标吞吐量是他们的*收听速度*——而不是阅读速度。对于以~75 token/秒（Encodec）进行token化的16 kHz音频，服务器必须为每位用户生成≥75个token/秒才能保持播放流畅。

这带来两个架构后果：

- **流匹配音频模型无法轻松实现流式传输。** Stable Audio 2.5和AudioCraft 2是一次性渲染固定长度的片段。要实现流式处理，你需要将片段分块并处理边界重叠——类似于滑动窗口扩散——这会增加100-300毫秒的延迟开销，相比编解码器自回归模型。

如果产品是“实时语音聊天”或“实时音乐续写”，请选择编解码器自回归路径。如果是“提交后渲染一个30秒片段”，流匹配在质量和总延迟方面胜出。

## 扩展阅读

- [Défossez 等 (2022). Encodec: 高保真神经音频压缩](https://arxiv.org/abs/2210.13438) — 编解码器标准。
- [Zeghidour 等 (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 首个被广泛使用的神经音频编解码器。
- [Kumar 等 (2023). 改进RVQGAN的高保真音频压缩 (DAC)](https://arxiv.org/abs/2306.06546) — DAC。
- [Wang 等 (2023). 神经编解码器语言模型是零样本文本到语音合成器 (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E。
- [Copet 等 (2023). 简单可控的音乐生成 (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen。
- [Liu 等 (2023). AudioLDM 2: 通过自监督预训练学习整体音频生成](https://arxiv.org/abs/2308.05734) — AudioLDM 2。
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — 2025年采用流匹配的文本到音乐模型。