# 音乐生成 — MusicGen、Stable Audio、Suno 与许可地震

> 2026年音乐生成格局：Suno v5 与 Udio v4 主导商业市场；MusicGen、Stable Audio Open 和 ACE-Step 引领开源领域。技术问题基本解决。法律问题（华纳音乐5亿美元和解案、环球音乐集团和解案）在2025-2026年重塑了该领域。

**类型:** 构建
**语言:** Python
**先决知识:** 第六阶段 · 02（频谱图）、第四阶段 · 10（扩散模型）
**时间:** 约75分钟

## 问题

文本 → 一段30秒至4分钟的音乐片段，包含歌词、人声和结构。三个子问题：

1.  **器乐生成。** 将文本如"温暖钢琴与lo-fi嘻哈鼓点" → 生成音频。代表模型：MusicGen、Stable Audio、AudioLDM。
2.  **歌曲生成（带人声+歌词）。** 将"关于雨夜德克萨斯的乡村歌曲" → 生成完整歌曲。代表模型：Suno、Udio、YuE、ACE-Step。
3.  **条件/可控生成。** 延长现有片段、重新生成过渡段、转换流派、音轨分离或修复。Udio的修复功能+音轨分离是2026年需要对标的功能特性。

## 概念

![音乐生成：基于token的语言模型 vs 扩散模型，2026年模型版图](../assets/music-generation.svg)

### 基于神经编码token的Token LM

Meta的 **MusicGen**（2023，MIT许可）及其衍生模型：以文本/旋律嵌入为条件，自回归预测EnCodec token（32 kHz, 4个码本），并使用EnCodec解码。参数量3亿至33亿。强大的基线模型；在超过30秒时效果会下降。

**ACE-Step**（开源，40亿参数XL版本于2026年4月发布）在此基础上扩展，用于完整歌曲的歌词条件生成。开源社区中最接近Suno的产品。

### 基于梅尔频谱或潜在空间的扩散模型

**Stable Audio (2023)** 与 **Stable Audio Open (2024)**：在压缩音频上进行潜在扩散。擅长循环段落、音效设计、环境氛围纹理。不擅长结构化的完整歌曲。

**AudioLDM / AudioLDM2**：通过类T2I风格的潜在扩散实现文本到音频生成，可泛化至音乐、音效和语音。

### 混合模型（生产级）— Suno、Udio、Lyria

闭源权重。可能结合了AR编码器LM和基于扩散的声码器，并配备专门的人声/鼓点/旋律生成头。Suno v5（2026）是ELO 1293分的质量领导者。Udio v4增加了修复功能+音轨分离（可单独下载贝斯、鼓、人声）。

### 评估

-   **FAD（Fréchet音频距离）。** 使用VGGish或PANNs特征计算生成音频分布与真实音频分布在嵌入空间的距离。越低越好。MusicGen small在MusicCaps上FAD为4.5；当前最优约3.0。
-   **音乐性（主观）。** 人类偏好度。Suno v5的ELO 1293分领先。
-   **文本-音频对齐度。** 提示词与输出之间的CLAP分数。
-   **音乐性伪影。** 节拍错位的过渡、人声乐句漂移、超过30秒后结构丢失。

## 2026年模型版图

| 模型 | 参数量 | 长度 | 人声 | 许可证 |
|------|--------|------|------|--------|
| MusicGen-large | 33亿 | 30秒 | 否 | MIT |
| Stable Audio Open | 12亿 | 47秒 | 否 | Stability 非商业许可 |
| ACE-Step XL (2026年4月) | 40亿 | > 2分钟 | 是 | Apache-2.0 |
| YuE | 70亿 | > 2分钟 | 是，多语言 | Apache-2.0 |
| Suno v5 (闭源) | 未知 | 4分钟 | 是，ELO 1293 | 商业 |
| Udio v4 (闭源) | 未知 | 4分钟 | 是 + 音轨分离 | 商业 |
| Google Lyria 3 (闭源) | 未知 | 实时 | 是 | 商业 |
| MiniMax Music 2.5 | 未知 | 4分钟 | 是 | 商业API |

## 法律环境（2025-2026）

-   **华纳音乐 vs Suno 和解案。** 5亿美元。华纳音乐集团现可监督Suno上的AI形象、音乐版权及用户生成曲目。环球音乐集团与Udio达成了类似和解。
-   **欧盟人工智能法案** + **加州SB 942**：AI生成的音乐必须被披露。
-   **Riffusion / MusicGen** 采用MIT许可，无合规负担，但也无商业人声功能。

可安全落地的模式：

1.  仅生成器乐（使用MusicGen、Stable Audio Open，输出为MIT/CC0许可）。
2.  使用商业API（Suno、Udio、ElevenLabs Music），按次生成获得许可。
3.  使用自有或已获授权的音乐库进行训练（多数企业最终会选择此路径）。
4.  在生成内容中添加水印+元数据标签。

## 动手构建

### 步骤1：使用MusicGen生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三种规模：`small`（3亿参数，快速）、`medium`（15亿参数）、`large`（33亿参数）。小模型足以验证"想法是否可行"。

### 步骤2：旋律条件生成

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody接受色度图输入，在更换音色的同时保留旋律。适用于"将这段旋律用弦乐四重奏演奏"的场景。

### 步骤3：FAD评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算VGGish嵌入距离。适用于流派级别的回归测试；但不能替代人类听众的评估。

### 步骤4：集成到LLM-音乐工作流中

结合第7-8课的理念：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 使用场景

| 目标 | 技术栈 |
|------|--------|
| 器乐音效设计 | Stable Audio Open |
| 游戏/自适应音乐 | Google Lyria RealTime (闭源) |
| 带人声的完整歌曲（商业） | Suno v5 或 Udio v4（需明确许可） |
| 带人声的完整歌曲（开源） | ACE-Step XL 或 YuE |
| 短广告曲 | 基于哼唱参考的MusicGen旋律条件生成 |
| MV背景音乐 | MusicGen + Stable Video Diffusion |

## 2026年仍会遇到的坑

-   **版权洗白提示词。** "泰勒·斯威夫特风格的歌曲" — 现在的商业Suno/Udio会过滤此类提示，开源模型不会。请自行添加过滤列表。
-   **超过30秒后的重复/漂移。** AR模型会循环。可交叉淡入淡出多个生成结果，或使用ACE-Step以保证结构连贯性。
-   **节奏漂移。** 模型会偏离BPM。在提示词中使用BPM标签，并用librosa的`beat_track`进行后处理过滤。
-   **人声清晰度。** Suno表现优异；开源模型的歌词往往含糊不清。若歌词至关重要，请使用商业API或进行微调。
-   **单声道输出。** 开源模型生成单声道或假立体声。使用立体声重建工具（如ezst，Cartesia的立体声扩散）进行升级。

## 落地部署

保存为`outputs/skill-music-designer.md`。选择模型、许可策略、长度/结构规划，以及用于音乐生成部署的披露元数据。

## 练习

1.  **简单。** 运行`code/main.py`。它会生成“生成式”的和弦进行+鼓点模式的ASCII符号 — 一个音乐生成的卡通演示。如果需要，可通过任意MIDI渲染器播放。
2.  **中等。** 安装`audiocraft`，使用MusicGen-small针对4种流派提示生成10秒片段，并对照参考流派集测量FAD。
3.  **困难。** 使用ACE-Step（或MusicGen-melody），使用不同的音色提示生成同一旋律的三个变体。计算它们与提示词的CLAP相似度以验证对齐效果。

## 关键术语

| 术语 | 常用说法 | 实际含义 |
|------|----------|----------|
| FAD | 音频FID | 真实与生成音频嵌入分布之间的Fréchet距离。 |
| 色度图 | 旋律作为音高 | 每帧12维向量；用于旋律条件生成。 |
| 音轨 | 乐器轨道 | 分离的贝斯/鼓/人声/旋律的WAV文件。 |
| 修复 | 重新生成某段 | 遮罩一个时间窗口；模型仅重新生成该部分。 |
| CLAP | 文本-音频CLIP | 对比学习的音频-文本嵌入；用于评估文本-音频对齐度。 |
| EnCodec | 音乐编码器 | Meta的神经编码器，MusicGen使用；32 kHz, 4个码本。 |

## 扩展阅读

-   [Copet 等 (2023). MusicGen](https://arxiv.org/abs/2306.05284) — 开放的自回归基准模型。
-   [Evans 等 (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — 音效设计的默认选择。
-   [ACE-Step](https://github.com/ace-step/ACE-Step) — 开源的40亿参数完整歌曲生成器，2026年4月发布。
-   [Suno v5 平台文档](https://suno.com) — 商业质量领导者。
-   [AudioLDM2](https://arxiv.org/abs/2308.05734) — 用于音乐和音效的潜在扩散模型。
-   [华纳-Suno和解案报道](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — 2025年11月的判例。