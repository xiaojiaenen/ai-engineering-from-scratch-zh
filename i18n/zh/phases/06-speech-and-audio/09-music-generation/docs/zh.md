# 音乐生成 — MusicGen、Stable Audio、Suno 以及许可领域的地震

> 2026 年音乐生成：Suno v5 和 Udio v4 主导商业市场；MusicGen、Stable Audio Open 和 ACE-Step 引领开源阵营。技术问题已基本解决。法律难题（华纳音乐 5 亿美元和解、UMG 和解）在 2025-2026 年重塑了整个领域。

**类型：** 构建实践
**语言：** Python
**前置知识：** 第 6 阶段 · 02（频谱图）、第 4 阶段 · 10（扩散模型）
**预计时间：** 约 75 分钟

## 问题描述

输入文字 → 一段 30 秒至 4 分钟的音乐片段，包含人声和歌词、完整结构。涉及三个子问题：

1. **器乐生成。** 如"温暖基调的 lo-fi hip-hop 鼓点" → 音频。代表模型：MusicGen、Stable Audio、AudioLDM。
2. **歌曲生成（含人声 + 歌词）。** 如"一首关于德州雨夜的乡村歌曲" → 完整歌曲。代表模型：Suno、Udio、YuE、ACE-Step。
3. **条件 / 可控生成。** 扩展已有片段、重新生成桥段、更换风格、分离音轨或局部重绘。Udio 的 inpainting（局部重绘）+ 音轨分离是 2026 年值得对齐的核心功能。

## 核心概念

![音乐生成：token LM vs 扩散模型，2026 年模型图谱](../assets/music-generation.svg)

### 基于神经编码 token 的 Token LM

Meta 的 **MusicGen**（2023，MIT 许可）及众多衍生模型：以文本/旋律嵌入为条件，自回归地预测 EnCodec token（32 kHz，4 个码本），再通过 EnCodec 解码。参数量 3 亿至 33 亿。是最强基线模型；但超过 30 秒时质量明显下降。

**ACE-Step**（开源，4B XL 版本于 2026 年 4 月发布）将上述方法扩展到完整歌曲的歌词条件生成，是开源社区中最接近 Suno 的方案。

### 基于 mel 谱或潜变量的扩散模型

**Stable Audio（2023）** 和 **Stable Audio Open（2024）**：在压缩音频潜变量上做扩散去噪。擅长循环乐段、声音设计、氛围质感，但不擅长有完整结构的长歌曲。

**AudioLDM / AudioLDM2**：基于 T2I（文本到图像）风格的潜变量扩散，泛化至音乐、音效和语音。

### 混合方案（商业产品）—— Suno、Udio、Lyria

闭源权重。大概采用 AR 编码 LM + 扩散式声码器，并配有专门的嗓音/鼓点/旋律头。Suno v5（2026）是 ELO 1293 评分的质量标杆。Udio v4 新增了局部重绘 + 音轨分离（可分别下载贝斯、鼓、人声轨道）。

### 评估指标

- **FAD（Fréchet Audio Distance，弗雷歇音频距离）。** 使用 VGGish 或 PANNs 特征，计算生成音频与真实音频分布之间的嵌入级距离。越低越好。MusicGen small 在 MusicCaps 上得分为 4.5 FAD；SOTA 约 3.0。
- **音乐性（主观）。** 人类偏好评分。Suno v5 ELO 1293 领先。
- **文本-音频对齐度。** prompt 与输出之间的 CLAP 分数。
- **音乐性伪影。** 节拍错位过渡、人声短语漂移、超过 30 秒后结构丢失。

## 2026 年模型图谱

| 模型 | 参数量 | 时长 | 人声 | 许可协议 |
|-------|--------|--------|--------|---------|
| MusicGen-large | 33 亿 | 30 秒 | 无 | MIT |
| Stable Audio Open | 12 亿 | 47 秒 | 无 | Stability 非商用 |
| ACE-Step XL（2026 年 4 月） | 40 亿 | > 2 分钟 | 有 | Apache-2.0 |
| YuE | 70 亿 | > 2 分钟 | 有，多语言 | Apache-2.0 |
| Suno v5（闭源） | — | 4 分钟 | 有，ELO 1293 | 商业 |
| Udio v4（闭源） | — | 4 分钟 | 有 + 音轨分离 | 商业 |
| Google Lyria 3（闭源） | — | 实时 | 有 | 商业 |
| MiniMax Music 2.5 | — | 4 分钟 | 有 | 商业 API |

## 法律格局（2025-2026）

- **华纳音乐 vs Suno 和解。** 5 亿美元。WMG 现在对 Suno 上 AI 拟真度、音乐版权及用户生成曲目拥有监管权。Udio 也面临类似的 UMG 和解。
- **欧盟《人工智能法案》+ 加州 SB 942 法案：** AI 生成音乐必须披露。
- **Riffusion / MusicGen** 虽为 MIT 许可、无合规负担，但也无法用于带人声的商业场景。

安全交付的模式：

1. 仅生成器乐（MusicGen、Stable Audio Open，MIT/CC0 输出）。
2. 使用商业 API（Suno、Udio、ElevenLabs Music），按生成次数获取许可。
3. 在自有或已授权曲库上训练（大多数企业最终走到这一步）。
4. 为生成内容添加水印 + 元数据标注。

```figure
sp-codec-tokens
```

## 动手实践

### 步骤 1：使用 MusicGen 生成音乐

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三种尺寸：`small`（3 亿参数，速度快）、`medium`（15 亿）、`large`（33 亿）。`small` 足够验证"想法是否成立"。

### 步骤 2：旋律条件生成

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody 接受一个音高图谱（chromagram），在变换音色的同时保留原旋律。适用于"把这段旋律换成弦乐四重奏"的场景。

### 步骤 3：FAD 评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算 VGGish 嵌入距离。适用于风格级别的回归测试；不能替代人类听测。

### 步骤 4：融入 LLM-音乐工作流

结合第 7-8 课的思路：

```python
prompt = "写一段 30 秒的爵士循环乐段。描述鼓、贝斯和钢琴的和弦编排。"
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 应用场景

| 目标 | 推荐方案 |
|------|-------|
| 器乐声音设计 | Stable Audio Open |
| 游戏 / 自适应音乐 | Google Lyria RealTime（闭源） |
| 完整歌曲（含人声，商业） | Suno v5 或 Udio v4（需明确获取许可） |
| 完整歌曲（含人声，开源） | ACE-Step XL 或 YuE |
| 短广告配乐 | MusicGen，以哼唱旋律为条件 |
| MV 背景音 | MusicGen + Stable Video Diffusion |

## 2026 年仍会踩中的坑

- **规避版权的检查提示词。** 如"泰勒·斯威夫特风格的歌曲"——商业版 Suno/Udio 现已过滤此类请求，开源模型则不会。请自行维护过滤列表。
- **超过 30 秒后的重复 / 漂移。** AR 模型容易出现循环重复。可对多次生成结果做交叉淡入淡出，或使用 ACE-Step 保证结构连贯。
- **节拍漂移。** 模型可能偏离指定 BPM。在 prompt 中显式标注 BPM，并用 librosa 的 `beat_track` 做后处理过滤。
- **人声清晰度。** Suno 表现优秀；开源模型的人声往往含糊不清。如果歌词重要，请使用商业 API 或进行微调。
- **单声道输出。** 开源模型通常生成单声道或假立体声。可通过专业立体声重构方案升级（如 ezst、Cartesia 的立体声扩散模型）。

## 交付建议

保存为 `outputs/skill-music-designer.md`。为你的音乐生成部署方案选择模型、许可策略、时长/结构设计及披露元数据。

## 练习题

1. **入门。** 运行 `code/main.py`。它会输出一段"生成式"和弦进行 + 鼓点图案（ASCII 字符表示）—— 一首音乐生成的漫画。如需听觉反馈，可用任意 MIDI 渲染器播放。
2. **中等。** 安装 `audiocraft`，用 MusicGen-small 针对 4 个风格 prompt 各生成 10 秒片段，并与参考风格集对比测量 FAD。
3. **进阶。** 使用 ACE-Step（或 MusicGen-melody），为同一段旋律生成三种不同音色 prompt 的变体，计算 CLAP 相似度以验证 prompt 对齐效果。

## 关键术语

| 术语 | 人们常说的 | 实际含义 |
|------|-----------------|---------|
| FAD | 音频 FID | 真实与生成音频嵌入分布之间的弗雷歇距离。 |
| Chromagram | 以音高表示的旋律 | 每帧 12 维向量；用于旋律条件输入的输入。 |
| Stems | 音轨 | 分离出的贝斯 / 鼓 / 人声 / 主旋律 WAV 文件。 |
| Inpainting | 重新生成某段落 | 掩码某个时间窗口；模型仅对该窗口重新生成。 |
| CLAP | 文本-音频 CLIP | 对比式音频-文本嵌入；用于评估文本-音频对齐度。 |
| EnCodec | 音乐编码 codec | Meta 的神经网络编解码器，被 MusicGen 使用；32 kHz，4 个码本。 |

## 延伸阅读

- [Copet 等（2023）。MusicGen](https://arxiv.org/abs/2306.05284) —— 开源自回归基准。
- [Evans 等（2024）。Stable Audio Open](https://arxiv.org/abs/2407.14358) —— 声音设计首选方案。
- [ACE-Step](https://github.com/ace-step/ACE-Step) —— 开源 40 亿参数完整歌曲生成器，2026 年 4 月发布。
- [Suno v5 平台文档](https://suno.com) —— 商业质量标杆。
- [AudioLDM2](https://arxiv.org/abs/2308.05734) —— 面向音乐与音效的潜变量扩散模型。
- [WMG-Suno 和解案例报道](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) —— 2025 年 11 月里程碑案例。
