# 音频语言模型：从 Whisper 到 Audio Flamingo 3 的演进弧线

> Whisper（Radford 等人，2022年12月）解决了语音识别问题——68万个小时的弱监督多语言语音数据、一个简单的编码器-解码器 Transformer，以及一个让每次后续 ASR 发布都引用它的数据集。但识别不等于推理。问"这段录音中有什么乐器"、"说话者表达了什么情绪"或"第3分钟发生了什么"需要的是音频理解，而非转录。Qwen-Audio、SALMONN、LTU 和 NVIDIA 的 Audio Flamingo 3（AF3，2025年7月）逐步构建了这一技术栈：保留 Whisper 级编码器，加装 Q-former，在音频-文本指令数据上训练，引入思维链推理。本课将带你走过这条演进弧线。

**类型：** Build
**语言：** Python（stdlib、对数 Mel 频谱图 + 音频 Q-former 骨架）
**前置条件：** Phase 6（语音与音频）、Phase 12 · 03（Q-Former）
**时间：** ~180 分钟

## 学习目标

- 从零计算对数 Mel 频谱图：加窗、FFT、滤波器组、对数变换。
- 比较编码器选项：Whisper 编码器、BEATs、AF-Whisper 混合架构。各自适用场景。
- 构建音频 Q-former：N 个可学习查询对频谱图 patch 进行交叉注意力。
- 解释级联（Whisper-then-LLM）与端到端音频 LLM 训练的区别：为什么端到端推理更能扩展。

## 问题所在

语音识别已被 Whisper 解决。音频 OCR 已成标配。但"标配"止步于转录。如果模型无法对所听内容进行推理——时序、说话人、情绪、音乐结构、环境音——仅靠转录无法驱动产品功能。

三条明显路径：

1. **级联方案**：Whisper 转录，LLM 基于文本推理。纯语音场景效果良好。但在音乐、环境音、多说话人重叠、情绪识别上失效。

2. **端到端音频 LLM**：音频编码器直接将音频 token 输入 LLM，跳过转录环节。保留了声学信息（情绪、说话人、环境）。需要新的训练数据。

3. **混合方案**：音频编码器 + 文本解码器，既能转录也能推理。Qwen-Audio 和 Audio Flamingo 选择了这条路。

## 概念解析

### 对数 Mel 频谱图：输入特征

每个音频编码器都从相同的特征开始：对数 Mel 频谱图。

1. 重采样到 16 kHz。
2. 使用 25ms 窗长、10ms 步长的短时傅里叶变换（STFT）。
3. 取 FFT 结果的幅度。
4. 应用 Mel 滤波器组（通常为 80 个对数间隔的滤波器，覆盖 0-8000 Hz）以映射到感知频率。
5. 对数压缩（log(1 + x)）以压缩动态范围。

结果：一个形状为 (T, 80) 的二维数组，其中 T 是时间帧数。对于 30 秒片段、100 Hz 帧率：(3000, 80)。

### Whisper 的编码器

Whisper 的编码器是一个 12 层的 ViT 风格 Transformer，将 log-Mel 频谱图作为时间帧序列处理。输出：每帧一个隐藏状态向量。

对于 ASR，Whisper 的解码器是一个交叉注意力 Transformer，根据编码器输出生成文本 token。标准的编码器-解码器架构。

对于 ALM（音频大语言模型），你需要将编码器输出作为不同 LLM 的输入。模式为：冻结 Whisper 编码器，训练 Q-former，冻结或微调 LLM。

### BEATs 与专用音频编码器

Whisper 是在以语音为主的数据上训练的。它在音乐和环境音频上表现较弱。

BEATs（Chen 等人，2022）是在 AudioSet 上训练的自监督 Transformer。在相同参数量下，对音乐和环境声音的捕捉优于 Whisper。

AF-Whisper（Audio Flamingo 3 的混合架构）：将 Whisper 和 BEATs 特征拼接作为音频输入。Whisper 承载语言信号，BEATs 承载声学信号。

### 音频 Q-former

与 BLIP-2 的视觉 Q-former 相同模式。固定数量的可学习查询（通常为 32 或 64 个）对音频编码器输出帧进行交叉注意力。这些查询变为 LLM 消费的音频 token。

训练阶段：先单独训练 Q-former，在音频-文本对上使用对比损失和描述损失（AudioCaps、Clotho）。指令阶段：端到端训练，解冻 LLM，在指令数据上训练。

### 演进弧线——SALMONN、Qwen-Audio、AF3

**SALMONN**（Tang 等人，2023）：Whisper + BEATs + Q-former + LLaMA。首个具备强大推理能力的开源音频 LLM。在 MMAU 基准上综合得分约 0.55。

**Qwen-Audio**（Chu 等人，2023）：相似架构，在更丰富的数据集上训练，针对多轮对话微调。MMAU 约 0.60。

**LTU — Listen, Think, Understand**（Gong 等人，2023）：显式推理数据，专注于音频片段的思维链推理。规模较小但更专注。

**Audio Flamingo 3**（Goel 等人，2025年7月）：当前开源 SOTA。8B LLM 主干（Qwen2 7B）、Whisper-large 编码器拼接 BEATs、64 查询 Q-former、在 100 万+ 音频-文本指令对上训练。MMAU 0.72，在某些子任务上匹敌闭源前沿模型。

AF3 还引入了按需思维链：模型可在最终答案之前选择性地输出思考 token（"让我先识别乐器：..."）。启用推理时，复杂推理任务的准确率提升 3-5 分。

### 级联 vs 端到端

**级联流水线：**
1. Whisper 转录音频 → 文本
2. LLM 基于文本推理

适用于："总结这段播客"。在以下场景失效：
- "这首歌的氛围是什么？"——氛围在声音中，不在文字里。
- "谁在说话，Alice 还是 Bob？"——需要说话人识别。
- "爆炸发生在第几秒？"——时序定位在文本中丢失。
- "这是真实还是合成的音频？"——深度伪造检测需要声学特征。

端到端保留声学信号。Qwen-Audio 和 AF3 原生处理音乐、环境和情绪。

### 2026 年生产配方

对于新的音频理解产品：

- **级联方案**：如果目标是转录，不涉及音乐、情绪推理。
- **AF3 / Qwen-Audio 系列**：如果涉及音乐、情绪、多说话人或复杂音频推理。

级联方案更便宜简单。端到端能力更强。

### MMAU——音频推理基准

MMAU（Massive Multimodal Audio Understanding）是 2024-2025 年的音频推理基准：

- 涵盖语音、音乐、环境声音的 10,000 个音频-文本 QA 对。
- 覆盖分类、时序推理、因果推理、开放式 QA。
- 测试级联流水线系统性遗漏的能力。

开源 SOTA（AF3）达 0.72；闭源前沿约 0.78（Gemini 2.5 Pro、Claude Opus 4.7）。差距小于 VideoMME 的开源-闭源差值，表明音频 LLM 正在成熟。

```figure
audio-text-ctc
```

## 实战

`code/main.py`：

- 使用 stdlib 实现 log-Mel 频谱图计算：加窗、朴素 DFT、Mel 滤波器组。
- 音频 Q-former 骨架：给定编码器输出帧，计算 Q、K、V、注意力，并输出 N 个 token。
- 在玩具任务上比较级联与端到端方案。

## 交付物

本课产出 `outputs/skill-audio-llm-pipeline-picker.md`。给定一个音频任务（转录、音乐标签、情绪推理、多说话人 diarization、环境分类），它会选择级联、端到端 AF3 或混合方案。

## 练习

1. 计算 16kHz、25ms 窗长、10ms 步长、80 个 Mel  bins 下 30 秒片段的 log-Mel 频谱图维度。48kHz 时如何变化？

2. 为什么 Whisper 在音乐上表现不佳？BEATs 捕捉了哪些 Whisper 未能捕捉的音频特征？

3. 64 查询 vs 32 查询的音频 Q-former：在何种任务复杂度下 64 查询更有优势？32 查询为哪些场景节省计算？

4. 阅读 AF3 第 4 节的按需推理。提出三个最受益于思维链的音频任务。

5. 使用 AF3 输出实现一个最小化的说话人分离流水线。如何标记说话人切换？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Log-Mel spectrogram | "Mel 特征" | 经过 Mel 滤波器组后的对数幅度二维（时间，频率）数组 |
| Audio Q-former | "Audio Perceiver" | 从音频编码器输出到固定长度查询的交叉注意力瓶颈，为 LLM 提供输入 |
| Cascaded | "ASR-then-LLM" | Whisper 转录、文本 LLM 推理的流水线；丢失声学信息 |
| End-to-end | "Audio-LLM" | 音频特征通过 Q-former 直接进入 LLM；保留声学信号 |
| BEATs | "Audio AudioSet encoder" | 在 AudioSet 上训练的 SSL Transformer；对音乐和环境声音表现强劲 |
| MMAU | "音频推理基准" | 涵盖语音、音乐、环境的 1 万 QA 对；2024 年评估标准 |
| On-demand thinking | "Audio CoT" | 模型可选择在最终答案前输出推理 token，准确率提升 3-5 分 |

## 延伸阅读

- [Radford et al. — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu et al. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel et al. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang et al. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong et al. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
