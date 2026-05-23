# 毕业设计 12 — 视频理解流水线（场景、问答、搜索）

> Twelve Labs 将 Marengo + Pegasus 产品化。VideoDB 提供了视频的增删改查 API。AI2 的 Molmo 2 发布了开源 VLM 检查点。Gemini 的长上下文可原生处理数小时的视频。TimeLens-100K 定义了大规模时间定位。2026 年的流水线已确定：场景分割、每场景字幕+嵌入、转录对齐、多向量索引，以及一个返回（起始时间，结束时间）时间戳和帧预览的查询。毕业设计需处理 100 小时视频，达到公开基准测试指标，并测量在计数和动作类问题上的幻觉率。

**类型：** 毕业设计
**语言：** Python（流水线），TypeScript（UI）
**先修课程：** 阶段 4（计算机视觉），阶段 6（语音），阶段 7（Transformer），阶段 11（大语言模型工程），阶段 12（多模态），阶段 17（基础设施）
**涵盖阶段：** P4 · P6 · P7 · P11 · P12 · P17
**时间：** 30 小时

## 问题

在 2026 年的规模下，长视频问答是带宽需求最大的多模态问题。Gemini 2.5 Pro 可以原生读取 2 小时的视频，但将 100 小时视频摄取到可查询的语料库中仍需要场景级索引。生产形态结合了场景分割（TransNetV2 或 PySceneDetect）、使用 VLM（Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）为每场景生成字幕、转录对齐（带词语时间戳的 Whisper-v3-turbo），以及一个并行存储字幕、帧嵌入和转录的多向量索引。查询流水线返回（起始时间，结束时间）时间戳和帧预览。

基准测试是公开的（ActivityNet-QA、NeXT-GQA）加上你自己的 100 个查询的自定义集。在计数和动作类问题上的幻觉是已知的难点故障类；本毕业设计会明确测量它。

## 概念

摄取时三条流水线并行运行。**场景分割**将视频切割成场景。**VLM 字幕**为每个场景生成字幕并从关键帧生成帧嵌入。**ASR 对齐**生成词级时间戳。三条流通过（scene_id，时间范围）连接。每个场景在多向量索引（Qdrant）中拥有三种向量类型：字幕嵌入、关键帧嵌入、转录嵌入。

查询时，自然语言问题同时对三个向量发起检索；结果通过 RRF 合并；一个时间定位适配器（TimeLens 风格）在最佳场景内细化（起始，结束）窗口。VLM 合成器（Gemini 2.5 Pro 或 Qwen3-VL-Max）接收查询 + 最佳场景 + 裁剪帧，并返回带有引用时间戳和帧预览的答案。

幻觉测量很重要。计数（“多少人进入房间？”）和动作类（“厨师是先倒后搅拌吗？”）问题以不可靠著称。毕业设计要求将此类问题的准确率与描述性问题分开报告。

## 架构

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## 技术栈

- 场景分割：TransNetV2（2024-26 最先进）或 PySceneDetect
- ASR：通过 faster-whisper 使用 Whisper-v3-turbo，带词级时间戳
- VLM 字幕+回答：Gemini 2.5 Pro 或 Qwen3-VL-Max 或 Molmo 2
- 时间定位：TimeLens-100K 训练的适配器或 VideoITG
- 索引：支持多向量（字幕/帧/转录）的 Qdrant
- UI：Next.js 15，带 HTML5 视频播放器和场景缩略图
- 评估：ActivityNet-QA，NeXT-GQA，手动标注的 100 个查询自定义集
- 幻觉基准：带手动标注的计数和动作类子集

## 构建它

1.  **摄取器。** 接受 YouTube URL 或本地 MP4 文件。如有需要，降采样至 720p。持久化 `{video_id, file_path}`。
2.  **场景分割。** 运行 TransNetV2 或 PySceneDetect 生成 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目标处理 100 小时视频：约 6k-8k 个场景。
3.  **ASR 通道。** 对音频运行 Whisper-v3-turbo；导出词级时间戳；分割为每个场景的转录片段。
4.  **VLM 字幕。** 对每个场景，使用关键帧和简短字幕模板调用 Gemini 2.5 Pro（或 Qwen3-VL-Max）。生成字幕 + 帧嵌入。
5.  **多向量索引。** Qdrant 集合包含三个命名向量。负载：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。
6.  **查询。** 自然语言问题触发三个稠密检索；通过互惠排名融合合并；top-k=5 个场景。
7.  **时间定位。** 在最佳场景上运行 TimeLens 风格适配器，以细化场景内的（起始，结束）窗口。
8.  **VLM 合成。** 使用查询 + 最佳 3 个场景片段（作为图像或短片）+ 转录调用 Gemini 2.5 Pro。要求 `(video_id, start_ms, end_ms)` 引用。
9.  **评估。** 运行 ActivityNet-QA 和 NeXT-GQA。构建一个 100 个查询的自定义集。报告总体准确率及分类细分（计数、动作、描述性）。

## 使用它

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 交付它

`outputs/skill-video-qa.md` 是最终交付物。给定一个 YouTube URL 或上传的视频，流水线将索引场景并回答带有带时间戳引用的问题。

| 权重 | 标准 | 测量方法 |
|:-:|---|---|
| 25 | 时间定位 IoU | 在预留的定位集上计算交并比 |
| 20 | QA 准确率 | 在 NeXT-GQA 和自定义 100 查询集上 |
| 20 | 摄取吞吐量 | 每美元花费处理的视频小时数 |
| 20 | UI 和引用用户体验 | 时间戳链接、缩略图条、跳转到帧 |
| 15 | 幻觉率 | 单独计算计数和动作类问题的准确率 |
| **100** | | |

## 练习

1.  将字幕通道的 Gemini 2.5 Pro 替换为 Qwen3-VL-Max。在人工评级的 50 个场景样本上报告字幕质量差异。
2.  将每场景的帧嵌入减少为一个池化向量，而不是多向量。衡量检索效果的回退。
3.  构建一个“严格计数”模式：合成器提取每个计数的实例及其时间戳，用户点击验证。测量用户验证是否降低幻觉率。
4.  对摄取成本进行基准测试：在三种 VLM 选择下，比较每美元处理的视频小时数。选择最佳平衡点。
5.  添加说话人分离的转录：在音频上运行 pyannote 说话人分离，并嵌入每位说话人的转录。演示“爱丽丝说了关于 X 的什么？”类查询。

## 关键术语

| 术语 | 人们怎么说 | 它实际指什么 |
|------|-----------------|------------------------|
| 场景分割 | “镜头检测” | 在镜头边界处将视频切割成场景 |
| 多向量索引 | “字幕+帧+转录” | 为每种表示形式设置命名向量的 Qdrant 集合 |
| 时间定位 | “它到底发生在什么时候” | 为查询答案细化（起始，结束）窗口 |
| 帧嵌入 | “视觉表示” | 关键帧的向量嵌入；用于场景视觉相似度 |
| RRF 融合 | “互惠排名融合” | 合并多个排名列表的策略；一种经典的混合检索技巧 |
| 计数幻觉 | “数错” | VLM 在“有多少个 X”问题上的已知故障模式 |
| ActivityNet-QA | “视频问答基准测试” | 长视频问答准确率基准测试 |

## 延伸阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 开源 VLM 检查点
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — 大规模时间定位
- [Gemini 视频长上下文](https://deepmind.google/technologies/gemini) — 托管参考
- [VideoDB](https://videodb.io) — 视频增删改查 API 参考
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — 商业参考
- [TransNetV2](https://github.com/soCzech/TransNetV2) — 场景分割模型
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — 经典开源替代方案
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — 参考评估基准测试