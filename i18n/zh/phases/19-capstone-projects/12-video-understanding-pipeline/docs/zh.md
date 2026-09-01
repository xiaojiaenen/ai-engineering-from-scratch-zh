# Capstone 12 — 视频理解管道（场景、问答、搜索）

> Twelve Labs 将 Marengo + Pegasus 产品化。VideoDB 推出了视频的 CRUD API。AI2 的 Molmo 2 发布了开源 VLM 检查点。Gemini 长上下文原生支持数小时视频。TimeLens-100K 定义了大规模时间定位。2026 年的管道已定型：场景分割、逐场景标题 + 嵌入、转录对齐、多向量索引，以及能回答（开始、结束）时间戳并附带帧预览的查询。本 Capstone 要摄入 100 小时视频、在公开基准上评测，并测量计数类和动作类问题的幻觉率。

**类型：** Capstone
**语言：** Python（管道）、TypeScript（UI）
**前置条件：** Phase 4（CV）、Phase 6（语音）、Phase 7（transformers）、Phase 11（LLM 工程）、Phase 12（多模态）、Phase 17（基础设施）
**涉及阶段：** P4 · P6 · P7 · P11 · P12 · P17
**时间：** 30 小时

## 问题

长视频问答是 2026 年规模下最耗带宽的多模态问题。Gemini 2.5 Pro 可以原生"阅读"一段 2 小时视频，但要把 100 小时视频摄入到可查询的语料库中，仍然需要一个场景级索引。生产形态需要将场景分割（TransNetV2 或 PySceneDetect）、基于 VLM 的逐场景标题生成与帧嵌入（Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）、带词级时间戳的转录对齐（Whisper-v3-turbo），以及一个并排存储标题、帧嵌入和转录的多向量索引结合起来。查询管道通过（开始、结束）时间戳加帧预览来回答。

评测基准有公开的 ActivityNet-QA、NeXT-GQA，以及我们自己构造的 100 题自定义集。计数类和动作类问题的幻觉是已知的强失败类别；本 Capstone 会对其进行专门测量。

## 概念

摄入时并行跑三条管道：**场景分割**将视频切分为多个场景。**VLM 标题生成**为每个场景生成一句标题，并从关键帧提取帧嵌入。**语音识别对齐**产生词级时间戳。三路流通过 `(scene_id, 时间范围)` 进行关联。每个场景在 Qdrant 多向量索引中存储三种向量：标题嵌入、关键帧嵌入、转录嵌入。

查询时，自然语言问题会同时打在三个向量上；结果用 RRF 合并；时序定位适配器（TimeLens 风格）会在最优场景内进一步收窄 `(开始, 结束)` 窗口。VLM 合成器（Gemini 2.5 Pro 或 Qwen3-VL-Max）以查询 + 最优场景 + 裁剪帧为输入，输出一条带引用时间戳和帧预览的回答。

幻觉测量很关键。计数类（"有多少人进入房间？"）和动作类型（"厨师是先倒还是先搅拌？"）问题在业内素以不可靠著称，报告时需要与描述性问题分开统计准确率。

## 架构

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (场景分割)
      |
      +--- 逐场景关键帧 --- VLM 标题 + 帧嵌入
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- 音频声道 --- Whisper-v3-turbo ASR + 词级时间戳
      |
      v
多向量 Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
查询:
  对三路向量同时做密集查询 -> RRF 合并 -> top-k 场景
      |
      v
TimeLens / VideoITG 时序定位（在场景内精化 start/end）
      |
      v
VLM 合成: 查询 + 最优场景 + 帧预览
      |
      v
回答 + (start, end) 时间戳 + 帧缩略图 + 引用
```

## 技术栈

- 场景分割：TransNetV2（2024-26 SOTA）或 PySceneDetect
- ASR：通过 faster-whisper 使用 Whisper-v3-turbo，输出词级时间戳
- VLM 标题/回答：Gemini 2.5 Pro 或 Qwen3-VL-Max 或 Molmo 2
- 时序定位：TimeLens-100K 训练适配器或 VideoITG
- 索引：Qdrant（支持多向量：标题 / 帧 / 转录）
- UI：Next.js 15，含 HTML5 视频播放器与场景缩略图
- 评测：ActivityNet-QA、NeXT-GQA、自定义 100 题手工标注集
- 幻觉基准：计数和动作类型子集，手工标注

```figure
cf-scene-index
```

## 构建步骤

1. **摄入入口。** 接受 YouTube 链接或本地 MP4；必要时下采样到 720p。持久化 `{video_id, file_path}`。

2. **场景分割。** 运行 TransNetV2 或 PySceneDetect，输出 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目标 100 小时：约 6000-8000 个场景。

3. **ASR 遍。** 对音频运行 Whisper-v3-turbo；导出词级时间戳；按场景切分为逐场景转录片段。

4. **VLM 标题生成。** 对每个场景，用关键帧和简短标题模板调用 Gemini 2.5 Pro（或 Qwen3-VL-Max）。产出标题 + 帧嵌入。

5. **多向量索引。** Qdrant 集合中配置三个命名向量。载荷：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **查询。** 自然语言问题同时发起三路密集查询；用互逆秩融合（RRF）合并；top-k=5 场景。

7. **时序定位。** 在最优场景上运行 TimeLens 风格适配器，精化该场景内的 `(start, end)` 窗口。

8. **VLM 合成。** 调用 Gemini 2.5 Pro，输入查询 + top-3 场景片段（图片或短视频片段）+ 转录。要求输出 `(video_id, start_ms, end_ms)` 引用。

9. **评测。** 跑 ActivityNet-QA 和 NeXT-GQA。构建 100 题自定义集。报告整体准确率 + 每类细分（计数、动作、描述）。

## 使用方式

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

## 交付物

`outputs/skill-video-qa.md` 是交付物。给定 YouTube 链接或上传的视频，管道会对场景建立索引，并以带时间戳引用的方式回答问题。

| 权重 | 维度 | 衡量方式 |
|:-:|---|---|
| 25 | 时序定位 IoU | 在保留的定位集上交并比 |
| 20 | 问答准确率 | NeXT-GQA 与自定义 100 题 |
| 20 | 摄入吞吐 | 每美元可处理的视频小时数 |
| 20 | UI 与引用体验 | 时间戳跳转链接、缩略图条、跳帧 |
| 15 | 幻觉率 | 计数和动作类准确率分开报告 |
| **100** | | |

## 练习

1. 将 Gemini 2.5 Pro 替换为 Qwen3-VL-Max 做标题生成遍。在人工标注的 50 个场景样本上汇报标题质量差异。

2. 将每场景的帧嵌入压缩为单池化向量，去掉多向量。衡量检索效果回退。

3. 构建"严格计数"模式：合成器提取每个被计数实例的时间戳，用户点击验证。衡量用户验证是否能降低幻觉。

4. 做摄入成本基准测试：对比三种 VLM 的"每小时视频/美元"开销。找到甜点。

5. 加入说话人分离转录：在音频上运行 pyannote speaker diarization，并为每个说话人单独嵌入转录。演示 "Alice 关于 X 说了什么？" 类查询。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| Scene segmentation | "镜头检测" | 在镜头边界处把视频切分为场景 |
| Multi-vector index | "标题 + 帧 + 转录" | Qdrant 集合中每个表示对应命名向量 |
| Temporal grounding | "具体发生在何时" | 对查询答案的时间窗口 `(start, end)` 做精化 |
| Frame embedding | "视觉表示" | 关键帧的向量嵌入；用于场景-视觉相似度 |
| RRF fusion | "互逆秩融合" | 跨多个排序列表的合并策略；经典混合检索技巧 |
| Counting hallucination | "误数" | VLM 在"多少个 X"问题上已知的失败模式 |
| ActivityNet-QA | "视频问答基准" | 长视频问答准确率评测基准 |

## 延伸阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 开源 VLM 检查点
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — 大规模时序定位
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — 托管参考实现
- [VideoDB](https://videodb.io) — 视频 CRUD API 参考
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — 商业参考
- [TransNetV2](https://github.com/soCzech/TransNetV2) — 场景分割模型
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — 经典开源替代
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — 参考评测基准
