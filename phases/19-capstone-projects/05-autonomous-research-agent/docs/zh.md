# 毕业设计 05 —— 自主研究智能体（AI-科学家级）

> Sakana 的 AI-Scientist-v2 已发表完整论文。Agent Laboratory 执行了实验。Allen AI 分享了运行轨迹。2026年的形态是：基于计划-执行-验证的树搜索，管理实验预算，沙盒化代码执行，一个带视觉反馈的 LaTeX 写作器，以及一个自动化 NeurIPS 风格的审稿人集成系统。本毕业设计的目标是构建这样一个系统，在每篇论文 30 美元的成本内端到端运行它，并通过 Sakana 文档中记录的沙盒逃逸红队测试。

**类型:** 毕业设计
**语言:** Python (智能体 + 沙盒), LaTeX (输出)
**先决条件:** 阶段 2 (机器学习), 阶段 3 (深度学习), 阶段 7 (Transformer), 阶段 10 (从零构建大语言模型), 阶段 14 (智能体), 阶段 15 (自主系统), 阶段 16 (多智能体), 阶段 18 (安全)
**涉及的阶段:** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**时间:** 40 小时

## 问题

自主研究智能体在 2026 年跨越了一个门槛。Sakana AI 的 AI-Scientist-v2 发表在《自然》杂志上，其生成的论文通过了研讨会同行评审。ShinkaEvolve (ICLR 2026) 将这条路线延伸到了演化假设。AMD 的 Agent Laboratory 发布了可复现的运行轨迹。这些智能体并非魔法——它们是一个运行在候选实验树上的计划-执行-验证循环，配有成本上限、种子绑定沙盒和自动化评审。其核心工艺在于循环、预算和安全方案。

你将通过在一个狭窄领域（例如，对一个 1 亿参数 Transformer 进行注意力稀疏性消融）针对一个种子想法实现一个这样的智能体，来学习这个循环。价值不在于第一次运行就发现新事物。价值在于基础设施：树搜索、实验沙盒、写作-评审循环、红队报告。Sakana 团队记录了沙盒逃逸失败案例；你的智能体必须通过同样的红队测试。

## 概念

该智能体是一个最佳优先树搜索。节点是实验规格：（假设、配置、代码、预期结果）。扩展步骤通过微小改动（交换优化器、调整批量大小、消融某个组件）提出子节点。每个子节点在一个全新的沙盒中运行，并设有严格的资源上限。结果反馈到一个评分函数中，该函数根据（新颖性 × 质量 × 剩余预算）对节点进行排名。树持续生长直到预算耗尽，然后对最佳分支进行总结报告。

写作者是多模态的。它生成 LaTeX 草稿，编译它，渲染图表，并将渲染后的 PDF 反馈给 Claude Opus 4.7 的视觉模式，对排版、图表可读性和主张-证据一致性进行评判。一个由五名 LLM 评委组成的集成系统发出 NeurIPS 风格的评分（新颖性、严谨性、清晰性、可复现性、影响力）；如果平均分低于阈值，论文将连同评审意见返回给写作者。

安全是基础。每个实验都在一个 E2B 或 Daytona 沙盒中运行，无网络出口、有固定的墙钟时间和固定的资源限制。智能体的代码生成步骤会经过一个策略层，该层会阻止可能逃逸沙盒的系统调用。红队报告复现了 Sakana 记录的攻击面（fork 炸弹、文件系统逃逸、LLM 编写的网络调用）。

## 架构

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## 技术栈

- 编排：带有检查点和人工批准闸门的 LangGraph
- 树搜索：针对实验节点的自定义最佳优先搜索（来自 Sakana v2 的 AB-MCTS 风格）
- 沙盒：每个实验使用 E2B，Docker-in-Docker 作为后备；通过 cgroups 进行资源限制
- 文献：Semantic Scholar Graph API + OpenAlex + 摘要的本地 FAISS 缓存
- 写作者：LaTeX 模板 + Claude Opus 4.7（视觉模式）用于图表评审和排版
- 评审者：5 名评委的集成（Opus 4.7, GPT-5.4, Gemini 3 Pro, DeepSeek R1, Qwen3-Max），进行加权聚合
- 实验框架：物理实验使用 PyTorch 2.5，使用 W&B 记录日志
- 可观测性：使用 Langfuse 追踪智能体轨迹，每篇论文有 30 美元硬性预算

## 构建步骤

1.  **种子和领域界定。** 采用一个种子想法（例如，“研究低于 1B 参数的 Transformer 中注意力图的稀疏模式”）。定义搜索空间：模型、数据集、计算预算。

2.  **文献检索。** 查询 Semantic Scholar + OpenAlex 获取 50 篇被引次数最高的相关论文；在本地缓存摘要；生成一份 1 页的领域摘要。

3.  **树结构搭建。** 用种子假设初始化根节点。实现 `expand(node) -> children` 并提出小幅度编辑建议（每个子节点只改变一个配置）。将 `score(node)` 实现为一个加权的（新颖性 × 质量 × 预算）项。

4.  **沙盒封装。** 每个实验运行 `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only`（或等效的 E2B 策略）。种子写入沙盒；输出以只读方式挂载回主机。

5.  **计划-执行-验证循环。** `plan` 提出子节点。`execute` 运行沙盒，捕获日志和指标。`verify` 对指标运行单元检查（损失是否降低？消融是否隔离了效果？）。失败的节点会在树上存储失败原因。

6.  **写作者。** 预算耗尽后，选择最佳分支。使用 matplotlib 渲染图表。通过 Claude Opus 4.7 并在上下文中提供分支轨迹，生成 LaTeX 草稿。编译。将编译后的 PDF 反馈给 Opus 4.7 视觉模式进行评审。迭代。

7.  **评审者集成。** 五名评委使用 NeurIPS 风格的评分标准对草稿在（新颖性、严谨性、清晰性、可复现性、影响力）方面进行评分。如果平均分 < 4.0/5，则连同评审意见返回给写作者。3 次重写后硬性停止。

8.  **红队测试。** 构建或集成一组针对沙盒的对抗性任务：fork 炸弹、网络数据渗漏尝试、文件系统逃逸、LLM 编写的 shell 元字符。确认所有攻击均被阻止。撰写发现报告。

9.  **可复现性。** 每篇论文附带其树搜索轨迹 JSON、种子、W&B 运行链接、沙盒配置以及一个可端到端复现的 README。

## 使用它

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## 交付物

`outputs/skill-ai-scientist.md` 是交付物。给定一个种子想法 + 一个领域 + 30 美元预算，它运行整个流程并输出一份可评审的论文加上一个可复现性包。

| 权重 | 标准 | 如何衡量 |
|:-:|---|---|
| 25 | 论文质量 | 根据已发表的研讨会论文进行盲评标准评审 |
| 20 | 实验严谨性 | 基线、种子、消融实验；每个主张都有结果表中的单元支持 |
| 20 | 成本和计算约束 | 强制执行 30 美元/篇论文的上限，通过 Langfuse 追踪 |
| 20 | 安全性 | 通过沙盒红队测试；验证网络策略和终止开关 |
| 15 | 可复现性 | 使用相同种子一键重跑可复现论文 |
| **100** | | |

## 练习

1.  在同一领域针对三个不同的种子想法运行该流程。比较树搜索的哪些部分重叠。识别出被浪费的重复计算。
2.  在估计成本超过 5 美元的实验执行前，增加一个人工在环闸门。衡量总成本下降了多少。
3.  将评审者集成换成单一评委。衡量在一组已知差论文（held-out set）上的误接受率。
4.  引入一个网络数据渗漏红队测试：智能体编写代码尝试 `curl` 一个外部地址。确认 `--network=none` 策略阻止了它。记录该尝试。
5.  将你的树搜索与一个扁平的随机基线（相同预算，无扩展策略）进行比较。报告新颖性 × 质量的增益。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|-----------------|------------------------|
| 树搜索 | "AB-MCTS 风格的扩展" | 基于新颖性×质量×预算得分的最佳优先实验节点探索 |
| 沙盒 | "实验隔离" | 无网络、受限 CPU/内存、固定种子、输入只读的容器 |
| 视觉评审 | "先渲染后阅读" | 将论文编译为 PDF，将 PDF 反馈给视觉语言模型进行排版和主张-证据评审 |
| 评审者集成 | "自动化同行评审" | 多个 LLM 评委使用 NeurIPS 评分标准为论文打分；加权聚合作为流程闸门 |
| 新颖性得分 | "这个新吗？" | 惩罚与 50 篇文献缓存相近性的启发式方法 |
| 成本上限 | "$ 预算" | 每篇论文总支出的硬性上限；Langfuse 计数器 + 预运行估算 |
| 红队 | "沙盒逃逸审计" | 如果策略错误则可能逃逸沙盒的对抗性任务 |

## 扩展阅读

- [Sakana AI-Scientist-v2 代码库](https://github.com/SakanaAI/AI-Scientist-v2) — 参考生产级研究智能体
- [Sakana AI-Scientist-v1 论文 (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) — 原始方法论
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) — 演化扩展
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) — 多角色研究实验室框架
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — 参考编排层
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — 文献搜索
- [E2B 沙盒](https://e2b.dev) — 参考实验隔离方案
- [NeurIPS 评审指南](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — 评审者集成所编码的评分标准