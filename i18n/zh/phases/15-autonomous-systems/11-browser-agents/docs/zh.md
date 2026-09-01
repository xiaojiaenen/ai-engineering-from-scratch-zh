# 浏览器代理与长周期网页任务

> ChatGPT agent（2025年7月）将 Operator 和深度研究合并为一个浏览器/终端代理，并将 BrowseComp 的 SOTA 提升至 68.9%。OpenAI 于 2025年8月31日关闭了 Operator——在产品线层进行整合。Anthropic 收购 Vercept 后，Claude Sonnet 在 OSWorld 上的表现从不足 15% 跃升至 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修正了原始 WebArena 中 11.3 个百分点的假阴性率，并推出了 258 题的 Hard 子集。这些数字是真实的。攻击面同样真实：OpenAI 的准备负责人公开表示，对浏览器代理的间接提示注入"是一个无法完全修补的缺陷"。2025–2026年记录的攻击包括：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks），以及 Perplexity Comet 中的一键劫持。

**类型：** 学习
**语言：** Python（stdlib，间接提示注入攻击面模型）
**前置知识：** 阶段 15 · 10（权限模式），阶段 15 · 01（长周期代理）
**时间：** 约 45 分钟

## 问题所在

浏览器代理是一种长周期代理，它读取不受信任的内容并执行具有后果性的操作。代理访问的每个页面都是用户未编写的输入。每个页面上的每个表单都是一个潜在的命令通道。2025–2026年的攻击案例库表明这并非假设：Tainted Memories 让攻击者通过精心构造的页面将恶意指令绑定到代理的记忆中；HashJack 将命令隐藏在代理访问过的 URL 片段中；Perplexity Comet 劫持只需一次点击即可触发。

防御态势令人不安。OpenAI 的准备负责人说得很清楚：间接提示注入"是一个无法完全修补的缺陷"。这是因为攻击存在于代理的"读取 vs 执行"边界中，而这个边界在架构上是模糊的——模型读取的每个 token 理论上都可以被当作指令来读取。

本课程命名了攻击面，梳理了基准测试 Landscape（BrowseComp、OSWorld、WebArena-Verified），并对一个最小化的间接提示注入场景进行了建模，以便你在第 14 和第 18 课中能够推理出真正的防御方案。

## 核心概念

### 2026年 Landscape：各系统简述

**ChatGPT agent（OpenAI）。** 2025年7月发布。统一了 Operator（浏览）和 Deep Research（多小时研究）。于 2025年8月31日关闭独立版 Operator。在 BrowseComp 上达到 SOTA 68.9%，在 OSWorld 和 WebArena-Verified 上表现强劲。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic 收购 Vercept 聚焦于计算机使用能力。将 Claude Sonnet 在 OSWorld 上的分数从不足 15% 提升至 72.5%。Claude Computer Use 以工具 API 形式发布。

**Gemini 3 Pro with Browser Use（DeepMind）。** Browser Use 集成提供计算机使用控制；FSF v3（2026年4月，第 20 课）专门追踪 ML 研发领域的自主性。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 修正了一个众所周知的问题：原始 WebArena 约有 11.3% 的假阴性率（即实际已完成但被标记为失败的任务）。Verified 版本通过人工编排的成功标准重新评分，并新增 258 题的 Hard 子集（ICLR 2026 论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs OSWorld vs WebArena

| 基准测试 | 衡量内容 | 时间尺度 |
|---|---|---|
| BrowseComp | 在限时条件下从开放网页中查找特定事实 | 分钟级 |
| OSWorld | 代理操作完整桌面（鼠标、键盘、shell） | 十几分钟级 |
| WebArena-Verified | 模拟站点中的事务性网页任务 | 分钟级 |
| Hard 子集 | WebArena-Verified 中具有多页面状态转换的任务 | 十几分钟级 |

不同的评估维度。高 BrowseComp 分数说明代理能查找事实，但不代表它能订机票。OSWorld 分数更接近"在我的桌面上能否工作"。WebArena-Verified 更接近"能否完成一个流程"。任何生产决策都需要选择与任务分布匹配的基准测试。

### 命名攻击面

1. **间接提示注入。** 不受信任的页面内容中包含指令。代理读取了它们，然后执行了它们。公开案例：2024年 Kai Greshake 等人、2025年 Tainted Memories 论文、2026年 HashJack（Cato Networks）。
2. **URL 片段 / 查询参数注入。** 被爬取 URL 的 `#fragment` 或查询字符串中包含命令。不可见渲染，但仍在代理的上下文内。
3. **记忆绑定攻击。** 页面指示代理写入持久化记忆（第 12 课涵盖持久化状态）。下一次会话中，记忆触发载荷，无需可见触发器。
4. **针对已认证会话的 CSRF 形态攻击。** Tainted Memories 类别：代理在某处已登录；攻击者的页面发起状态变更请求，代理使用用户的 Cookie 执行这些请求。
5. **一键劫持。** 一个视觉上无害的按钮承载了代理跟随执行的载荷。Comet 类别。
6. **代理宿主层面的 Content-Security-Policy 漏洞。** 渲染层和工具层本身可能成为攻击向量；浏览器嵌套在浏览器代理中的堆栈非常广泛。

### 为何"无法完全修补"

该攻击与代理的能力是同构的。代理必须读取不受信任的内容才能完成工作。代理读取的任何内容都可能包含指令。代理遵循的任何指令都可能与用户的实际请求不一致。防御措施（信任边界、分类器、工具白名单、关键操作人工审批）提高了攻击成本并缩小了其爆炸半径。但它们无法消除这一攻击类别。

这与第 8 课中 Lob 定理的推理模式相同：代理无法证明下一个 token 是安全的；它只能建立一个系统，使不安全 token 更易于检测。

### 真正能落地的防御姿态

- **读/写边界。** 读取永远不具后果性。写入（提交表单、发布内容、调用有副作用的工具）如果触发内容来自信任边界之外，则需要再次获得人工批准。
- **按任务配置工具白名单。** 代理可以浏览；除非该工具被明确启用，否则不能发起汇款。第 13 课涵盖预算。
- **会话隔离。** 浏览器代理会话仅使用受限凭据运行。不使用生产认证，不使用个人邮箱。保留每条 HTTP 请求的日志以供审计。
- **内容过滤器。** 获取的 HTML 在被拼接进模型上下文之前，先剥离已知危险模式。（降低简单攻击的成功率；无法阻止复杂载荷。）
- **关键操作 HITL。** 提议-提交模式（第 15 课）。
- **记忆上的金丝雀令牌。** 如果某条记忆条目被触发，用户会看到它（第 14 课）。

```figure
injection-boundary
```

## 动手实践

`code/main.py` 对一个浏览器代理在三个合成页面上运行进行了建模。其中一个页面是良性的，一个在可见文本中包含直接提示注入块，一个包含 URL 片段注入（不可见但位于代理上下文内）。脚本展示了：(a) 朴素代理会做什么，(b) 读/写边界能拦截什么，(c) 过滤器能拦截什么，(d) 两者都拦截不了什么。

## 部署

`outputs/skill-browser-agent-trust-boundary.md` 规划了一个拟议的浏览器代理部署范围：涉及哪些信任区域、被授权写入什么内容、首次运行前必须部署哪些防御措施。

## 练习

1. 运行 `code/main.py`。识别过滤器能拦截但读/写边界不能拦截的攻击，以及读/写边界能拦截而过滤器不能拦截的攻击。

2. 扩展过滤器以检测一类 HashJack 风格的 URL 片段注入。测量在包含合法片段的良性 URL 上的误报率。

3. 选择一个你熟悉的真实浏览器代理工作流（例如"预订机票"）。列出每一条读操作和每一条写操作。标注哪些写操作需要 HITL 及其原因。

4. 阅读 WebArena-Verified ICLR 2026 论文。找出原始 WebArena 评分不可靠的一类任务，并解释 Verified 子集如何解决这一问题。

5. 为浏览器代理场景设计一个记忆金丝雀令牌。你会存储什么、存储在哪里、什么会触发警报？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|---|---|---|
| 间接提示注入 | "糟糕的页面文本" | 代理读取的页面中，不受信任的内容包含代理会执行的指令 |
| Tainted Memories | "记忆攻击" | 代理将攻击者提供的指令写入持久化记忆；在下一次会话中触发 |
| HashJack | "URL 片段攻击" | 隐藏在 URL 片段 / 查询字符串中的载荷位于代理上下文中但不可见渲染 |
| 一键劫持 | "糟糕的按钮" | 视觉上可交互的元素承载了代理会跟随执行的载荷 |
| BrowseComp | "网页搜索基准" | 在开放网页上查找特定事实；分钟级时间尺度 |
| OSWorld | "桌面基准" | 完整 OS 控制；多步 GUI 任务 |
| WebArena-Verified | "修正后的网页任务基准" | ServiceNow 重新评分的 WebArena，含 Hard 子集 |
| 读/写边界 | "副作用闸门" | 读取永远不具后果性；如果内容来自信任边界之外，写入需要新的批准 |

## 延伸阅读

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator 与深度研究的合并；BrowseComp SOTA。
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — Operator 技术 lineage 及演变为 ChatGPT agent 的架构。
- [Zhou et al. — WebArena](https://webarena.dev/) — 原始基准测试。
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 修正子集论文。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 包含对计算机使用代理攻击面的讨论。
