# 浏览器智能体与长周期网络任务

> ChatGPT 智能体（2025 年 7 月）将 Operator 和深度研究功能整合为一个浏览器/终端智能体，并在 BrowseComp 基准测试上取得了 68.9% 的 SOTA 成绩。OpenAI 于 2025 年 8 月 31 日关闭了 Operator —— 这是产品层的整合。Anthropic 收购 Vercept 后，将 Claude Sonnet 在 OSWorld 上的得分从不足 15% 提升到了 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修复了原始 WebArena 中 11.3 个百分点的假阴性率问题，并推出了包含 258 个任务的 Hard 子集。这些数字是真实的。攻击面也同样真实：OpenAI 的准备负责人公开表示，针对浏览器智能体的间接提示注入"是一个无法被彻底修补的漏洞"。记录在案的 2025-2026 年攻击案例包括：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks），以及 Perplexity Comet 中的一键劫持。

**类型：** 学习  
**语言：** Python（标准库，间接提示注入攻击面模型）  
**前置要求：** 阶段 15 · 10（权限模式），阶段 15 · 01（长周期智能体）  
**时间：** 约 45 分钟

## 问题所在

浏览器智能体是一种长周期智能体，它读取不可信内容并执行具有重大后果的操作。智能体访问的每一个页面，都是用户未曾编写的输入。每个页面上的每个表单都是一个潜在的指令通道。2025-2026 年的攻击案例表明这并非假设：Tainted Memories 允许攻击者通过精心构造的页面将恶意指令绑定到智能体的记忆中；HashJack 将指令隐藏在智能体访问的 URL 片段中；Perplexity Comet 劫持则通过单次点击即可完成。

防御前景令人不安。OpenAI 的准备负责人道出了一个不愿明言的事实：间接提示注入"是一个无法被彻底修补的漏洞"。这是因为攻击存在于智能体的"读取-行动"边界上，而这一边界在架构上是模糊的 —— 模型读取的每一个 token 原则上都可能被当作指令来执行。

本课程将阐明攻击面，介绍基准测试格局（BrowseComp、OSWorld、WebArena-Verified），并对一个最小的间接提示注入场景进行建模，以便你在第 14 和 18 课中能够推演真实的防御措施。

## 核心概念

### 2026 年格局概览（每个系统一段）

**ChatGPT 智能体（OpenAI）。** 于 2025 年 7 月发布。将 Operator（浏览）和 Deep Research（多小时研究）统一起来。于 2025 年 8 月 31 日关闭了独立的 Operator。在 BrowseComp 上达到 68.9% 的 SOTA 成绩；在 OSWorld 和 WebArena-Verified 上表现强劲。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic 收购 Vercept 专注于计算机使用能力。将 Claude Sonnet 在 OSWorld 上的得分从 <15% 提升到 72.5%。Claude Computer Use 作为工具 API 发布。

**Gemini 3 Pro with Browser Use（DeepMind）。** 浏览器使用集成提供了计算机使用控制；FSF v3（2026 年 4 月，第 20 课）专门跟踪 ML 研发领域的自主性。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 修复了一个有据可查的问题：原始 WebArena 有约 11.3% 的假阴性率（实际已解决但被标记为失败的任务）。Verified 版本使用人工制定的成功标准重新评分，并增加了一个包含 258 个任务的 Hard 子集（ICLR 2026 论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs OSWorld vs WebArena

| 基准测试 | 衡量内容 | 时间跨度 |
|---|---|---|
| BrowseComp | 在时间压力下从开放网络中查找特定事实 | 分钟级 |
| OSWorld | 智能体操作完整桌面（鼠标、键盘、命令行） | 数十分钟级 |
| WebArena-Verified | 在模拟网站中完成事务性网络任务 | 分钟级 |
| Hard 子集 | 具有多页状态转换的 WebArena-Verified 任务 | 数十分钟级 |

不同的衡量维度。BrowseComp 高分说明智能体能查找事实，但不能说明它能预订航班。OSWorld 分数更接近"它能否在我的桌面上工作"。WebArena-Verified 更接近"它能否完成一个流程"。任何生产决策都需要选择与任务分布相匹配的基准测试。

### 攻击面详解

1.  **间接提示注入。** 不受信任的页面内容包含指令。智能体读取这些指令。智能体执行这些指令。公开案例：2024 年 Kai Greshake 等人、2025 年 Tainted Memories 论文、2026 年 HashJack（Cato Networks）。
2.  **URL 片段/查询注入。** 被爬取 URL 的 `#fragment` 或查询字符串中包含指令。不会被视觉渲染；但仍在智能体的上下文中。
3.  **记忆绑定攻击。** 页面指示智能体写入持久化记忆（第 12 课介绍持久状态）。在下一次会话中，记忆无需可见触发即可执行有效载荷。
4.  **针对已认证会话的 CSRF 攻击。** Tainted Memories 类：智能体已登录某处；攻击者的页面发出状态变更请求，智能体使用用户的 cookie 执行该请求。
5.  **一键劫持。** 一个视觉上无害的按钮搭载了智能体会遵循的有效载荷。Comet 类。
6.  **智能体宿主层的内容安全策略漏洞。** 渲染层和工具层本身可能成为攻击向量；浏览器-在-浏览器-智能体栈的攻击面很大。

### 为何"无法彻底修补"

攻击与智能体的能力是同构的。智能体必须读取不受信任的内容才能完成工作。智能体读取的任何内容都可能包含指令。智能体遵循的任何指令都可能偏离用户的实际请求。防御措施（信任边界、分类器、工具白名单、对重大后果操作的人机回环）提高了攻击的成本并减小了影响范围，但并未消除这一类攻击。

这与 Lob 定理（第 8 课）的推理模式相同：智能体无法证明下一个 token 是安全的；它只能建立一个系统，使不安全的 token 更易被检测到。

### 实际部署的防御态势

-   **读/写边界。** 读取永远不会产生重大后果。写入（提交表单、发布内容、调用具有副作用的工具）如果发起内容来自信任边界之外，则需要新的人工批准。
-   **按任务配置工具白名单。** 智能体可以浏览；但除非为该任务明确启用了转账工具，否则它不能发起转账。第 13 课介绍预算。
-   **会话隔离。** 浏览器智能体会话仅使用限定范围的凭证运行。不使用生产环境认证，不使用个人邮箱。保留每个 HTTP 请求的日志以供审计。
-   **内容净化器。** 获取的 HTML 在拼接到模型上下文之前，会剥离已知的恶意模式。（能减少简单攻击；无法阻止复杂的有效载荷。）
-   **对重大后果操作执行人机回环。** 提议-然后-提交模式（第 15 课）。
-   **记忆金丝雀。** 如果某个记忆条目被触发，用户会看到它（第 14 课）。

## 动手实践

`code/main.py` 模拟了一个小型浏览器智能体对三个合成页面的运行过程。一个页面是良性的，一个在可见文本中包含直接的提示注入代码块，一个包含 URL 片段注入（不可见但位于智能体的上下文中）。该脚本展示了（a）一个朴素智能体会做什么，（b）读/写边界捕获了什么，（c）净化器捕获了什么，（d）两者都未捕获什么。

## 部署规划

`outputs/skill-browser-agent-trust-boundary.md` 对一个拟议的浏览器智能体部署进行范围界定：它涉及哪些信任区域，它被授权写入什么，以及在首次运行前必须具备哪些防御措施。

## 练习

1.  运行 `code/main.py`。识别哪种攻击被净化器捕获但读/写边界未捕获，以及哪种攻击只有读/写边界能捕获。

2.  扩展净化器以检测一类 HashJack 式的 URL 片段注入。测量对具有合法片段的良性 URL 的误报率。

3.  选择一个你了解的真实浏览器智能体工作流（例如“预订航班”）。列出每一个读操作和每一个写操作。标记哪些写操作需要人机回环以及原因。

4.  阅读 WebArena-Verified ICLR 2026 论文。找出一个原始 WebArena 评分不可靠的任务类别，并解释 Verified 子集如何解决该问题。

5.  为浏览器智能体场景设计一个记忆金丝雀。你会存储什么，存储在哪里，以及什么会触发警报？

## 关键术语

| 术语 | 人们通常说 | 实际含义 |
|---|---|---|
| 间接提示注入 | “恶意页面文本” | 智能体读取的页面中，不受信任的内容包含智能体执行的指令 |
| Tainted Memories | “记忆攻击” | 智能体将攻击者提供的指令写入持久记忆；在下一次会话中触发 |
| HashJack | “URL 片段攻击” | 隐藏在 URL 片段/查询字符串中的有效载荷位于智能体的上下文中，但不会被视觉渲染 |
| 一键劫持 | “恶意按钮” | 可见的操作元素搭载了智能体会执行的后续有效载荷 |
| BrowseComp | “网络搜索基准” | 在开放网络中查找特定事实；分钟级时间跨度 |
| OSWorld | “桌面基准” | 完整的操作系统控制；多步骤 GUI 任务 |
| WebArena-Verified | “修复后的网络任务基准” | ServiceNow 重新评分的 WebArena，包含 Hard 子集 |
| 读/写边界 | “副作用门槛” | 读取永远不会产生重大后果；如果内容超出信任范围，写入需要新的批准 |

## 延伸阅读

-   [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator 和深度研究的合并；BrowseComp SOTA。
-   [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — Operator 的谱系及其演变为 ChatGPT 智能体的架构。
-   [Zhou et al. — WebArena](https://webarena.dev/) — 原始基准测试。
-   [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 修复子集论文。
-   [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 包含计算机使用智能体的攻击面讨论。