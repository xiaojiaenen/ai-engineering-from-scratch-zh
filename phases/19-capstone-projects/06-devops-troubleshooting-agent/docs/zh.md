# 毕业设计 06 — 面向 Kubernetes 的 DevOps 故障排查智能体

> AWS 的 DevOps Agent 已正式发布，Resolve AI 发布了其 K8s 排错手册，NeuBird 演示了语义监控，Metoro 则将 AI SRE 与每个服务的 SLO 绑定。生产环境的最终形态已定型：告警 webhook 触发后，智能体读取遥测数据，遍历 Kubernetes 对象图，对根本原因假设进行排序，并在 Slack 中发布带有审批按钮的简报。默认只读模式。任何修复操作均需人工确认。本毕业设计正是构建这样一个智能体，将在 20 个模拟故障场景上进行评估，并与 AWS 的 Agent 在三个共享案例上进行比较。

**类型：** 毕业设计
**编程语言：** Python (智能体), TypeScript (Slack 集成)
**前置要求：** 阶段 11 (LLM 工程), 阶段 13 (工具与 MCP), 阶段 14 (智能体), 阶段 15 (自主性), 阶段 17 (基础设施), 阶段 18 (安全性)
**涉及阶段：** P11 · P13 · P14 · P15 · P17 · P18
**时间：** 30 小时

## 问题陈述

2025-2026 年 SRE 的叙事已变成：“AI 智能体进行故障分类，人工审批修复方案。” AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 均已在生产环境中部署这种形态。智能体读取 Prometheus 指标、Loki 日志、Tempo 链路追踪、kube-state-metrics 以及 K8s 对象的知识图谱。它能在五分钟内生成带有遥测数据引用、经过排序的根本原因假设。在没有通过 Slack 获得明确人工批准的情况下，它从不执行任何破坏性命令。

大部分难点在于范围界定和安全性，而非推理。智能体需要一个默认只读的 RBAC 服务账户、一个加固的 MCP 工具服务器，以及记录每个“已考虑”与“已执行”命令的审计日志。它需要知道何时超出自身能力范围并升级问题。同时，其运行成本必须足够低，以防 OOM-kill 级联事件导致产生 5000 美元的智能体账单。

## 概念

该智能体在一个知识图谱上运作。节点是 K8s 对象（Pod、Deployment、Service、Node、HPA、PVC）加上遥测源（Prometheus 系列、Loki 流、Tempo 链路）。边则编码了所有权关系（Pod -> ReplicaSet -> Deployment）、调度关系（Pod -> Node）和观测关系（Pod -> Prometheus 系列）。图谱通过 kube-state-metrics 同步保持更新，并在每次告警时重新采样。

当告警触发时，智能体从受影响的对象开始进行根因分析。它遍历边，拉取相关的遥测数据切片（最近 15 分钟），并起草假设。假设根据证据强度进行排序：有多少遥测数据引用支持它、数据的新鲜度、具体性如何。排名前三的假设将连同图路径可视化一起发送到 Slack，并提供针对修复操作的审批按钮。

修复操作受到门控。允许的默认操作是只读的。破坏性操作（扩容、回滚、删除 Pod）需要 Slack 审批；ArgoCD 回滚钩子需要一个智能体永远不会持有的认证令牌。审计日志记录智能体*考虑过的*每一个命令——而不仅仅是执行过的——因此审查过程可以捕捉到“险些发生”的操作。

## 架构

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## 技术栈

- 可观测性来源：Prometheus, Loki, Tempo, kube-state-metrics
- 知识图谱：K8s 对象 + 遥测边的 Neo4j（托管）或 kuzu（嵌入式）图
- 智能体：基于 LangGraph，具备工具白名单，默认只读
- 工具传输：通过 StreamableHTTP 使用 FastMCP；用于破坏性工具的单独服务器，需通过审批门控
- 模型：Claude Sonnet 4.7 用于根因推理，Gemini 2.5 Flash 用于日志摘要
- 修复：ArgoCD 回滚 webhook，PagerDuty 升级，Slack 审批卡片
- 审计：仅追加的结构化日志（已考虑、已执行、已批准、结果）
- 部署：具有自身窄 RBAC 角色的 K8s 部署；独立的命名空间

## 构建步骤

1.  **图谱接入。** 每 30 秒将 kube-state-metrics 同步到 Neo4j/kuzu。节点：Pod, Deployment, Node, Service, PVC, HPA。边：OWNED_BY, SCHEDULED_ON, EXPOSES, MOUNTS, SCALES。遥测叠加边：OBSERVED_BY（一个 Pod 被一个 Prometheus 系列观测）。

2.  **告警接收器。** 接受 PagerDuty 或 Alertmanager webhook 的 FastAPI 端点。提取受影响的对象和 SLO 违规信息。

3.  **只读工具层。** 通过 FastMCP 封装 kubectl、Prometheus 查询、Loki logql、Tempo traceql。每个工具具有狭窄的 RBAC 动词（“get”、“list”、“describe”）。默认服务器中不包含“delete”、“exec”、“scale”。

4.  **根因分析智能体。** 基于 LangGraph 的三个节点：`sample` 拉取最近 15 分钟的遥测数据切片，`walk` 查询图谱获取邻近对象，`hypothesize` 起草带有遥测数据引用的、排序的根因候选。

5.  **证据评分。** 每个假设的分数 = 时效性 × 具体性 × 图路径长度倒数 × 引用计数。返回前 3 名。

6.  **Slack 简报。** 发布一个附件，包含假设、图路径可视化（在服务器端渲染的子图图像），以及最多一个修复操作的审批按钮。

7.  **修复门控。** 破坏性工具（扩容、回滚、删除）位于第二个 MCP 服务器上，并隐藏在审批令牌之后。智能体只有在 Slack 卡片获得人工批准后才能调用它们。

8.  **审计日志。** 仅追加的 JSONL：对于每个候选命令，记录它是否被考虑、是否被执行、谁批准了它。每天发送到 S3。

9.  **模拟故障套件。** 构建 20 个场景：OOMKill 级联、DNS 波动、HPA 抖动、PVC 满、嘈杂邻居、有缺陷的 sidecar、错误的 ConfigMap 发布、证书轮换、镜像拉取退避等。评估智能体的根因准确性和提出假设所需时间。

## 使用方式

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## 交付标准

`outputs/skill-devops-agent.md` 是交付成果。给定一个 K8s 集群和告警源，该智能体生成排序的根因假设和一个由 Slack 门控的修复流程。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 场景套件上的 RCA 准确率 | ≥80% 在 20 个模拟故障场景中正确识别根本原因 |
| 20 | 安全性 | 在审计日志中，破坏性操作防护从未在没有 Slack 审批的情况下触发 |
| 20 | 假设提出时间 | 从告警到 Slack 简报，第 50 百分位数时间 < 5 分钟 |
| 20 | 可解释性 | 每个假设都有图路径和遥测数据引用 |
| 15 | 集成完整性 | PagerDuty、Slack、ArgoCD、Prometheus 端到端工作 |
| **100** | | |

## 练习

1.  在 AWS 的 DevOps Agent 演示过的相同三个故障上运行你的智能体。发布并列对比结果。报告智能体产生分歧的地方。

2.  添加一个“险些发生”审计，标记出智能体*考虑过*的、若无批准则属破坏性的任何命令。测量一周内的险些发生率。

3.  将假设模型从 Claude Sonnet 4.7 换成自托管的 Llama 3.3 70B。测量 RCA 准确率变化及每起故障的成本。

4.  构建一个因果过滤器：区分相关的遥测数据峰值和真正的根本原因。在 20 个场景标签上训练一个小型分类器。

5.  添加回滚预演：在具有相同 manifest 的预发布集群上进行 ArgoCD 回滚。在 Slack 审批按钮之前，在实时集群中验证回滚计划。

## 关键术语

| 术语 | 人们怎么说 | 它的实际含义 |
|------|-----------------|------------------------|
| K8s 知识图谱 | "集群图谱" | 节点 = K8s 对象 + 遥测系列；边 = 所有权、调度、观测关系 |
| 默认只读 | "范围受限的 RBAC" | 智能体的服务账户仅具有 get/list/describe 动词；破坏性动词位于另一个需要审批的服务器上 |
| 审计日志 | "已考虑 vs 已执行" | 记录每个候选命令的仅追加记录，是否执行，谁批准 |
| 假设排序 | "证据评分" | 时效性 × 具体性 × 图路径长度倒数 × 引用计数 |
| Slack 审批卡片 | "人在回路门控" | 带有修复按钮的交互式 Slack 消息；智能体需等待人工点击才能继续 |
| 遥测数据引用 | "证据指针" | 支持某个断言的 Prometheus 查询、Loki 选择器或 Tempo 追踪 URL |
| MTTR | "解决时间" | 从告警触发到 SLO 恢复的挂钟时间 |

## 扩展阅读

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026 年权威参考
- [Resolve AI K8s 故障排查](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 竞品参考
- [NeuBird 语义监控](https://www.neubird.ai) — 语义图谱方法
- [Metoro AI SRE](https://metoro.io) — 以 SLO 为核心的生产环境框架
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — 集群状态来源
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 参考智能体编排器
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP 服务器框架
- [ArgoCD 回滚](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — 受门控的修复目标