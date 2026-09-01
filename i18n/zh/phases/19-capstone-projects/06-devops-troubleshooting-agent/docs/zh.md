# Capstone 06 — Kubernetes DevOps 故障排查代理

> AWS 的 DevOps Agent 已正式商用，Resolve AI 发布了 K8s 操作手册，NeuBird 演示了语义监控，Metoro 将 AI SRE 与每服务 SLO 绑定。生产形态已定型：告警 Webhook 触发，代理读取遥测数据，遍历 K8s 对象图谱，对根因假设进行排序，并发送附带审批按钮的 Slack 简报。默认只读，所有修复操作须经人工审批。本 Capstone 就是这个代理，在 20 个合成故障上评估，并与 AWS Agent 在三个共享案例上进行对比。

**类型：** Capstone 项目
**语言：** Python（代理），TypeScript（Slack 集成）
**前置知识：** Phase 11（LLM 工程）、Phase 13（工具与 MCP）、Phase 14（代理）、Phase 15（自主化）、Phase 17（基础设施）、Phase 18（安全）
**涉及阶段：** P11 · P13 · P14 · P15 · P17 · P18
**时间：** 30 小时

## 问题

2025-2026 年的 SRE 叙事变成了："AI 代理处理告警分诊，人工审批修复操作。" AWS DevOps Agent、Resolve AI、NeuBird、Metoro、PagerDuty AIOps 都在生产环境中实现了这一形态。代理读取 Prometheus 指标、Loki 日志、Tempo 链路追踪、kube-state-metrics，以及 K8s 对象的知识图谱。它在五分钟内生成带遥测引用的排序根因假设。它永远不会在无人工 Slack 明确批准的情况下执行破坏性命令。

大部分难点在于范围界定和安全，而非推理。代理需要默认只读的 RBAC 接口、加固的 MCP 工具服务器，以及记录每条"考虑过但未执行"命令的审计日志。它需要知道自己能力边界在哪并向上级上报。它还得足够便宜，避免 OOM-Kill 级联引发 $5k 的代理账单。

## 概念

代理基于知识图谱运行。节点是 K8s 对象（Pods、Deployments、Services、Nodes、HPAs、PVCs）加上遥测来源（Prometheus 序列、Loki 流、Tempo 链路）。边编码了所有权关系（Pod -> ReplicaSet -> Deployment）、调度关系（Pod -> Node）和观测关系（Pod -> Prometheus 序列）。图谱通过 kube-state-metrics 同步保持新鲜，每次告警时重新采样。

告警触发后，代理从受影响对象出发进行根因推导。它遍历边，拉取相关遥测切片（最近 15 分钟），草拟假设。假设按证据强度排序：多少遥测引用支持它、多久之前、多具体。前三名假设发送到 Slack，附带图谱路径可视化和修复操作的审批按钮。

修复操作设有门禁。默认允许的操作仅为只读。破坏性操作（缩容、回滚、删除 Pods）需要 Slack 审批；ArgoCD 回滚钩子需要的认证令牌，代理永远不持有。审计日志记录代理*考虑过*的每条命令——不仅仅是执行的——以便审查流程能捕获"差一点就出事"的情况。

## 架构

```
PagerDuty / Alertmanager Webhook
           |
           v
     FastAPI 接收器
           |
           v
   LangGraph 根因代理
           |
           +---- 只读 MCP 工具 ----+
           |                       |
           v                       v
   K8s 知识图谱               遥测切片
     (Neo4j / kuzu)        Prometheus, Loki, Tempo
   所有权 + 调度关系         最近 15 分钟，已限定范围
           |
           v
   假设排序（证据权重）
           |
           v
   Slack 简报 + 审批按钮
           |
           v (已批准)
   ArgoCD 回滚钩子 / PagerDuty 升级
           |
           v
   审计日志：考虑过 vs 已执行，每条命令
```

## 技术栈

- 可观测性来源：Prometheus、Loki、Tempo、kube-state-metrics
- 知识图谱：Neo4j（托管）或 kuzu（嵌入式）存储 K8s 对象 + 遥测边
- 代理：LangGraph，带每工具允许列表，默认只读
- 工具传输：FastMCP over StreamableHTTP；破坏性工具放在单独服务器，受审批门控
- 模型：Claude Sonnet 4.7 用于根因推理，Gemini 2.5 Flash 用于日志摘要
- 修复操作：ArgoCD 回滚 Webhook、PagerDuty 升级、Slack 审批卡片
- 审计：追加式结构化日志（考虑过、已执行、已批准、结果）
- 部署：K8s Deployment，带自己的窄 RBAC 角色；独立命名空间

```figure
ce-rootcause-walk
```

## 构建步骤

1. **图谱摄取。** 每 30 秒将 kube-state-metrics 同步到 Neo4j/kuzu。节点：Pod、Deployment、Node、Service、PVC、HPA。边：OWNED_BY、SCHEDULED_ON、EXPOSES、MOUNTS、SCALES。遥测覆盖边：OBSERVED_BY（一个 Pod 被一个 Prometheus 序列观测）。

2. **告警接收器。** FastAPI 端点，接受 PagerDuty 或 Alertmanager Webhook。提取受影响对象和 SLO 违规。

3. **只读工具接口。** 通过 FastMCP 封装 kubectl、Prometheus 查询、Loki logql、Tempo traceql。每个工具都有窄 RBAC 动词（"get"、"list"、"describe"）。默认服务器中不含 "delete"、"exec"、"scale"。

4. **根因代理。** LangGraph 包含三个节点：`sample` 拉取最近 15 分钟的遥测切片，`walk` 查询图谱中的邻居对象，`hypothesize` 草拟带遥测引用的排序根因候选。

5. **证据评分。** 每个假设的分数 = 时效性 × 具体性 × 图谱路径长度倒数 × 引用数量。返回前三名。

6. **Slack 简报。** 发布包含假设、图谱路径可视化（服务器端渲染的子图图片）、最多一个修复操作的审批按钮的附件。

7. **修复门禁。** 破坏性工具（缩容、回滚、删除）位于第二个 MCP 服务器上，受审批令牌保护。只有经过 Slack 卡片人工批准后，代理才能调用它们。

8. **审计日志。** 追加式 JSONL：对每个候选命令，记录是否被考虑、是否被执行、谁批准的。每日导出到 S3。

9. **合成故障套件。** 构建 20 个场景：OOM-Kill 级联、DNS 抖动、HPA 震荡、PVC 填满、嘈杂邻居、故障 Sidecar、错误的 ConfigMap 发布、证书轮换、镜像拉取退避等。在根因准确性和假设生成时间上对代理进行评分。

## 使用示例

```
Webhook: alert.pagerduty.com -> checkout-api SLO 违规，错误率 14%
[图谱]   受影响: Deployment checkout-api (3 个 Pods，节点 ip-10-2-3-4)
[遍历]   邻居: ReplicaSet checkout-api-abc, Service checkout-api,
          14 分钟前有最近发布
[采样]   prometheus error_rate 14%，上升趋势；loki 在 /api/v2/pay 上出现 500 错误
[假设]   #1 发布错误：最新镜像 checkout-api:v2.41 的 /healthz 失败
          引用: deploy.yaml (rev 42)、prometheus errorRate、loki 500 堆栈
[Slack]  [回滚到 v2.40]  [升级]  [忽略]
          （需要审批；代理不会单方面回滚）
```

## 交付物

`outputs/skill-devops-agent.md` 为交付件。给定 K8s 集群和告警来源，代理生成排序的根因假设，并通过 Slack 门控修复流程。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 场景套件上的 RCA 准确率 | 20 个合成故障中 ≥80% 的根因正确 |
| 20 | 安全性 | 审计日志中破坏性操作从未未经 Slack 批准而触发 |
| 20 | 假设生成时间 | p50 从告警到 Slack 简报低于 5 分钟 |
| 20 | 可解释性 | 每个假设都带图谱路径和遥测引用 |
| 15 | 集成完整度 | PagerDuty、Slack、ArgoCD、Prometheus 端到端可用 |
| **100** | | |

## 练习

1. 在 AWS DevOps Agent 演示的三个相同故障上运行你的代理。发布对比结果，报告代理的分歧点。

2. 添加"近失"审计，标记代理*考虑过*但未经批准就是破坏性的任何命令。测量一周内的近失率。

3. 将假设模型从 Claude Sonnet 4.7 换成自托管 Llama 3.3 70B。测量 RCA 准确率变化幅度和本故障成本。

4. 构建因果过滤器：区分相关遥测尖峰和真正的根因。在 20 场景标签上训练小型分类器。

5. 添加回滚试运行：在相同清单的预发集群上进行 ArgoCD 回滚。在 Slack 审批按钮之前，在现网集群中验证回滚计划。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|------------------------|
| K8s 知识图谱 | "集群图谱" | 节点 = K8s 对象 + 遥测序列；边 = 所有权、调度、观测 |
| 默认只读 | "受限 RBAC" | 代理的服务账号只有 get/list/describe 动词；破坏性动词位于另一服务器，受审批保护 |
| 审计日志 | "考虑过 vs 已执行" | 追加式记录每个候选命令、是否运行、谁批准 |
| 假设排序 | "证据评分" | 时效性 × 具体性 × 图谱路径长度倒数 × 引用数量 |
| Slack 审批卡片 | "HITL 门控" | 带修复按钮的交互式 Slack 消息；代理在人工点击前无法继续 |
| 遥测引用 | "证据指针" | 支持声明的 Prometheus 查询、Loki 选择器或 Tempo 链路 URL |
| MTTR | "解决时间" | 从告警触发到 SLO 恢复的墙钟时间 |

## 延伸阅读

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026 年权威参考
- [Resolve AI K8s 故障排查](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 竞品参考
- [NeuBird 语义监控](https://www.neubird.ai) — 语义图谱方法
- [Metoro AI SRE](https://metoro.io) — 以 SLO 为首的生产框架
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — 集群状态来源
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 参考代理编排器
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP 服务器框架
- [ArgoCD 回滚](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — 门控修复目标
