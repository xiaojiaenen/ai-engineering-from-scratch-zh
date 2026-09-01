# 安全性 — 密钥、API Key 轮转、审计日志、护栏

> 通过集中式密钥库（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除密钥泛滥。严禁将凭据存储在配置文件、VCS 中的 env 文件或电子表格中。使用 IAM 角色替代静态密钥；CI/CD 使用 OIDC。AI 网关模式是 2026 年的解决方案：应用 → 网关 → 模型提供商，网关在运行时从密钥库获取凭据。在密钥库中轮转密钥，所有应用在几分钟内即可生效——无需重新部署，无需 Slack 上"谁有新版 key"的询问消息。轮转策略 ≤90 天；每次提交时使用 TruffleHog / GitGuardian / Gitleaks 扫描。零信任：MFA、SSO、RBAC/ABAC、短生命周期令牌、设备合规性。PII 脱敏使用实体识别技术在转发前遮蔽 PHI/PII；一致的令牌化（Mesh 方法）将敏感值映射为稳定的占位符，使 LLM 保留代码/关系语义。网络出口：LLM 服务位于专用 VPC/VNet 子网，仅白名单允许 `api.openai.com`、`api.anthropic.com` 等；阻止其他所有出站流量。2026 年事故驱动因素：Vercel 供应链攻击，通过被利用的 CI/CD 凭据在数千个客户部署中泄露了环境变量。

**类型：** 学习
**语言：** Python（标准库、玩具版 PII 脱敏器 + 审计日志写入器）
**前置条件：** 第 17 阶段 · 19（AI 网关）、第 17 阶段 · 13（可观测性）
**时间：** 约 60 分钟

## 学习目标

- 列举四种密钥管理反模式（VCS 中的配置文件、硬编码环境变量、电子表格、静态密钥），并说明其替代方案。
- 解释 AI 网关从密钥库拉取凭据的模式作为 2026 生产标准。
- 实现具有稳定令牌化的 PII 脱敏器（相同值 → 相同占位符），以确保语义得以保留。
- 说出 2026 年 Vercel 供应链事件及其对 CI/CD 凭据卫生的启示。

## 问题所在

一名实习生提交了包含 API 密钥的 `.env` 文件。他很快删除了它。但密钥已经存在于 git 历史中——GitGuardian 扫描捕捉到了它，你的轮转流程是"在 Slack 上通知团队、更新 40 个配置文件、重新部署所有服务"。8 小时后，一半服务已上线，另一半正在等待部署窗口。

与此同时，用户提示词中包含"我的社保号是 123-45-6789"。提示词被发送到 OpenAI。你签署了 BAA，但内部政策要求在转发前遮蔽 PII。你没有这样做。

另外，你的 EKS 集群中的 LLM pod 可以访问任何互联网主机。有人通过 DNS 查询将数据泄露给攻击者控制的域名。没有任何拦截措施。

LLM 服务的安全性必须应对这三个向量。密钥库支持凭据、PII 脱敏、网络出口过滤、审计日志。

## 概念解析

### 集中式密钥库 + IAM 角色拉取

**密钥库**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。一个单一事实来源。

**IAM 角色**：应用/网关通过其 IAM 身份认证，而非静态密钥。密钥库在令牌有效期内返回密钥。

**AI 网关模式**：网关在请求时从密钥库拉取 `OPENAI_API_KEY`。在密钥库中轮转密钥；下次请求获取新密钥。无需重新部署。

### 轮转策略 ≤ 90 天

所有 API 密钥、密钥库根令牌、CI/CD 凭据。尽可能自动化轮转。手动轮转需记录并跟踪。

### 密钥扫描

- **TruffleHog** — 基于正则表达式 + 熵值检测提交中的密钥。
- **GitGuardian** — 商业产品，高准确率。
- **Gitleaks** — 开源，可在 CI 中运行。

每次提交时运行。若检测到新密钥，则阻止 PR。

### 零信任配置

- 所有账户要求 MFA。
- 通过 SAML/OIDC 实现 SSO。
- 基于角色的访问控制（RBAC）或基于属性的访问控制（ABAC）以实现细粒度访问。
- 短生命周期令牌（以小时计，而非天）。
- 设备合规性——仅限带有磁盘加密的公司设备。

### PII / PHI 脱敏

在提示词离开你的基础设施之前：

1. 实体识别（spaCy NER、Presidio、商业方案）。
2. 遮蔽匹配的实体：`"我的 SSN 是 123-45-6789"` → `"我的 SSN 是 [SSN_TOKEN_A3F]"`。
3. 一致的令牌化（Mesh 方法）：相同值映射到相同占位符，使 LLM 保留关系。
4. 可选的反向映射用于 LLM 响应。

静态正则过滤器可捕获基本模式；NER 可捕获更多。两者结合使用。

### 输入与输出护栏

输入：阻止已知越狱攻击、禁止主题；按用户限流。

输出：对泄露的密钥进行正则脱敏（API 密钥模式、拒绝上下文中的电子邮件模式），使用分类器检测政策违规。

### 网络出口白名单

LLM 服务位于专用子网：
- 白名单：`api.openai.com`、`api.anthropic.com`、向量数据库端点、密钥库端点。
- 其他所有内容：丢弃。
- DNS 通过仅允许列表的解析器（避免 DNS 隧道泄露）。

### 审计日志

每条 LLM 调用的不可变日志，包含：
- 时间戳。
- 用户/租户。
- 提示词哈希值（非原始提示词，以保护隐私）。
- 模型 + 版本。
- Token 数量。
- 成本。
- 响应哈希值。
- 任何护栏触发记录。

按监管要求保留（SOC 2 为 1 年，HIPAA 为 6 年）。

### 2026 年 Vercel 事件

供应链攻击：被利用的 CI/CD 凭据在数千个客户部署中泄露了环境变量。教训：CI/CD 凭据等同于生产环境凭据。存储在密钥库中。范围尽量缩小。积极轮转。

### 你需要记住的数据

- 轮转策略：≤ 90 天。
- 每次提交时扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026：CI/CD 凭据被泄露 → 数千个客户环境变量泄露。
- 审计日志保留：SOC 2 = 1 年，HIPAA = 6 年。

```figure
i4-vault-rotation
```

## 使用它

`code/main.py` 实现了一个带有稳定令牌化的玩具版 PII 脱敏器和仅追加审计日志。

## 交付它

本课程产出 `outputs/skill-llm-security-plan.md`。根据监管范围和当前状态，规划密钥库迁移、脱敏器、出口控制和审计日志。

## 练习

1. 运行 `code/main.py`。发送两条引用相同 SSN 的提示词。确认两者获得相同的占位符。
2. 为 vLLM-on-EKS 部署设计网络出口策略，该部署调用 OpenAI + Anthropic + Weaviate。
3. 你在 git 历史中发现了一个密钥（2 年前）。正确的响应是什么——轮转密钥、清理历史，还是两者都做？请说明理由。
4. 你的审计日志每天增长 10 GB。设计保留层级（热数据 30 天、温数据 12 个月、冷数据 6 年）。
5. 论证反向令牌化（将真实值替换回 LLM 响应）是否值得，与保持占位符可见相比。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Vault | "密钥存储" | 集中式凭据管理服务 |
| IAM 角色 | "基于身份的认证" | 应用扮演的角色；返回短生命周期凭据 |
| OIDC for CI/CD | "云颁发的令牌" | CI 中无静态密钥——通过 OIDC 进行身份验证 |
| TruffleHog / GitGuardian / Gitleaks | "密钥扫描器" | 提交时的密钥检测 |
| RBAC / ABAC | "访问控制" | 基于角色 vs 基于属性 |
| PII 脱敏 | "数据遮蔽" | 移除或令牌化敏感实体 |
| 一致的令牌化 | "稳定占位符" | 相同值 → 每次相同的令牌 |
| Mesh 方法 | "Mesh 令牌化" | 保留语义的令牌化模式 |
| 出口白名单 | "出站允许列表" | 仅允许访问的域名可达 |
| 审计日志 | "不可变历史" | 用于合规的仅追加记录 |

## 延伸阅读

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII 检测和匿名化。
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)
