# 缓存、速率限制与成本优化

> 多数AI初创公司并非死于糟糕的模型，而是死于糟糕的单位经济性。单次GPT-4o调用的成本仅为零点几美分，但当一万名用户每天发起十次调用时，仅输入token就会消耗250美元——在你收取任何费用之前。能存活下来的公司是那些将每次API调用视为金融交易而非函数调用的企业。

**类型：** 实战构建
**语言：** Python
**前置课程：** 第11阶段课程09（函数调用）
**时间：** 约45分钟
**相关课程：** 第11阶段 · 15（提示缓存）——本课程涵盖应用层缓存（语义缓存、精确哈希缓存、模型路由），课程15涵盖提供商层提示缓存（Anthropic的cache_control、OpenAI自动缓存、Gemini CachedContent）。结合两者可降低50-95%成本。

## 学习目标

- 实现语义缓存，从缓存中处理重复或相似查询，而非发起新API调用
- 跨提供商计算单次请求成本，并实现令牌感知的速率限制与预算告警
- 构建包含提示压缩、模型路由（昂贵模型与廉价模型）和响应缓存的成本优化层
- 设计分层缓存策略，针对不同查询类型采用精确匹配、语义相似度与前缀缓存

## 问题所在

你构建了一个RAG聊天机器人，效果出众，深受用户喜爱。

然后账单来了。

GPT-5的输入token价格为每百万5美元，输出token每百万15美元。Claude Opus 4.7的输入价格为15美元/输出75美元。Gemini 3 Pro输入价格1.25美元/输出5美元。GPT-5-mini则为0.25美元/2美元。以下价格仅为示例，请务必查阅提供商的最新定价页面。

以下计算足以拖垮初创公司：

- 每日活跃用户10,000人
- 每用户每日查询10次
- 每次查询1,000个输入token（系统提示+上下文+用户消息）
- 每次响应500个输出token

**每日输入成本：** 10,000 × 10 × 1,000 / 1,000,000 × $2.50 = **每日$250**
**每日输出成本：** 10,000 × 10 × 500 / 1,000,000 × $10.00 = **每日$500**
**每月总计：** **每月$22,500**

这还只是LLM的成本。加上嵌入、向量数据库托管、基础设施，一个聊天机器人的月成本将高达30,000美元。

残酷的是：40-60%的查询都是近似重复的。用户会用略微不同的措辞询问相同问题。你的系统提示——在每个请求中完全相同——每次都会被计费。由RAG检索的上下文文档会在询问相同主题的用户间重复出现。

你在为冗余计算支付全价。

## 核心概念

### LLM调用的成本剖析

每次API调用包含五个成本组成部分。

```mermaid
graph LR
    A[User Query] --> B[System Prompt<br/>500-2000 tokens]
    A --> C[Retrieved Context<br/>500-4000 tokens]
    A --> D[User Message<br/>50-500 tokens]
    B --> E[Input Cost<br/>$2.50/1M tokens]
    C --> E
    D --> E
    E --> F[Model Processing]
    F --> G[Output Cost<br/>$10.00/1M tokens]
```

系统提示是隐形的杀手。一个1,500个token的系统提示随每个请求发送，仅此前缀部分每百万请求成本就达3.75美元。按每日10万请求计算，这意味着每日375美元——每月11,250美元——用于永不变化的文本。

### 提供商缓存：内置折扣

2026年，三大主要提供商都提供服务端提示缓存，但机制各异。详细内容请参见第11阶段·15。

| 提供商 | 机制 | 折扣 | 最低要求 | 缓存时长 |
|----------|-----------|----------|---------|----------------|
| Anthropic | 显式cache_control标记 | 缓存命中时90%折扣（写入时额外支付25%） | 1,024 tokens (Sonnet/Opus), 2,048 (Haiku) | 默认5分钟；延长至1小时（写入费用翻倍） |
| OpenAI | 自动前缀匹配 | 缓存命中时50%折扣 | 1,024 tokens | 最佳保证最长1小时 |
| Google Gemini | 显式CachedContent API | 约75%减免（加存储费用） | 4,096 (Flash) / 32,768 (Pro) | 用户可配置TTL |

**Anthropic的方式**是显式的。你用`cache_control: {"type": "ephemeral"}`标记提示的部分内容。首次请求支付25%写入溢价。后续具有相同前缀的请求获得90%折扣。一个通常花费0.005美元的2,000个token系统提示，在缓存命中时仅需0.000625美元。超过10万请求，每日可节省437.50美元。

**OpenAI的方式**是自动的。任何与先前请求匹配的提示前缀都能获得50%折扣。无需标记。权衡是：折扣更少，控制力更弱，但无需任何实施工作。

### 语义缓存：你的自定义层

提供商缓存仅适用于相同前缀。语义缓存处理更复杂的情况：具有相同含义的不同查询。

"What is the return policy?"和"How do I return an item?"是不同的字符串，但意图相同。语义缓存会嵌入两个查询，计算余弦相似度，如果相似度超过阈值（通常为0.92-0.95）则返回缓存响应。

```mermaid
flowchart TD
    A[User Query] --> B[Embed Query]
    B --> C{Similar query<br/>in cache?}
    C -->|sim > 0.95| D[Return Cached Response]
    C -->|sim < 0.95| E[Call LLM API]
    E --> F[Cache Response<br/>with Embedding]
    F --> G[Return Response]
    D --> G
```

嵌入成本可忽略不计。OpenAI的text-embedding-3-small每百万token仅需0.02美元。与完整的LLM调用相比，检查缓存的成本几乎为零。

### 精确缓存：哈希与匹配

对于确定性调用（temperature=0，相同模型，相同提示），精确缓存更简单快速。对完整提示进行哈希处理，检查缓存，若找到则返回。

这完美适用于：
- 系统提示+固定上下文+完全相同的用户查询
- 具有相同工具定义的函数调用
- 同一文档需要多次处理的批处理

### 速率限制：保护你的预算

速率限制不仅关乎公平，更关乎生存。

**令牌桶算法：** 每个用户拥有一个容量为N个令牌的桶，以每秒R个令牌的速率补充。请求消耗桶中的令牌。若桶为空则拒绝请求。这允许突发流量（一次性使用整个桶），同时强制执行平均速率。

**每用户配额：** 根据用户层级设置每日/每月令牌限制。

| 层级 | 每日令牌限制 | 最大请求数/分钟 | 模型访问权限 |
|------|------------------|------------------|-------------|
| 免费 | 50,000 | 10 | 仅限GPT-4o-mini |
| 专业版 | 500,000 | 60 | GPT-4o、Claude Sonnet |
| 企业版 | 5,000,000 | 300 | 所有模型 |

### 模型路由：为合适的任务选择合适的模型

并非每个查询都需要GPT-4o。

"What time does the store close?"不需要使用每百万输出token收费10美元的模型。GPT-4o-mini每百万输出token0.60美元即可完美处理。Claude Haiku每百万输出token1.25美元也能胜任。一个简单的分类器可将廉价查询路由至廉价模型，复杂查询路由至昂贵模型。

```mermaid
flowchart TD
    A[User Query] --> B[Complexity Classifier]
    B -->|Simple: lookup, FAQ| C[GPT-4o-mini<br/>$0.15/$0.60 per 1M]
    B -->|Medium: analysis, summary| D[Claude Sonnet<br/>$3.00/$15.00 per 1M]
    B -->|Complex: reasoning, code| E[GPT-4o / Claude Opus<br/>$2.50/$10.00+]
```

一个调优良好的路由器仅在模型成本上就能节省40-70%。

### 成本追踪：了解资金去向

你无法优化无法测量的东西。记录每次API调用的以下信息：

- 时间戳
- 模型名称
- 输入token数
- 输出token数
- 延迟（毫秒）
- 计算成本（$）
- 用户ID
- 缓存命中/未命中
- 请求类别

这些数据能揭示哪些功能成本高昂、哪些用户消耗资源最多、缓存在哪里产生最大影响。

### 批处理：批量折扣

OpenAI的批处理API以50%的折扣异步处理请求。你提交一批最多50,000个请求，结果将在24小时内返回。

适用于：
- 每夜文档处理
- 批量分类
- 评估运行
- 数据增强管道

不适用于：面向用户的实时查询（延迟至关重要）。

### 预算告警与断路器

当达到限制时，断路器会停止支出。没有它，一个错误或滥用可能在几小时内耗尽你的月度预算。

设置三个阈值：
1. **警告**（预算的70%）：发送告警
2. **限流**（预算的85%）：仅切换至更廉价的模型
3. **停止**（预算的95%）：拒绝新请求，仅返回缓存响应

### 优化技术栈

按顺序应用这些技术。每一层都建立在前一层的基础上。

| 层级 | 技术 | 典型节省 | 实施难度 |
|-------|-----------|----------------|----------------------|
| 1 | 提供商提示缓存 | 30-50% | 低（添加缓存标记） |
| 2 | 精确缓存 | 10-20% | 低（哈希+字典） |
| 3 | 语义缓存 | 15-30% | 中等（嵌入+相似度） |
| 4 | 模型路由 | 40-70% | 中等（分类器） |
| 5 | 速率限制 | 预算保护 | 低（令牌桶） |
| 6 | 提示压缩 | 10-30% | 中等（重写提示） |
| 7 | 批处理 | 合格项50% | 低（批处理API） |

应用1-5层的RAG应用通常将成本从每月22,500美元降至4,000-6,000美元。这正是烧钱与构建可持续业务之间的区别。

### 实际节省：优化前后对比

以下是服务于10,000日活用户的RAG聊天机器人的实际成本分解。

| 指标 | 优化前 | 优化后 | 节省 |
|--------|--------------------|--------------------|---------|
| 每月LLM成本 | $22,500 | $5,200 | 77% |
| 每次查询平均成本 | $0.0075 | $0.0017 | 77% |
| 缓存命中率 | 0% | 52% | -- |
| 路由至mini模型的查询比例 | 0% | 65% | -- |
| P95延迟 | 2,800ms | 900ms（缓存命中：50ms） | 68% |
| 每月嵌入成本 | $0 | $180 | （新增成本） |
| 每月总成本 | $22,500 | $5,380 | 76% |

语义缓存的嵌入成本（每月180美元）在首次缓存命中时就开始产生回报。

## 构建实施

### 第一步：成本计算器

构建一个了解主要模型当前定价的令牌成本计算器。

```python
import hashlib
import time
import json
import math
from dataclasses import dataclass, field


MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
    "gpt-4.1": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cached_input": 0.10},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cached_input": 0.025},
    "o3": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
    "o3-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.55},
    "o4-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.275},
    "claude-opus-4": {"input": 15.00, "output": 75.00, "cached_input": 1.50},
    "claude-sonnet-4": {"input": 3.00, "output": 15.00, "cached_input": 0.30},
    "claude-haiku-3.5": {"input": 0.80, "output": 4.00, "cached_input": 0.08},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00, "cached_input": 0.3125},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60, "cached_input": 0.0375},
}


def calculate_cost(model, input_tokens, output_tokens, cached_input_tokens=0):
    if model not in MODEL_PRICING:
        return {"error": f"Unknown model: {model}"}
    pricing = MODEL_PRICING[model]
    non_cached = input_tokens - cached_input_tokens
    input_cost = (non_cached / 1_000_000) * pricing["input"]
    cached_cost = (cached_input_tokens / 1_000_000) * pricing["cached_input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total = input_cost + cached_cost + output_cost
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "input_cost": round(input_cost, 6),
        "cached_input_cost": round(cached_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total, 6),
    }
```

### 第二步：精确缓存

对完整提示进行哈希处理，为完全相同的请求返回缓存响应。

```python
class ExactCache:
    def __init__(self, max_size=1000, ttl_seconds=3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _hash(self, model, messages, temperature):
        key_data = json.dumps({"model": model, "messages": messages, "temperature": temperature}, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, model, messages, temperature=0.0):
        if temperature > 0:
            self.misses += 1
            return None
        key = self._hash(model, messages, temperature)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                self.hits += 1
                entry["access_count"] += 1
                return entry["response"]
            del self.cache[key]
        self.misses += 1
        return None

    def put(self, model, messages, temperature, response):
        if temperature > 0:
            return
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache, key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
        key = self._hash(model, messages, temperature)
        self.cache[key] = {
            "response": response,
            "timestamp": time.time(),
            "access_count": 1,
        }

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0,
            "cache_size": len(self.cache),
        }
```

### 第三步：语义缓存

嵌入查询，当相似度超过阈值时返回缓存响应。

```python
def simple_embed(text):
    words = text.lower().split()
    vocab = {}
    for w in words:
        vocab[w] = vocab.get(w, 0) + 1
    norm = math.sqrt(sum(v * v for v in vocab.values()))
    if norm == 0:
        return {}
    return {k: v / norm for k, v in vocab.items()}


def cosine_similarity(a, b):
    if not a or not b:
        return 0.0
    all_keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in all_keys)
    return dot


class SemanticCache:
    def __init__(self, similarity_threshold=0.85, max_size=500, ttl_seconds=3600):
        self.entries = []
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def get(self, query):
        query_embedding = simple_embed(query)
        now = time.time()
        best_match = None
        best_sim = 0.0
        for entry in self.entries:
            if now - entry["timestamp"] > self.ttl:
                continue
            sim = cosine_similarity(query_embedding, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_match = entry
        if best_match and best_sim >= self.threshold:
            self.hits += 1
            best_match["access_count"] += 1
            return {"response": best_match["response"], "similarity": round(best_sim, 4), "original_query": best_match["query"]}
        self.misses += 1
        return None

    def put(self, query, response):
        if len(self.entries) >= self.max_size:
            self.entries.sort(key=lambda e: e["timestamp"])
            self.entries.pop(0)
        self.entries.append({
            "query": query,
            "embedding": simple_embed(query),
            "response": response,
            "timestamp": time.time(),
            "access_count": 1,
        })

    def stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total > 0 else 0,
            "cache_size": len(self.entries),
        }
```

### 第四步：速率限制器

带每用户配额的令牌桶速率限制器。

```python
class TokenBucketRateLimiter:
    def __init__(self):
        self.buckets = {}
        self.tiers = {
            "free": {"capacity": 50_000, "refill_rate": 500, "max_requests_per_min": 10},
            "pro": {"capacity": 500_000, "refill_rate": 5_000, "max_requests_per_min": 60},
            "enterprise": {"capacity": 5_000_000, "refill_rate": 50_000, "max_requests_per_min": 300},
        }

    def _get_bucket(self, user_id, tier="free"):
        if user_id not in self.buckets:
            tier_config = self.tiers.get(tier, self.tiers["free"])
            self.buckets[user_id] = {
                "tokens": tier_config["capacity"],
                "capacity": tier_config["capacity"],
                "refill_rate": tier_config["refill_rate"],
                "last_refill": time.time(),
                "request_timestamps": [],
                "max_rpm": tier_config["max_requests_per_min"],
                "tier": tier,
                "total_tokens_used": 0,
            }
        return self.buckets[user_id]

    def _refill(self, bucket):
        now = time.time()
        elapsed = now - bucket["last_refill"]
        refill = int(elapsed * bucket["refill_rate"])
        if refill > 0:
            bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + refill)
            bucket["last_refill"] = now

    def check(self, user_id, tokens_needed, tier="free"):
        bucket = self._get_bucket(user_id, tier)
        self._refill(bucket)
        now = time.time()
        bucket["request_timestamps"] = [t for t in bucket["request_timestamps"] if now - t < 60]
        if len(bucket["request_timestamps"]) >= bucket["max_rpm"]:
            return {"allowed": False, "reason": "rate_limit", "retry_after_seconds": 60 - (now - bucket["request_timestamps"][0])}
        if bucket["tokens"] < tokens_needed:
            deficit = tokens_needed - bucket["tokens"]
            wait = deficit / bucket["refill_rate"]
            return {"allowed": False, "reason": "token_limit", "tokens_available": bucket["tokens"], "retry_after_seconds": round(wait, 1)}
        return {"allowed": True, "tokens_available": bucket["tokens"]}

    def consume(self, user_id, tokens_used, tier="free"):
        bucket = self._get_bucket(user_id, tier)
        bucket["tokens"] -= tokens_used
        bucket["request_timestamps"].append(time.time())
        bucket["total_tokens_used"] += tokens_used

    def get_usage(self, user_id):
        if user_id not in self.buckets:
            return {"error": "User not found"}
        b = self.buckets[user_id]
        return {
            "user_id": user_id,
            "tier": b["tier"],
            "tokens_remaining": b["tokens"],
            "capacity": b["capacity"],
            "total_tokens_used": b["total_tokens_used"],
            "utilization": round(b["total_tokens_used"] / b["capacity"], 4) if b["capacity"] else 0,
        }
```

### 第五步：成本追踪器

记录每次调用并计算累计总量。

```python
class CostTracker:
    def __init__(self, monthly_budget=1000.0):
        self.logs = []
        self.monthly_budget = monthly_budget
        self.alerts = []

    def log_call(self, model, input_tokens, output_tokens, cached_input_tokens=0, latency_ms=0, user_id="anonymous", cache_status="miss"):
        cost = calculate_cost(model, input_tokens, output_tokens, cached_input_tokens)
        entry = {
            "timestamp": time.time(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_input_tokens": cached_input_tokens,
            "latency_ms": latency_ms,
            "cost": cost["total_cost"],
            "user_id": user_id,
            "cache_status": cache_status,
        }
        self.logs.append(entry)
        self._check_budget()
        return entry

    def _check_budget(self):
        total = self.total_cost()
        pct = total / self.monthly_budget if self.monthly_budget > 0 else 0
        if pct >= 0.95 and not any(a["level"] == "stop" for a in self.alerts):
            self.alerts.append({"level": "stop", "message": f"Budget 95% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})
        elif pct >= 0.85 and not any(a["level"] == "throttle" for a in self.alerts):
            self.alerts.append({"level": "throttle", "message": f"Budget 85% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})
        elif pct >= 0.70 and not any(a["level"] == "warning" for a in self.alerts):
            self.alerts.append({"level": "warning", "message": f"Budget 70% consumed: ${total:.2f}/${self.monthly_budget:.2f}", "timestamp": time.time()})

    def total_cost(self):
        return round(sum(e["cost"] for e in self.logs), 6)

    def cost_by_model(self):
        by_model = {}
        for e in self.logs:
            m = e["model"]
            if m not in by_model:
                by_model[m] = {"calls": 0, "cost": 0, "input_tokens": 0, "output_tokens": 0}
            by_model[m]["calls"] += 1
            by_model[m]["cost"] = round(by_model[m]["cost"] + e["cost"], 6)
            by_model[m]["input_tokens"] += e["input_tokens"]
            by_model[m]["output_tokens"] += e["output_tokens"]
        return by_model

    def cache_savings(self):
        cache_hits = [e for e in self.logs if e["cache_status"] == "hit"]
        if not cache_hits:
            return {"saved": 0, "cache_hits": 0}
        saved = 0
        for e in cache_hits:
            full_cost = calculate_cost(e["model"], e["input_tokens"], e["output_tokens"])
            saved += full_cost["total_cost"]
        return {"saved": round(saved, 4), "cache_hits": len(cache_hits)}

    def summary(self):
        if not self.logs:
            return {"total_calls": 0, "total_cost": 0}
        total_latency = sum(e["latency_ms"] for e in self.logs)
        cache_hits = sum(1 for e in self.logs if e["cache_status"] == "hit")
        return {
            "total_calls": len(self.logs),
            "total_cost": self.total_cost(),
            "avg_cost_per_call": round(self.total_cost() / len(self.logs), 6),
            "avg_latency_ms": round(total_latency / len(self.logs), 1),
            "cache_hit_rate": round(cache_hits / len(self.logs), 4),
            "cost_by_model": self.cost_by_model(),
            "cache_savings": self.cache_savings(),
            "budget_remaining": round(self.monthly_budget - self.total_cost(), 2),
            "budget_utilization": round(self.total_cost() / self.monthly_budget, 4) if self.monthly_budget > 0 else 0,
            "alerts": self.alerts,
        }
```

### 第六步：模型路由器

将查询路由到能够处理它们的最廉价模型。

```python
SIMPLE_KEYWORDS = ["what time", "hours", "address", "phone", "price", "return policy", "hello", "hi", "thanks", "yes", "no"]
COMPLEX_KEYWORDS = ["analyze", "compare", "explain why", "write code", "debug", "architect", "design", "trade-off", "evaluate"]


def classify_complexity(query):
    q = query.lower()
    if len(q.split()) <= 5 or any(kw in q for kw in SIMPLE_KEYWORDS):
        return "simple"
    if any(kw in q for kw in COMPLEX_KEYWORDS):
        return "complex"
    return "medium"


def route_model(query, tier="pro"):
    complexity = classify_complexity(query)
    routing_table = {
        "simple": {"free": "gpt-4.1-nano", "pro": "gpt-4o-mini", "enterprise": "gpt-4o-mini"},
        "medium": {"free": "gpt-4o-mini", "pro": "claude-sonnet-4", "enterprise": "claude-sonnet-4"},
        "complex": {"free": "gpt-4o-mini", "pro": "gpt-4o", "enterprise": "claude-opus-4"},
    }
    model = routing_table[complexity].get(tier, "gpt-4o-mini")
    return {"query": query, "complexity": complexity, "model": model, "tier": tier}
```

### 第七步：运行演示

```python
def simulate_llm_call(model, query):
    input_tokens = len(query.split()) * 4 + 500
    output_tokens = 150 + (len(query.split()) * 2)
    latency = 200 + (output_tokens * 2)
    return {
        "model": model,
        "response": f"[Simulated {model} response to: {query[:50]}...]",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency,
    }


def run_demo():
    print("=" * 60)
    print("  Caching, Rate Limiting & Cost Optimization Demo")
    print("=" * 60)

    print("\n--- Model Pricing ---")
    for model, pricing in list(MODEL_PRICING.items())[:6]:
        cost_1k = calculate_cost(model, 1000, 500)
        print(f"  {model}: ${cost_1k['total_cost']:.6f} per 1K in + 500 out")

    print("\n--- Cost Comparison: 100K Requests ---")
    for model in ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4", "claude-haiku-3.5"]:
        cost = calculate_cost(model, 1000 * 100_000, 500 * 100_000)
        print(f"  {model}: ${cost['total_cost']:.2f}")

    print("\n--- Anthropic Cache Savings ---")
    no_cache = calculate_cost("claude-sonnet-4", 2000, 500, 0)
    with_cache = calculate_cost("claude-sonnet-4", 2000, 500, 1500)
    saving = no_cache["total_cost"] - with_cache["total_cost"]
    print(f"  Without cache: ${no_cache['total_cost']:.6f}")
    print(f"  With 1500 cached tokens: ${with_cache['total_cost']:.6f}")
    print(f"  Savings per call: ${saving:.6f} ({saving/no_cache['total_cost']*100:.1f}%)")

    exact_cache = ExactCache(max_size=100, ttl_seconds=300)
    semantic_cache = SemanticCache(similarity_threshold=0.75, max_size=100)
    rate_limiter = TokenBucketRateLimiter()
    tracker = CostTracker(monthly_budget=100.0)

    print("\n--- Exact Cache ---")
    messages_1 = [{"role": "user", "content": "What is the return policy?"}]
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.0)
    print(f"  First lookup: {'HIT' if result else 'MISS'}")
    exact_cache.put("gpt-4o-mini", messages_1, 0.0, "You can return items within 30 days.")
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.0)
    print(f"  Second lookup: {'HIT' if result else 'MISS'} -> {result}")
    result = exact_cache.get("gpt-4o-mini", messages_1, 0.7)
    print(f"  With temp=0.7: {'HIT' if result else 'MISS (non-deterministic, skip cache)'}")
    print(f"  Stats: {exact_cache.stats()}")

    print("\n--- Semantic Cache ---")
    test_queries = [
        ("What is the return policy?", "Items can be returned within 30 days with receipt."),
        ("How do I return an item?", None),
        ("What are your store hours?", "We are open 9am-9pm Monday through Saturday."),
        ("When does the store open?", None),
        ("Tell me about quantum computing", "Quantum computers use qubits..."),
        ("Explain quantum mechanics", None),
    ]
    for query, response in test_queries:
        cached = semantic_cache.get(query)
        if cached:
            print(f"  '{query[:40]}' -> CACHE HIT (sim={cached['similarity']}, original='{cached['original_query'][:40]}')")
        elif response:
            semantic_cache.put(query, response)
            print(f"  '{query[:40]}' -> MISS (stored)")
        else:
            print(f"  '{query[:40]}' -> MISS (no match)")
    print(f"  Stats: {semantic_cache.stats()}")

    print("\n--- Rate Limiting ---")
    for i in range(12):
        check = rate_limiter.check("user_1", 1000, "free")
        if check["allowed"]:
            rate_limiter.consume("user_1", 1000, "free")
        status = "OK" if check["allowed"] else f"BLOCKED ({check['reason']})"
        if i < 5 or not check["allowed"]:
            print(f"  Request {i+1}: {status}")
    print(f"  Usage: {rate_limiter.get_usage('user_1')}")

    print("\n--- Model Routing ---")
    routing_queries = [
        "What time do you close?",
        "Summarize this quarterly earnings report",
        "Analyze the trade-offs between microservices and monoliths",
        "Hello",
        "Write code for a binary search tree with deletion",
    ]
    for q in routing_queries:
        route = route_model(q, "pro")
        print(f"  '{q[:50]}' -> {route['model']} ({route['complexity']})")

    print("\n--- Full Pipeline: Before vs After Optimization ---")
    queries = [
        "What is the return policy?",
        "How do I return something?",
        "What are your hours?",
        "When do you open?",
        "Explain the difference between TCP and UDP",
        "Compare TCP vs UDP protocols",
        "Hello",
        "What is your phone number?",
        "Write a Python function to sort a list",
        "Analyze the pros and cons of serverless architecture",
    ]

    print("\n  [Before: no caching, single model (gpt-4o)]")
    tracker_before = CostTracker(monthly_budget=1000.0)
    for q in queries:
        result = simulate_llm_call("gpt-4o", q)
        tracker_before.log_call("gpt-4o", result["input_tokens"], result["output_tokens"], latency_ms=result["latency_ms"], cache_status="miss")
    before = tracker_before.summary()
    print(f"  Total cost: ${before['total_cost']:.6f}")
    print(f"  Avg cost/call: ${before['avg_cost_per_call']:.6f}")
    print(f"  Avg latency: {before['avg_latency_ms']}ms")

    print("\n  [After: caching + routing + rate limiting]")
    exact_c = ExactCache()
    semantic_c = SemanticCache(similarity_threshold=0.75)
    tracker_after = CostTracker(monthly_budget=1000.0)

    for q in queries:
        messages = [{"role": "user", "content": q}]
        cached = exact_c.get("gpt-4o", messages, 0.0)
        if cached:
            tracker_after.log_call("gpt-4o-mini", 0, 0, latency_ms=5, cache_status="hit")
            continue
        sem_cached = semantic_c.get(q)
        if sem_cached:
            tracker_after.log_call("gpt-4o-mini", 0, 0, latency_ms=15, cache_status="hit")
            continue
        route = route_model(q)
        result = simulate_llm_call(route["model"], q)
        tracker_after.log_call(route["model"], result["input_tokens"], result["output_tokens"], latency_ms=result["latency_ms"], cache_status="miss")
        exact_c.put(route["model"], messages, 0.0, result["response"])
        semantic_c.put(q, result["response"])

    after = tracker_after.summary()
    print(f"  Total cost: ${after['total_cost']:.6f}")
    print(f"  Avg cost/call: ${after['avg_cost_per_call']:.6f}")
    print(f"  Avg latency: {after['avg_latency_ms']}ms")
    print(f"  Cache hit rate: {after['cache_hit_rate']:.0%}")

    if before["total_cost"] > 0:
        savings_pct = (1 - after["total_cost"] / before["total_cost"]) * 100
        print(f"\n  SAVINGS: {savings_pct:.1f}% cost reduction")
        print(f"  Latency improvement: {(1 - after['avg_latency_ms'] / before['avg_latency_ms']) * 100:.1f}% faster")

    print("\n--- Budget Alerts Demo ---")
    alert_tracker = CostTracker(monthly_budget=0.01)
    for i in range(5):
        alert_tracker.log_call("gpt-4o", 5000, 2000, latency_ms=500)
    print(f"  Total spent: ${alert_tracker.total_cost():.6f} / ${alert_tracker.monthly_budget}")
    for alert in alert_tracker.alerts:
        print(f"  ALERT [{alert['level'].upper()}]: {alert['message']}")

    print("\n--- Cost Breakdown by Model ---")
    multi_tracker = CostTracker(monthly_budget=500.0)
    for _ in range(50):
        multi_tracker.log_call("gpt-4o-mini", 800, 200, latency_ms=150)
    for _ in range(30):
        multi_tracker.log_call("claude-sonnet-4", 1500, 500, latency_ms=400)
    for _ in range(10):
        multi_tracker.log_call("gpt-4o", 2000, 800, latency_ms=600)
    for _ in range(10):
        multi_tracker.log_call("claude-opus-4", 3000, 1000, latency_ms=1200)
    breakdown = multi_tracker.cost_by_model()
    for model, data in sorted(breakdown.items(), key=lambda x: x[1]["cost"], reverse=True):
        print(f"  {model}: {data['calls']} calls, ${data['cost']:.6f}, {data['input_tokens']:,} in / {data['output_tokens']:,} out")
    print(f"  Total: ${multi_tracker.total_cost():.6f}")

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
```

## 使用指南

### Anthropic提示缓存

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     system=[
#         {
#             "type": "text",
#             "text": "You are a helpful customer support agent for Acme Corp...",
#             "cache_control": {"type": "ephemeral"},
#         }
#     ],
#     messages=[{"role": "user", "content": "What is the return policy?"}],
# )
#
# print(f"Input tokens: {response.usage.input_tokens}")
# print(f"Cache creation tokens: {response.usage.cache_creation_input_tokens}")
# print(f"Cache read tokens: {response.usage.cache_read_input_tokens}")
```

首次调用写入缓存（25%溢价）。后续所有具有相同系统提示前缀的调用从缓存读取（90%折扣）。缓存持续5分钟，每次命中时重置计时器。

### OpenAI自动缓存

```python
# from openai import OpenAI
#
# client = OpenAI()
#
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "system", "content": "You are a helpful customer support agent..."},
#         {"role": "user", "content": "What is the return policy?"},
#     ],
# )
#
# print(f"Prompt tokens: {response.usage.prompt_tokens}")
# print(f"Cached tokens: {response.usage.prompt_tokens_details.cached_tokens}")
# print(f"Completion tokens: {response.usage.completion_tokens}")
```

OpenAI自动缓存。任何超过1,024个token且与近期请求匹配的提示前缀都能获得50%折扣。无需代码更改——只需在响应中检查`prompt_tokens_details.cached_tokens`以验证是否正常工作。

### OpenAI批处理API

```python
# import json
# from openai import OpenAI
#
# client = OpenAI()
#
# requests = []
# for i, query in enumerate(queries):
#     requests.append({
#         "custom_id": f"request-{i}",
#         "method": "POST",
#         "url": "/v1/chat/completions",
#         "body": {
#             "model": "gpt-4o-mini",
#             "messages": [{"role": "user", "content": query}],
#         },
#     })
#
# with open("batch_input.jsonl", "w") as f:
#     for r in requests:
#         f.write(json.dumps(r) + "\n")
#
# batch_file = client.files.create(file=open("batch_input.jsonl", "rb"), purpose="batch")
# batch = client.batches.create(input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h")
# print(f"Batch ID: {batch.id}, Status: {batch.status}")
```

批处理API对所有token提供统一50%折扣。结果在24小时内送达。非常适合非实时工作负载：评估、数据标注、批量摘要。

### 使用Redis的生产环境语义缓存

```python
# import redis
# import numpy as np
# from openai import OpenAI
#
# r = redis.Redis()
# client = OpenAI()
#
# def get_embedding(text):
#     response = client.embeddings.create(model="text-embedding-3-small", input=text)
#     return response.data[0].embedding
#
# def semantic_cache_lookup(query, threshold=0.95):
#     query_emb = np.array(get_embedding(query))
#     keys = r.keys("cache:emb:*")
#     best_sim, best_key = 0, None
#     for key in keys:
#         stored_emb = np.frombuffer(r.get(key), dtype=np.float32)
#         sim = np.dot(query_emb, stored_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(stored_emb))
#         if sim > best_sim:
#             best_sim, best_key = sim, key
#     if best_sim >= threshold and best_key:
#         response_key = best_key.decode().replace("cache:emb:", "cache:resp:")
#         return r.get(response_key).decode()
#     return None
```

在生产环境中，用向量索引（Redis向量搜索、Pinecone或pgvector）替代线性扫描。线性扫描适用于少于1,000个条目的情况。超过此数量，使用ANN（近似最近邻）实现O(log n)查找。

## 交付成果

本课程产出`outputs/prompt-cost-optimizer.md`——一个可重用的提示，用于分析你的LLM应用并推荐具体的成本优化措施及预计节省。

同时产出`outputs/skill-cost-patterns.md`——一个决策框架，用于为你的用例选择合适的缓存策略、速率限制配置和模型路由规则。

## 练习

1. **为语义缓存实现LRU淘汰策略。** 用最近最少使用策略替代先进先出策略。跟踪每个条目的最后访问时间，当缓存已满时淘汰访问时间最旧的条目。比较两种策略在100次查询中的命中率。

2. **构建成本预测工具。** 给定API调用日志（来自CostTracker的日志），基于最近7天的平均值预测月度成本。考虑工作日/周末模式。如果预测月度成本超过预算20%以上，则触发告警。

3. **实现分层语义缓存。** 使用两个相似度阈值：0.98用于高置信度命中（立即返回），0.90用于中等置信度命中（返回时附带说明："基于类似的历史问题..."）。跟踪每个命中来自哪个层级，并测量用户满意度的差异。

4. **构建模型路由分类器。** 用基于嵌入的分类器替代基于关键词的分类器。嵌入50个已标注查询（简单/中等/复杂），然后通过寻找最近的已标注示例对新查询进行分类。在包含20个查询的测试集上测量分类准确率。

5. **实现带降级层级的断路器。** 在预算使用率达70%时记录警告。85%时自动将所有路由切换至最廉价模型（gpt-4o-mini）。95%时仅提供缓存响应并拒绝新查询。通过模拟1,000个请求对1.00美元预算进行测试，验证每个阈值是否正确触发。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 提示缓存 | “缓存系统提示” | 提供商级别的缓存，重复的提示前缀可获得折扣（Anthropic 90%、OpenAI 50%）——OpenAI无需代码更改，Anthropic需显式标记 |
| 语义缓存 | “智能缓存” | 嵌入查询，计算与历史查询的相似度，若超过阈值则返回缓存响应——能捕获精确匹配遗漏的改写表述 |
| 精确缓存 | “哈希缓存” | 对完整提示（模型+消息+temperature）进行哈希处理，为完全相同的输入返回缓存响应——仅适用于temperature=0的确定性调用 |
| 令牌桶 | “速率限制器” | 一种算法，每个用户拥有一个容量为N个令牌的桶，以每秒R个令牌的速率补充——允许突发流量（最多N个），同时强制执行平均速率R |
| 模型路由 | “精打细算路由” | 使用分类器将简单查询发送至廉价模型（GPT-4o-mini、Haiku），复杂查询发送至昂贵模型（GPT-4o、Opus）——在模型成本上节省40-70% |
| 成本追踪 | “计量” | 记录每次API调用的模型、token数、延迟、成本和用户ID，以便确切了解资金去向和哪些功能成本高昂 |
| 断路器 | “紧急停止开关” | 当支出接近预算限制时，自动降级服务（更廉价模型、仅缓存响应）或完全停止请求 |
| 批处理API | “批量折扣” | OpenAI的异步处理服务，享受50%折扣——最多提交50,000个请求，24小时内获得结果 |
| 提示压缩 | “token瘦身” | 重写系统提示和上下文以使用更少token，同时保留含义——更短的提示成本更低且通常性能更好 |
| 缓存命中率 | “缓存效率” | 从缓存提供服务而非调用LLM的请求百分比——生产环境聊天机器人通常为40-60%，按比例节省成本 |

## 扩展阅读

- [Anthropic提示缓存指南](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) —— Anthropic显式cache_control标记、定价和缓存生命周期行为的官方文档
- [OpenAI提示缓存](https://platform.openai.com/docs/guides/prompt-caching) —— OpenAI的自动缓存、如何通过使用字段验证缓存命中、最低前缀长度
- [OpenAI批处理API](https://platform.openai.com/docs/guides/batch) —— 异步处理50%折扣、JSONL格式、24小时完成窗口和5万请求限制
- [GPTCache](https://github.com/zilliztech/GPTCache) —— 开源语义缓存库，支持多种嵌入后端、向量存储和淘汰策略
- [Martian模型路由器](https://docs.withmartian.com) —— 生产环境模型路由，自动选择能够处理每个查询的最廉价模型
- [Not Diamond](https://www.notdiamond.ai) —— 基于机器学习的模型路由器，从你的流量模式中学习以优化跨提供商的成本/质量权衡
- [Helicone](https://www.helicone.ai) —— LLM可观测性平台，提供成本追踪、缓存、速率限制和预算告警作为代理层
- [Dean & Barroso, "The Tail at Scale" (CACM 2013)](https://research.google/pubs/the-tail-at-scale/) —— 延迟、吞吐量、TTFT/TPOT百分位和对冲请求；"选择仍满足P95的最廉价模型"背后的成本模型
- [Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023)](https://arxiv.org/abs/2309.06180) —— vLLM论文；分页KV缓存+持续批处理为何在吞吐量上比朴素服务器高24倍，"缓存和成本"下的基础设施层
- [Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (ICLR 2024)](https://arxiv.org/abs/2307.08691) —— 内核级成本降低，与提示缓存正交；结合推测解码和GQA阅读以了解完整的成本曲线图。