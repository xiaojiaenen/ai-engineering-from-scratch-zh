# 函数调用与工具使用

> LLM（大语言模型）本身什么也做不了。它们只生成文本。这就是全部能力。它们无法查看天气、查询数据库、发送邮件、运行代码或读取文件。你见过的每一个“AI 智能体”，本质上都是一个 LLM 生成一段 JSON，指出需要调用哪个函数——然后由你的代码真正去执行它。模型是大脑，工具是双手，函数调用是连接两者的神经系统。

**类型：** Build
**语言：** Python
**前置知识：** Phase 11 Lesson 03（结构化输出）
**预计时间：** 约 75 分钟
**相关内容：** Phase 11 · 14（模型上下文协议 MCP）——当工具需要跨主机共享时，从内联函数调用升级为 MCP 服务器。本课覆盖内联场景；MCP 课程覆盖协议场景。

## 学习目标

- 实现函数调用循环：定义工具 Schema、解析模型的 tool-call JSON、执行函数并返回结果
- 设计具有清晰描述和类型化参数的工具 Schema，确保模型能够可靠调用
- 构建多轮智能体循环，通过链式调用多个函数来回答复杂查询
- 处理函数调用的边界情况：并行工具调用、错误传播以及防止无限工具循环

## 问题所在

你正在构建一个聊天机器人。用户问：“东京现在天气怎么样？”

模型回复：“我无法访问实时天气数据，但根据季节来看，东京可能在 15 摄氏度左右……”

这是一种披上了免责声明的幻觉。模型并不知道天气。它也永远不会知道。天气每小时都在变化，而模型的训练数据是几个月前的。

正确答案需要调用 OpenWeatherMap API，获取当前温度，然后返回真实数据。模型无法调用 API，但你的代码可以。缺少的只是一个结构化协议，让模型能够说“我需要用这些参数调用天气 API”，并让你的代码执行它并将结果反馈回来。

这就是函数调用。模型输出描述需要调用哪个函数及参数的结构化 JSON。你的应用程序执行该函数，结果返回给对话上下文，模型利用结果生成最终答案。

没有函数调用，LLM 只是百科全书；有了它，它们才成为智能体。

## 核心概念

### 函数调用循环

每次工具使用交互都遵循相同的五步循环。

```mermaid
sequenceDiagram
    participant U as 用户
    participant A as 应用
    participant M as 模型
    participant T as 工具

    U->>A: "What's the weather in Tokyo?"
    A->>M: messages + tool definitions
    M->>A: tool_call: get_weather(city="Tokyo")
    A->>T: Execute get_weather("Tokyo")
    T->>A: {"temp": 18, "condition": "cloudy"}
    A->>M: tool_result + conversation
    M->>A: "It's 18C and cloudy in Tokyo."
    A->>U: Final response
```

第 1 步：用户发送消息。第 2 步：模型接收消息以及工具定义（描述可用函数的 JSON Schema）。第 3 步：模型不直接回复文本，而是输出一个 tool call——一个包含函数名和参数的结构化 JSON 对象。第 4 步：你的代码执行该函数并捕获结果。第 5 步：结果返回给模型，模型此时拥有了真实数据，从而生成最终答案。

模型永远不会执行任何操作。它只负责决定调用什么函数以及传入什么参数。你的代码才是执行者。

### 工具定义：JSON Schema 契约

每个工具都由一个 JSON Schema 定义，用于告诉模型该函数的作用、接受的参数以及参数的类型要求。

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a city. Returns temperature in Celsius and conditions.",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "City name, e.g. 'Tokyo' or 'San Francisco'"
        },
        "units": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "Temperature units"
        }
      },
      "required": ["city"]
    }
  }
}
```

`description` 字段至关重要。模型会读取这些描述来决定何时以及如何使用该工具。模糊的描述（如“获取天气”）比清晰的描述（如“获取某城市的当前天气，返回摄氏温度和天气状况”）会导致更差的选择效果。描述本质上就是针对工具选择的提示词。

### 提供商对比

所有主流提供商都支持函数调用，但 API 接口存在差异。

| 提供商 | API 参数 | 工具调用格式 | 并行调用 | 强制调用 |
|----------|--------------|-----------------|---------------|----------------|
| OpenAI (GPT-5, o4) | `tools` | `tool_calls[].function` | 支持（单次可返回多个） | `tool_choice="required"` |
| Anthropic (Claude 4.6/4.7) | `tools` | `content[].type="tool_use"` | 支持（多个 block） | `tool_choice={"type":"any"}` |
| Google (Gemini 3) | `function_declarations` | `functionCall` | 支持 | `function_calling_config` |
| Open-weight (Llama 4, Qwen3, DeepSeek-V3) | Llama 4 原生 `tools`；其他模型使用 Hermes 或 ChatML | 混合 | 因模型而异 | 基于提示或 `tool_choice`（若支持） |

到 2026 年，三大闭源提供商已收敛于几乎相同的基于 JSON Schema 的格式。Llama 4 自带原生 `tools` 字段，其结构与 OpenAI 一致。开源微调模型仍各有差异——Hermes 格式（NousResearch）在第三方微调模型中最常见。对于跨主机共享的工具，推荐使用 MCP（Phase 11 · 14）而非内联函数调用——它们的服务器是统一的。

### 工具选择：Auto、Required、Specific

你控制模型何时使用工具。

**Auto（默认）**：模型自行决定是调用工具还是直接回复。例如：“2+2 等于几？”——直接回复。“东京天气怎样？”——调用工具。

**Required**：模型必须调用至少一个工具。当你确定用户意图需要调用工具时使用。可防止模型凭空猜测而不是查询真实数据。

**Specific（指定函数）**：强制模型调用特定函数。`tool_choice={"type":"function", "function": {"name": "get_weather"}}` 可保证无论查询内容如何都会调用天气工具。适用于路由场景——当上游逻辑已确定需要哪个工具时。

### 并行函数调用

GPT-4o 和 Claude 可在单轮中调用多个函数。用户问：“东京和纽约的天气怎么样？”模型同时输出两个 tool call：

```json
[
  {"name": "get_weather", "arguments": {"city": "Tokyo"}},
  {"name": "get_weather", "arguments": {"city": "New York"}}
]
```

你的代码应并发执行这两个调用，返回两个结果，由模型合成单一回复。这将往返次数从 2 次降为 1 次。对于每次查询包含 5-10 次工具调用的智能体，并行调用可降低 60-80% 的延迟。

### 结构化输出 vs 函数调用

第 03 课讲解了结构化输出。函数调用使用了相同的 JSON Schema 机制，但目的不同。

**结构化输出**：强制模型以特定格式产出数据。输出即为最终产品。示例：从文本中提取产品信息，输出 `{name, price, in_stock}`。

**函数调用**：模型声明执行某个动作的意图。输出是中间步骤。示例：`get_weather(city="Tokyo")`——模型是在请求执行动作，而非直接生成最终答案。

需要数据提取时使用结构化输出；需要模型与外部系统交互时使用函数调用。

### 安全性：不可妥协的规则

函数调用是你所能赋予 LLM 的最危险能力。模型决定执行什么。如果你的工具集中包含数据库查询，模型就会构造查询语句；如果包含 Shell 命令，模型就会写入命令。

**规则 1：切勿将模型生成的 SQL 直接传入数据库。** 模型可能生成 DROP TABLE、UNION 注入或返回全表数据的查询。始终使用参数化查询，始终进行校验，始终使用操作白名单。

**规则 2：仅限白名单函数。** 模型只能调用你显式定义的函数。切勿构建通用的“按名称执行任意函数”工具。如果你有 50 个内部函数，只暴露用户需要的 5 个。

**规则 3：校验参数。** 模型可能会传入 `"; DROP TABLE users; --"` 这样的城市名。执行前必须对所有参数进行类型、范围和格式校验。

**规则 4：过滤工具结果。** 如果工具返回敏感数据（API 密钥、PII、内部错误），在返回给模型前必须进行过滤。模型会将工具结果原样包含在其回复中。

**规则 5：限制工具调用频率。** 陷入循环的模型可能连续调用数百次工具。设置上限（每轮对话 10-20 次为合理范围），打断无限循环。

### 错误处理

工具会失败。API 会超时。数据库会宕机。文件可能不存在。模型需要知道工具何时失败以及原因。

将错误作为结构化工具结果返回，而非抛出异常：

```json
{
  "error": true,
  "message": "City 'Toky' not found. Did you mean 'Tokyo'?",
  "code": "CITY_NOT_FOUND"
}
```

模型读取后调整参数并重试。模型擅长根据结构化错误消息自我修正，但不擅长从空响应或泛泛的“出错了”中恢复。

### MCP：模型上下文协议

MCP 是 Anthropic 推出的工具互操作性开放标准。与其让每个应用各自定义工具，MCP 提供了一个通用协议：工具由 MCP 服务器提供，供 MCP 客户端（如 Claude Code、Cursor 或你的应用）消费。

一个 MCP 服务器可向任意兼容客户端暴露工具。Postgres MCP 服务器让任何 MCP 兼容智能体都能访问数据库；GitHub MCP 服务器让任何智能体都能访问仓库。工具定义一次，处处复用。

MCP 之于函数调用，犹如 HTTP 之于网络。它标准化了传输层，使工具具备可移植性。

```figure
mx-tool-call-loop
```

## 构建实践

### 步骤 1：定义工具注册表

构建一个注册表，存储工具定义及其实现。每个工具包含一个 JSON Schema 定义（模型所见）和一个 Python 函数（你的代码执行）。

```python
import json
import math
import time
import hashlib


TOOL_REGISTRY = {}


def register_tool(name, description, parameters, function):
    TOOL_REGISTRY[name] = {
        "definition": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        },
        "function": function,
    }
```

### 步骤 2：实现 5 个工具

构建计算器、天气查询、网页搜索模拟器、文件读取器和代码执行器。

```python
def calculator(expression, precision=2):
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": True, "message": f"Invalid characters in expression: {expression}"}
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return {"result": round(float(result), precision), "expression": expression}
    except Exception as e:
        return {"error": True, "message": str(e)}


WEATHER_DB = {
    "tokyo": {"temp_c": 18, "condition": "cloudy", "humidity": 72, "wind_kph": 14},
    "new york": {"temp_c": 22, "condition": "sunny", "humidity": 45, "wind_kph": 8},
    "london": {"temp_c": 12, "condition": "rainy", "humidity": 88, "wind_kph": 22},
    "san francisco": {"temp_c": 16, "condition": "foggy", "humidity": 80, "wind_kph": 18},
    "sydney": {"temp_c": 25, "condition": "sunny", "humidity": 55, "wind_kph": 10},
}


def get_weather(city, units="celsius"):
    key = city.lower().strip()
    if key not in WEATHER_DB:
        suggestions = [c for c in WEATHER_DB if c.startswith(key[:3])]
        return {
            "error": True,
            "message": f"City '{city}' not found.",
            "suggestions": suggestions,
            "code": "CITY_NOT_FOUND",
        }
    data = WEATHER_DB[key].copy()
    if units == "fahrenheit":
        data["temp_f"] = round(data["temp_c"] * 9 / 5 + 32, 1)
        del data["temp_c"]
    data["city"] = city
    return data


SEARCH_DB = {
    "python function calling": [
        {"title": "OpenAI Function Calling Guide", "url": "https://platform.openai.com/docs/guides/function-calling", "snippet": "Learn how to connect LLMs to external tools."},
        {"title": "Anthropic Tool Use", "url": "https://docs.anthropic.com/en/docs/tool-use", "snippet": "Claude can interact with external tools and APIs."},
    ],
    "MCP protocol": [
        {"title": "Model Context Protocol", "url": "https://modelcontextprotocol.io", "snippet": "An open standard for connecting AI models to data sources."},
    ],
    "weather API": [
        {"title": "OpenWeatherMap API", "url": "https://openweathermap.org/api", "snippet": "Free weather API with current, forecast, and historical data."},
    ],
}


def web_search(query, max_results=3):
    key = query.lower().strip()
    for db_key, results in SEARCH_DB.items():
        if db_key in key or key in db_key:
            return {"query": query, "results": results[:max_results], "total": len(results)}
    return {"query": query, "results": [], "total": 0}


FILE_SYSTEM = {
    "data/config.json": '{"model": "gpt-4o", "temperature": 0.7, "max_tokens": 4096}',
    "data/users.csv": "name,email,role\nAlice,alice@example.com,admin\nBob,bob@example.com,user",
    "README.md": "# My Project\nA tool-use agent built from scratch.",
}


def read_file(path):
    if ".." in path or path.startswith("/"):
        return {"error": True, "message": "Path traversal not allowed.", "code": "FORBIDDEN"}
    if path not in FILE_SYSTEM:
        available = list(FILE_SYSTEM.keys())
        return {"error": True, "message": f"File '{path}' not found.", "available_files": available, "code": "NOT_FOUND"}
    content = FILE_SYSTEM[path]
    return {"path": path, "content": content, "size_bytes": len(content), "lines": content.count("\n") + 1}


def run_code(code, language="python"):
    if language != "python":
        return {"error": True, "message": f"Language '{language}' not supported. Only 'python' is available."}
    forbidden = ["import os", "import sys", "import subprocess", "exec(", "eval(", "__import__", "open("]
    for pattern in forbidden:
        if pattern in code:
            return {"error": True, "message": f"Forbidden operation: {pattern}", "code": "SECURITY_VIOLATION"}
    try:
        local_vars = {}
        exec(code, {"__builtins__": {"print": print, "range": range, "len": len, "str": str, "int": int, "float": float, "list": list, "dict": dict, "sum": sum, "min": min, "max": max, "abs": abs, "round": round, "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "math": math}}, local_vars)
        result = local_vars.get("result", None)
        return {"success": True, "result": result, "variables": {k: str(v) for k, v in local_vars.items() if not k.startswith("_")}}
    except Exception as e:
        return {"error": True, "message": f"{type(e).__name__}: {e}"}
```

### 步骤 3：注册所有工具

```python
def register_all_tools():
    register_tool(
        "calculator", "Evaluate a mathematical expression. Supports +, -, *, /, parentheses, and decimals. Returns the numeric result.",
        {"type": "object", "properties": {"expression": {"type": "string", "description": "Math expression, e.g. '(10 + 5) * 3'"}, "precision": {"type": "integer", "description": "Decimal places in result", "default": 2}}, "required": ["expression"]},
        calculator,
    )
    register_tool(
        "get_weather", "Get current weather for a city. Returns temperature, condition, humidity, and wind speed.",
        {"type": "object", "properties": {"city": {"type": "string", "description": "City name, e.g. 'Tokyo' or 'San Francisco'"}, "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units, defaults to celsius"}}, "required": ["city"]},
        get_weather,
    )
    register_tool(
        "web_search", "Search the web for information. Returns a list of results with title, URL, and snippet.",
        {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}, "max_results": {"type": "integer", "description": "Maximum results to return", "default": 3}}, "required": ["query"]},
        web_search,
    )
    register_tool(
        "read_file", "Read the contents of a file. Returns the file content, size, and line count.",
        {"type": "object", "properties": {"path": {"type": "string", "description": "Relative file path, e.g. 'data/config.json'"}}, "required": ["path"]},
        read_file,
    )
    register_tool(
        "run_code", "Execute Python code in a sandboxed environment. Set a 'result' variable to return output.",
        {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute"}, "language": {"type": "string", "enum": ["python"], "description": "Programming language"}}, "required": ["code"]},
        run_code,
    )
```

### 步骤 4：构建函数调用循环

这是核心引擎。它模拟模型决定调用哪个工具、执行工具并将结果反馈的过程。

```python
def simulate_model_decision(user_message, tools, conversation_history):
    msg = user_message.lower()

    if any(word in msg for word in ["weather", "temperature", "forecast"]):
        cities = []
        for city in WEATHER_DB:
            if city in msg:
                cities.append(city)
        if not cities:
            for word in msg.split():
                if word.capitalize() in [c.title() for c in WEATHER_DB]:
                    cities.append(word)
        if not cities:
            cities = ["tokyo"]
        calls = []
        for city in cities:
            calls.append({"name": "get_weather", "arguments": {"city": city.title()}})
        return calls

    if any(word in msg for word in ["calculate", "compute", "math", "what is", "how much"]):
        for token in msg.split():
            if any(c in token for c in "+-*/"):
                return [{"name": "calculator", "arguments": {"expression": token}}]
        if "+" in msg or "-" in msg or "*" in msg or "/" in msg:
            expr = "".join(c for c in msg if c in "0123456789+-*/.() ")
            if expr.strip():
                return [{"name": "calculator", "arguments": {"expression": expr.strip()}}]
        return [{"name": "calculator", "arguments": {"expression": "0"}}]

    if any(word in msg for word in ["search", "find", "look up", "google"]):
        query = msg.replace("search for", "").replace("look up", "").replace("find", "").strip()
        return [{"name": "web_search", "arguments": {"query": query}}]

    if any(word in msg for word in ["read", "file", "open", "cat", "show"]):
        for path in FILE_SYSTEM:
            if path.split("/")[-1].split(".")[0] in msg:
                return [{"name": "read_file", "arguments": {"path": path}}]
        return [{"name": "read_file", "arguments": {"path": "README.md"}}]

    if any(word in msg for word in ["run", "execute", "code", "python"]):
        return [{"name": "run_code", "arguments": {"code": "result = 'Hello from the sandbox!'", "language": "python"}}]

    return []


def execute_tool_call(tool_call):
    name = tool_call["name"]
    args = tool_call["arguments"]

    if name not in TOOL_REGISTRY:
        return {"error": True, "message": f"Unknown tool: {name}", "code": "UNKNOWN_TOOL"}

    tool = TOOL_REGISTRY[name]
    func = tool["function"]
    start = time.time()

    try:
        result = func(**args)
    except TypeError as e:
        result = {"error": True, "message": f"Invalid arguments: {e}"}

    elapsed_ms = round((time.time() - start) * 1000, 2)
    return {"tool": name, "result": result, "execution_time_ms": elapsed_ms}


def run_function_calling_loop(user_message, max_iterations=5):
    conversation = [{"role": "user", "content": user_message}]
    tool_definitions = [t["definition"] for t in TOOL_REGISTRY.values()]
    all_tool_results = []

    for iteration in range(max_iterations):
        tool_calls = simulate_model_decision(user_message, tool_definitions, conversation)

        if not tool_calls:
            break

        results = []
        for call in tool_calls:
            result = execute_tool_call(call)
            results.append(result)

        conversation.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

        for result in results:
            conversation.append({"role": "tool", "content": json.dumps(result["result"]), "tool_name": result["tool"]})

        all_tool_results.extend(results)
        break

    return {"conversation": conversation, "tool_results": all_tool_results, "iterations": iteration + 1 if tool_calls else 0}
```

### 步骤 5：参数校验

构建一个校验器，在执行前根据 JSON Schema 检查工具调用参数。

```python
def validate_tool_arguments(tool_name, arguments):
    if tool_name not in TOOL_REGISTRY:
        return [f"Unknown tool: {tool_name}"]

    schema = TOOL_REGISTRY[tool_name]["definition"]["function"]["parameters"]
    errors = []

    if not isinstance(arguments, dict):
        return [f"Arguments must be an object, got {type(arguments).__name__}"]

    for required_field in schema.get("required", []):
        if required_field not in arguments:
            errors.append(f"Missing required argument: {required_field}")

    properties = schema.get("properties", {})
    for arg_name, arg_value in arguments.items():
        if arg_name not in properties:
            errors.append(f"Unknown argument: {arg_name}")
            continue

        prop_schema = properties[arg_name]
        expected_type = prop_schema.get("type")

        type_checks = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
        if expected_type in type_checks:
            if not isinstance(arg_value, type_checks[expected_type]):
                errors.append(f"Argument '{arg_name}': expected {expected_type}, got {type(arg_value).__name__}")

        if "enum" in prop_schema and arg_value not in prop_schema["enum"]:
            errors.append(f"Argument '{arg_name}': '{arg_value}' not in {prop_schema['enum']}")

    return errors
```

### 步骤 6：运行演示

```python
def run_demo():
    register_all_tools()

    print("=" * 60)
    print("  Function Calling & Tool Use Demo")
    print("=" * 60)

    print("\n--- Registered Tools ---")
    for name, tool in TOOL_REGISTRY.items():
        desc = tool["definition"]["function"]["description"][:60]
        params = list(tool["definition"]["function"]["parameters"].get("properties", {}).keys())
        print(f"  {name}: {desc}...")
        print(f"    params: {params}")

    print(f"\n--- Argument Validation ---")
    validation_tests = [
        ("get_weather", {"city": "Tokyo"}, "Valid call"),
        ("get_weather", {}, "Missing required arg"),
        ("get_weather", {"city": "Tokyo", "units": "kelvin"}, "Invalid enum value"),
        ("calculator", {"expression": 123}, "Wrong type (int for string)"),
        ("unknown_tool", {"x": 1}, "Unknown tool"),
    ]
    for tool_name, args, label in validation_tests:
        errors = validate_tool_arguments(tool_name, args)
        status = "VALID" if not errors else f"ERRORS: {errors}"
        print(f"  {label}: {status}")

    print(f"\n--- Tool Execution ---")
    direct_tests = [
        {"name": "calculator", "arguments": {"expression": "(10 + 5) * 3 / 2"}},
        {"name": "get_weather", "arguments": {"city": "Tokyo"}},
        {"name": "get_weather", "arguments": {"city": "Mars"}},
        {"name": "web_search", "arguments": {"query": "python function calling"}},
        {"name": "read_file", "arguments": {"path": "data/config.json"}},
        {"name": "read_file", "arguments": {"path": "../etc/passwd"}},
        {"name": "run_code", "arguments": {"code": "result = sum(range(1, 101))"}},
        {"name": "run_code", "arguments": {"code": "import os; os.system('rm -rf /')"}},
    ]
    for call in direct_tests:
        result = execute_tool_call(call)
        print(f"\n  {call['name']}({json.dumps(call['arguments'])})")
        print(f"    -> {json.dumps(result['result'], indent=None)[:100]}")
        print(f"    time: {result['execution_time_ms']}ms")

    print(f"\n--- Full Function Calling Loop ---")
    test_queries = [
        "What's the weather in Tokyo?",
        "Calculate (100 + 250) * 0.15",
        "Search for MCP protocol",
        "Read the config file",
        "Run some Python code",
        "Tell me a joke",
    ]
    for query in test_queries:
        print(f"\n  User: {query}")
        result = run_function_calling_loop(query)
        if result["tool_results"]:
            for tr in result["tool_results"]:
                print(f"    Tool: {tr['tool']} ({tr['execution_time_ms']}ms)")
                print(f"    Result: {json.dumps(tr['result'], indent=None)[:90]}")
        else:
            print(f"    [No tool called -- direct response]")
        print(f"    Iterations: {result['iterations']}")

    print(f"\n--- Parallel Tool Calls ---")
    multi_city_query = "What's the weather in tokyo and london?"
    print(f"  User: {multi_city_query}")
    result = run_function_calling_loop(multi_city_query)
    print(f"  Tool calls made: {len(result['tool_results'])}")
    for tr in result["tool_results"]:
        city = tr["result"].get("city", "unknown")
        temp = tr["result"].get("temp_c", "N/A")
        print(f"    {city}: {temp}C, {tr['result'].get('condition', 'N/A')}")

    print(f"\n--- Security Checks ---")
    security_tests = [
        ("read_file", {"path": "../../etc/passwd"}),
        ("run_code", {"code": "import subprocess; subprocess.run(['ls'])"}),
        ("calculator", {"expression": "__import__('os').system('ls')"}),
    ]
    for tool_name, args in security_tests:
        result = execute_tool_call({"name": tool_name, "arguments": args})
        blocked = result["result"].get("error", False)
        print(f"  {tool_name}({list(args.values())[0][:40]}): {'BLOCKED' if blocked else 'ALLOWED'}")
```

## 实际使用

### OpenAI 函数调用

```python
# 导入 OpenAI 客户端
# from openai import OpenAI
#
# # 初始化客户端
# client = OpenAI()
#
# # 定义工具列表
# tools = [{
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "description": "Get current weather for a city",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }
# }]
#
# # 发起请求
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
#     tools=tools,
#     tool_choice="auto",
# )
#
# # 提取工具调用
# tool_call = response.choices[0].message.tool_calls[0]
# # 解析参数
# args = json.loads(tool_call.function.arguments)
# # 执行工具
# result = get_weather(**args)
#
# # 发送最终请求
# final = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         response.choices[0].message,
#         {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)},
#     ],
# )
# # 输出结果
# print(final.choices[0].message.content)
```

OpenAI 会将工具调用返回在 `response.choices[0].message.tool_calls` 中。每次调用都有一个 `id`，你在返回结果时必须包含该 ID。模型依靠此 ID 将结果与调用匹配。GPT-4o 可在单次响应中返回多个工具调用——遍历并执行所有调用即可。

### Anthropic 工具使用

```python
# 导入 Anthropic 客户端
# import anthropic
#
# # 初始化客户端
# client = anthropic.Anthropic()
#
# # 发起请求
# response = client.messages.create(
#     model="claude-sonnet-5",
#     max_tokens=1024,
#     tools=[{
#         "name": "get_weather",
#         "description": "Get current weather for a city",
#         "input_schema": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string"},
#                 "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
#             },
#             "required": ["city"]
#         }
#     }],
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
# )
#
# # 提取工具调用块
# tool_block = next(b for b in response.content if b.type == "tool_use")
# # 执行工具
# result = get_weather(**tool_block.input)
#
# # 发送最终请求
# final = client.messages.create(
#     model="claude-sonnet-5",
#     max_tokens=1024,
#     tools=[...],
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         {"role": "assistant", "content": response.content},
#         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": json.dumps(result)}]},
#     ],
# )
```

Anthropic 将工具调用作为 `type: "tool_use"` 的内容块返回。工具结果应放入类型为 `type: "tool_result"` 的用户消息中。注意关键区别：Anthropic 使用 `input_schema` 定义工具
