# 函数调用与工具使用

> LLMs无法执行任何操作。它们只生成文本。这是它们的全部能力。它们不能检查天气、查询数据库、发送邮件、运行代码或读取文件。你所见过的每个“AI智能体”本质上都是一个LLM，它生成描述应调用哪个函数的JSON——而你的代码才真正执行调用。模型是大脑，工具是双手，函数调用则是连接二者的神经系统。

**类型：** 构建
**语言：** Python
**先修课程：** 第11阶段第03课（结构化输出）
**时间：** 约75分钟
**相关课程：** 第11阶段·14（模型上下文协议）——当工具需要在主机间共享时，从内联函数调用升级到MCP服务器。本课程涵盖内联场景；MCP涵盖协议场景。

## 学习目标

- 实现函数调用循环：定义工具模式、解析模型的工具调用JSON、执行函数并返回结果
- 设计具有清晰描述和类型化参数的工具模式，使模型能够可靠调用
- 构建多轮智能体循环，串联多个函数调用以回答复杂查询
- 处理函数调用边缘情况：并行工具调用、错误传播和防止无限工具循环

## 问题描述

你构建了一个聊天机器人。用户询问：“东京现在的天气如何？”

模型回答：“我无法访问实时天气数据，但根据季节判断，东京现在可能在15摄氏度左右……”

这是一种带着免责声明的幻觉。模型不知道天气，也永远不会知道。天气每小时都在变化。模型的训练数据已是数月前的信息。

正确答案需要调用OpenWeatherMap API，获取当前温度，并返回真实数据。模型无法调用API，但你的代码可以。缺失的关键环节是一个结构化协议，让模型能够表达“我需要使用这些参数调用天气API”，并让你的代码执行它、将结果反馈回去。

这就是函数调用。模型输出结构化的JSON，描述应调用哪个函数及使用什么参数。你的应用程序执行该函数，将结果返回对话上下文。模型利用结果生成最终答案。

没有函数调用，LLM就是百科全书。有了它，它们就变成了智能体。

## 核心概念

### 函数调用循环

每次工具使用交互都遵循相同的5步循环。

```mermaid
sequenceDiagram
    participant U as User
    participant A as Application
    participant M as Model
    participant T as Tool

    U->>A: "What's the weather in Tokyo?"
    A->>M: messages + tool definitions
    M->>A: tool_call: get_weather(city="Tokyo")
    A->>T: Execute get_weather("Tokyo")
    T->>A: {"temp": 18, "condition": "cloudy"}
    A->>M: tool_result + conversation
    M->>A: "It's 18C and cloudy in Tokyo."
    A->>U: Final response
```

步骤1：用户发送消息。步骤2：模型接收消息及工具定义（描述可用函数的JSON Schema）。步骤3：模型不直接回复文本，而是输出工具调用——一个包含函数名和参数的结构化JSON对象。步骤4：你的代码执行函数并捕获结果。步骤5：结果返回模型，模型现在拥有真实数据来生成最终答案。

模型从不执行任何操作，它只决定调用哪个函数及使用什么参数。你的代码才是执行者。

### 工具定义：JSON Schema契约

每个工具由JSON Schema定义，它告诉模型函数的功能、接受的参数以及参数的类型要求。

```mermaid
sequenceDiagram
    participant U as User
    participant A as Application
    participant M as Model
    participant T as Tool

    U->>A: "What's the weather in Tokyo?"
    A->>M: messages + tool definitions
    M->>A: tool_call: get_weather(city="Tokyo")
    A->>T: Execute get_weather("Tokyo")
    T->>A: {"temp": 18, "condition": "cloudy"}
    A->>M: tool_result + conversation
    M->>A: "It's 18C and cloudy in Tokyo."
    A->>U: Final response
```

`description`字段至关重要。模型读取这些字段来决定何时以及如何使用工具。模糊的描述如“获取天气”会导致工具选择效果变差，而“获取城市的当前天气。返回摄氏温度和天气状况。”这样的描述则更优。描述本质上是为工具选择提供的提示。

### 提供商比较

主流提供商都支持函数调用，但API接口存在差异。

| 提供商 | API参数 | 工具调用格式 | 并行调用 | 强制调用 |
|----------|--------------|-----------------|---------------|----------------|
| OpenAI (GPT-5, o4) | `tools` | `tool_calls[].function` | 是（每轮多个） | `tool_choice="required"` |
| Anthropic (Claude 4.6/4.7) | `tools` | `content[].type="tool_use"` | 是（多个块） | `tool_choice={"type":"any"}` |
| Google (Gemini 3) | `function_declarations` | `functionCall` | 是 | `function_calling_config` |
| 开源模型 (Llama 4, Qwen3, DeepSeek-V3) | Llama 4原生支持`tools`；其他使用Hermes或ChatML | 混合格式 | 取决于模型 | 基于提示或`tool_choice`（如支持） |

截至2026年，三大闭源提供商已趋近于采用几乎相同的基于JSON Schema的格式。Llama 4内置原生`tools`字段，与OpenAI格式一致。开源微调模型仍存在差异——Hermes格式（NousResearch）是第三方微调最常见的格式。对于跨主机的工具共享，建议使用MCP（第11阶段·14）而非内联函数调用——所有提供商使用相同的服务器。

### 工具选择：自动、强制、指定

你可以控制模型何时使用工具。

**自动**（默认）：模型自行决定是调用工具还是直接回答。“2+2等于几？”——直接回答。“天气怎么样？”——调用工具。

**强制**：模型必须至少调用一个工具。当你确定用户意图需要工具时使用此选项。防止模型猜测而非查询真实数据。

**指定函数**：强制模型调用特定函数。`tool_choice={"type":"function", "function": {"name": "get_weather"}}`保证会调用天气工具，无论查询内容如何。用于路由场景——当上游逻辑已确定需要哪个工具时。

### 并行函数调用

GPT-4o和Claude能在单轮中调用多个函数。用户询问：“东京和纽约的天气怎么样？”模型同时输出两个工具调用：

```json
[
  {"name": "get_weather", "arguments": {"city": "Tokyo"}},
  {"name": "get_weather", "arguments": {"city": "New York"}}
]
```

你的代码同时执行两个调用（理想情况下并发执行），返回两个结果，模型合成单一回复。这将往返次数从2次减少到1次。对于每个查询需要5-10次工具调用的智能体，并行调用可减少60-80%的延迟。

### 结构化输出 vs 函数调用

第03课已介绍结构化输出。函数调用使用相同的JSON Schema机制，但目的不同。

**结构化输出**：强制模型生成特定形状的数据。输出是最终产品。示例：从文本中提取产品信息为`{name, price, in_stock}`格式。

**函数调用**：模型声明执行某个操作的意图。输出是中间步骤。示例：`get_weather(city="Tokyo")`——模型请求执行操作，而非生成最终答案。

需要数据提取时使用结构化输出。需要模型与外部系统交互时使用函数调用。

### 安全：不可违背的规则

函数调用是你能赋予LLM的最危险能力。模型选择执行什么操作。如果你的工具集包含数据库查询，模型会构建查询语句；如果包含shell命令，模型会编写命令。

**规则1：绝不将模型生成的SQL直接传给数据库。** 模型可能会生成DROP TABLE、UNION注入或返回所有行的查询。务必参数化、验证、并使用操作白名单。

**规则2：函数白名单。** 模型只能调用你明确定义的函数。切勿构建“按名称执行任意函数”的通用工具。如果有50个内部函数，只暴露用户需要的5个。

**规则3：参数验证。** 模型可能传递如`"; DROP TABLE users; --"`这样的城市名。在执行前，根据预期类型、范围和格式验证每个参数。

**规则4：清理工具结果。** 如果工具返回敏感数据（API密钥、个人身份信息、内部错误），在发送回模型前进行过滤。模型会原样引用工具结果。

**规则5：限制工具调用频率。** 循环中的模型可能调用工具数百次。设置最大值（每会话10-20次调用较合理），中断无限循环。

### 错误处理

工具会失败。API会超时。数据库会宕机。文件不存在。模型需要知道工具何时失败及原因。

将错误作为结构化工具结果返回，而非异常：

```json
{
  "error": true,
  "message": "City 'Toky' not found. Did you mean 'Tokyo'?",
  "code": "CITY_NOT_FOUND"
}
```

模型读取错误信息后，会调整参数并重试。模型擅长从结构化错误信息中自我纠正，但难以从空响应或通用“出现错误”消息中恢复。

### MCP：模型上下文协议

MCP是Anthropic推出的工具互操作开放标准。MCP不依赖每个应用程序自定义工具，而是提供通用协议：工具由MCP服务器提供，由MCP客户端（如Claude Code、Cursor或你的应用程序）消费。

一个MCP服务器可向任何兼容客户端暴露工具。PostgreSQL MCP服务器为任何MCP兼容智能体提供数据库访问；GitHub MCP服务器提供代码仓库访问。工具一次定义，随处可用。

MCP之于函数调用，正如HTTP之于网络。它标准化传输层，使工具变得可移植。

## 构建实现

### 步骤1：定义工具注册表

构建一个存储工具定义及其实现的注册表。每个工具包含JSON Schema定义（模型所见）和Python函数（代码执行部分）。

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

### 步骤2：实现5个工具

构建计算器、天气查询、网页搜索模拟器、文件读取器和代码运行器。

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

### 步骤3：注册所有工具

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

### 步骤4：构建函数调用循环

这是核心引擎。它模拟模型决定调用哪个工具、执行工具并反馈结果。

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

### 步骤5：参数验证

构建验证器，在执行前检查工具调用参数是否符合JSON Schema。

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

### 步骤6：运行演示

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

## 使用实践

### OpenAI函数调用

```python
# from openai import OpenAI
#
# client = OpenAI()
#
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
# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[{"role": "user", "content": "Weather in Tokyo?"}],
#     tools=tools,
#     tool_choice="auto",
# )
#
# tool_call = response.choices[0].message.tool_calls[0]
# args = json.loads(tool_call.function.arguments)
# result = get_weather(**args)
#
# final = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         response.choices[0].message,
#         {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)},
#     ],
# )
# print(final.choices[0].message.content)
```

OpenAI以`response.choices[0].message.tool_calls`形式返回工具调用。每个调用包含一个`id`，返回结果时必须包含该ID。模型使用此ID将结果与调用匹配。GPT-4o可在单次响应中返回多个工具调用——需遍历并执行所有调用。

### Anthropic工具使用

```python
# import anthropic
#
# client = anthropic.Anthropic()
#
# response = client.messages.create(
#     model="claude-sonnet-4-20250514",
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
# tool_block = next(b for b in response.content if b.type == "tool_use")
# result = get_weather(**tool_block.input)
#
# final = client.messages.create(
#     model="claude-sonnet-4-20250514",
#     max_tokens=1024,
#     tools=[...],
#     messages=[
#         {"role": "user", "content": "Weather in Tokyo?"},
#         {"role": "assistant", "content": response.content},
#         {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": json.dumps(result)}]},
#     ],
# )
```

Anthropic以内容块形式返回工具调用，包含`type: "tool_use"`。工具结果需作为用户消息附带`type: "tool_result"`返回。注意关键区别：Anthropic使用`input_schema`定义工具参数，而OpenAI使用`parameters`。

### MCP集成

```python
# MCP servers expose tools over a standardized protocol.
# Any MCP-compatible client can discover and call these tools.
#
# Example: connecting to a Postgres MCP server
#
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
#
# server_params = StdioServerParameters(
#     command="npx",
#     args=["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"],
# )
#
# async with stdio_client(server_params) as (read, write):
#     async with ClientSession(read, write) as session:
#         await session.initialize()
#         tools = await session.list_tools()
#         result = await session.call_tool("query", {"sql": "SELECT count(*) FROM users"})
```

MCP将工具实现与工具消费解耦。PostgreSQL服务器了解SQL，GitHub服务器了解API。你的智能体只需发现和调用工具——无需为每个集成编写提供商特定代码。

## 产出成果

本课程生成`outputs/prompt-tool-designer.md`——一个用于设计工具定义的可复用提示模板。提供工具功能描述，即可生成包含描述、类型和约束的完整JSON Schema定义。

同时生成`outputs/skill-function-calling-patterns.md`——一个在生产环境中实现函数调用的决策框架，涵盖工具设计、错误处理、安全性和提供商特定模式。

## 练习题

1. **添加第6个工具：数据库查询。** 实现一个使用内存表的模拟SQL工具。该工具接受表名和过滤条件（非原始SQL）。验证表名在白名单中，过滤操作符限制为`=`、`>`、`<`、`>=`、`<=`。返回匹配行的JSON格式。

2. **实现带错误反馈的重试机制。** 当工具调用失败（如城市未找到），将错误信息反馈给模型决策函数，让其修正参数。跟踪每次调用的重试次数。设置每工具调用最多3次重试。

3. **构建多步智能体。** 某些查询需要串联工具调用：“读取配置文件并告诉我配置的模型，然后搜索该模型的定价信息。”实现一个循环运行直到模型决定无需更多工具的机制，将累积结果传入每个决策步骤。限制最多10次迭代以防止无限循环。

4. **测量工具选择准确率。** 创建30个带预期工具名的测试查询。在所有查询上运行决策函数，测量选择正确工具的百分比。识别哪些查询最易造成工具间的混淆。

5. **实现工具调用缓存。** 如果60秒内使用相同参数调用同一工具，返回缓存结果而非重新执行。使用以`(tool_name, frozenset(args.items()))`为键的字典。测量包含20个查询的会话中的缓存命中率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------------|----------------------|
| 函数调用 | “工具使用” | 模型输出结构化JSON描述要调用的函数及参数——由你的代码执行，而非模型 |
| 工具定义 | “函数模式” | 描述工具名称、用途、参数和类型的JSON Schema对象——模型据此决定何时及如何使用工具 |
| 工具选择 | “调用模式” | 控制模型必须调用工具（强制）、可以调用工具（自动）或必须调用特定工具（指定） |
| 并行调用 | “多工具” | 模型在单轮中输出多个工具调用，减少往返次数——GPT-4o和Claude均支持 |
| 工具结果 | “函数输出” | 执行工具的返回值，作为消息发送回模型供其使用真实数据生成回复 |
| 参数验证 | “输入检查” | 在执行工具前，验证模型生成的参数是否符合预期类型、范围和约束 |
| MCP | “工具协议” | 模型上下文协议——Anthropic的开放标准，通过服务器暴露工具供任何兼容客户端发现和调用 |
| 智能体循环 | “ReAct循环” | 模型决策工具→代码执行工具→结果反馈的迭代循环，直到模型获得足够信息生成回复 |
| 工具投毒 | “通过工具的提示注入” | 工具结果包含操纵模型行为的指令的攻击方式——需清理所有工具输出 |
| 频率限制 | “调用预算” | 设置每会话最大工具调用次数，防止无限循环和API成本失控 |

## 延伸阅读

- [OpenAI函数调用指南](https://platform.openai.com/docs/guides/function-calling) —— GPT-4o工具使用的权威参考，包括并行调用、强制调用和结构化参数
- [Anthropic工具使用指南](https://docs.anthropic.com/en/docs/tool-use) —— Claude的工具使用实现，包含input_schema、多工具响应和tool_choice配置
- [模型上下文协议规范](https://modelcontextprotocol.io) —— 跨AI应用的工具互操作开放标准，基于服务器/客户端架构
- [Schick等，2023年 —— “Toolformer：语言模型可自学使用工具”](https://arxiv.org/abs/2302.04761) —— 训练LLM决定何时及如何调用外部工具的基础论文
- [Patil等，2023年 —— “Gorilla：连接海量API的大语言模型”](https://arxiv.org/abs/2305.15334) —— 微调LLM以准确调用1,645个API并减少幻觉
- [伯克利函数调用排行榜](https://gorilla.cs.berkeley.edu/leaderboard.html) —— 对比GPT-4o、Claude、Gemini和开源模型函数调用准确率的实时基准测试
- [Yao等，“ReAct：在语言模型中协同推理与行动”（ICLR 2023）](https://arxiv.org/abs/2210.03629) —— 思考-行动-观察循环，是每个工具调用的外层智能体循环；本课程结束处，第14阶段将继续深入
- [Anthropic——构建有效智能体（2024年12月）](https://www.anthropic.com/research/building-effective-agents) —— 五种可组合模式（提示链、路由、并行化、编排器-工作者、评估器-优化器）均基于单一工具使用原语构建