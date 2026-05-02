#!/usr/bin/env python3
"""
s01_agent_loop_openai.py - Agent Loop (OpenAI 兼容版本)

核心思路：
  用户输入 → 模型思考 → 模型决定调用工具 → 执行工具 → 把工具结果反馈给模型 →
  模型继续思考 → ... → 模型不再调用工具，给出最终文本回复 → 结束

这是一个 while 循环驱动的 Agent 模式：
  - 模型每轮回复可能包含 tool_calls（表示想调用工具）
  - 如果有 tool_calls，执行对应工具，把结果 append 到 messages，继续循环
  - 如果没有 tool_calls，说明模型已经给出最终答案，循环结束

与 Anthropic Messages API 版本（s01_agent_loop.py）的区别：
  - Anthropic: 使用 "content blocks" 格式，tool_use / tool_result 是 content 数组中的元素
  - OpenAI:    使用 "tool_calls" / "tool" role 格式，tool_calls 在 message 对象上，
               工具结果用 role="tool" 的独立消息表示
"""

# ============================================================
# 导入部分
# ============================================================

import json  # 用于解析模型返回的 tool_call.arguments（JSON 字符串 → dict）
import os  # 用于读取环境变量（MODEL_ID、API_KEY 等）和获取当前工作目录
import subprocess  # 用于实际执行 bash 命令（模型的工具调用最终通过这个模块落地）
import sys  # 用于向 stderr 输出错误信息

# readline 模块提供终端输入的行编辑功能（上下箭头历史、Tab 补全等）
# macOS 的 libedit（不是 GNU readline）对 UTF-8 输入有 bug，
# 以下配置修复了 #143 号问题：在 macOS 上用退格键删除中文字符时会乱码
try:
    import readline

    readline.parse_and_bind("set bind-tty-special-chars off")  # 让终端特殊键（如 Ctrl+C）不被 readline 拦截
    readline.parse_and_bind("set input-meta on")  # 允许输入 8-bit 字符（UTF-8 的前提）
    readline.parse_and_bind("set output-meta on")  # 允许输出 8-bit 字符
    readline.parse_and_bind("set convert-meta off")  # 不要把 8-bit 字符转成 \M- 前缀格式
    readline.parse_and_bind("set enable-meta-keybindings on")  # 启用 meta 键绑定（Alt 键组合）
except ImportError:
    pass  # Windows 没有 readline 模块，跳过即可（Windows 用 pyreadline3 或不做行编辑）

# dotenv: 从 .env 文件加载环境变量到 os.environ
# 这样不用在系统层面设置环境变量，只需要在项目根目录放一个 .env 文件
from dotenv import load_dotenv

# OpenAI Python SDK —— 这是与 OpenAI 兼容 API 交互的核心库
# 所有遵循 OpenAI chat/completions 协议的服务（Dashscope、智谱、DeepSeek 等）
# 都可以用这个 SDK 调用，只需改 base_url 和 api_key
try:
    from openai import OpenAI
except ImportError:
    # 如果没安装 openai 包，给出明确提示并退出
    print("Error: openai package not installed. Run: pip install openai", file=sys.stderr)
    raise

# ============================================================
# 环境配置
# ============================================================

# override=True: .env 文件中的值会覆盖系统中已存在的同名环境变量
# 这确保了 .env 里的配置优先级最高，方便调试和切换不同 provider
load_dotenv(override=True)


def env_first(*names: str) -> str | None:
    """
    按优先级依次检查多个环境变量名，返回第一个有值的。

    用法示例：
      env_first("OPENAI_API_KEY", "API_KEY")
      → 先看 OPENAI_API_KEY，如果没设置再看 API_KEY

    这在兼容多个 provider 时很有用：不同 provider 可能用不同的变量名，
    但逻辑上都是"API 密钥"，用一个函数统一处理。
    """
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


# ---- 从环境变量读取配置 ----

# MODEL_ID: 必须设置，决定调用哪个模型（如 "glm-5.1"、"deepseek-chat"）
# 用 os.environ["MODEL_ID"] 而不是 os.getenv() → 如果没设置会直接抛 KeyError，
# 强制用户必须配置这个值，避免因遗漏配置而调用到错误模型
MODEL = os.environ["MODEL_ID"]

# OPENAI_API_KEY: API 密钥，用于鉴权
# 不同 provider 的密钥格式不同（Dashscope 用 sk-xxx，智谱用 zhipu.xxx 等）
API_KEY = env_first("OPENAI_API_KEY")

# OPENAI_BASE_URL: API 的基础地址
# 默认是 https://api.openai.com/v1，但我们通常用第三方兼容服务：
#   - Dashscope:  https://dashscope.aliyuncs.com/compatible-mode/v1
#   - 智谱:       https://open.bigmodel.cn/api/paas/v4/
#   - DeepSeek:   https://api.deepseek.com/v1
BASE_URL = env_first("OPENAI_BASE_URL")

# 创建 OpenAI 客户端 —— 所有后续 API 调用都通过这个 client 对象发起
# 内部会自动拼接 base_url + "/chat/completions" 等路径
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ============================================================
# 系统提示词（System Prompt）
# ============================================================

# 系统提示词定义了 Agent 的角色和行为准则
# os.getcwd() 让模型知道当前工作目录，方便它构造正确的文件路径
# "Act, don't explain" 是关键指令：要求模型直接执行操作而非写长篇解释
# 这对于 coding agent 很重要——用户要的是结果，不是教程
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# ============================================================
# 工具定义（Tool Definitions）
# ============================================================

# 这是 OpenAI function calling 格式的工具定义
# 模型在对话中看到这个定义后，就知道它可以调用哪些函数、每个函数需要什么参数
#
# 结构说明：
#   type: "function"           → 工具类型，目前 OpenAI 只支持 function
#   function.name              → 工具名称，模型在 tool_calls 中会引用这个名字
#   function.description       → 工具描述，模型根据这个决定什么时候该调用
#   function.parameters        → JSON Schema 格式的参数定义，模型据此构造 arguments
#
# 这个 Agent 只有一个工具：bash
# 让模型能执行任意 shell 命令，是 coding agent 的核心能力
TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"}  # 命令内容，如 "ls -la"、"python test.py"
            },
            "required": ["command"],  # command 是必填参数
        },
    },
}]


# ============================================================
# 工具执行函数
# ============================================================

def run_bash(command: str) -> str:
    """
    实际执行 bash 命令并返回输出结果。

    安全机制：
      - dangerous 列表定义了被禁止的命令模式（rm -rf /、sudo、shutdown 等）
      - 如果命令包含这些模式，直接返回错误信息，不执行
      - 这是防止 Agent 破坏系统的第一道防线（虽然简单，但有效）

    执行机制：
      - subprocess.run() 在子进程中执行命令
      - shell=True: 命令作为完整 shell 字符串执行（支持管道、重定向等）
      - cwd=os.getcwd(): 在当前工作目录执行
      - capture_output=True: 捕获 stdout 和 stderr
      - text=True: 输出以字符串而非字节返回
      - timeout=120: 命令最多执行 120 秒，防止死循环或长时间阻塞

    输出处理：
      - 合并 stdout + stderr（很多工具的关键信息在 stderr 中，如 warning、progress）
      - 截断到 50000 字符（防止超长输出撑爆 token budget）
      - 如果没有任何输出，返回 "(no output)"（让模型知道命令确实执行了但没产出）
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# ============================================================
# 消息格式转换函数
# ============================================================

def assistant_message_to_dict(message) -> dict:
    """
    将 OpenAI SDK 返回的 ChatCompletionMessage 对象转换为 dict，
    以便追加到 messages 列表中用于后续 API 调用。

    为什么需要转换？
      - OpenAI SDK 返回的 message 是一个 Pydantic 对象（ChatCompletionMessage）
      - 但 client.chat.completions.create() 的 messages 参数需要 list[dict]
      - 所以必须把对象转成 dict 格式

    转换规则：
      - role: 始终是 "assistant"（模型回复的角色）
      - content: 模型的文本回复，可能是 None（纯工具调用时没有文本内容）
        → 用 `or ""` 处理 None 的情况，确保 content 字段始终存在
      - tool_calls: 如果模型想调用工具，这个字段会存在
        → 把每个 tool_call 拆解成 dict，保留 id、type、function(name + arguments)
        → id 是关键：工具结果消息必须通过 tool_call_id 与对应的 tool_call 关联

    最终格式示例（有工具调用时）：
      {
        "role": "assistant",
        "content": "",
        "tool_calls": [
          {
            "id": "call_abc123",           ← 工具调用的唯一标识
            "type": "function",
            "function": {
              "name": "bash",              ← 要调用的工具名
              "arguments": '{"command": "ls"}'  ← 参数（JSON 字符串）
            }
          }
        ]
      }
    """
    item = {
        "role": "assistant",
        "content": message.content or "",
    }
    if message.tool_calls:
        item["tool_calls"] = [{
            "id": tool_call.id,
            "type": tool_call.type,
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        } for tool_call in message.tool_calls]
    return item


def parse_tool_arguments(arguments: str) -> dict:
    """
    解析模型返回的 tool_call.function.arguments 字符串为 Python dict。

    arguments 在 OpenAI 协议中是一个 JSON 字符串（不是已解析的对象），
    例如：'{"command": "ls -la"}'

    为什么不直接 json.loads？
      - 模型有时会返回空字符串或格式错误的 JSON
      - 如果 arguments 是 None/空字符串，用 "{}" 作为默认值
      - 如果 JSON 解析失败，不抛异常，而是返回带 "_error" 键的 dict
        → 这样错误信息会作为工具结果反馈给模型，让它知道参数有问题并自行修正
      - 如果解析成功但结果不是 dict（比如模型返回了一个数组），也标记为错误
        → OpenAI function calling 的参数必须是 object（dict）

    返回值约定：
      - 正常情况：返回解析后的 dict，如 {"command": "ls"}
      - 错误情况：返回 {"_error": "错误描述"}
        → 在 agent_loop 中，如果 args 包含 "_error" 键，就把错误信息当作工具输出
    """
    try:
        value = json.loads(arguments or "{}")
    except json.JSONDecodeError as e:
        return {"_error": f"Invalid JSON tool arguments: {e}"}
    if not isinstance(value, dict):
        return {"_error": "Invalid tool arguments: expected object"}
    return value


# ============================================================
# 核心：Agent Loop（工具调用循环）
# ============================================================

def agent_loop(messages: list):
    """
    Agent 的核心循环：反复调用模型，直到模型给出最终文本回复（不再调用工具）。

    循环流程图：
      ┌─────────────────────────────────┐
      │  调用模型 (chat.completions)     │
      │  传入: system + history + tools  │
      └────────────┬────────────────────┘
                   │
                   ▼
      ┌─────────────────────────────────┐
      │  模型回复 → 转为 dict → append   │
      │  到 messages                     │
      └────────────┬────────────────────┘
                   │
                   ▼
      ┌─────────────────┐
      │  有 tool_calls？ │
      └────┬───────┬────┘
           │ No    │ Yes
           ▼       ▼
      返回(结束)  ┌─────────────────────────┐
                 │  遍历每个 tool_call:      │
                 │  解析 name + arguments    │
                 │  执行对应工具函数          │
                 │  把结果 append 到 messages │
                 │  (role="tool")            │
                 └──────────┬────────────────┘
                            │
                            ▼
                     继续循环（回到顶部）

    关键设计点：
      1. 每次调用都把完整的 messages 列表传给模型
         → 模型能看到之前所有的对话、工具调用和工具结果
         → 这让模型能基于工具输出进行多步推理

      2. system prompt 每次都重新构造（不在 messages 列表中）
         → 保证 system prompt 不被历史对话"淹没"
         → OpenAI 协议要求 system 是 messages 数组的第一个元素

      3. tool_choice="auto"
         → 让模型自行决定是否调用工具（而非强制调用或禁止调用）

      4. 工具结果用 role="tool" 的消息表示
         → 必须包含 tool_call_id，与对应的 tool_call 关联
         → OpenAI 协议要求：每个 tool_call 必须有一个匹配的 tool 消息回复

      5. 打印 response 对象（调试用途）
         → 可以看到 API 返回的原始数据（model、usage、finish_reason 等）
         → 生产环境应移除或改为 logging
    """
    while True:
        # ---- 第1步：调用模型 ----
        # messages 参数：[system] + 用户历史 + 模型历史 + 工具结果历史
        # 用 *messages 解包把列表合并成一个完整对话
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, *messages],
            tools=TOOLS,
            tool_choice="auto",  # 模型自行决定是否调用工具
            max_tokens=8000,  # 限制单次回复的 token 数（防止超长输出浪费 token）
        )

        # ---- 第2步：提取模型回复 ----
        # choices[0].message 是模型的回复消息对象
        # .content 是文本内容，.tool_calls 是工具调用列表（可能为 None）
        message = response.choices[0].message

        # ---- 第3步：把模型回复存入历史 ----
        # 必须转为 dict 格式才能追加到 messages（OpenAI API 要求 messages 是 dict 列表）
        # 这样下次循环时，模型能看到自己之前的回复（包括之前想调什么工具）
        messages.append(assistant_message_to_dict(message))

        # ---- 第4步：判断是否继续循环 ----
        # 如果模型没有 tool_calls → 说明它已经给出了最终文本回复，循环结束
        # 如果有 tool_calls → 说明模型还想执行工具，继续循环
        if not message.tool_calls:
            return

        # ---- 第5步：执行所有工具调用 ----
        # 模型可能在一轮回复中同时调用多个工具（parallel tool calls）
        # 遍历每个 tool_call，依次执行并收集结果
        for tool_call in message.tool_calls:
            name = tool_call.function.name  # 工具名称，如 "bash"
            args = parse_tool_arguments(  # 解析参数 JSON 字符串为 dict
                tool_call.function.arguments
            )

            # 根据 args 解析结果决定输出
            if "_error" in args:
                # 参数解析失败 → 把错误信息作为工具输出反馈给模型
                # 模型看到错误后会尝试修正参数重新调用
                output = args["_error"]
            elif name == "bash":
                # 调用 bash 工具 → 执行 shell 命令
                command = args.get("command", "")
                # 用黄色(\033[33m)打印执行的命令，方便用户追踪 Agent 的操作
                print(f"\033[33m$ {command}\033[0m")
                output = run_bash(command)
                # 只打印输出前 200 字符（避免刷屏），完整结果会存入 messages 供模型看
                print(output[:200])
            else:
                # 模型试图调用未定义的工具 → 返回错误信息
                # 这防止模型"幻觉"出不存在工具并陷入死循环
                output = f"Error: Unknown tool {name}"

            # ---- 第6步：把工具结果追加到 messages ----
            # OpenAI 协议要求：
            #   - role 必须是 "tool"
            #   - tool_call_id 必须与对应的 tool_call.id 匹配
            #     → 这样模型才知道"这个结果是哪个工具调用产生的"
            #   - content 是工具的输出文本
            # 不满足这些要求会导致 API 返回 400 错误
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output,
            })


# ============================================================
# 主程序入口（交互式 REPL）
# ============================================================

if __name__ == "__main__":
    # history: 保存整个对话历史（用户消息 + 模型回复 + 工具调用 + 工具结果）
    # 每次 agent_loop 调用后会往 history 中追加新消息
    # 下次用户提问时，history 已经包含了之前所有的对话上下文
    # → 模型能看到完整的对话历史，实现多轮对话 + 跨轮次的工具调用记忆
    history = []

    # REPL 循环（Read-Eval-Print Loop）
    # 持续读取用户输入，调用 agent_loop，打印结果
    while True:
        try:
            query = input("\033[36m请输入： >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            # EOFError: 用户按了 Ctrl+D（标准输入结束）
            # KeyboardInterrupt: 用户按了 Ctrl+C（中断）
            # 两种情况都意味着用户想退出，跳出 REPL 循环
            break

        # 如果用户输入空行、q 或 exit → 退出
        # 这提供了两种退出方式：Ctrl+D/Ctrl+C 或输入退出命令
        if query.strip().lower() in ("q", "exit", ""):
            break

        # 把用户输入作为 "user" 消息追加到历史
        history.append({"role": "user", "content": query})

        # 调用 agent_loop —— 可能执行多轮工具调用
        # agent_loop 会往 history 中追加：
        #   - 模型的 assistant 消息（含 tool_calls 或纯文本）
        #   - 工具的 tool 消息（工具执行结果）
        # 最终 history 的最后一个 assistant 消息就是模型的最终回复
        agent_loop(history)

        # agent_loop 结束后，history 的最后一条应该是模型的最终文本回复
        # .get("content") 安全地获取内容（如果没有 content 也不会报错）
        # 如果 content 为空（如模型只做了工具调用但最终没说任何话），就不打印
        response_content = history[-1].get("content")
        if response_content:
            print(response_content)

        # 打印空行分隔不同轮次的对话，提升可读性
        print()