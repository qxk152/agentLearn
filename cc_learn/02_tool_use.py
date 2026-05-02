#!/usr/bin/env python3
"""
s02_tool_use_openai.py - 带工具调用能力的 Agent Loop（OpenAI 兼容版本）

这个文件把两个脚本的能力合在一起：
  - s01_agent_loop_openai.py 里的 OpenAI 兼容 Agent 循环
  - s02_tool_use.py 里的工具集合：bash、read_file、write_file、edit_file

OpenAI 协议里的关键点：
  - 工具用 type="function" 的格式声明
  - 模型回复里可能带 tool_calls，表示它想调用工具
  - 工具执行结果要作为 role="tool" 的消息追加回 messages
  - 每条工具结果必须带 tool_call_id，用来对应模型刚才发出的那次 tool_call
"""

import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import readline

    # 改善交互式终端的输入体验，尤其是 macOS/libedit 下的 UTF-8 输入。
    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    pass

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Run: pip install openai", file=sys.stderr)
    raise


load_dotenv(override=True)


def env_first(*names: str) -> str | None:
    """按顺序读取多个环境变量，返回第一个非空值。"""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


# 当前 Agent 运行的工作区。所有文件类工具都会被限制在这个目录下面。
WORKDIR = Path.cwd()

# MODEL_ID 是必需配置；没有它就直接报错，避免误用默认模型。
MODEL = os.environ["MODEL_ID"]

# OpenAI SDK 支持所有 OpenAI 兼容接口，只要配置 key 和 base_url 即可。
API_KEY = env_first("OPENAI_API_KEY")
BASE_URL = env_first("OPENAI_BASE_URL")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 系统提示词告诉模型：你是一个 coding agent，并且可以使用工具直接行动。
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


def safe_path(path: str) -> Path:
    """把用户传入的路径解析为工作区内路径，并阻止访问工作区外部文件。"""
    resolved = (WORKDIR / path).resolve()
    if not resolved.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path}")
    return resolved


def run_bash(command: str) -> str:
    """执行 shell 命令，并把 stdout/stderr 合并后返回给模型。"""
    # 这里只做非常基础的危险命令拦截。真实生产环境需要更严格的沙箱。
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous):
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        # 工具输出会进入模型上下文，所以截断超长输出，防止占满 token。
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def run_read(path: str, limit: int | None = None) -> str:
    """读取工作区内文件；limit 可限制最多返回多少行。"""
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """写入工作区内文件；父目录不存在时会自动创建。"""
    try:
        target = safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """在工作区内文件中替换第一处精确匹配的文本。"""
    try:
        target = safe_path(path)
        content = target.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        target.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# 工具分发表：模型给出工具名和参数后，agent_loop 通过这里找到实际 Python 函数。
TOOL_HANDLERS = {
    "bash": lambda **kwargs: run_bash(kwargs["command"]),
    "read_file": lambda **kwargs: run_read(kwargs["path"], kwargs.get("limit")),
    "write_file": lambda **kwargs: run_write(kwargs["path"], kwargs["content"]),
    "edit_file": lambda **kwargs: run_edit(
        kwargs["path"],
        kwargs["old_text"],
        kwargs["new_text"],
    ),
}


# OpenAI function calling 格式的工具定义。
# 模型会根据 name、description 和 parameters 自动决定什么时候调用哪个工具。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents from the current workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the current workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in a workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
]


def assistant_message_to_dict(message) -> dict:
    """把 OpenAI SDK 返回的消息对象转换成可再次传给 API 的 dict。"""
    item = {
        "role": "assistant",
        "content": message.content or "",
    }
    if message.tool_calls:
        # tool_calls 必须保留下来；下一轮 API 请求需要看到模型刚才调用了哪些工具。
        item["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
    return item


def parse_tool_arguments(arguments: str) -> dict:
    """把 tool_call.function.arguments 这段 JSON 字符串解析成 dict。"""
    try:
        value = json.loads(arguments or "{}")
    except json.JSONDecodeError as e:
        return {"_error": f"Invalid JSON tool arguments: {e}"}
    if not isinstance(value, dict):
        return {"_error": "Invalid tool arguments: expected object"}
    return value


def call_tool(name: str, args: dict) -> str:
    """根据工具名执行对应处理函数，并把异常转换成可反馈给模型的文本。"""
    if "_error" in args:
        return args["_error"]

    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return f"Error: Unknown tool {name}"

    try:
        return handler(**args)
    except KeyError as e:
        return f"Error: Missing required argument {e}"
    except Exception as e:
        return f"Error: {e}"


def agent_loop(messages: list):
    """持续调用模型和工具，直到模型不再请求 tool_calls。"""
    while True:
        # 每轮都把 system prompt 放在最前面，再附加完整历史消息。
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, *messages],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=8000,
        )

        message = response.choices[0].message
        messages.append(assistant_message_to_dict(message))

        # 没有 tool_calls 表示模型已经给出最终回答，本轮 Agent 循环结束。
        if not message.tool_calls:
            return

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = parse_tool_arguments(tool_call.function.arguments)
            output = call_tool(name, args)

            # 打印工具名和输出片段，方便人在终端里观察 Agent 做了什么。
            print(f"\033[33m> {name}\033[0m")
            print(output[:200])

            # OpenAI 协议要求工具结果使用 role="tool"，并匹配 tool_call_id。
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": output,
                }
            )


if __name__ == "__main__":
    # 简单 REPL：每次用户输入都会进入同一个 history，实现多轮上下文。
    history = []
    while True:
        try:
            query = input("\033[36ms02-openai >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        # 输入空行、q 或 exit 时退出。
        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        agent_loop(history)

        response_content = history[-1].get("content")
        if response_content:
            print(response_content)

        print()
