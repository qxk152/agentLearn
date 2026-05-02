#!/usr/bin/env python3
"""
s04_subagent_openai.py - Subagents（OpenAI 兼容版本）

整体流程：
  1. 用户向父 agent 提出任务。
  2. 父 agent 正常使用 bash/read_file/write_file/edit_file 处理任务。
  3. 如果父 agent 认为某个探索或子任务会污染主上下文，就调用 task 工具。
  4. task 工具内部启动 run_subagent(prompt)。
  5. 子 agent 使用全新的 sub_messages，只接收这次 prompt，不继承父 agent 历史。
  6. 子 agent 可以调用基础工具完成调查或修改，但不能再调用 task。
  7. 子 agent 停止调用工具后，最后一段文本作为 summary 返回父 agent。
  8. 父 agent 只把 summary 放进自己的 messages，子 agent 的完整上下文被丢弃。
  9. 父 agent 基于 summary 继续工作或回复用户。

这个设计的核心目的：
  - 让子任务拥有独立上下文，避免父 agent 的 messages 变得过长、过乱
  - 让父 agent 只接收压缩后的结果，而不是所有中间工具输出
  - 通过不给子 agent task 工具，避免无限递归创建子 agent

父 agent 可以通过 task 工具启动一个子 agent：
  - 子 agent 使用全新的 messages=[]，不会继承父 agent 的长上下文
  - 子 agent 和父 agent 共享同一个文件系统工作区
  - 子 agent 只能使用基础工具，不能继续调用 task 递归创建子 agent
  - 子 agent 完成后只把最终摘要返回给父 agent

OpenAI 协议要点：
  - 工具定义使用 type="function"
  - 模型请求工具调用时会在 assistant message 上产生 tool_calls
  - 工具结果使用 role="tool"，并通过 tool_call_id 对应具体调用
"""

import json
import os
import subprocess
import sys
from pathlib import Path

try:
    import readline

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
    """按顺序读取环境变量，返回第一个非空值。"""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


WORKDIR = Path.cwd()
MODEL = os.environ["MODEL_ID"]
API_KEY = env_first("OPENAI_API_KEY")
BASE_URL = env_first("OPENAI_BASE_URL")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 父 agent 的系统提示词：可以直接做事，也可以用 task 把子任务委派出去。
SYSTEM = f"You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks."

# 子 agent 的系统提示词：只完成收到的 prompt，并在结束时总结发现。
SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."


def safe_path(path: str) -> Path:
    """解析工作区内路径，并阻止文件工具访问工作区外部。"""
    resolved = (WORKDIR / path).resolve()
    if not resolved.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path}")
    return resolved


def run_bash(command: str) -> str:
    """执行 shell 命令，返回合并后的 stdout/stderr。"""
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
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def run_read(path: str, limit: int | None = None) -> str:
    """读取工作区内文件内容。"""
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """写入工作区内文件。"""
    try:
        target = safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """替换工作区内文件的第一处精确匹配文本。"""
    try:
        target = safe_path(path)
        content = target.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        target.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


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


# 子 agent 只能拿到基础工具。
# 注意这里故意不放 task，防止子 agent 继续创建孙 agent，导致上下文和流程失控。
CHILD_TOOLS = [
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


# 父 agent 的工具 = 基础工具 + task。
# task 不是普通文件工具，它会启动一个全新上下文的子 agent。
PARENT_TOOLS = CHILD_TOOLS + [
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "description": {
                        "type": "string",
                        "description": "Short description of the task",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
]


def assistant_message_to_dict(message) -> dict:
    """把 OpenAI SDK 的 assistant 消息对象转换成可复用的 dict。"""
    item = {
        "role": "assistant",
        "content": message.content or "",
    }
    if message.tool_calls:
        # 必须保留 tool_calls；下一轮请求中，OpenAI 需要看到 assistant 发起过哪些工具调用。
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
    """解析 OpenAI tool_call 里的 JSON 字符串参数。"""
    try:
        value = json.loads(arguments or "{}")
    except json.JSONDecodeError as e:
        return {"_error": f"Invalid JSON tool arguments: {e}"}
    if not isinstance(value, dict):
        return {"_error": "Invalid tool arguments: expected object"}
    return value


def call_base_tool(name: str, args: dict) -> str:
    """执行基础工具；task 不在这里处理。"""
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


def run_subagent(prompt: str) -> str:
    """用全新上下文启动子 agent，并只返回子 agent 最后的文本摘要。"""
    # 这是上下文隔离的关键：子 agent 从一个全新的消息列表开始。
    # 它看不到父 agent 之前读过的大量代码、工具输出和推理过程。
    sub_messages = [{"role": "user", "content": prompt}]
    response = None

    # 给子 agent 一个最大轮数，避免模型反复调用工具导致死循环。
    for _ in range(30):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SUBAGENT_SYSTEM}, *sub_messages],
            tools=CHILD_TOOLS,
            tool_choice="auto",
            max_tokens=8000,
        )

        message = response.choices[0].message
        # 子 agent 自己的 assistant 消息只进入 sub_messages，不进入父 agent messages。
        sub_messages.append(assistant_message_to_dict(message))

        # 没有 tool_calls 表示子 agent 已经给出最终摘要，可以结束。
        if not message.tool_calls:
            break

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = parse_tool_arguments(tool_call.function.arguments)
            output = call_base_tool(name, args)

            # 工具结果也只回写给子 agent 自己。
            # 父 agent 最后只会看到 summary，不会看到这些中间输出。
            sub_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(output)[:50000],
                },
            )

    if response is None:
        return "(no summary)"

    # 只取子 agent 最后一轮的文本内容返回给父 agent。
    # sub_messages 整体会随着函数返回被丢弃，实现“摘要返回，过程隔离”。
    final_text = response.choices[0].message.content or ""
    return final_text or "(no summary)"


def agent_loop(messages: list):
    """父 agent 循环；遇到 task 工具时启动隔离上下文的子 agent。"""
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, *messages],
            tools=PARENT_TOOLS,
            tool_choice="auto",
            max_tokens=8000,
        )

        message = response.choices[0].message
        # 父 agent 的 assistant 消息进入父 messages。
        # 如果里面有 task tool_call，后面会执行并把结果作为 tool 消息补回去。
        messages.append(assistant_message_to_dict(message))

        if not message.tool_calls:
            return

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = parse_tool_arguments(tool_call.function.arguments)

            if name == "task" and "_error" not in args:
                description = args.get("description", "subtask")
                prompt = args.get("prompt", "")
                print(f"\033[33m> task ({description}): {prompt[:80]}\033[0m")
                # task 工具的输出不是本地命令结果，而是子 agent 的最终摘要。
                output = run_subagent(prompt)
            else:
                # 非 task 的工具仍然由父 agent 在当前上下文里直接执行。
                output = call_base_tool(name, args)
                print(f"\033[33m> {name}\033[0m")

            print(str(output)[:200])

            # 无论是基础工具还是 task，返回给 OpenAI 的格式都一样：
            # role="tool" + tool_call_id + content。
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(output),
                },
            )


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms04-openai >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        agent_loop(history)

        response_content = history[-1].get("content")
        if response_content:
            print(response_content)

        print()
