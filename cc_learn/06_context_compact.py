#!/usr/bin/env python3
"""
s06_context_compact_openai.py - Compact（OpenAI 兼容版本）

整体流程：
  1. 每次调用模型前，先执行 micro_compact(messages)。
  2. micro_compact 会压缩较旧的 tool 消息内容，保留最近几个工具结果。
  3. 如果估算 token 数超过阈值，就触发 auto_compact(messages)。
  4. auto_compact 先把完整 messages 保存到 .transcripts/。
  5. 再调用模型生成会话摘要。
  6. 最后用一条包含摘要的 user message 替换整个历史。
  7. 模型也可以主动调用 compact 工具，触发手动压缩。

三层压缩：
  - Layer 1: micro_compact，每轮静默执行，压缩旧工具结果。
  - Layer 2: auto_compact，超过阈值自动保存 transcript 并总结。
  - Layer 3: compact tool，模型主动请求立即压缩。

OpenAI 协议差异：
  - Anthropic 的工具结果在 user content block 里，type="tool_result"。
  - OpenAI 的工具结果是独立消息，role="tool"。
  - OpenAI 工具结果通过 tool_call_id 对应 assistant message 里的 tool_calls。
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    import readline

    readline.parse_and_bind("set bind-tty-special-chars off")
    readline.parse_and_bind("set input-meta on")
    readline.parse_and_bind("set output-meta on")
    readline.parse_and_bind("set convert-meta off")
except ImportError:
    pass

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

SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."

THRESHOLD = 50000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 3
PRESERVE_RESULT_TOOLS = {"read_file"}


def estimate_tokens(messages: list) -> int:
    """粗略估算 token 数：按 4 个字符约等于 1 个 token。"""
    return len(str(messages)) // 4


def micro_compact(messages: list) -> list:
    """压缩较旧的 OpenAI tool 消息，降低上下文体积。"""
    tool_results = []
    for msg_idx, msg in enumerate(messages):
        if msg.get("role") == "tool":
            tool_results.append((msg_idx, msg))

    if len(tool_results) <= KEEP_RECENT:
        return messages

    # OpenAI 的工具名不在 role="tool" 消息里，需要从之前 assistant.tool_calls 反查。
    tool_name_map = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tool_call in msg.get("tool_calls", []) or []:
            tool_id = tool_call.get("id")
            function = tool_call.get("function", {})
            if tool_id:
                tool_name_map[tool_id] = function.get("name", "unknown")

    # 只压缩较旧的工具结果，最近 KEEP_RECENT 个保留原文。
    to_clear = tool_results[:-KEEP_RECENT]
    for _, result in to_clear:
        content = result.get("content", "")
        if not isinstance(content, str) or len(content) <= 100:
            continue

        tool_call_id = result.get("tool_call_id", "")
        tool_name = tool_name_map.get(tool_call_id, "unknown")
        if tool_name in PRESERVE_RESULT_TOOLS:
            continue

        result["content"] = f"[Previous: used {tool_name}]"

    return messages


def auto_compact(messages: list) -> list:
    """保存完整 transcript，调用模型总结，再用摘要替换历史消息。"""
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as file:
        for msg in messages:
            file.write(json.dumps(msg, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")

    conversation_text = json.dumps(messages, default=str)[-80000:]
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize this conversation for continuity. Include: "
                    "1) What was accomplished, 2) Current state, 3) Key decisions made. "
                    "Be concise but preserve critical details.\n\n"
                    + conversation_text
                ),
            },
        ],
        max_tokens=2000,
    )

    summary = response.choices[0].message.content or "No summary generated."
    return [
        {
            "role": "user",
            "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}",
        },
    ]


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
    "compact": lambda **kwargs: "Manual compression requested.",
}


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
    {
        "type": "function",
        "function": {
            "name": "compact",
            "description": "Trigger manual conversation compression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "What to preserve in the summary",
                    },
                },
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


def call_tool(name: str, args: dict) -> str:
    """执行工具并把错误转换成模型可读文本。"""
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
    """循环调用模型和工具，并在需要时压缩上下文。"""
    while True:
        micro_compact(messages)

        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, *messages],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=8000,
        )

        message = response.choices[0].message
        messages.append(assistant_message_to_dict(message))

        if not message.tool_calls:
            return

        manual_compact = False
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = parse_tool_arguments(tool_call.function.arguments)

            if name == "compact" and "_error" not in args:
                manual_compact = True
                output = "Compressing..."
            else:
                output = call_tool(name, args)

            print(f"\033[33m> {name}\033[0m")
            print(str(output)[:200])

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(output),
                },
            )

        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)
            return


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms06-openai >> \033[0m")
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
