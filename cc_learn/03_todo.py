#!/usr/bin/env python3
"""
s03_todo_openai.py - TodoWrite（OpenAI 兼容版本）

这个版本把 s03_todo_write.py 的 TODO 能力移植到 OpenAI tool calling 协议：
  - 模型可以调用 bash/read_file/write_file/edit_file/todo 五个工具
  - todo 工具会把任务列表保存到 TodoManager 的内存状态里
  - 如果模型连续 3 轮工具调用都没有更新 todo，就注入提醒消息

TODO是模型自己判断当前任务进行到哪一步，然后主动调用 todo 工具，把
  它认为的任务状态写进 Python 内存。
给大模型提供任务摘要，不要忘了最初的起点
  流程是这样：

  用户提出任务
    ↓
  模型思考：这个任务有几步
    ↓
  模型调用 todo 工具，提交任务列表
    ↓
  Python 的 TodoManager.update() 校验并保存到内存
    ↓
  模型继续调用 read_file / edit_file / bash 等工具做事
    ↓
  模型根据执行结果判断哪一步完成了(TodoManager 只根据模型返回的结果全量刷新内存)
    ↓
  模型再次调用 todo 工具，更新状态
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

SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose."""


class TodoManager:
    """保存模型维护的结构化 TODO 状态。"""

    def __init__(self):
        # TODO 状态只保存在当前 Python 进程内存里；程序退出后不会持久化。
        self.items = []

    def update(self, items: list) -> str:
        """校验并整体替换 TODO 列表，返回渲染后的列表文本。"""
        # 限制 todo 数量，避免模型一次写入过长任务列表占用太多上下文。
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")

        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            # 模型传入的参数可能不是严格字符串，这里统一转成字符串并清理空白。
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()

            # 如果模型没有显式给 id，就用当前位置生成一个稳定的默认 id。
            item_id = str(item.get("id", str(i + 1)))

            # 每个任务必须有内容，否则 TODO 列表对人和模型都没有意义。
            if not text:
                raise ValueError(f"Item {item_id}: text required")

            # status 只允许三种状态，防止模型发明 started/done/running 等非协议值。
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")

            # 后面会检查 in_progress 数量，确保同一时间只跟踪一个当前任务。
            if status == "in_progress":
                in_progress_count += 1

            # 只保存清洗后的字段，丢弃模型传入的多余字段，保持状态结构简单。
            validated.append({"id": item_id, "text": text, "status": status})

        # 这个约束让 TODO 像一个单线程执行计划：当前只能有一个正在做的任务。
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")

        # update 是“整体替换”语义，不是增量追加；模型每次都提交完整 TODO 列表。
        self.items = validated
        return self.render()

    def render(self) -> str:
        """把 TODO 状态渲染成人类和模型都容易读的文本。"""
        if not self.items:
            return "No todos."

        lines = []
        for item in self.items:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")

        # 追加完成进度汇总，方便终端用户和模型快速判断当前进展。
        done = sum(1 for item in self.items if item["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


# 全局 TODO 管理器。todo 工具每次被调用时都会更新这个对象。
TODO = TodoManager()


def safe_path(path: str) -> Path:
    """解析工作区内路径，并阻止工具访问工作区外部。"""
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
    # todo 工具的实际执行入口：把模型传入的 items 交给 TodoManager 校验并保存。
    "todo": lambda **kwargs: TODO.update(kwargs["items"]),
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
            "name": "todo",
            "description": "Update task list. Track progress on multi-step tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    # items 是完整 TODO 列表；模型每次更新都应该传入所有任务。
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "text": {"type": "string"},
                                # status 的 enum 要和 TodoManager.update() 里的校验保持一致。
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "pending",
                                        "in_progress",
                                        "completed",
                                    ],
                                },
                            },
                            "required": ["id", "text", "status"],
                        },
                    },
                },
                "required": ["items"],
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
    """循环调用模型和工具；必要时提醒模型更新 TODO。"""
    # 记录模型已经连续多少轮工具调用没有更新 todo。
    rounds_since_todo = 0

    while True:
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

        # 本轮只要出现过一次 todo 工具调用，就认为模型已经同步过进度。
        used_todo = False
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = parse_tool_arguments(tool_call.function.arguments)
            output = call_tool(name, args)

            print(f"\033[33m> {name}\033[0m")
            print(str(output)[:200])

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(output),
                }
            )

            if name == "todo":
                used_todo = True

        # 调用了 todo 就清零；没调用就累加。这样可以发现模型长时间忘记维护计划。
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        if rounds_since_todo >= 3:
            # reminder 作为普通 user 消息进入历史，让模型下一轮能看到并主动更新 TODO。
            messages.append(
                {
                    "role": "user",
                    "content": "<reminder>Update your todos.</reminder>",
                }
            )


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms03-openai >> \033[0m")
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
