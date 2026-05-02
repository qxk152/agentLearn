#!/usr/bin/env python3
"""
s07_task_system_openai.py - Tasks (OpenAI-compatible version)

Tasks persist as JSON files in .tasks/ so they survive context compression.
Each task has a dependency graph (blockedBy).

OpenAI protocol notes:
  - Tool definitions use the Chat Completions function-calling format.
  - Assistant tool requests are stored in message.tool_calls.
  - Tool results are appended as role="tool" messages with matching tool_call_id.
"""

import json
import os
import subprocess
import sys
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
    """Return the first non-empty environment variable from names."""
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

TASKS_DIR = WORKDIR / ".tasks"
SYSTEM = f"You are a coding agent at {WORKDIR}. Use task tools to plan and track work."


# -- TaskManager: 带依赖图的任务 CRUD，任务以 JSON 文件持久化 --
class TaskManager:
    def __init__(self, tasks_dir: Path):
        # 所有任务都保存在 .tasks/ 目录；程序重启后仍能恢复。
        self.dir = tasks_dir
        self.dir.mkdir(exist_ok=True)
        # 新任务 ID 基于当前目录里已有的最大 task_N.json 继续递增。
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        # 从 task_1.json 这类文件名里取出数字部分，计算当前最大任务 ID。
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        return max(ids) if ids else 0

    def _load(self, task_id: int) -> dict:
        # 任务不存在时直接抛错，让工具调用把错误返回给模型。
        path = self.dir / f"task_{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        # ensure_ascii=False 让中文 subject/description 能原样写入 JSON。
        path = self.dir / f"task_{task['id']}.json"
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False))

    def create(self, subject: str, description: str = "") -> str:
        # blockedBy 是依赖列表：当前任务被这些任务阻塞，直到它们完成。
        task = {
            "id": self._next_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "blockedBy": [],
            "owner": "",
        }
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2, ensure_ascii=False)

    def get(self, task_id: int) -> str:
        return json.dumps(self._load(task_id), indent=2, ensure_ascii=False)

    def update(
        self,
        task_id: int,
        status: str = None,
        add_blocked_by: list = None,
        remove_blocked_by: list = None,
    ) -> str:
        task = self._load(task_id)
        if status:
            # 状态只允许三种，防止模型写入 done/running 等不一致值。
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            if status == "completed":
                # 一个任务完成后，自动从其他任务的 blockedBy 中移除这个 ID。
                self._clear_dependency(task_id)
        if add_blocked_by:
            # 使用 set 去重，避免同一个依赖被重复加入。
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))
        if remove_blocked_by:
            # 显式移除依赖时，只保留不在 remove_blocked_by 里的 ID。
            task["blockedBy"] = [
                x for x in task["blockedBy"] if x not in remove_blocked_by
            ]
        self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    def _clear_dependency(self, completed_id: int):
        """从所有其他任务的 blockedBy 里移除已完成任务 ID。"""
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)

    def list_all(self) -> str:
        tasks = []
        # 按任务 ID 排序，保证每次展示顺序稳定。
        files = sorted(
            self.dir.glob("task_*.json"),
            key=lambda f: int(f.stem.split("_")[1]),
        )
        for f in files:
            tasks.append(json.loads(f.read_text()))
        if not tasks:
            return "No tasks."
        lines = []
        for t in tasks:
            # 用短标记渲染状态，方便模型和终端用户快速扫描任务板。
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }.get(t["status"], "[?]")
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{blocked}")
        return "\n".join(lines)


TASKS = TaskManager(TASKS_DIR)


# -- Base tool implementations --
def safe_path(path: str) -> Path:
    resolved = (WORKDIR / path).resolve()
    if not resolved.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path}")
    return resolved


def run_bash(command: str) -> str:
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


def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        target = safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
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
    "task_create": lambda **kwargs: TASKS.create(
        kwargs["subject"],
        kwargs.get("description", ""),
    ),
    "task_update": lambda **kwargs: TASKS.update(
        kwargs["task_id"],
        kwargs.get("status"),
        kwargs.get("addBlockedBy"),
        kwargs.get("removeBlockedBy"),
    ),
    "task_list": lambda **kwargs: TASKS.list_all(),
    "task_get": lambda **kwargs: TASKS.get(kwargs["task_id"]),
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
            "name": "task_create",
            "description": "Create a new persistent task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["subject"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_update",
            "description": "Update a task's status or dependencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                    },
                    "addBlockedBy": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "removeBlockedBy": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_list",
            "description": "List all tasks with status summary.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_get",
            "description": "Get full details of a task by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer"},
                },
                "required": ["task_id"],
            },
        },
    },
]


def assistant_message_to_dict(message) -> dict:
    """Convert an OpenAI assistant message object into a reusable dict."""
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
    """Parse the JSON string held in an OpenAI tool call."""
    try:
        value = json.loads(arguments or "{}")
    except json.JSONDecodeError as e:
        return {"_error": f"Invalid JSON tool arguments: {e}"}
    if not isinstance(value, dict):
        return {"_error": "Invalid tool arguments: expected object"}
    return value


def call_tool(name: str, args: dict) -> str:
    """Run a tool handler and convert errors into model-readable text."""
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
                },
            )


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms07-openai >> \033[0m")
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
