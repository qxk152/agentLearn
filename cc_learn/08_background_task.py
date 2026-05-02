#!/usr/bin/env python3
"""
08_back_ground_tasks_openai.py - Background Tasks (OpenAI-compatible version)

Run commands in background threads. A notification queue is drained before each
LLM call so completed background results are injected into the conversation.

OpenAI protocol notes:
  - Tool definitions use the Chat Completions function-calling format.
  - Assistant tool requests are stored in message.tool_calls.
  - Tool results are appended as role="tool" messages with matching tool_call_id.
"""

import json
import os
import subprocess
import sys
import threading
import uuid
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

SYSTEM = f"You are a coding agent at {WORKDIR}. Use background_run for long-running commands."


# -- BackgroundManager: 后台线程执行 + 完成通知队列 --
class BackgroundManager:
    def __init__(self):
        # tasks 保存所有后台任务的当前状态；check_background 会读取这里。
        self.tasks = {}  # task_id -> {status, result, command}
        # _notification_queue 只保存“已完成但还没有注入对话”的结果摘要。
        self._notification_queue = []  # completed task results
        # 后台线程和主 agent loop 会同时访问通知队列，用锁避免并发读写冲突。
        self._lock = threading.Lock()

    def run(self, command: str) -> str:
        """启动后台线程并立即返回 task_id，不阻塞 agent loop。"""
        # uuid 截短成 8 位，便于模型和用户在终端里引用任务。
        task_id = str(uuid.uuid4())[:8]
        # 先登记 running 状态，再启动线程；这样线程启动后可以被立即查询。
        self.tasks[task_id] = {"status": "running", "result": None, "command": command}
        thread = threading.Thread(
            target=self._execute,
            args=(task_id, command),
            # daemon=True 让主程序退出时不被后台线程卡住。
            daemon=True,
        )
        thread.start()
        return f"Background task {task_id} started: {command[:80]}"

    def _execute(self, task_id: str, command: str):
        """后台线程入口：执行命令、记录结果、把完成通知放入队列。"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=WORKDIR,
                capture_output=True,
                text=True,
                timeout=300,
            )
            # 后台任务可能输出很多内容；完整任务结果最多保留 50000 字符。
            output = (result.stdout + result.stderr).strip()[:50000]
            status = "completed"
        except subprocess.TimeoutExpired:
            output = "Error: Timeout (300s)"
            status = "timeout"
        except Exception as e:
            output = f"Error: {e}"
            status = "error"

        # tasks 中保存完整结果，供 check_background 查询。
        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = output or "(no output)"
        with self._lock:
            # 通知队列中只放短摘要，避免下一轮上下文突然膨胀。
            self._notification_queue.append(
                {
                    "task_id": task_id,
                    "status": status,
                    "command": command[:80],
                    "result": (output or "(no output)")[:500],
                },
            )

    def check(self, task_id: str = None) -> str:
        """查询单个后台任务；不传 task_id 时列出所有任务。"""
        if task_id:
            task = self.tasks.get(task_id)
            if not task:
                return f"Error: Unknown task {task_id}"
            return (
                f"[{task['status']}] {task['command'][:60]}\n"
                f"{task.get('result') or '(running)'}"
            )

        lines = []
        for task_id, task in self.tasks.items():
            lines.append(f"{task_id}: [{task['status']}] {task['command'][:60]}")
        return "\n".join(lines) if lines else "No background tasks."

    def drain_notifications(self) -> list:
        """取出并清空待注入的后台完成通知。"""
        with self._lock:
            # list(...) 复制当前通知，随后清空原队列，保证每个完成结果只注入一次。
            notifications = list(self._notification_queue)
            self._notification_queue.clear()
        return notifications


BG = BackgroundManager()


# -- Tool implementations --
def safe_path(path: str) -> Path:
    # 所有文件工具都限制在当前工作区内，防止模型访问或写入外部路径。
    resolved = (WORKDIR / path).resolve()
    if not resolved.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path}")
    return resolved


def run_bash(command: str) -> str:
    # 阻塞式命令适合短任务；长任务应使用 background_run。
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
        # limit 让模型可以只读取文件前 N 行，减少上下文占用。
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        target = safe_path(path)
        # 写文件前自动创建父目录，和前几章 openai 示例保持一致。
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        target = safe_path(path)
        content = target.read_text()
        # edit 是精确替换第一处匹配，避免模型做模糊修改造成不可预期变更。
        if old_text not in content:
            return f"Error: Text not found in {path}"
        target.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    # OpenAI tool_call 解析出来后，会按 function.name 分发到这里。
    "bash": lambda **kwargs: run_bash(kwargs["command"]),
    "read_file": lambda **kwargs: run_read(kwargs["path"], kwargs.get("limit")),
    "write_file": lambda **kwargs: run_write(kwargs["path"], kwargs["content"]),
    "edit_file": lambda **kwargs: run_edit(
        kwargs["path"],
        kwargs["old_text"],
        kwargs["new_text"],
    ),
    "background_run": lambda **kwargs: BG.run(kwargs["command"]),
    "check_background": lambda **kwargs: BG.check(kwargs.get("task_id")),
}


TOOLS = [
    # OpenAI Chat Completions 的工具 schema：外层 type=function，参数放在 function.parameters。
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command (blocking).",
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
            "name": "background_run",
            "description": "Run command in background thread. Returns task_id immediately.",
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
            "name": "check_background",
            "description": "Check background task status. Omit task_id to list all.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                },
            },
        },
    },
]


def assistant_message_to_dict(message) -> dict:
    """把 OpenAI SDK 的 assistant 消息对象转换成可再次发送的 dict。"""
    item = {
        "role": "assistant",
        "content": message.content or "",
    }
    if message.tool_calls:
        # 必须保留 tool_calls；后续 role="tool" 消息要用 tool_call_id 与它们配对。
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
    """解析 OpenAI tool_call 中的 JSON 字符串参数。"""
    try:
        value = json.loads(arguments or "{}")
    except json.JSONDecodeError as e:
        return {"_error": f"Invalid JSON tool arguments: {e}"}
    if not isinstance(value, dict):
        return {"_error": "Invalid tool arguments: expected object"}
    return value


def call_tool(name: str, args: dict) -> str:
    """执行工具，并把异常转换成模型能读懂的文本结果。"""
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


def inject_background_notifications(messages: list):
    """每次模型调用前，把已完成后台任务的结果注入对话。"""
    notifications = BG.drain_notifications()
    if not notifications or not messages:
        return

    # 通知作为普通 user 消息进入历史，让模型下一轮可以自然读取结果并继续决策。
    notification_text = "\n".join(
        f"[bg:{item['task_id']}] {item['status']}: {item['result']}"
        for item in notifications
    )
    messages.append(
        {
            "role": "user",
            "content": f"<background-results>\n{notification_text}\n</background-results>",
        },
    )


def agent_loop(messages: list):
    while True:
        # 关键机制：模型不等待 background_run 完成；完成结果在下一次 LLM 调用前补进上下文。
        inject_background_notifications(messages)

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, *messages],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=8000,
        )

        message = response.choices[0].message
        # assistant 消息必须先入历史，再追加 tool 结果，顺序要符合 OpenAI 协议。
        messages.append(assistant_message_to_dict(message))

        if not message.tool_calls:
            return

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = parse_tool_arguments(tool_call.function.arguments)
            output = call_tool(name, args)

            print(f"\033[33m> {name}\033[0m")
            print(str(output)[:200])

            # 每个 tool_call 必须有一个 role="tool" 消息用同一个 tool_call_id 回填。
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
            query = input("\033[36ms08-openai >> \033[0m")
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
