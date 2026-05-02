#!/usr/bin/env python3
"""
s09_agent_team_openai.py - Agent Teams（OpenAI 兼容版本）

本文件把 s09_agent_teams.py 的 Anthropic 版本改成 OpenAI Chat Completions
版本，同时保留“团队成员通过文件收件箱协作”的核心设计。

整体模型：
  1. 主 agent 是 team lead，和用户交互，拥有 spawn_teammate/list_teammates/
     send_message/read_inbox/broadcast 等团队管理工具。
  2. 每个 teammate 是一个持久命名 agent。它被 spawn 后会在单独线程里运行，
     拥有自己的 messages 历史、system prompt 和工具列表。
  3. teammate 之间不直接共享对话上下文，而是通过 .team/inbox/*.jsonl 传递消息。
     每个成员一个 JSONL 文件，写入是追加，读取时会 drain 并清空。
  4. .team/config.json 保存 team_name、成员名、角色和当前状态；程序重启后还能
     看到已有成员记录。

OpenAI 协议要点：
  - 工具定义必须是 {"type": "function", "function": {...}}。
  - 模型请求工具调用时，assistant 消息里会出现 message.tool_calls。
  - assistant 消息必须先放回 messages，并保留完整 tool_calls。
  - 每个工具执行结果都必须作为 role="tool" 消息追加，并使用对应的
    tool_call_id。否则下一轮请求会缺少工具调用配对关系。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
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
# OpenAI SDK 支持自定义 base_url，方便连接 OpenAI 兼容网关或本地代理。
# 这里没有强制要求 OPENAI_API_KEY 必须存在；某些兼容服务可能不校验 key。
API_KEY = env_first("OPENAI_API_KEY")
BASE_URL = env_first("OPENAI_BASE_URL")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"

# 多个 teammate 是线程并发运行的，如果它们同时调用 OpenAI，很容易触发
# limit_burst_rate。这里用一个进程内的轻量节流器把请求平滑发出。
LLM_REQUEST_LOCK = threading.Lock()
LLM_LAST_REQUEST_AT = 0.0
LLM_MIN_INTERVAL_SECONDS = float(os.getenv("OPENAI_MIN_REQUEST_INTERVAL_SECONDS", "0.8"))
LLM_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "4"))
LLM_RETRY_BASE_SECONDS = float(os.getenv("OPENAI_RETRY_BASE_SECONDS", "1.0"))

# lead 的 system prompt 只描述团队协作能力；具体工作由用户消息和工具调用驱动。
SYSTEM = f"You are a team lead at {WORKDIR}. Spawn teammates and communicate via inboxes."

# s10 会继续扩展 shutdown/approval 协议。这里先声明完整消息类型，
# 但当前文件主要处理普通 message 和 broadcast。
VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}


def function_tool(name: str, description: str, parameters: dict) -> dict:
    """构造一个 OpenAI Chat Completions function tool schema。"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def object_schema(properties: dict | None = None, required: list[str] | None = None) -> dict:
    """生成工具参数的 JSON Schema，减少每个工具定义里的重复样板代码。"""
    schema = {
        "type": "object",
        "properties": properties or {},
    }
    if required:
        schema["required"] = required
    return schema


def assistant_message_to_dict(message) -> dict:
    """把 OpenAI SDK 的 assistant 消息对象转换成可继续发送的 dict。

    关键点：如果 assistant 发起了工具调用，tool_calls 不能丢。OpenAI 下一轮
    请求会校验消息序列：每个 assistant.tool_calls 都必须有后续 role="tool"
    消息用相同 tool_call_id 回填结果。
    """
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
    """解析 OpenAI tool_call.function.arguments 里的 JSON 字符串。

    模型返回的 arguments 是字符串，不是 dict。这里统一解析并把 JSON 错误转换
    成工具输出文本，让 agent 能在下一轮读到错误并自行修正参数。
    """
    try:
        value = json.loads(arguments or "{}")
    except json.JSONDecodeError as e:
        return {"_error": f"Invalid JSON tool arguments: {e}"}
    if not isinstance(value, dict):
        return {"_error": "Invalid tool arguments: expected object"}
    return value


def is_rate_limit_error(error: Exception) -> bool:
    """判断异常是否是 OpenAI/兼容服务的限流错误。

    不同 OpenAI 兼容服务的异常类型不完全一致：有的会带 status_code=429，
    有的只把 limit_burst_rate 放在字符串里。这里用保守的文本/属性检查兼容两类。
    """
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True

    text = str(error).lower()
    return (
        "429" in text
        or "rate_limit" in text
        or "limit_burst_rate" in text
    )


def wait_for_llm_slot():
    """串行化并平滑 OpenAI 请求，避免多个线程瞬间打爆 burst rate。

    这个锁只保护“何时发请求”的时间窗口，不保护 agent 的 messages。
    每个 teammate 的 messages 仍然是独立的。
    """
    global LLM_LAST_REQUEST_AT

    with LLM_REQUEST_LOCK:
        now = time.monotonic()
        wait_seconds = LLM_MIN_INTERVAL_SECONDS - (now - LLM_LAST_REQUEST_AT)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        LLM_LAST_REQUEST_AT = time.monotonic()


def create_chat_completion(**kwargs):
    """调用 OpenAI Chat Completions，并对 429/burst rate 做退避重试。

    统一入口的好处是 lead 和所有 teammate 共用同一套节流/重试策略。否则每个线程
    各自直接调用 SDK，会在 spawn 多个 teammate 时同时发请求。
    """
    for attempt in range(LLM_MAX_RETRIES + 1):
        wait_for_llm_slot()
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            if not is_rate_limit_error(e) or attempt >= LLM_MAX_RETRIES:
                raise
            delay = LLM_RETRY_BASE_SECONDS * (2 ** attempt)
            print(f"OpenAI rate limit hit; retrying in {delay:.1f}s ({attempt + 1}/{LLM_MAX_RETRIES})")
            time.sleep(delay)

    raise RuntimeError("unreachable")


# -- MessageBus: 每个 teammate 一个 JSONL 收件箱 --
class MessageBus:
    def __init__(self, inbox_dir: Path):
        # 收件箱目录固定在工作区 .team/inbox 下，避免散落到项目其他位置。
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(
        self,
        sender: str,
        to: str,
        content: str,
        msg_type: str = "message",
        extra: dict | None = None,
    ) -> str:
        # msg_type 做白名单校验，防止模型随意发明协议字段导致后续 agent 无法理解。
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            # extra 预留给后续章节扩展，比如 shutdown approval、plan approval 等。
            msg.update(extra)
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:
            # 使用 JSONL 的原因：追加写入简单，多个消息天然按行分隔，
            # 即使程序中途退出，也比维护一个大 JSON 数组更不容易损坏整份数据。
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        """读取并清空指定成员的收件箱。

        drain 语义很重要：消息一旦被读入某个 agent 的上下文，就不再保留在文件里，
        避免每轮循环重复注入同一批消息。
        """
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        messages = []
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        # 清空文件表示这些消息已经交付给对应 agent。
        inbox_path.write_text("")
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        """向除 sender 以外的所有 teammate 发送 broadcast 消息。"""
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


BUS = MessageBus(INBOX_DIR)


# -- TeammateManager: 用 config.json 管理持久命名 teammate --
class TeammateManager:
    def __init__(self, team_dir: Path):
        # .team/config.json 保存成员列表和状态，in-memory threads 只保存本次进程里的线程。
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}

    def _load_config(self) -> dict:
        # 没有 config 时创建一个最小团队配置。后续 spawn 会把成员写进去。
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        # ensure_ascii=False 让中文角色/状态说明能原样保存，便于教学和调试。
        self.config_path.write_text(json.dumps(self.config, indent=2, ensure_ascii=False))

    def _find_member(self, name: str) -> dict | None:
        for member in self.config["members"]:
            if member["name"] == name:
                return member
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        """创建或唤醒一个 teammate，并在线程中启动它自己的 agent loop。"""
        member = self._find_member(name)
        if member:
            # 同名成员已经在工作时不允许再次 spawn，避免两个线程同时消费同一个 inbox。
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save_config()

        # daemon=True 表示主程序退出时不等待 teammate 线程；这符合教学示例的轻量设计。
        # 真正生产系统通常需要显式 shutdown/cleanup 协议。
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _teammate_loop(self, name: str, role: str, prompt: str):
        """teammate 的独立 agent loop。

        每个 teammate 拥有自己的 messages 列表，因此不会把 lead 的完整历史带进来。
        它只通过 prompt、自己的工具结果、以及 inbox 消息更新上下文。
        """
        sys_prompt = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"Use send_message to communicate. Complete your task."
        )
        messages = [{"role": "user", "content": prompt}]
        tools = self._teammate_tools()

        # 设置最大轮数，防止模型持续调用工具导致线程无限运行。
        for _ in range(50):
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                # inbox 消息作为普通 user 消息注入 teammate 上下文。
                # 这里用 JSON 保留 from/type/timestamp 等结构化字段。
                messages.append({"role": "user", "content": json.dumps(msg, ensure_ascii=False)})
            try:
                response = create_chat_completion(
                    model=MODEL,
                    messages=[{"role": "system", "content": sys_prompt}, *messages],
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=8000,
                )
            except Exception as e:
                print(f"  [{name}] Error: {e}")
                break

            message = response.choices[0].message
            # 先保存 assistant 消息，再追加工具结果，顺序必须符合 OpenAI 协议。
            messages.append(assistant_message_to_dict(message))

            # 没有工具调用表示模型已经给出最终回复，本 teammate 本轮任务结束。
            if not message.tool_calls:
                break

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                args = parse_tool_arguments(tool_call.function.arguments)
                output = self._exec(name, tool_name, args)
                print(f"  [{name}] {tool_name}: {str(output)[:120]}")
                # 每个 tool_call 都必须回填一个 role="tool" 消息，且 ID 要一一对应。
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(output)[:50000],
                    },
                )

        member = self._find_member(name)
        if member and member["status"] != "shutdown":
            # 当前文件只实现“完成任务后回到 idle”。shutdown 状态预留给 s10 协议。
            member["status"] = "idle"
            self._save_config()

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        """执行 teammate 可用工具，并把异常转换成模型能读懂的字符串。"""
        if "_error" in args:
            return args["_error"]
        try:
            if tool_name == "bash":
                return run_bash(args["command"])
            if tool_name == "read_file":
                return run_read(args["path"], args.get("limit"))
            if tool_name == "write_file":
                return run_write(args["path"], args["content"])
            if tool_name == "edit_file":
                return run_edit(args["path"], args["old_text"], args["new_text"])
            if tool_name == "send_message":
                return BUS.send(
                    sender,
                    args["to"],
                    args["content"],
                    args.get("msg_type", "message"),
                )
            if tool_name == "read_inbox":
                return json.dumps(BUS.read_inbox(sender), indent=2, ensure_ascii=False)
            return f"Unknown tool: {tool_name}"
        except KeyError as e:
            return f"Error: Missing required argument {e}"
        except Exception as e:
            return f"Error: {e}"

    def _teammate_tools(self) -> list:
        """teammate 工具集 = 基础文件/命令工具 + 团队通信工具。

        teammate 不能 spawn 其他 teammate，也不能 broadcast 给全队；这样可以控制
        示例复杂度，避免子成员无限扩张团队。
        """
        return BASE_TOOLS + [
            function_tool(
                "send_message",
                "Send message to a teammate.",
                object_schema(
                    {
                        "to": {"type": "string"},
                        "content": {"type": "string"},
                        "msg_type": {
                            "type": "string",
                            "enum": sorted(VALID_MSG_TYPES),
                        },
                    },
                    ["to", "content"],
                ),
            ),
            function_tool(
                "read_inbox",
                "Read and drain your inbox.",
                object_schema(),
            ),
        ]

    def list_all(self) -> str:
        """返回当前团队成员列表，供 /team 命令和 list_teammates 工具使用。"""
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for member in self.config["members"]:
            lines.append(f"  {member['name']} ({member['role']}): {member['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        """只返回成员名，broadcast 工具用它枚举收件人。"""
        return [member["name"] for member in self.config["members"]]


# -- Base tool implementations: 基础命令/文件工具 --
def safe_path(path: str) -> Path:
    """把用户/模型给出的相对路径限制在当前工作区内。

    这个保护用于 read/write/edit，避免模型通过 ../ 访问或修改工作区外的文件。
    """
    resolved = (WORKDIR / path).resolve()
    if not resolved.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {path}")
    return resolved


def run_bash(command: str) -> str:
    """执行短命令并返回 stdout/stderr。

    这是教学示例里的简化版 shell 工具：只做非常基础的危险命令拦截，真实系统需要
    更严格的 sandbox、权限审批、超时和审计。
    """
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
    """读取工作区内文件，可用 limit 限制行数以减少上下文占用。"""
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """写入工作区内文件；父目录不存在时自动创建。"""
    try:
        target = safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """精确替换文件中第一处 old_text。

    使用精确替换可以降低模型“模糊编辑”带来的误改风险；找不到文本时直接返回错误。
    """
    try:
        target = safe_path(path)
        content = target.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        target.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


BASE_TOOLS = [
    # lead 和 teammate 共用的基础工具。OpenAI schema 的 parameters 字段就是 JSON Schema。
    function_tool(
        "bash",
        "Run a shell command.",
        object_schema({"command": {"type": "string"}}, ["command"]),
    ),
    function_tool(
        "read_file",
        "Read file contents from the current workspace.",
        object_schema(
            {
                "path": {"type": "string"},
                "limit": {"type": "integer"},
            },
            ["path"],
        ),
    ),
    function_tool(
        "write_file",
        "Write content to a file in the current workspace.",
        object_schema(
            {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            ["path", "content"],
        ),
    ),
    function_tool(
        "edit_file",
        "Replace exact text in a workspace file.",
        object_schema(
            {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            ["path", "old_text", "new_text"],
        ),
    ),
]


TEAM = TeammateManager(TEAM_DIR)


TOOL_HANDLERS = {
    # lead 可用工具的分发表。OpenAI 返回 function.name 后，agent_loop 会查这里执行。
    "bash": lambda **kwargs: run_bash(kwargs["command"]),
    "read_file": lambda **kwargs: run_read(kwargs["path"], kwargs.get("limit")),
    "write_file": lambda **kwargs: run_write(kwargs["path"], kwargs["content"]),
    "edit_file": lambda **kwargs: run_edit(
        kwargs["path"],
        kwargs["old_text"],
        kwargs["new_text"],
    ),
    "spawn_teammate": lambda **kwargs: TEAM.spawn(
        kwargs["name"],
        kwargs["role"],
        kwargs["prompt"],
    ),
    "list_teammates": lambda **kwargs: TEAM.list_all(),
    "send_message": lambda **kwargs: BUS.send(
        "lead",
        kwargs["to"],
        kwargs["content"],
        kwargs.get("msg_type", "message"),
    ),
    "read_inbox": lambda **kwargs: json.dumps(
        BUS.read_inbox("lead"),
        indent=2,
        ensure_ascii=False,
    ),
    "broadcast": lambda **kwargs: BUS.broadcast(
        "lead",
        kwargs["content"],
        TEAM.member_names(),
    ),
}


TOOLS = BASE_TOOLS + [
    # lead 比 teammate 多团队管理能力：可以 spawn、list、broadcast。
    function_tool(
        "spawn_teammate",
        "Spawn a persistent teammate that runs in its own thread.",
        object_schema(
            {
                "name": {"type": "string"},
                "role": {"type": "string"},
                "prompt": {"type": "string"},
            },
            ["name", "role", "prompt"],
        ),
    ),
    function_tool(
        "list_teammates",
        "List all teammates with name, role, status.",
        object_schema(),
    ),
    function_tool(
        "send_message",
        "Send a message to a teammate's inbox.",
        object_schema(
            {
                "to": {"type": "string"},
                "content": {"type": "string"},
                "msg_type": {
                    "type": "string",
                    "enum": sorted(VALID_MSG_TYPES),
                },
            },
            ["to", "content"],
        ),
    ),
    function_tool(
        "read_inbox",
        "Read and drain the lead's inbox.",
        object_schema(),
    ),
    function_tool(
        "broadcast",
        "Send a message to all teammates.",
        object_schema({"content": {"type": "string"}}, ["content"]),
    ),
]


def call_tool(name: str, args: dict) -> str:
    """执行 lead 工具，并把错误转换成模型可读文本。

    不把异常抛出到 agent_loop，是为了让模型能在下一轮看到“缺少参数/未知工具”等
    问题，并尝试修正工具调用。
    """
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
    """lead 的主 agent loop。

    每轮先 drain lead 的 inbox，把 teammate 发来的消息注入上下文；然后调用 OpenAI。
    如果模型请求工具，就执行工具并把结果按 tool_call_id 回填；如果没有工具调用，
    说明模型已经给出面向用户的最终回复，本轮结束。
    """
    while True:
        inbox = BUS.read_inbox("lead")
        if inbox:
            # lead 的 inbox 结果用 XML-ish 包裹，便于模型识别这是一批团队消息。
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "<inbox>"
                        f"{json.dumps(inbox, indent=2, ensure_ascii=False)}"
                        "</inbox>"
                    ),
                },
            )

        response = create_chat_completion(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, *messages],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=8000,
        )

        message = response.choices[0].message
        # assistant 消息必须保存在历史里；如果丢掉 tool_calls，后续 tool 消息会失配。
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
                    "content": str(output)[:50000],
                },
            )


if __name__ == "__main__":
    # 简单 REPL：用户每输入一条消息，就让 lead agent 循环到不再调用工具为止。
    # /team 和 /inbox 是本地调试命令，不经过模型。
    history = []
    while True:
        try:
            query = input("\033[36ms09-openai >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2, ensure_ascii=False))
            continue

        history.append({"role": "user", "content": query})
        agent_loop(history)

        response_content = history[-1].get("content")
        if response_content:
            print(response_content)

        print()
