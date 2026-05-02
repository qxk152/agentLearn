#!/usr/bin/env python3
# Harness: autonomy -- models that find work without being told.
"""
s11_autonomous_agents_openai.py - 自主 Agent（OpenAI 兼容版本）

本文件是 s11_autonomous_agents.py 的 OpenAI Chat Completions 版本。
s11 在 s10 “团队协议”的基础上增加了自主性：teammate 完成当前任务后不会立刻消亡，
而是进入 idle 轮询阶段，自己寻找后续工作。

整体流程：

  1. 用户在 REPL 中输入任务。
  2. lead agent 调用 OpenAI，决定是否 spawn teammate、发消息、查看任务板等。
  3. lead 通过 spawn_teammate 创建持久 teammate。
  4. 每个 teammate 在自己的线程里运行，拥有独立 messages 历史。
  5. teammate 先进入 WORK 阶段：
       - 每轮读取自己的 inbox。
       - 调用 OpenAI。
       - 执行模型请求的工具。
       - 如果模型调用 idle 工具，主动进入 IDLE 阶段。
  6. teammate 进入 IDLE 阶段后会定期轮询：
       - inbox 里是否有新消息。
       - .tasks/ 中是否有未领取、未阻塞的 pending task。
  7. 如果发现新消息，teammate 把消息注入上下文并恢复 WORK。
  8. 如果发现可领取任务，teammate 原子 claim task，然后把任务说明注入上下文并恢复 WORK。
  9. 如果 idle 超时仍没有新工作，teammate 状态变为 shutdown。

生命周期图：

    spawn
      |
      v
    WORK  <------------------------------+
      |                                  |
      | 模型调用 idle / 当前轮结束        |
      v                                  |
    IDLE -- inbox 有消息 ----------------+
      |
      +-- .tasks 有可领取任务 ------------+
      |
      +-- 超时 -> shutdown

类与模块设计：

  MessageBus
    文件消息总线。每个 teammate 一个 JSONL inbox，send 追加一行，read_inbox 读取后清空。
    它让 lead 和 teammate 通过显式消息通信，而不是共享 Python 内存对象。

  Task board helpers
    scan_unclaimed_tasks 扫描 .tasks/task_*.json。
    claim_task 用 _claim_lock 保护领取动作，避免多个 teammate 抢同一个任务。

  TeammateManager
    管理团队配置、成员状态、线程生命周期和 teammate 的 WORK/IDLE 主循环。
    它是本文件的核心运行时。

  OpenAI protocol helpers
    openai_tools 把原 Anthropic 风格工具定义转换为 OpenAI function tools。
    assistant_message_to_dict 保留 assistant.tool_calls。
    create_chat_completion 统一做请求节流和 429 重试。

OpenAI 协议要点：
  - 工具传给 OpenAI 前必须是 {"type": "function", "function": {...}}。
  - 模型请求工具调用时，assistant 消息里会出现 message.tool_calls。
  - assistant 消息必须先放回 messages，并保留 tool_calls。
  - 每个工具结果都必须作为 role="tool" 消息追加，并使用对应 tool_call_id。
    否则下一轮 OpenAI 请求会因为工具调用没有配对结果而失败。
"""

import json
import os
import subprocess
import sys
import threading
import time
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
    """按顺序读取环境变量，返回第一个非空值。"""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


# 当前工作区是所有文件工具、团队目录和任务板目录的基准路径。
WORKDIR = Path.cwd()
MODEL = os.environ["MODEL_ID"]
# 支持 OpenAI 官方 API，也支持通过 OPENAI_BASE_URL 接入兼容网关。
API_KEY = env_first("OPENAI_API_KEY")
BASE_URL = env_first("OPENAI_BASE_URL")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"
TASKS_DIR = WORKDIR / ".tasks"

# teammate 进入 idle 后的轮询策略：
# 每 POLL_INTERVAL 秒检查一次 inbox 和任务板；超过 IDLE_TIMEOUT 仍无工作则 shutdown。
POLL_INTERVAL = 5
IDLE_TIMEOUT = 60

# 多个 autonomous teammate 会在不同线程里同时调用模型。
# 这个轻量节流器用来平滑 OpenAI 请求，降低 burst rate 限流概率。
LLM_REQUEST_LOCK = threading.Lock()
LLM_LAST_REQUEST_AT = 0.0
LLM_MIN_INTERVAL_SECONDS = float(os.getenv("OPENAI_MIN_REQUEST_INTERVAL_SECONDS", "0.8"))
LLM_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "4"))
LLM_RETRY_BASE_SECONDS = float(os.getenv("OPENAI_RETRY_BASE_SECONDS", "1.0"))

SYSTEM = f"You are a team lead at {WORKDIR}. Teammates are autonomous -- they find work themselves."

# 消息类型沿用 s09/s10 的团队协议。s11 主要新增 autonomous idle/claim_task，
# 但仍然保留 shutdown 和 plan approval 的通信格式。
VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}

# -- Request trackers --
# shutdown_requests 和 plan_requests 是 request_id -> 状态 的内存索引。
# 它们负责把异步 inbox 消息重新关联回原始请求。
shutdown_requests = {}
plan_requests = {}
_tracker_lock = threading.Lock()
# 任务领取必须加锁，否则多个 teammate 可能同时看到同一个 pending task 并重复领取。
_claim_lock = threading.Lock()


def openai_tools(tools: list) -> list:
    """把 Anthropic 风格工具定义转换成 OpenAI function tools。

    仓库早期章节使用 {"name": ..., "input_schema": ...} 的工具格式。
    为了少改动 s11 原有工具列表，这里在调用 OpenAI 前做一次格式转换。
    """
    converted = []
    for tool in tools:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                },
            },
        )
    return converted


def assistant_message_to_dict(message) -> dict:
    """把 OpenAI SDK 的 assistant message 转成可再次发送的 dict。

    关键点是保留 tool_calls。OpenAI 要求后续 role="tool" 消息用 tool_call_id
    与这些 tool_calls 一一配对。
    """
    item = {"role": "assistant", "content": message.content or ""}
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
    """解析 tool_call.function.arguments。

    OpenAI 返回的 arguments 是 JSON 字符串。这里把解析错误转换成普通工具输出，
    让模型下一轮能看到错误并修正，而不是让本地程序崩掉。
    """
    try:
        value = json.loads(arguments or "{}")
    except json.JSONDecodeError as e:
        return {"_error": f"Invalid JSON tool arguments: {e}"}
    if not isinstance(value, dict):
        return {"_error": "Invalid tool arguments: expected object"}
    return value


def is_rate_limit_error(error: Exception) -> bool:
    """兼容 OpenAI 官方和兼容服务的限流错误判断。"""
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True
    text = str(error).lower()
    return "429" in text or "rate_limit" in text or "limit_burst_rate" in text


def wait_for_llm_slot():
    """全局串行化模型请求的发送时间。

    注意：这里不是锁住 agent 的上下文，只是锁住“什么时候能发请求”。
    每个 teammate 的 messages 仍然独立。
    """
    global LLM_LAST_REQUEST_AT

    with LLM_REQUEST_LOCK:
        now = time.monotonic()
        wait_seconds = LLM_MIN_INTERVAL_SECONDS - (now - LLM_LAST_REQUEST_AT)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        LLM_LAST_REQUEST_AT = time.monotonic()


def create_chat_completion(**kwargs):
    """统一调用 OpenAI，并对 429/burst rate 做指数退避重试。"""
    for attempt in range(LLM_MAX_RETRIES + 1):
        wait_for_llm_slot()
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            if not is_rate_limit_error(e) or attempt >= LLM_MAX_RETRIES:
                raise
            delay = LLM_RETRY_BASE_SECONDS * (2 ** attempt)
            print(
                "OpenAI rate limit hit; "
                f"retrying in {delay:.1f}s ({attempt + 1}/{LLM_MAX_RETRIES})"
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")


# -- MessageBus: 每个 teammate 一个 JSONL 收件箱 --
class MessageBus:
    """文件消息总线。

    设计目的：
      - 每个 agent 的上下文彼此隔离。
      - 通信必须通过显式消息完成。
      - 消息留在 .team/inbox/*.jsonl 中，便于调试和观察。

    JSONL 的每一行都是一条独立 JSON 消息。send 负责追加，read_inbox 负责读取并清空。
    """

    def __init__(self, inbox_dir: Path):
        # 收件箱目录固定在 .team/inbox 下。
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        """向指定 teammate 的 inbox 追加一条消息。"""
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            # extra 用于 request_id、approve、feedback 等协议字段。
            msg.update(extra)
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        """读取并清空指定 teammate 的 inbox。

        这是 drain 语义：消息一旦被注入 agent 上下文，就不再留在 inbox 文件里，
        避免下一轮重复读取。
        """
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        messages = []
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        inbox_path.write_text("")
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        """向除 sender 外的所有 teammate 发送 broadcast 消息。"""
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


BUS = MessageBus(INBOX_DIR)


# -- Task board scanning: autonomous teammate 的任务发现机制 --
def scan_unclaimed_tasks() -> list:
    """扫描 .tasks/，返回可以被 autonomous teammate 自动领取的任务。

    可领取条件：
      - status == pending
      - owner 为空
      - blockedBy 为空
    """
    TASKS_DIR.mkdir(exist_ok=True)
    unclaimed = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text())
        if (task.get("status") == "pending"
                and not task.get("owner")
                and not task.get("blockedBy")):
            unclaimed.append(task)
    return unclaimed


def claim_task(task_id: int, owner: str) -> str:
    """领取任务并把任务状态改成 in_progress。

    _claim_lock 保护整个 read-check-write 区间，避免两个线程同时领取同一任务。
    """
    with _claim_lock:
        path = TASKS_DIR / f"task_{task_id}.json"
        if not path.exists():
            return f"Error: Task {task_id} not found"
        task = json.loads(path.read_text())
        if task.get("owner"):
            existing_owner = task.get("owner") or "someone else"
            return f"Error: Task {task_id} has already been claimed by {existing_owner}"
        if task.get("status") != "pending":
            status = task.get("status")
            return f"Error: Task {task_id} cannot be claimed because its status is '{status}'"
        if task.get("blockedBy"):
            return f"Error: Task {task_id} is blocked by other task(s) and cannot be claimed yet"
        task["owner"] = owner
        task["status"] = "in_progress"
        path.write_text(json.dumps(task, indent=2))
    return f"Claimed task #{task_id} for {owner}"


# -- Identity re-injection after compression --
def make_identity_block(name: str, role: str, team_name: str) -> dict:
    """生成身份注入消息。

    当 teammate 在 idle 阶段自动领取新任务时，可能已经经历了较长上下文。
    这个 identity block 用来重新提醒模型自己的姓名、角色和团队，降低身份漂移。
    """
    return {
        "role": "user",
        "content": f"<identity>You are '{name}', role: {role}, team: {team_name}. Continue your work.</identity>",
    }


# -- Autonomous TeammateManager --
class TeammateManager:
    """自主 teammate 的运行时管理器。

    负责三类事情：

      1. 团队名册
         .team/config.json 保存成员 name/role/status，供 /team 和 list_teammates 使用。

      2. 线程生命周期
         spawn() 创建 daemon thread，每个 teammate 在自己的线程里运行 _loop()。

      3. 自主工作循环
         teammate 在 WORK 和 IDLE 两个阶段之间切换：
           - WORK：正常调用模型和工具。
           - IDLE：轮询 inbox 和 .tasks，发现工作后恢复 WORK。
    """

    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}

    def _load_config(self) -> dict:
        """读取团队配置；不存在时返回默认团队。"""
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        """保存团队配置。"""
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
        """按 name 查找 teammate 配置。"""
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def _set_status(self, name: str, status: str):
        """更新 teammate 状态并写回 config.json。"""
        member = self._find_member(name)
        if member:
            member["status"] = status
            self._save_config()

    def spawn(self, name: str, role: str, prompt: str) -> str:
        """创建或重新唤醒一个 autonomous teammate。"""
        member = self._find_member(name)
        if member:
            # 同名 teammate 正在工作时禁止重复启动，避免两个线程消费同一个 inbox。
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save_config()
        # daemon=True 让教学 REPL 退出时不用等待 teammate 线程。
        # 生产系统一般需要更完整的 shutdown/join 流程。
        thread = threading.Thread(
            target=self._loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _loop(self, name: str, role: str, prompt: str):
        """teammate 的主循环。

        这个函数是 s11 的核心：
          - 外层 while True 让 teammate 可以多次 WORK -> IDLE -> WORK。
          - WORK 阶段最多连续跑 50 轮工具调用，避免无限循环。
          - IDLE 阶段轮询 inbox 和任务板，发现工作后恢复 WORK。
          - IDLE 超时后自动 shutdown。
        """
        team_name = self.config["team_name"]
        sys_prompt = (
            f"You are '{name}', role: {role}, team: {team_name}, at {WORKDIR}. "
            f"Use idle tool when you have no more work. You will auto-claim new tasks."
        )
        messages = [{"role": "user", "content": prompt}]
        tools = self._teammate_tools()

        while True:
            # -- WORK PHASE: standard agent loop --
            for _ in range(50):
                # 每次模型调用前先 drain inbox，让 teammate 能响应 lead 或其他队友。
                inbox = BUS.read_inbox(name)
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        # s11 中 shutdown_request 在 teammate 侧是立即停止信号。
                        self._set_status(name, "shutdown")
                        return
                    messages.append({"role": "user", "content": json.dumps(msg)})
                try:
                    response = create_chat_completion(
                        model=MODEL,
                        messages=[{"role": "system", "content": sys_prompt}, *messages],
                        tools=openai_tools(tools),
                        tool_choice="auto",
                        max_tokens=8000,
                    )
                except Exception:
                    self._set_status(name, "idle")
                    return

                message = response.choices[0].message
                # OpenAI 要求 assistant tool_calls 先进入历史，再追加 tool 结果。
                messages.append(assistant_message_to_dict(message))
                if not message.tool_calls:
                    break

                idle_requested = False
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    args = parse_tool_arguments(tool_call.function.arguments)
                    if tool_name == "idle":
                        # idle 是 teammate 主动告诉运行时“我暂时没活了”。
                        idle_requested = True
                        output = "Entering idle phase. Will poll for new tasks."
                    else:
                        output = self._exec(name, tool_name, args)
                    print(f"  [{name}] {tool_name}: {str(output)[:120]}")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(output)[:50000],
                        },
                    )
                if idle_requested:
                    break

            # -- IDLE PHASE: poll for inbox messages and unclaimed tasks --
            self._set_status(name, "idle")
            resume = False
            polls = IDLE_TIMEOUT // max(POLL_INTERVAL, 1)
            for _ in range(polls):
                time.sleep(POLL_INTERVAL)
                inbox = BUS.read_inbox(name)
                if inbox:
                    # 有新消息就恢复 WORK，让模型根据消息继续决策。
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            return
                        messages.append({"role": "user", "content": json.dumps(msg)})
                    resume = True
                    break
                unclaimed = scan_unclaimed_tasks()
                if unclaimed:
                    # 自动领取第一个可领取任务。claim_task 内部有锁，避免重复领取。
                    task = unclaimed[0]
                    result = claim_task(task["id"], name)
                    if result.startswith("Error:"):
                        continue
                    task_prompt = (
                        f"<auto-claimed>Task #{task['id']}: {task['subject']}\n"
                        f"{task.get('description', '')}</auto-claimed>"
                    )
                    if len(messages) <= 3:
                        # 上下文很短时补身份块，强调“我是谁、我的角色是什么”。
                        messages.insert(0, make_identity_block(name, role, team_name))
                        messages.insert(1, {"role": "assistant", "content": f"I am {name}. Continuing."})
                    messages.append({"role": "user", "content": task_prompt})
                    messages.append({"role": "assistant", "content": f"Claimed task #{task['id']}. Working on it."})
                    resume = True
                    break

            if not resume:
                # idle 超时说明没有新消息也没有新任务，teammate 自行关闭。
                self._set_status(name, "shutdown")
                return
            self._set_status(name, "working")

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        """执行 teammate 工具。

        teammate 可用工具包括基础文件/命令工具、团队消息工具、协议工具、
        idle 和 claim_task。这里把所有异常前的参数解析错误也作为字符串返回给模型。
        """
        if "_error" in args:
            return args["_error"]
        if tool_name == "bash":
            return _run_bash(args["command"])
        if tool_name == "read_file":
            return _run_read(args["path"])
        if tool_name == "write_file":
            return _run_write(args["path"], args["content"])
        if tool_name == "edit_file":
            return _run_edit(args["path"], args["old_text"], args["new_text"])
        if tool_name == "send_message":
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            with _tracker_lock:
                if req_id in shutdown_requests:
                    shutdown_requests[req_id]["status"] = "approved" if args["approve"] else "rejected"
            BUS.send(
                sender, "lead", args.get("reason", ""),
                "shutdown_response", {"request_id": req_id, "approve": args["approve"]},
            )
            return f"Shutdown {'approved' if args['approve'] else 'rejected'}"
        if tool_name == "plan_approval":
            plan_text = args.get("plan", "")
            req_id = str(uuid.uuid4())[:8]
            with _tracker_lock:
                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}
            BUS.send(
                sender, "lead", plan_text, "plan_approval_response",
                {"request_id": req_id, "plan": plan_text},
            )
            return f"Plan submitted (request_id={req_id}). Waiting for approval."
        if tool_name == "claim_task":
            return claim_task(args["task_id"], sender)
        return f"Unknown tool: {tool_name}"

    def _teammate_tools(self) -> list:
        """返回 teammate 可用工具列表。

        与 lead 不同，teammate 拥有 idle 和 claim_task，用于自主进入 idle 和主动领取任务。
        """
        return [
            {"name": "bash", "description": "Run a shell command.",
             "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
            {"name": "read_file", "description": "Read file contents.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
            {"name": "write_file", "description": "Write content to file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            {"name": "edit_file", "description": "Replace exact text in file.",
             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
            {"name": "send_message", "description": "Send message to a teammate.",
             "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
            {"name": "read_inbox", "description": "Read and drain your inbox.",
             "input_schema": {"type": "object", "properties": {}}},
            {"name": "shutdown_response", "description": "Respond to a shutdown request.",
             "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "reason": {"type": "string"}}, "required": ["request_id", "approve"]}},
            {"name": "plan_approval", "description": "Submit a plan for lead approval.",
             "input_schema": {"type": "object", "properties": {"plan": {"type": "string"}}, "required": ["plan"]}},
            {"name": "idle", "description": "Signal that you have no more work. Enters idle polling phase.",
             "input_schema": {"type": "object", "properties": {}}},
            {"name": "claim_task", "description": "Claim a task from the task board by ID.",
             "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
        ]

    def list_all(self) -> str:
        """列出团队成员状态。"""
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        """返回所有 teammate 名称，供 broadcast 使用。"""
        return [m["name"] for m in self.config["members"]]


TEAM = TeammateManager(TEAM_DIR)


# -- Base tool implementations: 基础命令/文件工具 --
def _safe_path(p: str) -> Path:
    """把模型传入路径限制在当前工作区内，阻止 ../ 越界访问。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _run_bash(command: str) -> str:
    """执行短命令，返回 stdout/stderr。

    这是教学版轻量防护，只拦截少量明显危险命令。生产环境应使用更严格 sandbox。
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def _run_read(path: str, limit: int = None) -> str:
    """读取工作区内文件，可用 limit 限制行数。"""
    try:
        lines = _safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def _run_write(path: str, content: str) -> str:
    """写入工作区内文件，必要时创建父目录。"""
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def _run_edit(path: str, old_text: str, new_text: str) -> str:
    """精确替换文件中第一处 old_text。"""
    try:
        fp = _safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- Lead-specific protocol handlers: lead 侧协议工具 --
def handle_shutdown_request(teammate: str) -> str:
    """向 teammate 发送 shutdown_request，并登记 request_id。"""
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send(
        "lead", teammate, "Please shut down gracefully.",
        "shutdown_request", {"request_id": req_id},
    )
    return f"Shutdown request {req_id} sent to '{teammate}'"


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    """审批 teammate 提交的计划，并把结果发回提交者 inbox。"""
    with _tracker_lock:
        req = plan_requests.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"
    with _tracker_lock:
        req["status"] = "approved" if approve else "rejected"
    BUS.send(
        "lead", req["from"], feedback, "plan_approval_response",
        {"request_id": request_id, "approve": approve, "feedback": feedback},
    )
    return f"Plan {req['status']} for '{req['from']}'"


def _check_shutdown_status(request_id: str) -> str:
    """按 request_id 查询 shutdown 请求状态。"""
    with _tracker_lock:
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}))


# -- Lead tool dispatch (14 tools) --
# lead 可用工具分发表。OpenAI 返回 function.name 后，agent_loop 会在这里查找 handler。
TOOL_HANDLERS = {
    "bash":              lambda **kw: _run_bash(kw["command"]),
    "read_file":         lambda **kw: _run_read(kw["path"], kw.get("limit")),
    "write_file":        lambda **kw: _run_write(kw["path"], kw["content"]),
    "edit_file":         lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "spawn_teammate":    lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":    lambda **kw: TEAM.list_all(),
    "send_message":      lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":        lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":         lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
    "shutdown_request":  lambda **kw: handle_shutdown_request(kw["teammate"]),
    "shutdown_response": lambda **kw: _check_shutdown_status(kw.get("request_id", "")),
    "plan_approval":     lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
    "idle":              lambda **kw: "Lead does not idle.",
    "claim_task":        lambda **kw: claim_task(kw["task_id"], "lead"),
}

# lead 暴露给模型的工具定义。这里仍保留早期章节的 input_schema 格式，
# 调用 OpenAI 前由 openai_tools() 转换成 function.parameters。
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "spawn_teammate", "description": "Spawn an autonomous teammate.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["name", "role", "prompt"]}},
    {"name": "list_teammates", "description": "List all teammates.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "send_message", "description": "Send a message to a teammate.",
     "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}}, "required": ["to", "content"]}},
    {"name": "read_inbox", "description": "Read and drain the lead's inbox.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "broadcast", "description": "Send a message to all teammates.",
     "input_schema": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]}},
    {"name": "shutdown_request", "description": "Request a teammate to shut down.",
     "input_schema": {"type": "object", "properties": {"teammate": {"type": "string"}}, "required": ["teammate"]}},
    {"name": "shutdown_response", "description": "Check shutdown request status.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}}, "required": ["request_id"]}},
    {"name": "plan_approval", "description": "Approve or reject a teammate's plan.",
     "input_schema": {"type": "object", "properties": {"request_id": {"type": "string"}, "approve": {"type": "boolean"}, "feedback": {"type": "string"}}, "required": ["request_id", "approve"]}},
    {"name": "idle", "description": "Enter idle state (for lead -- rarely used).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "claim_task", "description": "Claim a task from the board by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
]


def agent_loop(messages: list):
    """lead 的主 agent loop。

    每轮先读取 lead inbox，把 teammate 汇报注入上下文；再调用 OpenAI。
    如果模型请求工具，就执行工具并把结果用 role="tool" 回填。
    没有 tool_calls 时表示本轮可以回复用户。
    """
    while True:
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append({
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
            })
        response = create_chat_completion(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, *messages],
            tools=openai_tools(TOOLS),
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
            handler = TOOL_HANDLERS.get(name)
            try:
                if "_error" in args:
                    output = args["_error"]
                else:
                    output = handler(**args) if handler else f"Unknown tool: {name}"
            except Exception as e:
                output = f"Error: {e}"
            print(f"> {name}:")
            print(str(output)[:200])
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(output)[:50000],
                },
            )


if __name__ == "__main__":
    # 简单 REPL：
    #   /team  查看 teammate 状态
    #   /inbox 查看 lead inbox
    #   /tasks 查看 .tasks 任务板
    # 普通输入会进入 lead agent_loop。
    history = []
    while True:
        try:
            query = input("\033[36ms11 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
            continue
        if query.strip() == "/tasks":
            TASKS_DIR.mkdir(exist_ok=True)
            for f in sorted(TASKS_DIR.glob("task_*.json")):
                t = json.loads(f.read_text())
                marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
                owner = f" @{t['owner']}" if t.get("owner") else ""
                print(f"  {marker} #{t['id']}: {t['subject']}{owner}")
            continue
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1].get("content")
        if response_content:
            print(response_content)
        print()
