#!/usr/bin/env python3
"""
s10_team-protocols_opai.py - 团队协议（OpenAI兼容版本）

这是 s10_team_protocols.py 的 OpenAI Chat Completions 兼容版本。
它保留了 s10 的协议行为：

  - 持久化队友，使用 JSONL 格式的收件箱（inbox）进行消息通信
  - 关闭（shutdown）请求/响应，通过 request_id 进行请求关联追踪
  - 计划审批（plan approval）请求/审核，通过 request_id 进行请求关联追踪

OpenAI 协议注意事项：
  - 工具定义格式为 {"type": "function", "function": {...}}
  - 助手的工具调用请求存储在 message.tool_calls 中
  - 每个工具结果以 role="tool" 的形式追加到消息列表，并携带相同的 tool_call_id
"""

from __future__ import annotations

import json  # JSON 编解码，用于消息序列化与工具参数解析
import os  # 环境变量读取与路径操作
import subprocess  # 执行 shell 命令
import sys  # 系统相关操作（标准错误输出等）
import threading  # 多线程支持，用于队友的独立运行循环和锁机制
import time  # 时间相关功能（睡眠、时间戳等）
import uuid  # 生成唯一请求 ID
from pathlib import Path  # 路径操作，比 os.path 更面向对象

from dotenv import load_dotenv  # 从 .env 文件加载环境变量

# ===== readline 配置 =====
# 尝试导入 readline 模块（仅在类 Unix 系统可用），配置终端输入行为
# 这些设置确保在交互式输入时能正确处理元字符和中文等非 ASCII 字符
try:
    import readline

    readline.parse_and_bind("set bind-tty-special-chars off")  # 关闭 TTY 特殊字符绑定，避免输入冲突
    readline.parse_and_bind("set input-meta on")  # 允许输入元字符（如中文）
    readline.parse_and_bind("set output-meta on")  # 允许输出元字符
    readline.parse_and_bind("set convert-meta off")  # 不将元字符转换为转义序列
except ImportError:
    pass  # Windows 等系统不支持 readline，跳过即可

# ===== OpenAI SDK 导入 =====
try:
    from openai import OpenAI  # OpenAI Python SDK 客户端
except ImportError:
    print("Error: openai package not installed. Run: pip install openai", file=sys.stderr)
    raise

# 加载 .env 文件中的环境变量（override=True 表示覆盖已存在的同名变量）
load_dotenv(override=True)


def env_first(*names: str) -> str | None:
    """
    依次检查多个环境变量名，返回第一个非空值。
    用于支持多个备选环境变量名（如 OPENAI_API_KEY 和其他兼容名称）。

    参数:
        *names: 多个环境变量名，按优先级顺序传入
    返回:
        第一个找到的非空环境变量值，若全部为空则返回 None
    """
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


# ===== 全局配置常量 =====
WORKDIR = Path.cwd()  # 当前工作目录，作为团队的工作空间根路径
MODEL = os.environ["MODEL_ID"]  # 必须设置的环境变量，指定使用的模型 ID
API_KEY = env_first("OPENAI_API_KEY")  # API 密钥，支持多种环境变量名
BASE_URL = env_first("OPENAI_BASE_URL")  # API 基础 URL，用于兼容其他 OpenAI 格式的服务

# 初始化 OpenAI 客户端，传入 API 密钥和基础 URL
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ===== 团队文件系统路径 =====
TEAM_DIR = WORKDIR / ".team"  # 团队元数据目录（存放配置等）
INBOX_DIR = TEAM_DIR / "inbox"  # 收件箱目录（存放各成员的 JSONL 消息文件）

# ===== LLM 请求速率限制配置 =====
# 使用线程锁和最小请求间隔来避免并发请求导致速率限制错误
LLM_REQUEST_LOCK = threading.Lock()  # 线程锁，确保请求间隔计算的原子性
LLM_LAST_REQUEST_AT = 0.0  # 上一次 LLM 请求的时间戳（单调时钟）
LLM_MIN_INTERVAL_SECONDS = float(os.getenv("OPENAI_MIN_REQUEST_INTERVAL_SECONDS", "0.8"))  # 两次请求之间的最小间隔秒数
LLM_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "4"))  # 速率限制错误时的最大重试次数
LLM_RETRY_BASE_SECONDS = float(os.getenv("OPENAI_RETRY_BASE_SECONDS", "1.0"))  # 重试的基础延迟秒数（指数退避）

# ===== 系统提示词 =====
# 定义团队领导（lead）的角色和行为约束
SYSTEM = (
    f"You are a team lead at {WORKDIR}. "
    "Manage teammates with shutdown and plan approval protocols."
)

# ===== 合法消息类型集合 =====
# 定义消息总线中允许的消息类型，用于校验发送消息的类型合法性
VALID_MSG_TYPES = {
    "message",  # 普通点对点消息
    "broadcast",  # 广播消息（发给所有队友）
    "shutdown_request",  # 关闭请求（领导发给队友）
    "shutdown_response",  # 关闭响应（队友回复领导）
    "plan_approval_response",  # 计划审批响应（队友提交计划或领导审批结果）
}

# ===== 全局请求追踪器 =====
# 用于跟踪 shutdown 请求和 plan 审批请求的状态
# key: request_id (短 UUID), value: 包含请求详情和状态的字典
shutdown_requests = {}  # 关闭请求追踪字典
plan_requests = {}  # 计划审批请求追踪字典
_tracker_lock = threading.Lock()  # 保护上述两个字典的线程锁，防止并发读写冲突


def function_tool(name: str, description: str, parameters: dict) -> dict:
    """
    构建 OpenAI function 类型的工具定义字典。

    OpenAI API 的工具格式要求：
      - type: "function"
      - function: 包含 name, description, parameters

    参数:
        name: 工具名称
        description: 工具功能描述（供 LLM 理解何时调用）
        parameters: 工具参数的 JSON Schema 定义
    返回:
        符合 OpenAI tools 格式的字典
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def object_schema(properties: dict | None = None, required: list[str] | None = None) -> dict:
    """
    构建 JSON Schema 的 object 类型定义，用于工具参数描述。

    参数:
        properties: 参数属性定义字典，key 为参数名，value 为类型描述
        required: 必需参数名列表
    返回:
        JSON Schema object 类型定义字典
    """
    schema = {"type": "object", "properties": properties or {}}
    if required:
        schema["required"] = required
    return schema


def assistant_message_to_dict(message) -> dict:
    """
    将 OpenAI API 返回的助手消息对象转换为可序列化的字典格式。

    这是必要的，因为 OpenAI SDK 返回的消息是自定义对象，
    需要转换为字典才能存入消息历史列表中。

    处理逻辑：
      - 基本字段: role, content
      - 如果消息包含工具调用（tool_calls），将其一并转换保存
        - 每个 tool_call 包含: id, type, function(name, arguments)

    参数:
        message: OpenAI API 返回的 ChatCompletionMessage 对象
    返回:
        可序列化的消息字典
    """
    item = {
        "role": "assistant",
        "content": message.content or "",
    }
    if message.tool_calls:
        # 将每个工具调用对象转换为字典格式
        item["tool_calls"] = [
            {
                "id": tool_call.id,  # 工具调用的唯一标识，用于关联 tool 角色的响应
                "type": tool_call.type,  # 类型，通常为 "function"
                "function": {
                    "name": tool_call.function.name,  # 工具名称
                    "arguments": tool_call.function.arguments,  # 工具参数（JSON 字符串）
                },
            }
            for tool_call in message.tool_calls
        ]
    return item


def parse_tool_arguments(arguments: str) -> dict:
    """
    解析工具调用参数字符串为字典。

    LLM 返回的工具参数是 JSON 字符串格式，需要解析为 Python 字典。
    如果解析失败，返回包含 _error 键的字典，后续逻辑会检测该错误标记。

    参数:
        arguments: 工具参数的 JSON 字符串（可能为空或 None）
    返回:
        解析后的参数字典，或包含 _error 的错误字典
    """
    try:
        value = json.loads(arguments or "{}")
    except json.JSONDecodeError as e:
        return {"_error": f"Invalid JSON tool arguments: {e}"}  # JSON 解析失败
    if not isinstance(value, dict):
        return {"_error": "Invalid tool arguments: expected object"}  # 参数不是对象类型
    return value


def is_rate_limit_error(error: Exception) -> bool:
    """
    判断异常是否为速率限制（rate limit）错误。

    检测方式：
      1. 检查异常的 status_code 属性是否为 429（HTTP Too Many Requests）
      2. 检查异常文本中是否包含 "429"、"rate_limit"、"limit_burst_rate" 等关键词

    参数:
        error: 捕获的异常对象
    返回:
        True 表示是速率限制错误，需要重试；False 表示其他错误
    """
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True
    text = str(error).lower()
    return "429" in text or "rate_limit" in text or "limit_burst_rate" in text


def wait_for_llm_slot():
    """
    等待 LLM 请求间隔_slot，避免并发请求过快触发速率限制。

    工作原理：
      1. 获取线程锁，确保间隔计算的原子性
      2. 计算自上次请求以来的等待时间
      3. 如果等待时间不足最小间隔，则睡眠补齐
      4. 更新最后请求时间戳

    使用单调时钟（time.monotonic）而非系统时钟，
    避免系统时间调整导致的计算异常。
    """
    global LLM_LAST_REQUEST_AT

    with LLM_REQUEST_LOCK:
        now = time.monotonic()
        # 计算还需要等待的秒数
        wait_seconds = LLM_MIN_INTERVAL_SECONDS - (now - LLM_LAST_REQUEST_AT)
        if wait_seconds > 0:
            time.sleep(wait_seconds)  # 等待补齐最小间隔
        LLM_LAST_REQUEST_AT = time.monotonic()  # 更新最后请求时间


def create_chat_completion(**kwargs):
    """
    创建 Chat Completion 请求，带有速率限制保护和重试机制。

    工作流程：
      1. 每次请求前调用 wait_for_llm_slot() 确保请求间隔
      2. 调用 OpenAI API 创建 completion
      3. 如果遇到速率限制错误，使用指数退避策略重试
         - 重试延迟 = LLM_RETRY_BASE_SECONDS * 2^attempt
         - 最多重试 LLM_MAX_RETRIES 次
      4. 非速率限制错误直接抛出

    参数:
        **kwargs: 传递给 client.chat.completions.create() 的所有参数
    返回:
        OpenAI API 的 ChatCompletion 响应对象
    """
    for attempt in range(LLM_MAX_RETRIES + 1):
        wait_for_llm_slot()  # 确保请求间隔
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            # 如果不是速率限制错误，或已达到最大重试次数，直接抛出
            if not is_rate_limit_error(e) or attempt >= LLM_MAX_RETRIES:
                raise
            # 指数退避延迟：基础延迟 * 2^当前重试次数
            delay = LLM_RETRY_BASE_SECONDS * (2 ** attempt)
            print(
                "OpenAI rate limit hit; "
                f"retrying in {delay:.1f}s ({attempt + 1}/{LLM_MAX_RETRIES})"
            )
            time.sleep(delay)  # 等待后重试
    raise RuntimeError("unreachable")  # 逻辑上不应到达此处


class MessageBus:
    """
    消息总线（MessageBus）- 团队成员间的通信系统。

    核心设计：
      - 每个成员拥有一个独立的 JSONL 收件箱文件（如 inbox/alice.jsonl）
      - 消息以 JSON 行格式追加写入，支持高效并发读写
      - read_inbox 读取后会清空文件（"drain"模式），确保消息不重复处理
      - 支持多种消息类型：普通消息、广播、关闭请求/响应、计划审批

    JSONL（JSON Lines）格式的优势：
      - 每行一条独立消息，追加写入无需重写整个文件
      - 读取时逐行解析，天然支持并发写入
      - 文件结构简单，易于调试和监控
    """

    def __init__(self, inbox_dir: Path):
        """
        初始化消息总线。

        参数:
            inbox_dir: 收件箱目录路径，会自动创建该目录
        """
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)  # 确保收件箱目录存在

    def send(
            self,
            sender: str,
            to: str,
            content: str,
            msg_type: str = "message",
            extra: dict | None = None,
    ) -> str:
        """
        向指定成员的收件箱发送一条消息。

        流程：
          1. 校验消息类型是否合法
          2. 构造消息字典（包含 type, from, content, timestamp）
          3. 合入额外字段（如 request_id, approve 等）
          4. 以 JSON 行格式追加到目标成员的收件箱文件

        参数:
            sender: 发送者名称
            to: 接收者名称
            content: 消息内容
            msg_type: 消息类型，默认为 "message"
            extra: 额外字段字典，会被合并进消息中
        返回:
            发送结果描述字符串，或错误信息
        """
        # 校验消息类型合法性
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        # 构造消息结构
        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),  # Unix 时间戳，记录消息发送时间
        }
        # 合入额外字段（如 request_id, approve, plan 等）
        if extra:
            msg.update(extra)
        # 写入目标成员的收件箱文件（追加模式）
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")  # ensure_ascii=False 保留中文等非 ASCII 字符
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        """
        读取并清空指定成员的收件箱（drain 模式）。

        "drain"模式意味着读取后文件会被清空，
        确保同一条消息不会被重复处理。

        流程：
          1. 检查收件箱文件是否存在
          2. 逐行解析 JSONL 文件，还原为消息字典列表
          3. 清空收件箱文件（写入空字符串）

        参数:
            name: 成员名称
        返回:
            消息字典列表，如果没有消息则返回空列表
        """
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []  # 收件箱不存在，返回空列表
        messages = []
        # 逐行解析 JSONL 文件
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        # 清空收件箱文件（drain 模式的关键步骤）
        inbox_path.write_text("")
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        """
        向所有队友广播消息（排除发送者自己）。

        广播实质上是逐个调用 send() 方法，
        向每个非发送者的队友发送一条 broadcast 类型的消息。

        参数:
            sender: 广播发起者名称
            content: 广播内容
            teammates: 所有队友名称列表
        返回:
            广播结果描述（如 "Broadcast to 3 teammates"）
        """
        count = 0
        for name in teammates:
            if name != sender:  # 不向自己发送广播
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


# 创建全局消息总线实例，使用 INBOX_DIR 作为收件箱目录
BUS = MessageBus(INBOX_DIR)


class TeammateManager:
    """
    队友管理器（TeammateManager）- 管理团队成员的生命周期和通信。

    核心职责：
      - 加载/保存团队配置（config.json），记录成员名称、角色、状态
      - 创建（spawn）新队友，为其启动独立线程运行 LLM 循环
      - 在队友线程中执行工具调用（bash, 文件操作, 消息通信等）
      - 管理队友状态：working（工作中）, idle（空闲）, shutdown（已关闭）

    配置文件格式 (config.json):
      {
        "team_name": "default",
        "members": [
          {"name": "alice", "role": "coder", "status": "working"},
          {"name": "bob", "role": "reviewer", "status": "idle"}
        ]
      }
    """

    def __init__(self, team_dir: Path):
        """
        初始化队友管理器。

        参数:
            team_dir: 团队元数据目录路径，存放配置文件等
        """
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)  # 确保团队目录存在
        self.config_path = self.dir / "config.json"  # 配置文件路径
        self.config = self._load_config()  # 加载团队配置
        self.threads = {}  # 成员名称 -> 线程对象的映射字典

    def _load_config(self) -> dict:
        """
        从配置文件加载团队配置。

        如果配置文件不存在，返回默认配置（空成员列表）。

        返回:
            团队配置字典
        """
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}  # 默认配置

    def _save_config(self):
        """
        将当前团队配置保存到配置文件。

        使用 JSON 格式，indent=2 便于人类阅读，
        ensure_ascii=False 保留中文等非 ASCII 字符。
        """
        self.config_path.write_text(json.dumps(self.config, indent=2, ensure_ascii=False))

    def _find_member(self, name: str) -> dict | None:
        """
        在成员列表中查找指定名称的成员。

        参数:
            name: 成员名称
        返回:
            成员配置字典，如果未找到则返回 None
        """
        for member in self.config["members"]:
            if member["name"] == name:
                return member
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        """
        创建（spawn）一个新队友，并启动其独立线程。

        工作流程：
          1. 检查成员是否已存在：
             - 如果存在且状态为 idle/shutdown，更新其角色和状态为 working
             - 如果存在且状态为 working，返回错误（不允许重复 spawn）
             - 如果不存在，创建新成员记录
          2. 保存配置文件
          3. 创建守护线程，运行队友的 LLM 循环（_teammate_loop）
          4. 记录线程对象并启动

        参数:
            name: 队友名称（唯一标识）
            role: 队友角色描述（如 "coder", "reviewer"）
            prompt: 初始任务提示（发给队友的第一条用户消息）
        返回:
            操作结果描述字符串
        """
        member = self._find_member(name)
        if member:
            # 成员已存在，检查其当前状态
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"  # 正在工作，不允许重复 spawn
            # 从空闲或已关闭状态恢复，更新角色和状态
            member["status"] = "working"
            member["role"] = role
        else:
            # 新成员，创建配置记录
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save_config()  # 保存更新后的配置

        # 创建守护线程（daemon=True，主线程退出时自动终止）
        thread = threading.Thread(
            target=self._teammate_loop,  # 线程运行目标函数
            args=(name, role, prompt),  # 传递参数
            daemon=True,  # 守护线程，主程序退出时自动终止
        )
        self.threads[name] = thread  # 记录线程对象
        thread.start()  # 启动线程
        return f"Spawned '{name}' (role: {role})"

    def _teammate_loop(self, name: str, role: str, prompt: str):
        """
        队友的 LLM 运行循环 - 在独立线程中持续运行。

        这是队友的核心执行逻辑，工作流程：
          1. 构造系统提示词（告知队友自己的名称、角色和工作目录）
          2. 初始化消息列表（包含初始任务提示）
          3. 进入循环（最多 50 次迭代，防止无限循环）：
             a. 读取收件箱消息，转为用户消息追加到历史
             b. 如果收到关闭批准（shutdown approved），标记退出
             c. 调用 LLM API 生成回复
             d. 将助手回复追加到消息历史
             e. 如果没有工具调用，结束循环（队友完成任务）
             f. 执行每个工具调用，将结果追加到消息历史
             g. 如果是 shutdown_response 且批准关闭，标记退出
          4. 循环结束后，更新成员状态（shutdown 或 idle）

        参数:
            name: 队友名称
            role: 队友角色
            prompt: 初始任务提示
        """
        # 构造系统提示词，告知队友自身身份和行为规范
        sys_prompt = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            "Submit plans via plan_approval before major work. "  # 重要工作前需提交计划审批
            "Respond to shutdown_request with shutdown_response."  # 收到关闭请求需回复
        )
        # 初始化消息历史，包含初始任务提示
        messages = [{"role": "user", "content": prompt}]
        # 获取队友可用的工具列表
        tools = self._teammate_tools()
        # 退出标志：当收到关闭批准时设为 True
        should_exit = False

        # 最多 50 次迭代，防止无限消耗 LLM API
        for _ in range(50):
            # 读取收件箱，将每条消息转为用户消息追加到历史
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                messages.append({"role": "user", "content": json.dumps(msg, ensure_ascii=False)})
            # 如果已标记退出，跳出循环
            if should_exit:
                break

            # 调用 LLM API 生成回复
            try:
                response = create_chat_completion(
                    model=MODEL,
                    messages=[{"role": "system", "content": sys_prompt}, *messages],  # 系统提示词 + 消息历史
                    tools=tools,  # 队友可用的工具列表
                    tool_choice="auto",  # LLM 自行决定是否调用工具
                    max_tokens=8000,  # 最大生成 token 数
                )
            except Exception as e:
                print(f"  [{name}] Error: {e}")
                break  # API 错误，终止循环

            # 获取助手回复消息
            message = response.choices[0].message
            # 将助手回复转为字典并追加到消息历史
            messages.append(assistant_message_to_dict(message))

            # 如果没有工具调用，说明队友已完成任务，结束循环
            if not message.tool_calls:
                break

            # 逐个执行工具调用
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name  # 工具名称
                args = parse_tool_arguments(tool_call.function.arguments)  # 解析工具参数
                output = self._exec(name, tool_name, args)  # 执行工具，获取结果
                # 打印工具调用摘要（限制120字符，避免刷屏）
                print(f"  [{name}] {tool_name}: {str(output)[:120]}")
                # 将工具结果追加到消息历史（role="tool"，携带 tool_call_id 关联）
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,  # 关联到对应的工具调用请求
                        "content": str(output)[:50000],  # 工具结果内容（限制长度避免超出 API 限制）
                    },
                )
                # 如果工具是 shutdown_response 且批准关闭，标记退出
                if (
                        tool_name == "shutdown_response"
                        and "_error" not in args  # 参数解析无错误
                        and args.get("approve")  # 队友同意关闭
                ):
                    should_exit = True

        # 循环结束后，更新成员状态
        member = self._find_member(name)
        if member:
            # 如果因关闭批准退出，状态为 shutdown；否则为 idle
            member["status"] = "shutdown" if should_exit else "idle"
            self._save_config()  # 保存状态变更

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        """
        执行单个工具调用 - 队友工具调用的统一执行入口。

        根据工具名称分发到对应的执行函数：
          - bash: 执行 shell 命令
          - read_file: 读取文件内容
          - write_file: 写入文件内容
          - edit_file: 替换文件中的文本
          - send_message: 发送消息给其他队友
          - read_inbox: 读取自己的收件箱
          - shutdown_response: 回复关闭请求
          - plan_approval: 提交计划审批

        参数:
            sender: 执行者名称（用于消息发送的来源标识）
            tool_name: 工具名称
            args: 工具参数字典
        返回:
            工具执行结果字符串
        """
        # 如果参数解析有错误，直接返回错误信息
        if "_error" in args:
            return args["_error"]
        try:
            # ===== 文件与命令操作工具 =====
            if tool_name == "bash":
                return run_bash(args["command"])  # 执行 shell 命令
            if tool_name == "read_file":
                return run_read(args["path"], args.get("limit"))  # 读取文件，可选限制行数
            if tool_name == "write_file":
                return run_write(args["path"], args["content"])  # 写入文件
            if tool_name == "edit_file":
                return run_edit(args["path"], args["old_text"], args["new_text"])  # 替换文件文本

            # ===== 消息通信工具 =====
            if tool_name == "send_message":
                # 发送消息给指定队友
                return BUS.send(
                    sender,
                    args["to"],
                    args["content"],
                    args.get("msg_type", "message"),  # 默认消息类型为 "message"
                )
            if tool_name == "read_inbox":
                # 读取自己的收件箱（drain 模式）
                return json.dumps(BUS.read_inbox(sender), indent=2, ensure_ascii=False)

            # ===== 关闭协议工具 =====
            if tool_name == "shutdown_response":
                """
                回复关闭请求的处理逻辑：
                  1. 从参数获取 request_id 和 approve（是否同意关闭）
                  2. 更新全局 shutdown_requests 追踪器中的请求状态
                  3. 通过消息总线发送 shutdown_response 给领导（lead）
                  4. 返回操作结果
                """
                req_id = args["request_id"]  # 关闭请求 ID，用于关联原始请求
                approve = args["approve"]  # 是否同意关闭（True/False）
                # 在线程锁保护下更新请求追踪器状态
                with _tracker_lock:
                    if req_id in shutdown_requests:
                        shutdown_requests[req_id]["status"] = (
                            "approved" if approve else "rejected"
                        )
                # 通过消息总线发送关闭响应给领导
                BUS.send(
                    sender,
                    "lead",
                    args.get("reason", ""),  # 关闭原因或拒绝原因（可选）
                    "shutdown_response",
                    {"request_id": req_id, "approve": approve},  # 额外字段
                )
                return f"Shutdown {'approved' if approve else 'rejected'}"

            # ===== 计划审批工具 =====
            if tool_name == "plan_approval":
                """
                提交计划审批的处理逻辑：
                  1. 从参数获取计划文本
                  2. 生成唯一的 request_id（短 UUID）
                  3. 在全局 plan_requests 追踪器中记录请求
                  4. 通过消息总线发送 plan_approval_response 给领导
                  5. 返回提交结果（包含 request_id）
                """
                plan_text = args.get("plan", "")  # 计划文本内容
                req_id = str(uuid.uuid4())[:8]  # 生成短 UUID 作为请求 ID
                # 在线程锁保护下记录请求
                with _tracker_lock:
                    plan_requests[req_id] = {
                        "from": sender,  # 请求来源
                        "plan": plan_text,  # 计划内容
                        "status": "pending",  # 初始状态为待审批
                    }
                # 通过消息总线发送计划审批请求给领导
                BUS.send(
                    sender,
                    "lead",
                    plan_text,
                    "plan_approval_response",
                    {"request_id": req_id, "plan": plan_text},  # 额外字段
                )
                return f"Plan submitted (request_id={req_id}). Waiting for lead approval."

            # 未知工具
            return f"Unknown tool: {tool_name}"
        except KeyError as e:
            # 缺少必需参数
            return f"Error: Missing required argument {e}"
        except Exception as e:
            # 其他执行错误
            return f"Error: {e}"

    def _teammate_tools(self) -> list:
        """
        获取队友可用的工具列表。

        队友的工具 = 基础工具（bash, 文件操作）+ 团队协作工具
        团队协作工具包括：
          - send_message: 发送消息给其他队友
          - read_inbox: 读取自己的收件箱
          - shutdown_response: 回复关闭请求
          - plan_approval: 提交计划审批

        返回:
            工具定义列表（符合 OpenAI tools 格式）
        """
        return BASE_TOOLS + [
            # 发送消息工具 - 队友之间通信的基础
            function_tool(
                "send_message",
                "Send message to a teammate.",
                object_schema(
                    {
                        "to": {"type": "string"},  # 目标队友名称
                        "content": {"type": "string"},  # 消息内容
                        "msg_type": {"type": "string", "enum": sorted(VALID_MSG_TYPES)},  # 消息类型枚举
                    },
                    ["to", "content"],  # 必需参数
                ),
            ),
            # 读取收件箱工具 - drain 模式，读取后清空
            function_tool("read_inbox", "Read and drain your inbox.", object_schema()),
            # 关闭响应工具 - 队友回复领导的关闭请求
            function_tool(
                "shutdown_response",
                "Respond to a shutdown request. Approve to shut down, reject to keep working.",
                object_schema(
                    {
                        "request_id": {"type": "string"},  # 关闭请求 ID
                        "approve": {"type": "boolean"},  # 是否同意关闭
                        "reason": {"type": "string"},  # 原因说明（可选）
                    },
                    ["request_id", "approve"],  # 必需参数
                ),
            ),
            # 计划审批工具 - 队友提交计划等待领导审批
            function_tool(
                "plan_approval",
                "Submit a plan for lead approval. Provide plan text.",
                object_schema({"plan": {"type": "string"}}, ["plan"]),  # 必需参数：计划文本
            ),
        ]

    def list_all(self) -> str:
        """
        列出所有团队成员的信息。

        格式示例：
          Team: default
            alice (coder): working
            bob (reviewer): idle

        返回:
            格式化的团队成员列表字符串，如果没有成员则返回 "No teammates."
        """
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for member in self.config["members"]:
            lines.append(f"  {member['name']} ({member['role']}): {member['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        """
        获取所有成员名称列表。

        用于广播消息时获取所有目标成员。

        返回:
            成员名称字符串列表
        """
        return [member["name"] for member in self.config["members"]]


def safe_path(path: str) -> Path:
    """
    安全路径验证 - 防止路径逃逸（path traversal）攻击。

    工作原理：
      1. 将传入路径与工作目录拼接并解析为绝对路径
      2. 检查解析后的路径是否仍在工作目录下
      3. 如果路径逃逸出工作目录，抛出 ValueError

    例如：path="../../etc/passwd" 会被拒绝，
    因为 resolve() 后的路径不在 WORKDIR 下。

    参数:
        path: 相对路径字符串
    返回:
        解析后的安全绝对路径
    异常:
        ValueError: 如果路径逃逸出工作目录
    """
    resolved = (WORKDIR / path).resolve()  # 拼接工作目录并解析为绝对路径
    if not resolved.is_relative_to(WORKDIR):  # 检查是否在工作目录下
        raise ValueError(f"Path escapes workspace: {path}")
    return resolved


def run_bash(command: str) -> str:
    """
    执行 shell 命令 - 队友的工具之一。

    安全措施：
      1. 检查命令是否包含危险关键词（rm -rf /, sudo, shutdown, reboot 等）
      2. 如果包含危险关键词，直接拒绝执行
      3. 命令在工作目录（WORKDIR）下执行
      4. 设置 120 秒超时，防止长时间运行

    参数:
        command: shell 命令字符串
    返回:
        命令输出（stdout + stderr），或错误信息
    """
    # 危险命令关键词列表
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous):
        return "Error: Dangerous command blocked"  # 拒绝危险命令
    try:
        result = subprocess.run(
            command,
            shell=True,  # 通过 shell 执行（支持管道等特性）
            cwd=WORKDIR,  # 在工作目录下执行
            capture_output=True,  # 捕获 stdout 和 stderr
            text=True,  # 以文本模式返回输出
            timeout=120,  # 120 秒超时限制
        )
        # 合并 stdout 和 stderr，并截取前 50000 字符
        output = (result.stdout + result.stderr).strip()
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"  # 超时错误
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"  # 其他执行错误


def run_read(path: str, limit: int | None = None) -> str:
    """
    读取文件内容 - 队友的工具之一。

    流程：
      1. 验证路径安全性（调用 safe_path）
      2. 读取文件全部内容
      3. 如果指定了 limit，只返回前 limit 行
      4. 截取前 50000 字符返回

    参数:
        path: 相对文件路径
        limit: 可选，返回的最大行数
    返回:
        文件内容字符串，或错误信息
    """
    try:
        lines = safe_path(path).read_text().splitlines()  # 安全路径 + 逐行读取
        if limit and limit < len(lines):
            # 截取前 limit 行，并标注剩余行数
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]  # 截取前 50000 字符
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """
    写入文件内容 - 队友的工具之一。

    流程：
      1. 验证路径安全性（调用 safe_path）
      2. 自动创建父目录（如果不存在）
      3. 写入内容到文件
      4. 返回写入字节数

    参数:
        path: 相对文件路径
        content: 要写入的内容
    返回:
        操作结果描述，或错误信息
    """
    try:
        target = safe_path(path)  # 验证路径安全性
        target.parent.mkdir(parents=True, exist_ok=True)  # 自动创建父目录
        target.write_text(content)  # 写入内容
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    编辑文件内容 - 队友的工具之一，进行精确文本替换。

    流程：
      1. 验证路径安全性（调用 safe_path）
      2. 读取文件内容
      3. 检查 old_text 是否存在于文件中
      4. 如果不存在，返回错误
      5. 替换第一次出现的 old_text 为 new_text
      6. 写回文件

    注意：只替换第一次出现（replace(..., 1)），
    避免意外替换多处相同文本。

    参数:
        path: 相对文件路径
        old_text: 要替换的原始文本
        new_text: 替换后的新文本
    返回:
        操作结果描述，或错误信息
    """
    try:
        target = safe_path(path)  # 验证路径安全性
        content = target.read_text()  # 读取当前文件内容
        if old_text not in content:
            return f"Error: Text not found in {path}"  # 未找到要替换的文本
        # 只替换第一次出现的匹配文本
        target.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def handle_shutdown_request(teammate: str) -> str:
    """
    处理关闭请求 - 领导向队友发送关闭请求。

    流程：
      1. 生成唯一的 request_id（短 UUID）
      2. 在全局 shutdown_requests 追踪器中记录请求（状态为 pending）
      3. 通过消息总线发送 shutdown_request 给目标队友
      4. 返回请求 ID 和状态

    参数:
        teammate: 目标队友名称
    返回:
        操作结果描述（包含 request_id）
    """
    req_id = str(uuid.uuid4())[:8]  # 生成短 UUID 作为请求 ID
    # 在线程锁保护下记录关闭请求
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    # 通过消息总线发送关闭请求给目标队友
    BUS.send(
        "lead",
        teammate,
        "Please shut down gracefully.",  # 优雅关闭请求内容
        "shutdown_request",
        {"request_id": req_id},  # 额外字段：请求 ID
    )
    return f"Shutdown request {req_id} sent to '{teammate}' (status: pending)"


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    """
    处理计划审批 - 领导审批或驳回队友提交的计划。

    流程：
      1. 在全局 plan_requests 追踪器中查找请求
      2. 如果找不到，返回错误
      3. 更新请求状态为 approved 或 rejected
      4. 通过消息总线发送审批结果给提交计划的队友
      5. 返回审批结果

    参数:
        request_id: 计划请求 ID（由队友提交时生成）
        approve: 是否批准计划
        feedback: 审批反馈意见（可选）
    返回:
        审批结果描述字符串，或错误信息
    """
    # 查找计划请求
    with _tracker_lock:
        req = plan_requests.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"  # 未找到请求

    # 更新请求状态
    with _tracker_lock:
        req["status"] = "approved" if approve else "rejected"
    # 通过消息总线发送审批结果给队友
    BUS.send(
        "lead",
        req["from"],  # 发给提交计划的队友
        feedback,  # 审批反馈内容
        "plan_approval_response",
        {"request_id": request_id, "approve": approve, "feedback": feedback},  # 额外字段
    )
    return f"Plan {req['status']} for '{req['from']}'"


def check_shutdown_status(request_id: str) -> str:
    """
    查询关闭请求的状态 - 领导检查之前发送的关闭请求是否已被处理。

    参数:
        request_id: 关闭请求 ID
    返回:
        请求状态的 JSON 字符串，如果请求不存在则返回错误信息
    """
    with _tracker_lock:
        return json.dumps(
            shutdown_requests.get(request_id, {"error": "not found"}),
            ensure_ascii=False,
        )


# ===== 基础工具定义列表 =====
# 这些工具对所有角色（领导和队友）都可用
BASE_TOOLS = [
    # bash 工具 - 执行 shell 命令
    function_tool(
        "bash",
        "Run a shell command.",
        object_schema({"command": {"type": "string"}}, ["command"]),  # 必需参数：命令字符串
    ),
    # read_file 工具 - 读取文件内容
    function_tool(
        "read_file",
        "Read file contents from the current workspace.",
        object_schema(
            {"path": {"type": "string"}, "limit": {"type": "integer"}},  # path 必需，limit 可选
            ["path"],  # 必需参数
        ),
    ),
    # write_file 工具 - 写入文件内容
    function_tool(
        "write_file",
        "Write content to a file in the current workspace.",
        object_schema(
            {"path": {"type": "string"}, "content": {"type": "string"}},  # path 和 content 必需
            ["path", "content"],  # 必需参数
        ),
    ),
    # edit_file 工具 - 精确替换文件中的文本
    function_tool(
        "edit_file",
        "Replace exact text in a workspace file.",
        object_schema(
            {
                "path": {"type": "string"},  # 文件路径
                "old_text": {"type": "string"},  # 要替换的原始文本
                "new_text": {"type": "string"},  # 替换后的新文本
            },
            ["path", "old_text", "new_text"],  # 必需参数
        ),
    ),
]

# 创建全局队友管理器实例
TEAM = TeammateManager(TEAM_DIR)

# ===== 领导的工具调用处理器映射表 =====
# key: 工具名称, value: lambda 处理函数
# 领导的工具与队友的工具有所不同：
#   - 领导可以 spawn/list/broadcast/shutdown_request 等管理操作
#   - 队友可以 send_message/read_inbox/shutdown_response/plan_approval 等协作操作
TOOL_HANDLERS = {
    # ===== 基础文件与命令操作 =====
    "bash": lambda **kwargs: run_bash(kwargs["command"]),
    "read_file": lambda **kwargs: run_read(kwargs["path"], kwargs.get("limit")),
    "write_file": lambda **kwargs: run_write(kwargs["path"], kwargs["content"]),
    "edit_file": lambda **kwargs: run_edit(
        kwargs["path"],
        kwargs["old_text"],
        kwargs["new_text"],
    ),
    # ===== 团队管理操作 =====
    "spawn_teammate": lambda **kwargs: TEAM.spawn(
        kwargs["name"],
        kwargs["role"],
        kwargs["prompt"],
    ),
    "list_teammates": lambda **kwargs: TEAM.list_all(),
    # ===== 消息通信操作 =====
    "send_message": lambda **kwargs: BUS.send(
        "lead",  # 领导作为发送者
        kwargs["to"],
        kwargs["content"],
        kwargs.get("msg_type", "message"),
    ),
    "read_inbox": lambda **kwargs: json.dumps(
        BUS.read_inbox("lead"),  # 读取领导的收件箱
        indent=2,
        ensure_ascii=False,
    ),
    "broadcast": lambda **kwargs: BUS.broadcast(
        "lead",  # 领导作为广播发起者
        kwargs["content"],
        TEAM.member_names(),  # 广播给所有成员
    ),
    # ===== 关闭协议操作 =====
    "shutdown_request": lambda **kwargs: handle_shutdown_request(kwargs["teammate"]),
    "shutdown_response": lambda **kwargs: check_shutdown_status(
        kwargs.get("request_id", ""),  # 查询关闭请求状态
    ),
    # ===== 计划审批操作 =====
    "plan_approval": lambda **kwargs: handle_plan_review(
        kwargs["request_id"],
        kwargs["approve"],
        kwargs.get("feedback", ""),
    ),
}

# ===== 领导可用的完整工具定义列表 =====
# 领导的工具 = 基础工具 + 团队管理工具
TOOLS = BASE_TOOLS + [
    # spawn_teammate 工具 - 创建新队友
    function_tool(
        "spawn_teammate",
        "Spawn a persistent teammate.",
        object_schema(
            {
                "name": {"type": "string"},  # 队友名称（唯一标识）
                "role": {"type": "string"},  # 队友角色描述
                "prompt": {"type": "string"},  # 初始任务提示
            },
            ["name", "role", "prompt"],  # 必需参数
        ),
    ),
    # list_teammates 工具 - 列出所有队友
    function_tool("list_teammates", "List all teammates.", object_schema()),
    # send_message 工具 - 发送消息给指定队友
    function_tool(
        "send_message",
        "Send a message to a teammate.",
        object_schema(
            {
                "to": {"type": "string"},  # 目标队友名称
                "content": {"type": "string"},  # 消息内容
                "msg_type": {"type": "string", "enum": sorted(VALID_MSG_TYPES)},  # 消息类型枚举
            },
            ["to", "content"],  # 必需参数
        ),
    ),
    # read_inbox 工具 - 读取领导的收件箱
    function_tool("read_inbox", "Read and drain the lead's inbox.", object_schema()),
    # broadcast 工具 - 向所有队友广播消息
    function_tool(
        "broadcast",
        "Send a message to all teammates.",
        object_schema({"content": {"type": "string"}}, ["content"]),  # 必需参数：广播内容
    ),
    # shutdown_request 工具 - 请求队友关闭
    function_tool(
        "shutdown_request",
        "Request a teammate to shut down gracefully. Returns a request_id for tracking.",
        object_schema({"teammate": {"type": "string"}}, ["teammate"]),  # 必需参数：队友名称
    ),
    # shutdown_response 工具 - 查询关闭请求状态
    function_tool(
        "shutdown_response",
        "Check the status of a shutdown request by request_id.",
        object_schema({"request_id": {"type": "string"}}, ["request_id"]),  # 必需参数：请求 ID
    ),
    # plan_approval 工具 - 审批或驳回队友的计划
    function_tool(
        "plan_approval",
        "Approve or reject a teammate's plan. Provide request_id + approve + optional feedback.",
        object_schema(
            {
                "request_id": {"type": "string"},  # 计划请求 ID
                "approve": {"type": "boolean"},  # 是否批准
                "feedback": {"type": "string"},  # 审批反馈（可选）
            },
            ["request_id", "approve"],  # 必需参数
        ),
    ),
]


def call_tool(name: str, args: dict) -> str:
    """
    调用指定工具 - 领导侧的工具调用统一入口。

    流程：
      1. 如果参数解析有错误（args 中包含 _error），直接返回错误
      2. 从 TOOL_HANDLERS 映射表中查找对应处理器
      3. 如果未找到，返回"未知工具"错误
      4. 执行处理器，传入参数
      5. 捕获 KeyError（缺少必需参数）和其他异常

    参数:
        name: 工具名称
        args: 工具参数字典
    返回:
        工具执行结果字符串，或错误信息
    """
    # 参数解析错误检测
    if "_error" in args:
        return args["_error"]

    # 查找工具处理器
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return f"Error: Unknown tool {name}"

    try:
        return handler(**args)  # 执行处理器
    except KeyError as e:
        return f"Error: Missing required argument {e}"  # 缺少必需参数
    except Exception as e:
        return f"Error: {e}"  # 其他执行错误


def agent_loop(messages: list):
    """
    领导的 LLM 运行循环 - 处理一次用户查询的完整交互。

    工作流程：
      1. 读取领导的收件箱，如果有消息则转为用户消息追加到历史
      2. 调用 LLM API 生成回复（带上系统提示词和工具定义）
      3. 将助手回复追加到消息历史
      4. 如果没有工具调用，循环结束（领导完成回答）
      5. 逐个执行工具调用：
         a. 解析工具名称和参数
         b. 调用 call_tool() 执行工具
         c. 打印工具调用摘要（黄色标记）
         d. 将工具结果追加到消息历史
      6. 回到步骤 1 继续（因为工具执行可能产生新的收件箱消息）

    注意：这是一个循环而非递归，领导会持续执行直到
    LLM 不再调用工具（即产生纯文本回复）。

    参数:
        messages: 消息历史列表，会在循环中被修改（追加新消息）
    """
    while True:
        # 读取领导的收件箱消息
        inbox = BUS.read_inbox("lead")
        if inbox:
            # 将收件箱消息转为用户消息，用 XML 标签包裹便于 LLM 理解
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

        # 调用 LLM API 生成回复
        response = create_chat_completion(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, *messages],  # 系统提示词 + 消息历史
            tools=TOOLS,  # 领导可用的工具列表
            tool_choice="auto",  # LLM 自行决定是否调用工具
            max_tokens=8000,  # 最大生成 token 数
        )

        # 获取助手回复消息并追加到历史
        message = response.choices[0].message
        messages.append(assistant_message_to_dict(message))

        # 如果没有工具调用，领导完成回答，结束循环
        if not message.tool_calls:
            return

        # 逐个执行工具调用
        for tool_call in message.tool_calls:
            name = tool_call.function.name  # 工具名称
            args = parse_tool_arguments(tool_call.function.arguments)  # 解析工具参数
            output = call_tool(name, args)  # 执行工具调用

            # 打印工具调用摘要（黄色高亮）
            print(f"\033[33m> {name}\033[0m")  # \033[33m = 黄色, \033[0m = 恢复默认色
            print(str(output)[:200])  # 限制 200 字符显示

            # 将工具结果追加到消息历史
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,  # 关联到对应的工具调用请求
                    "content": str(output)[:50000],  # 工具结果内容（截取前 50000 字符）
                },
            )


# ===== 主程序入口 =====
if __name__ == "__main__":
    # 初始化空的对话历史列表
    history = []
    # 交互式循环：持续接受用户输入
    while True:
        try:
            # 读取用户输入，显示青色提示符
            # \033[36m = 青色, \033[0m = 恢复默认色
            query = input("\033[36ms10-openai >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break  # Ctrl+C 或 Ctrl+D 退出

        # ===== 内置命令处理 =====
        # 空输入、q、exit 命令：退出程序
        if query.strip().lower() in ("q", "exit", ""):
            break
        # /team 命令：显示团队成员列表
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        # /inbox 命令：显示领导的收件箱消息（注意：这会 drain 收件箱）
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2, ensure_ascii=False))
            continue

        # ===== LLM 交互处理 =====
        # 将用户输入追加到对话历史
        history.append({"role": "user", "content": query})
        # 运行领导的 LLM 循环
        agent_loop(history)

        # 显示领导的最终文本回复
        # history[-1] 是最后一次追加的消息，可能是助手回复或工具结果
        # 如果最后一条是助手回复且有文本内容，则显示
        response_content = history[-1].get("content")
        if response_content:
            print(response_content)

        print()  # 输出一个空行，增加可读性