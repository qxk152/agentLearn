#!/usr/bin/env python3
"""
s05_skill_loading_openai.py - Skills（OpenAI 兼容版本）

整体流程：
  1. 启动时扫描 WORKDIR/skills/**/SKILL.md。
  2. SkillLoader 只把每个 skill 的 name/description/tags 摘要放进 system prompt。
  3. 模型看到“可用 skill 列表”，但不会一开始就加载所有完整正文。
  4. 当模型需要某个专业知识时，调用 load_skill(name) 工具。
  5. Python 从内存里的 SkillLoader 取出完整 skill body。
  6. 完整 skill 内容作为 tool 结果返回给模型。
  7. 模型基于刚加载的 skill 继续完成用户任务。

这个设计的核心目的：
  - 第一层只注入 metadata，保持 system prompt 短。
  - 第二层按需加载完整 skill，避免无关知识污染上下文。
  - skill 文件仍然是普通 Markdown + YAML frontmatter，方便维护。

OpenAI 协议要点：
  - 工具定义使用 type="function"
  - 模型请求工具时在 assistant message 上产生 tool_calls
  - 工具结果使用 role="tool"，并通过 tool_call_id 对应具体调用
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml
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
SKILLS_DIR = WORKDIR / "skills"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


class SkillLoader:
    """扫描 skills 目录，并按需返回 skill 的完整正文。"""

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills = {}
        self._load_all()

    def _load_all(self):
        """加载所有 SKILL.md，但启动时只会把 metadata 放进 system prompt。"""
        if not self.skills_dir.exists():
            return

        for file_path in sorted(self.skills_dir.rglob("SKILL.md")):
            text = file_path.read_text()
            meta, body = self._parse_frontmatter(text)
            name = meta.get("name", file_path.parent.name)
            self.skills[name] = {
                "meta": meta,
                "body": body,
                "path": str(file_path),
            }

    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        """解析 SKILL.md 开头的 YAML frontmatter。"""
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not match:
            return {}, text

        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}

        return meta, match.group(2).strip()

    def get_descriptions(self) -> str:
        """第一层：返回简短 skill 描述，注入 system prompt。"""
        if not self.skills:
            return "(no skills available)"

        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "No description")
            tags = skill["meta"].get("tags", "")
            line = f"  - {name}: {desc}"
            if tags:
                line += f" [{tags}]"
            lines.append(line)
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        """第二层：返回指定 skill 的完整正文，作为 load_skill 工具结果。"""
        skill = self.skills.get(name)
        if not skill:
            available = ", ".join(self.skills.keys())
            return f"Error: Unknown skill '{name}'. Available: {available}"
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"


SKILL_LOADER = SkillLoader(SKILLS_DIR)

SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.

Skills available:
{SKILL_LOADER.get_descriptions()}"""


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
    "load_skill": lambda **kwargs: SKILL_LOADER.get_content(kwargs["name"]),
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
            "name": "load_skill",
            "description": "Load specialized knowledge by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name to load",
                    },
                },
                "required": ["name"],
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
    """循环调用模型和工具，直到模型不再请求 tool_calls。"""
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
            query = input("\033[36ms05-openai >> \033[0m")
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
