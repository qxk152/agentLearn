# s05_skill_loading_openai.py 流程说明

`s05_skill_loading_openai.py` 的核心作用是：先只告诉模型有哪些 skill，需要时再通过 `load_skill` 工具加载完整 skill 内容。

## 整体流程

```text
程序启动
  ↓
扫描 skills/**/SKILL.md
  ↓
解析每个 SKILL.md 的 frontmatter 和正文
  ↓
把 skill 简短描述放进 system prompt
  ↓
用户提问
  ↓
模型判断是否需要某个 skill
  ↓
如果需要，模型调用 load_skill(name)
  ↓
Python 返回该 skill 的完整正文
  ↓
模型基于完整 skill 继续完成任务
```

## 1. 启动时创建 SkillLoader

```python
SKILL_LOADER = SkillLoader(SKILLS_DIR)
```

`SKILLS_DIR` 指向当前工作目录下的 `skills` 目录：

```python
SKILLS_DIR = WORKDIR / "skills"
```

创建 `SkillLoader` 时会执行：

```python
self._load_all()
```

也就是启动时先扫描所有可用的 skill。

## 2. 扫描所有 SKILL.md

```python
for file_path in sorted(self.skills_dir.rglob("SKILL.md")):
```

这行会递归查找：

```text
skills/**/SKILL.md
```

例如：

```text
skills/pdf/SKILL.md
skills/code-review/SKILL.md
```

`sorted(...)` 的作用是让加载顺序稳定。

## 3. 解析 frontmatter 和正文

```python
text = file_path.read_text()
meta, body = self._parse_frontmatter(text)
```

一个 `SKILL.md` 通常长这样：

```markdown
---
name: pdf
description: Process PDF files
tags: pdf,docs
---

# PDF Skill

完整说明...
```

解析后会得到：

```python
meta = {
    "name": "pdf",
    "description": "Process PDF files",
    "tags": "pdf,docs",
}
```

以及：

```python
body = "# PDF Skill\n\n完整说明..."
```

## 4. 存入内存

```python
name = meta.get("name", file_path.parent.name)
self.skills[name] = {
    "meta": meta,
    "body": body,
    "path": str(file_path),
}
```

这里会优先使用 frontmatter 里的 `name`。

如果没有写 `name`，就用 `SKILL.md` 所在目录名兜底。

最终内存结构类似：

```python
self.skills = {
    "pdf": {
        "meta": {
            "name": "pdf",
            "description": "Process PDF files",
            "tags": "pdf,docs",
        },
        "body": "# PDF Skill\n\n完整说明...",
        "path": "skills/pdf/SKILL.md",
    }
}
```

## 5. 第一层：只把简介放进 system prompt

```python
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.

Skills available:
{SKILL_LOADER.get_descriptions()}"""
```

`get_descriptions()` 只返回简短描述：

```text
  - pdf: Process PDF files [pdf,docs]
  - code-review: Review code changes
```

这样模型一开始只知道有哪些 skill，不会一次性看到所有 skill 的完整正文。

## 6. 第二层：模型按需调用 load_skill

工具注册在：

```python
"load_skill": lambda **kwargs: SKILL_LOADER.get_content(kwargs["name"]),
```

如果模型调用：

```json
{
  "name": "pdf"
}
```

Python 实际执行：

```python
SKILL_LOADER.get_content("pdf")
```

返回：

```xml
<skill name="pdf">
完整 PDF skill 内容...
</skill>
```

## 7. OpenAI 工具结果回填

工具执行结果会以 OpenAI tool message 的形式放回 `messages`：

```python
messages.append(
    {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(output),
    },
)
```

`tool_call_id` 用来告诉 OpenAI API：这条工具结果对应刚才哪一次工具调用。

模型下一轮就能看到完整 skill 内容，并按这个 skill 的要求继续工作。

## 核心思想

```text
不要一开始把所有 skill 全塞进 system prompt。
只塞 skill 简介，需要哪个再加载哪个。
```

这种方式叫两层 skill injection：

- 第一层：metadata，便宜、短、常驻 system prompt
- 第二层：完整正文，按需通过 `load_skill` 加载
