"""工具注册表（Tool Registry）。

- `ToolSpec`:             工具对 prompt 的呈现（名字 / 描述 / 入参示例 / 严格 JSON Schema）
- `ToolExecutionResult`:  工具执行结果（ok 标志、content、是否终止、answer）
- `ToolRegistry`:         把 specs（给模型看的）和 handlers（给 Python 调的）绑成一体
- `create_default_tool_registry()`: 默认 8 件套工具集合

每个 `_xxx` 私有函数是一个工具的 handler，签名统一为 `(task, action_input) → ToolExecutionResult`，
方便按名字查表分发。`answer` 是唯一 `is_terminal=True` 的工具，触发后 ReAct 循环结束。

`input_schema` vs `json_schema` 的区分：
- `input_schema` 是**给文本 ReAct prompt 看的演示示例**（不是严格 JSON Schema），用于提示模型
  入参长什么样；保持为非严格形式，是为了实测里发现其比严谨 schema 更不易让模型出错
- `json_schema` 是**给 OpenAI 原生 function calling 用的严格 JSON Schema**，由 `to_openai_tools`
  拼进请求体，vLLM 侧可在生成时做 constrained decoding
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.filesystem import (
    list_context_tree,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
    resolve_context_path,
)
from data_agent_baseline.tools.python_exec import execute_python_code
from data_agent_baseline.tools.sqlite import execute_read_only_sql, inspect_sqlite_schema

# Agent 任意代码的硬超时。值太小会扼杀合法的 pandas 任务；太大会让 run-benchmark 显著变慢
EXECUTE_PYTHON_TIMEOUT_SECONDS = 30


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """工具对模型的 "规格说明"，同时承载 prompt 文案与原生 function calling 的严格 schema。

    - name:          Agent 在 action 字段里写的名字
    - description:   一句话说明用途
    - input_schema:  action_input 的最小示例（文本 ReAct prompt 用，非严格 JSON Schema）
    - json_schema:   严格 JSON Schema，供 native function calling 下推给模型；
                     None 表示不支持原生调用（默认工具都已填充）
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    json_schema: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """工具执行的统一返回结构。

    - ok:          True 表示工具正常完成；即使业务层"失败"（如 SQL 查空）也是 True
    - content:     回传给模型看的结果 JSON
    - is_terminal: 是否应让 ReAct 循环立即终止（仅 answer 置 True）
    - answer:      终止工具附带的答案表（非终止工具为 None）
    """

    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


# 统一的 handler 签名：方便 ToolRegistry.execute 按名字查表调用
ToolHandler = Callable[[PublicTask, dict[str, Any]], ToolExecutionResult]


# ------------------------- 工具 handler 实现 -------------------------
# 每个 _xxx 都把 action_input 的原始 dict 映射到底层实现函数，做必要的类型转换。


def _list_context(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """`list_context`: 展示 context/ 目录树，默认深度 4。"""
    max_depth = int(action_input.get("max_depth", 4))
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=max_depth))


def _read_csv(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """`read_csv`: 读取 CSV 预览，默认前 20 行。"""
    path = str(action_input["path"])
    max_rows = int(action_input.get("max_rows", 20))
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, path, max_rows=max_rows))


def _read_json(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """`read_json`: 读取 JSON 并按字符数截断，默认 4000 字符。"""
    path = str(action_input["path"])
    max_chars = int(action_input.get("max_chars", 4000))
    return ToolExecutionResult(ok=True, content=read_json_preview(task, path, max_chars=max_chars))


def _read_doc(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """`read_doc`: 按行分页读取任意文本（md/txt），可选关键词检索。"""
    path = str(action_input["path"])
    start_line = int(action_input.get("start_line", 1))
    line_count = int(action_input.get("line_count", 80))
    raw_keyword = action_input.get("keyword")
    keyword = None if raw_keyword is None or raw_keyword == "" else str(raw_keyword)
    return ToolExecutionResult(
        ok=True,
        content=read_doc_preview(
            task,
            path,
            start_line=start_line,
            line_count=line_count,
            keyword=keyword,
        ),
    )


def _inspect_sqlite_schema(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """`inspect_sqlite_schema`: 查看 sqlite 数据库的全部表结构。"""
    # resolve_context_path 同时做越界校验 + 存在性检查
    path = resolve_context_path(task, str(action_input["path"]))
    return ToolExecutionResult(ok=True, content=inspect_sqlite_schema(path))


def _execute_context_sql(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """`execute_context_sql`: 对 context/ 下的 sqlite 跑只读 SQL，默认 200 行上限。"""
    path = resolve_context_path(task, str(action_input["path"]))
    sql = str(action_input["sql"])
    limit = int(action_input.get("limit", 200))
    return ToolExecutionResult(ok=True, content=execute_read_only_sql(path, sql, limit=limit))


def _execute_python(task: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """`execute_python`: 在子进程中跑任意 Python 代码，30s 超时。

    注意这里根据 `content.success` 决定 `ok` 的真假——和其他工具不同，
    这个工具本身可能因用户代码抛错而失败，模型需要一个明确的 ok 信号。
    """
    code = str(action_input["code"])
    content = execute_python_code(
        context_root=task.context_dir,
        code=code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _answer(_: PublicTask, action_input: dict[str, Any]) -> ToolExecutionResult:
    """`answer`: 终止工具，提交最终答案表。

    严格校验结构——答案表一旦落地即为最终产物，此处不合法比静默生成错误 CSV 更好：
    - columns 必须是非空字符串列表
    - rows 是列表，每行也是列表，且列数与 columns 一致
    校验失败抛 ValueError，由 ReAct 循环捕获为一条 __error__ 步，让模型有机会重提。
    """
    columns = action_input.get("columns")
    rows = action_input.get("rows")
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(item, str) for item in columns)
    ):
        raise ValueError("answer.columns must be a non-empty list of strings.")
    if not isinstance(rows, list):
        raise ValueError("answer.rows must be a list.")

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list):
            raise ValueError("Each answer row must be a list.")
        # 列数匹配：防止模型少填或多填字段导致 CSV 对齐错乱
        if len(row) != len(columns):
            raise ValueError("Each answer row must match the number of columns.")
        # list(row) 拷贝：避免 answer_input 被后续代码误修改
        normalized_rows.append(list(row))

    answer = AnswerTable(columns=list(columns), rows=normalized_rows)
    return ToolExecutionResult(
        ok=True,
        content={
            # 给模型一个简洁的提交确认信号，不把完整表回显（避免浪费 token）
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(normalized_rows),
        },
        is_terminal=True,  # 触发 ReAct 循环退出
        answer=answer,  # 答案对象交给循环，最终写入 prediction.csv
    )


# ------------------------- 严格 JSON Schema（native function calling 用） -------------------------
# 所有 schema 保持 `additionalProperties: false`，让 vLLM 可做 constrained decoding。
# 注意：对纯整数计数/上限参数，明确允许的最小值以兜住模型误填 0 / 负数。


# 所有工具的 `path` 字段共用同一条路径约定：
# - 必须相对于任务的 context 目录
# - **不要** 带 `context/` 前缀，哪怕 system prompt 里提到过 `context/`
# 用常量收敛文案，后续调整只改一处，保证 8 个 schema 描述一致，减少模型的歧义。
_PATH_CONVENTION_NOTE = (
    "Paths are relative to the task context directory; DO NOT prepend 'context/'. "
    "For example use 'knowledge.md' or 'csv/foo.csv', not 'context/knowledge.md'."
)


_LIST_CONTEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "max_depth": {
            "type": "integer",
            "minimum": 1,
            "default": 4,
            "description": "Maximum directory depth to walk when listing files.",
        },
    },
    "required": [],
    "additionalProperties": False,
}


_READ_CSV_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": (
                "Relative path to a CSV file. "
                "Examples: 'csv/member.csv', 'data/train.csv'. " + _PATH_CONVENTION_NOTE
            ),
        },
        "max_rows": {
            "type": "integer",
            "minimum": 1,
            "default": 20,
            "description": "Maximum number of preview rows to return.",
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}


_READ_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": (
                "Relative path to a JSON file. "
                "Examples: 'json/Patient.json', 'zip_code.json'. " + _PATH_CONVENTION_NOTE
            ),
        },
        "max_chars": {
            "type": "integer",
            "minimum": 1,
            "default": 4000,
            "description": "Maximum number of characters to return from the JSON preview.",
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}


_READ_DOC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": (
                "Relative path to a text document (md/txt/...). "
                "Examples: 'knowledge.md', 'docs/readme.txt'. " + _PATH_CONVENTION_NOTE
            ),
        },
        "start_line": {
            "type": "integer",
            "minimum": 1,
            "default": 1,
            "description": "1-based line number to start reading from.",
        },
        "line_count": {
            "type": "integer",
            "minimum": 1,
            "default": 80,
            "description": "Number of lines to return in this page.",
        },
        "keyword": {
            "type": ["string", "null"],
            "default": None,
            "description": "Optional keyword; when set, returns a context window around the first match.",
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}


_INSPECT_SQLITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": (
                "Relative path to a sqlite database file. "
                "Examples: 'db/clinical.sqlite', 'main.db'. " + _PATH_CONVENTION_NOTE
            ),
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}


_EXECUTE_CONTEXT_SQL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": (
                "Relative path to the sqlite/db file. "
                "Examples: 'db/clinical.sqlite', 'main.db'. " + _PATH_CONVENTION_NOTE
            ),
        },
        "sql": {
            "type": "string",
            "description": "Read-only SQL query to execute.",
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "default": 200,
            "description": "Maximum number of rows to return.",
        },
    },
    "required": ["path", "sql"],
    "additionalProperties": False,
}


_EXECUTE_PYTHON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": (
                "Python source to execute with the task context directory as CWD. "
                f"Execution timeout is {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
        },
    },
    "required": ["code"],
    "additionalProperties": False,
}


# answer.rows 的单元格允许 string / number / boolean / null 这四种 JSON 标量
_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "columns": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
            "description": "Column header names (non-empty).",
        },
        "rows": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "number"},
                        {"type": "boolean"},
                        {"type": "null"},
                    ]
                },
            },
            "description": "Answer rows; each row's length must match the columns array.",
        },
    },
    "required": ["columns", "rows"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class ToolRegistry:
    """把 ToolSpec（prompt 侧）与 ToolHandler（执行侧）绑在一起的查找表。

    同一个工具的 spec 和 handler 是分开维护的：prompt 需要自然语言描述，
    执行只需要函数引用——解耦能让测试替换 handler 而不重写 prompt。
    """

    specs: dict[str, ToolSpec]
    handlers: dict[str, ToolHandler]

    def describe_for_prompt(self) -> str:
        """把 specs 渲染成 "- name: desc\\n  input_schema: {...}" 的多行文案。

        按 name 排序保证输出稳定，便于 prompt cache 命中与 diff 审查。
        """
        lines = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            lines.append(f"- {spec.name}: {spec.description}")
            lines.append(f"  input_schema: {spec.input_schema}")
        return "\n".join(lines)

    def execute(
        self, task: PublicTask, action: str, action_input: dict[str, Any]
    ) -> ToolExecutionResult:
        """按工具名分发到对应 handler。

        未知工具名直接 KeyError，由 ReAct 循环吞入记录成错误步——模型下一轮能看到纠错信号。
        """
        if action not in self.handlers:
            raise KeyError(f"Unknown tool: {action}")
        return self.handlers[action](task, action_input)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """把所有带 `json_schema` 的 specs 打包成 OpenAI `tools=[...]` 请求字段。

        - 按工具名排序，保证请求体稳定（便于 prompt cache 命中）
        - 缺少 `json_schema` 的工具会被跳过 —— 即不暴露给 native function calling
        """
        tools: list[dict[str, Any]] = []
        for name in sorted(self.specs):
            spec = self.specs[name]
            if spec.json_schema is None:
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.json_schema,
                    },
                }
            )
        return tools


def create_default_tool_registry() -> ToolRegistry:
    """构造默认 8 件套工具注册表。

    工具清单（按字母序）：
    - answer                 终止 + 提交答案（唯一终止工具）
    - execute_context_sql    sqlite 只读 SQL
    - execute_python         子进程执行任意 Python
    - inspect_sqlite_schema  sqlite 表结构
    - list_context           列目录树
    - read_csv               预览 CSV
    - read_doc               预览文本文档
    - read_json              预览 JSON

    `input_schema` 里的示例值是"演示"形式（不是真实 JSON Schema），模型看到
    `"columns": ["column_name"]` 会按模式填自己的数据，实测比严谨 schema 更不易出错。
    `json_schema` 则是给 OpenAI function calling 用的严格 schema。
    """
    specs = {
        "answer": ToolSpec(
            name="answer",
            description="Submit the final answer table. This is the only valid terminating action.",
            input_schema={
                "columns": ["column_name"],
                "rows": [["value_1"]],
            },
            json_schema=_ANSWER_SCHEMA,
        ),
        "execute_context_sql": ToolSpec(
            name="execute_context_sql",
            description="Run a read-only SQL query against a sqlite/db file inside context.",
            input_schema={
                "path": "relative/path/to/file.sqlite",
                "sql": "SELECT ...",
                "limit": 200,
            },
            json_schema=_EXECUTE_CONTEXT_SQL_SCHEMA,
        ),
        "execute_python": ToolSpec(
            name="execute_python",
            description=(
                "Execute arbitrary Python code with the task context directory as the "
                "working directory. The tool returns the code's captured stdout as `output`. "
                # 动态把超时秒数写进描述，避免改常量后忘记更新文案
                f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
            input_schema={
                "code": "import os\nprint(sorted(os.listdir('.')))",
            },
            json_schema=_EXECUTE_PYTHON_SCHEMA,
        ),
        "inspect_sqlite_schema": ToolSpec(
            name="inspect_sqlite_schema",
            description="Inspect tables and columns in a sqlite/db file inside context.",
            input_schema={"path": "relative/path/to/file.sqlite"},
            json_schema=_INSPECT_SQLITE_SCHEMA,
        ),
        "list_context": ToolSpec(
            name="list_context",
            description="List files and directories available under context.",
            input_schema={"max_depth": 4},
            json_schema=_LIST_CONTEXT_SCHEMA,
        ),
        "read_csv": ToolSpec(
            name="read_csv",
            description="Read a preview of a CSV file inside context.",
            input_schema={"path": "relative/path/to/file.csv", "max_rows": 20},
            json_schema=_READ_CSV_SCHEMA,
        ),
        "read_doc": ToolSpec(
            name="read_doc",
            description=(
                "Read a text-like document inside context by line page, or search for "
                "the first case-insensitive keyword match. Use `has_more` and the next "
                "`start_line` to continue reading."
            ),
            input_schema={
                "path": "relative/path/to/file.md",
                "start_line": 1,
                "line_count": 80,
                "keyword": "optional search text",
            },
            json_schema=_READ_DOC_SCHEMA,
        ),
        "read_json": ToolSpec(
            name="read_json",
            description="Read a preview of a JSON file inside context.",
            input_schema={"path": "relative/path/to/file.json", "max_chars": 4000},
            json_schema=_READ_JSON_SCHEMA,
        ),
    }
    # handlers 的键必须与 specs 对齐；新增工具时同步改两处
    handlers = {
        "answer": _answer,
        "execute_context_sql": _execute_context_sql,
        "execute_python": _execute_python,
        "inspect_sqlite_schema": _inspect_sqlite_schema,
        "list_context": _list_context,
        "read_csv": _read_csv,
        "read_doc": _read_doc,
        "read_json": _read_json,
    }
    return ToolRegistry(specs=specs, handlers=handlers)
