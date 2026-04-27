"""八个工具入参的 Pydantic 模型与对应严格 JSON Schema 常量。

每个 `XxxInput` 同时承担三种角色的"单一真相源"：
- `XxxInput.example_value()`：给文本 ReAct prompt 的演示示例（非严格 schema）
- `XxxInput.model_validate(...)`：`ToolRegistry.execute` 边界处的入参校验
- `XXX_SCHEMA = normalize_for_vllm(XxxInput.model_json_schema())`：原生 function
  calling 的 OpenAI tools[i].function.parameters，需保持字节稳定（被
  tests/test_tool_schemas_byte_fixture 锁定）。

`extra="forbid"` → JSON Schema 中 `additionalProperties=false`，是 vLLM 约束解码必备。
`EXECUTE_PYTHON_TIMEOUT_SECONDS` 在此处定义，因为 `ExecutePythonInput` 的字段描述
直接拼了它的值；`handlers.py` 与 `registry.py` 只是消费方。
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from data_agent_baseline.tools.schema_normalize import normalize_for_vllm

# Agent 任意代码的硬超时。值太小会扼杀合法的 pandas 任务；太大会让 run-benchmark 显著变慢
EXECUTE_PYTHON_TIMEOUT_SECONDS = 30


# 所有工具的 `path` 字段共用同一条路径约定：
# - 必须相对于任务的 context 目录
# - **不要** 带 `context/` 前缀，哪怕 system prompt 里提到过 `context/`
# 用常量收敛文案，后续调整只改一处，保证 8 个 schema 描述一致，减少模型的歧义。
_PATH_CONVENTION_NOTE = (
    "Paths are relative to the task context directory; DO NOT prepend 'context/'. "
    "For example use 'knowledge.md' or 'csv/foo.csv', not 'context/knowledge.md'."
)


def _path_field(*, file_kind: str, examples: tuple[str, ...]) -> Any:
    """Path 字段统一描述工厂：保证 5 个 path 入参的描述风格一致。"""
    example_text = ", ".join(repr(ex) for ex in examples)
    return Field(
        description=(
            f"Relative path to a {file_kind}. Examples: {example_text}. " + _PATH_CONVENTION_NOTE
        ),
    )


def _positive_int_field(*, default: int, description: str) -> Any:
    """非负整数字段工厂：minimum=1 + 默认值，避免模型误填 0/负数。"""
    return Field(default=default, ge=1, description=description)


class ListContextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_depth: Annotated[
        int,
        _positive_int_field(
            default=4,
            description="Maximum directory depth to walk when listing files.",
        ),
    ]

    @classmethod
    def example_value(cls) -> dict[str, Any]:
        return {"max_depth": 4}


# `inspect_files` 入参刻意保持空模型：`extra="forbid"` 在 schema 中变 `additionalProperties=false`，
# 把"无配置全量扫描"语义写进契约——模型即使凭空填字段也会立刻拿到可操作的 ValidationError。
# 不写 class docstring 是为了避免 Pydantic 把它注入到 JSON Schema 顶层 description（破坏字节稳定性）。
class InspectFilesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def example_value(cls) -> dict[str, Any]:
        return {}


class ReadCsvInput(BaseModel):
    """`read_csv` 入参：仅 path，**刻意没有** max_rows / head / tail 旋钮。

    设计契约：read_csv 是 inspect-style 工具——返回固定的 schema + head(20) + tail(5)；
    模型无法通过加大窗口绕过样本语义。需要全量过滤/聚合时唯一出口是 execute_python。
    任何 max_rows/limit/offset 字段都会因 `extra="forbid"` 显式报错，把这个边界打到
    Pydantic 校验层，让模型立刻拿到可操作的反馈。
    """

    model_config = ConfigDict(extra="forbid")

    path: Annotated[
        str,
        _path_field(file_kind="CSV file", examples=("csv/member.csv", "data/train.csv")),
    ]

    @classmethod
    def example_value(cls) -> dict[str, Any]:
        return {"path": "relative/path/to/file.csv"}


class ReadJsonInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Annotated[
        str,
        _path_field(file_kind="JSON file", examples=("json/Patient.json", "zip_code.json")),
    ]
    max_chars: Annotated[
        int,
        _positive_int_field(
            default=4000,
            description="Maximum number of characters to return from the JSON preview.",
        ),
    ]

    @classmethod
    def example_value(cls) -> dict[str, Any]:
        return {"path": "relative/path/to/file.json", "max_chars": 4000}


class ReadDocInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Annotated[
        str,
        _path_field(
            file_kind="text document (md/txt/...)",
            examples=("knowledge.md", "docs/readme.txt"),
        ),
    ]
    start_line: Annotated[
        int,
        _positive_int_field(
            default=1,
            description="1-based line number to start reading from.",
        ),
    ]
    line_count: Annotated[
        int,
        _positive_int_field(
            default=80,
            description="Number of lines to return in this page.",
        ),
    ]
    keyword: str | None = Field(
        default=None,
        description="Optional keyword; when set, returns a context window around the first match.",
    )

    @classmethod
    def example_value(cls) -> dict[str, Any]:
        return {
            "path": "relative/path/to/file.md",
            "start_line": 1,
            "line_count": 80,
            "keyword": "optional search text",
        }


class InspectSqliteSchemaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Annotated[
        str,
        _path_field(
            file_kind="sqlite database file",
            examples=("db/clinical.sqlite", "main.db"),
        ),
    ]

    @classmethod
    def example_value(cls) -> dict[str, Any]:
        return {"path": "relative/path/to/file.sqlite"}


class ExecuteContextSqlInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # 这一项的"the sqlite/db file"措辞与其它 4 个 path 字段不同，沿用历史 schema 文案
    # 以保持 to_openai_tools() 的字节稳定（_path_field 的 "Relative path to a {kind}." 模板
    # 不适配此处的 "the"，因此手写一个 Field）。
    path: Annotated[
        str,
        Field(
            description=(
                "Relative path to the sqlite/db file. "
                "Examples: 'db/clinical.sqlite', 'main.db'. " + _PATH_CONVENTION_NOTE
            ),
        ),
    ]
    sql: Annotated[str, Field(description="Read-only SQL query to execute.")]
    limit: Annotated[
        int,
        _positive_int_field(
            default=200,
            description="Maximum number of rows to return.",
        ),
    ]

    @classmethod
    def example_value(cls) -> dict[str, Any]:
        return {
            "path": "relative/path/to/file.sqlite",
            "sql": "SELECT ...",
            "limit": 200,
        }


class ExecutePythonInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: Annotated[
        str,
        Field(
            description=(
                "Python source to execute with the task context directory as CWD. "
                f"Execution timeout is {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
        ),
    ]

    @classmethod
    def example_value(cls) -> dict[str, Any]:
        return {"code": "import os\nprint(sorted(os.listdir('.')))"}


class AnswerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: Annotated[
        list[str],
        Field(min_length=1, description="Column header names (non-empty)."),
    ]
    rows: Annotated[
        # `int` 必须排在 `float` 之前：Pydantic v2 union 按声明顺序匹配，否则
        # JSON 整数（如主键 ID 163109）会被先匹中 float 而落地成 163109.0，
        # 写进 prediction.csv 的字面值就跟 gold 的 "163109" 失配。
        list[list[str | int | float | bool | None]],
        Field(description="Answer rows; each row's length must match the columns array."),
    ]

    @model_validator(mode="after")
    def _check_row_widths(self) -> AnswerInput:
        # answer 一旦提交即为最终产物；列数错位会让 prediction.csv 对齐崩坏，必须早抛。
        width = len(self.columns)
        for index, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(f"row {index} has {len(row)} cells but columns has {width}")
        return self

    @classmethod
    def example_value(cls) -> dict[str, Any]:
        return {"columns": ["column_name"], "rows": [["value_1"]]}


LIST_CONTEXT_SCHEMA = normalize_for_vllm(ListContextInput.model_json_schema())
INSPECT_FILES_SCHEMA = normalize_for_vllm(InspectFilesInput.model_json_schema())
READ_CSV_SCHEMA = normalize_for_vllm(ReadCsvInput.model_json_schema())
READ_JSON_SCHEMA = normalize_for_vllm(ReadJsonInput.model_json_schema())
READ_DOC_SCHEMA = normalize_for_vllm(ReadDocInput.model_json_schema())
INSPECT_SQLITE_SCHEMA = normalize_for_vllm(InspectSqliteSchemaInput.model_json_schema())
EXECUTE_CONTEXT_SQL_SCHEMA = normalize_for_vllm(ExecuteContextSqlInput.model_json_schema())
EXECUTE_PYTHON_SCHEMA = normalize_for_vllm(ExecutePythonInput.model_json_schema())
ANSWER_SCHEMA = normalize_for_vllm(AnswerInput.model_json_schema())
