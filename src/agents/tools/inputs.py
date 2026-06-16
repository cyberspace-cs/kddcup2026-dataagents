"""工具入参的 Pydantic 模型与对应严格 JSON Schema 常量。

每个 `XxxInput` 同时承担两种角色的"单一真相源"：
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

from agents.prompts.rules import PATH_CONVENTION_NOTE
from agents.tools.schema_normalize import normalize_for_vllm

# Agent 任意代码的硬超时。值太小会扼杀合法的 pandas 任务；太大会让 run-benchmark 显著变慢
EXECUTE_PYTHON_TIMEOUT_SECONDS = 30


# `answer` inline 路径的硬闸阈值。超过任一阈值即拒绝 inline 提交，强制 `from_csv`。
# 动机：模型把上百行表用 `rows=[...]` 自抄进 tool_call JSON 时，长结构化生成会出现
# attention drift——run 20260430-015 task_8 即是 140 行被抄成 143 行（多 3 行幻觉 +
# 57 行 amount 串味）。execute_python 已写正确 CSV 时仍走 inline 是这条规则的目标。
# 数值与 prompt `_DATA_OUTPUT_RULES` 的 "10+ rows OR 50+ cells" 文案保持一致，
# 后续两边一起调（grep "10+ rows" 即可定位）。
# 阈值执行点在 `handlers.handle_answer`（artifact-exists 检查**之后**），不在本文件的
# `_check_payload`——回归参照 run 20260614-009/task_31：模型先写好 `_answer/answer.csv`、
# 再发 10 行 inline，旧顺序下 Pydantic 把不可执行的"阈值错"挤掉了 handler 那条精准的
# "artifact-exists, 用 from_csv 指 <path>" 提示，模型死循环耗光预算。
INLINE_ANSWER_ROW_LIMIT = 10
INLINE_ANSWER_CELL_LIMIT = 50


def _path_field(*, file_kind: str, examples: tuple[str, ...]) -> Any:
    """Path 字段统一描述工厂：保证 5 个 path 入参的描述风格一致。"""
    example_text = ", ".join(repr(ex) for ex in examples)
    return Field(
        description=(
            f"Relative path to a {file_kind}. Examples: {example_text}. " + PATH_CONVENTION_NOTE
        ),
    )


def _positive_int_field(*, default: int, description: str) -> Any:
    """非负整数字段工厂：minimum=1 + 默认值，避免模型误填 0/负数。"""
    return Field(default=default, ge=1, description=description)


# `inspect_files` 入参刻意保持空模型：`extra="forbid"` 在 schema 中变 `additionalProperties=false`，
# 把"无配置全量扫描"语义写进契约——模型即使凭空填字段也会立刻拿到可操作的 ValidationError。
# 不写 class docstring 是为了避免 Pydantic 把它注入到 JSON Schema 顶层 description（破坏字节稳定性）。
class InspectFilesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExploreInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


# `explore_video` 入参：explorer 专属（不进主 registry）。
class ExploreVideoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Annotated[
        str,
        _path_field(file_kind="video file", examples=("video.mp4", "media/demo.mov")),
    ]
    instructions: str = Field(
        description=(
            "Analysis brief for the video agent: what the task needs from this video — "
            "which charts, tables, values, or text to transcribe, plus any IDs or labels "
            "to look for."
        ),
    )


EXPLORE_VIDEO_SCHEMA = normalize_for_vllm(ExploreVideoInput.model_json_schema())


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


class ReadJsonInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Annotated[
        str,
        _path_field(file_kind="JSON file", examples=("json/Patient.json", "zip_code.json")),
    ]


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
            default=100,
            description="Number of lines to return in this page.",
        ),
    ]
    keyword: str | None = Field(
        default=None,
        description=(
            "Optional case-insensitive keyword. When set, returns ALL paragraphs "
            "(blank-line-separated) containing the keyword across the whole file "
            "(start_line / line_count are ignored), capped at 20 matches / 8KB total. "
            "Prefer this over manual paging when searching for an ID/name/term — "
            "one call replaces a sequential scan."
        ),
    )


class SearchContextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Search query for the task-local retrieval index. Use this for "
                "entity names, IDs, phrases, or evidence in long documents."
            ),
        ),
    ]
    path: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Optional exact path filter, relative to context for source files "
                "or the absolute ETL CSV path shown by inspect_files."
            ),
        ),
    ] = None
    limit: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Optional result cap. The runtime also caps this at the configured "
                "run.retrieval_index.max_results."
            ),
        ),
    ] = None


class LookupFieldInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    term: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Field, unit, code, alias, threshold, table, or column term to look up "
                "in knowledge.md and schema summaries."
            ),
        ),
    ]
    limit: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Optional result cap. The runtime also caps this at the configured "
                "run.retrieval_index.max_results."
            ),
        ),
    ] = None


class InspectSqliteSchemaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Annotated[
        str,
        _path_field(
            file_kind="sqlite database file",
            examples=("db/clinical.sqlite", "main.db"),
        ),
    ]


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
                "Examples: 'db/clinical.sqlite', 'main.db'. " + PATH_CONVENTION_NOTE
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


class GrepContextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Regex pattern to search for across all context files. "
                "Case-insensitive. Supports full Python regex syntax."
            ),
        ),
    ]
    path: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Optional path substring filter. When set, only files whose "
                "path contains this string are searched."
            ),
        ),
    ] = None


GREP_CONTEXT_SCHEMA = normalize_for_vllm(GrepContextInput.model_json_schema())


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


class AnswerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Column header names. Leave empty when using from_csv. "
                "Inline is rejected for 10+ rows — use from_csv instead."
            ),
        ),
    ]
    rows: Annotated[
        # `int` 必须排在 `float` 之前：Pydantic v2 union 按声明顺序匹配，否则
        # JSON 整数（如主键 ID 163109）会被先匹中 float 而落地成 163109.0，
        # 写进 prediction.csv 的字面值就跟 gold 的 "163109" 失配。
        list[list[str | int | float | bool | None]],
        Field(
            default_factory=list,
            description=(
                "Answer rows; each row's length must match the columns array. "
                "Leave empty when using from_csv. "
                "Inline is rejected for 10+ rows — use from_csv instead."
            ),
        ),
    ]
    from_csv: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Absolute path to a CSV file containing the answer table "
                "(header row + data rows). REQUIRED for 10+ rows or 50+ cells "
                "(inline is rejected above this threshold). When set, `columns` "
                "and `rows` MUST be empty. Convention: write the CSV from "
                "`execute_python` under `os.environ['DABENCH_ANSWER_DIR']`."
            ),
        ),
    ] = None

    @model_validator(mode="after")
    def _check_payload(self) -> AnswerInput:
        # answer 一旦提交即为最终产物；XOR 校验外加列数错位都会让 prediction.csv
        # 对齐崩坏，必须在边界处一次性早抛。
        # inline 体量硬闸**不在这里**：参见模块顶 INLINE_ANSWER_ROW_LIMIT 注释。
        if self.from_csv is not None:
            if self.columns or self.rows:
                raise ValueError("provide either from_csv or columns+rows, not both")
            return self
        if not self.columns:
            raise ValueError("columns must be non-empty when from_csv is not set")
        width = len(self.columns)
        for index, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(f"row {index} has {len(row)} cells but columns has {width}")
        return self


INSPECT_FILES_SCHEMA = normalize_for_vllm(InspectFilesInput.model_json_schema())
READ_CSV_SCHEMA = normalize_for_vllm(ReadCsvInput.model_json_schema())
READ_JSON_SCHEMA = normalize_for_vllm(ReadJsonInput.model_json_schema())
READ_DOC_SCHEMA = normalize_for_vllm(ReadDocInput.model_json_schema())
SEARCH_CONTEXT_SCHEMA = normalize_for_vllm(SearchContextInput.model_json_schema())
LOOKUP_FIELD_SCHEMA = normalize_for_vllm(LookupFieldInput.model_json_schema())
INSPECT_SQLITE_SCHEMA = normalize_for_vllm(InspectSqliteSchemaInput.model_json_schema())
EXECUTE_CONTEXT_SQL_SCHEMA = normalize_for_vllm(ExecuteContextSqlInput.model_json_schema())
EXECUTE_PYTHON_SCHEMA = normalize_for_vllm(ExecutePythonInput.model_json_schema())
ANSWER_SCHEMA = normalize_for_vllm(AnswerInput.model_json_schema())
