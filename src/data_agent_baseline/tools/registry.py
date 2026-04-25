"""工具注册表（Tool Registry）。

- `ToolDefinition`:        合并 prompt 视图与执行视图于一体（name/description/schema/handler/is_terminal）
- `ToolExecutionResult`:   工具执行结果（ok 标志、content、是否终止、answer）
- `ToolRegistry`:          name → ToolDefinition 的不可变只读映射
- `create_default_tool_registry()`: 默认 8 件套工具集合（按注册顺序）

每个 `_xxx` 私有函数是一个工具的 handler，签名统一为 `(task, action_input) → ToolExecutionResult`，
方便按名字查表分发。`answer` 是唯一 `is_terminal=True` 的工具，触发后 ReAct 循环结束。

`input_schema` vs `json_schema` 的区分：
- `input_schema` 是**给文本 ReAct prompt 看的演示示例**（不是严格 JSON Schema），用于提示模型
  入参长什么样；保持为非严格形式，是为了实测里发现其比严谨 schema 更不易让模型出错
- `json_schema` 是**给 OpenAI 原生 function calling 用的严格 JSON Schema**，由 `to_openai_tools`
  拼进请求体，vLLM 侧可在生成时做 constrained decoding
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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


# 统一的 handler 签名：第二个参数对内置工具是已验证的 BaseModel 实例（详见 _INPUT_MODELS），
# 对未在 _INPUT_MODELS 注册的用户自定义工具则透传原始 dict。
ToolHandler = Callable[[PublicTask, Any], ToolExecutionResult]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """单个工具的全量定义：把 prompt 侧与执行侧合并成同一对象。

    - name:          Agent 在 action 字段里写的名字
    - description:   一句话说明用途
    - input_schema:  action_input 的最小示例（文本 ReAct prompt 用，非严格 JSON Schema）
    - json_schema:   严格 JSON Schema，供 native function calling 下推给模型；
                     None 表示不暴露给原生调用（默认工具都已填充）
    - handler:       具体执行函数；内置工具签名为 `(task, args: <ModelClass>) -> ToolExecutionResult`，
                     未在 `_INPUT_MODELS` 注册的用户工具沿用 `(task, action_input: dict)`
    - is_terminal:   True 时 ReAct 循环会在该工具成功返回后立即终止；默认仅 answer 为 True
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    json_schema: dict[str, Any] | None
    handler: ToolHandler
    is_terminal: bool = False


# ------------------------- 工具 handler 实现 -------------------------
# 每个 _xxx 接收一个已被对应 Pydantic 模型校验过的 args 实例（参见 _INPUT_MODELS）；
# 类型转换 / 必填校验 / 边界检查全部在 ToolRegistry.execute 边界处完成。


def _list_context(task: PublicTask, args: ListContextInput) -> ToolExecutionResult:
    """`list_context`: 展示 context/ 目录树，默认深度 4。"""
    return ToolExecutionResult(ok=True, content=list_context_tree(task, max_depth=args.max_depth))


def _read_csv(task: PublicTask, args: ReadCsvInput) -> ToolExecutionResult:
    """`read_csv`: 读取 CSV 预览，默认前 20 行。"""
    return ToolExecutionResult(
        ok=True, content=read_csv_preview(task, args.path, max_rows=args.max_rows)
    )


def _read_json(task: PublicTask, args: ReadJsonInput) -> ToolExecutionResult:
    """`read_json`: 读取 JSON 并按字符数截断，默认 4000 字符。"""
    return ToolExecutionResult(
        ok=True, content=read_json_preview(task, args.path, max_chars=args.max_chars)
    )


def _read_doc(task: PublicTask, args: ReadDocInput) -> ToolExecutionResult:
    """`read_doc`: 按行分页读取任意文本（md/txt），可选关键词检索。"""
    # 沿用历史"空字符串与 None 同义"行为，避免下层把空 keyword 当作精确空匹配
    keyword = None if not args.keyword else args.keyword
    return ToolExecutionResult(
        ok=True,
        content=read_doc_preview(
            task,
            args.path,
            start_line=args.start_line,
            line_count=args.line_count,
            keyword=keyword,
        ),
    )


def _inspect_sqlite_schema(task: PublicTask, args: InspectSqliteSchemaInput) -> ToolExecutionResult:
    """`inspect_sqlite_schema`: 查看 sqlite 数据库的全部表结构。"""
    # resolve_context_path 同时做越界校验 + 存在性检查
    path = resolve_context_path(task, args.path)
    return ToolExecutionResult(ok=True, content=inspect_sqlite_schema(path))


def _execute_context_sql(task: PublicTask, args: ExecuteContextSqlInput) -> ToolExecutionResult:
    """`execute_context_sql`: 对 context/ 下的 sqlite 跑只读 SQL，默认 200 行上限。"""
    path = resolve_context_path(task, args.path)
    return ToolExecutionResult(
        ok=True, content=execute_read_only_sql(path, args.sql, limit=args.limit)
    )


def _execute_python(task: PublicTask, args: ExecutePythonInput) -> ToolExecutionResult:
    """`execute_python`: 在子进程中跑任意 Python 代码，30s 超时。

    注意这里根据 `content.success` 决定 `ok` 的真假——和其他工具不同，
    这个工具本身可能因用户代码抛错而失败，模型需要一个明确的 ok 信号。
    """
    content = execute_python_code(
        context_root=task.context_dir,
        code=args.code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def _answer(_task: PublicTask, args: AnswerInput) -> ToolExecutionResult:
    """`answer`: 终止工具，提交最终答案表。

    所有结构性校验（columns 非空、每行列数 == len(columns)、单元格类型）由
    `AnswerInput` 在 ToolRegistry.execute 边界处完成；handler 收到的 `args`
    可信，直接拷贝成 AnswerTable 落盘。
    """
    # list(...) 拷贝：让最终落盘的 AnswerTable 不与外部对象共享底层引用
    columns = list(args.columns)
    rows = [list(row) for row in args.rows]
    answer = AnswerTable(columns=columns, rows=rows)
    return ToolExecutionResult(
        ok=True,
        content={
            # 给模型一个简洁的提交确认信号，不把完整表回显（避免浪费 token）
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(rows),
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


# ------------------------- 字段工厂 + 工具入参模型 -------------------------
# 八个工具各自一个 Pydantic 输入模型，作为 input_schema/json_schema/handler 的"单一真相源"。
# `extra="forbid"` → 生成的 JSON Schema 中 additionalProperties=false（vLLM 约束解码必备）。


def _normalize_for_vllm(schema: dict[str, Any]) -> dict[str, Any]:
    """把 Pydantic 生成的 JSON Schema 重写成 vLLM 端的字节稳定形态。

    三处确定性变换（与历史手写 schema 字节等价）：
    1. 递归剥掉每一层 dict 的 `title` 键（Pydantic 会在每个 property 加 `title:
       "Field Name"` 字样；xgrammar 不消费它，但保留会改请求体字节）。
    2. 把 `{"anyOf": [{"type": T}, {"type": "null"}]}`（含两种顺序）折叠为
       `{"type": [T, "null"]}`，与历史 `_READ_DOC_SCHEMA` 中 `keyword` 一致；
       兄弟键（`default`/`description` 等）原样保留。
    3. 顶层 object 缺省 `required` 字段时补 `required: []` —— Pydantic 在没有
       必填字段时会省略该键，但历史手写 schema 显式写了 `required: []`。

    其它任何结构原样不动；该函数纯函数式，输入对象不被改写。
    """

    def _walk(node: Any) -> Any:
        if isinstance(node, list):
            return [_walk(item) for item in cast(list[Any], node)]
        if not isinstance(node, dict):
            return node
        original = cast(dict[str, Any], node)
        # 先做 anyOf-nullable 折叠：要在剥 title 之前判断结构，避免子元素被先改写。
        any_of_value = original.get("anyOf")
        if isinstance(any_of_value, list) and len(cast(list[Any], any_of_value)) == 2:
            entries = cast(list[Any], any_of_value)
            non_null_type: str | None = None
            null_seen = False
            ok = True
            for entry in entries:
                if not isinstance(entry, dict):
                    ok = False
                    break
                entry_dict = cast(dict[str, Any], entry)
                # 仅识别"形如单 type 的简形 anyOf 分支"，含其他键则跳过折叠
                if set(entry_dict.keys()) != {"type"}:
                    ok = False
                    break
                inner_type = entry_dict["type"]
                if not isinstance(inner_type, str):
                    ok = False
                    break
                if inner_type == "null":
                    null_seen = True
                else:
                    non_null_type = inner_type
            if ok and null_seen and non_null_type is not None:
                rewritten: dict[str, Any] = {}
                for key, value in original.items():
                    if key == "anyOf":
                        continue
                    if key == "title":
                        continue
                    rewritten[key] = _walk(value)
                rewritten["type"] = [non_null_type, "null"]
                return rewritten

        result: dict[str, Any] = {}
        for key, value in original.items():
            if key == "title":
                continue
            result[key] = _walk(value)
        if result.get("type") == "object" and "required" not in result:
            result["required"] = []
        return result

    walked = _walk(schema)
    return cast(dict[str, Any], walked)


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


class ReadCsvInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Annotated[
        str,
        _path_field(file_kind="CSV file", examples=("csv/member.csv", "data/train.csv")),
    ]
    max_rows: Annotated[
        int,
        _positive_int_field(
            default=20,
            description="Maximum number of preview rows to return.",
        ),
    ]

    @classmethod
    def example_value(cls) -> dict[str, Any]:
        return {"path": "relative/path/to/file.csv", "max_rows": 20}


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
        list[list[str | float | bool | None]],
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


_LIST_CONTEXT_SCHEMA = _normalize_for_vllm(ListContextInput.model_json_schema())
_READ_CSV_SCHEMA = _normalize_for_vllm(ReadCsvInput.model_json_schema())
_READ_JSON_SCHEMA = _normalize_for_vllm(ReadJsonInput.model_json_schema())
_READ_DOC_SCHEMA = _normalize_for_vllm(ReadDocInput.model_json_schema())
_INSPECT_SQLITE_SCHEMA = _normalize_for_vllm(InspectSqliteSchemaInput.model_json_schema())
_EXECUTE_CONTEXT_SQL_SCHEMA = _normalize_for_vllm(ExecuteContextSqlInput.model_json_schema())
_EXECUTE_PYTHON_SCHEMA = _normalize_for_vllm(ExecutePythonInput.model_json_schema())
_ANSWER_SCHEMA = _normalize_for_vllm(AnswerInput.model_json_schema())


# ------------------------- 校验边界：ValidationError → 结构化 ValueError -------------------------


# 工具名 → 对应入参模型；ToolRegistry.execute 用这张表把原始 dict 翻译成已验证模型实例。
# 仅内置工具登记于此；未登记的（用户自定义）工具走原始 dict 透传，保留向后兼容。
_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "list_context": ListContextInput,
    "read_csv": ReadCsvInput,
    "read_json": ReadJsonInput,
    "read_doc": ReadDocInput,
    "inspect_sqlite_schema": InspectSqliteSchemaInput,
    "execute_context_sql": ExecuteContextSqlInput,
    "execute_python": ExecutePythonInput,
    "answer": AnswerInput,
}


def _format_validation_error(tool_name: str, error: ValidationError) -> ValueError:
    """把 Pydantic ValidationError 翻译成单行 `<tool>: <field_path> <reason>` 的 ValueError。

    模型在下一轮 ReAct 看到这条消息要能立刻自我修正，所以：
    - 不把内部 model 名（如 `ReadCsvInput`）泄漏出去
    - 只取首条错误（多错并排时模型反而更糊涂）
    - 错误类别按 Pydantic 的 type code 分流，给出可操作描述
    """
    first = error.errors()[0]
    err_type = str(first["type"])
    loc = cast(tuple[Any, ...], first.get("loc", ()))
    field_path = ".".join(str(seg) for seg in loc) if loc else ""
    ctx = first.get("ctx", {})

    if err_type == "missing":
        return ValueError(f"{tool_name}: field '{field_path}' is required")
    if err_type == "extra_forbidden":
        return ValueError(f"{tool_name}: extra field '{field_path}' is not permitted")
    if err_type.endswith("_type"):
        # Pydantic 把基础类型不匹配编码成 `<expected>_type`（如 `string_type`/`int_type`）
        expected = err_type.removesuffix("_type")
        actual = type(first.get("input")).__name__
        return ValueError(f"{tool_name}: field '{field_path}' must be a {expected}, got {actual}")
    if err_type == "greater_than_equal":
        bound = ctx.get("ge")
        actual_value = first.get("input")
        return ValueError(
            f"{tool_name}: field '{field_path}' must be >= {bound}, got {actual_value}"
        )
    if err_type == "too_short":
        min_len = ctx.get("min_length")
        return ValueError(
            f"{tool_name}: field '{field_path}' must contain at least {min_len} item(s)"
        )
    if err_type == "value_error":
        # 用户自定义 ValueError 会被 Pydantic 加 "Value error, " 前缀；剥掉再贴 tool 名
        msg = str(first["msg"]).removeprefix("Value error, ")
        if field_path:
            return ValueError(f"{tool_name}: field '{field_path}' {msg}")
        return ValueError(f"{tool_name}: {msg}")
    # 兜底：未识别的 type code 直接贴原始 msg，仍保证首 token 是 tool 名
    return ValueError(f"{tool_name}: field '{field_path}' {first['msg']}")


@dataclass(frozen=True, slots=True)
class ToolRegistry:
    """name → ToolDefinition 的不可变只读映射。

    构造完成后无法增删替换：底层用 MappingProxyType 包住一个有序 dict，
    任何项级写入都会抛 TypeError，这样 NativeToolsOpenAIAdapter 等下游缓存不会变脏。
    实例本身是 frozen dataclass，对 `definitions` 字段的重新赋值会抛 FrozenInstanceError。
    """

    definitions: Mapping[str, ToolDefinition]

    def __post_init__(self) -> None:
        # 始终对入参做一次浅拷贝再包成 MappingProxyType：
        # 即便调用方传入的已经是 MappingProxyType，它内部仍可能指向一个外部仍在持有的
        # 可变 dict——`MappingProxyType(other_proxy)` 不会重建底层存储，于是对源 dict 的
        # 后续 add/remove/replace 会透过 proxy 漏进 registry，违反不可变契约。
        object.__setattr__(self, "definitions", MappingProxyType(dict(self.definitions)))

    def describe_for_prompt(self) -> str:
        """把 definitions 渲染成 "- name: desc\\n  input_schema: {...}" 的多行文案。

        按 name 排序保证输出稳定，便于 prompt cache 命中与 diff 审查。
        """
        lines: list[str] = []
        for name in sorted(self.definitions):
            definition = self.definitions[name]
            lines.append(f"- {definition.name}: {definition.description}")
            lines.append(f"  input_schema: {definition.input_schema}")
        return "\n".join(lines)

    def execute(
        self, task: PublicTask, action: str, action_input: dict[str, Any]
    ) -> ToolExecutionResult:
        """按工具名分发到对应 handler，对内置工具先过 Pydantic 校验。

        - 未知工具名 → KeyError（ReAct 循环会吞入记录成错误步）
        - 已知工具但 action_input 不合规 → 由 `_format_validation_error` 翻译为
          单行 `ValueError("<tool>: <field> <reason>")`，模型下一轮就能看到具体修复方向
        - `model_validate` 不修改入参 dict（Pydantic 复制后再校验），原 dict 身份/内容不变
        """
        if action not in self.definitions:
            raise KeyError(f"Unknown tool: {action}")
        model_cls = _INPUT_MODELS.get(action)
        if model_cls is None:
            # 用户自定义工具：未在 _INPUT_MODELS 注册时直接透传 dict（向后兼容）
            return self.definitions[action].handler(task, action_input)
        try:
            validated = model_cls.model_validate(action_input)
        except ValidationError as exc:
            raise _format_validation_error(action, exc) from exc
        return self.definitions[action].handler(task, validated)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """把所有带 `json_schema` 的 definitions 打包成 OpenAI `tools=[...]` 请求字段。

        - 按工具名排序，保证请求体字节稳定（prompt cache 命中 + diff 审查友好）
        - 缺少 `json_schema` 的工具会被跳过 —— 即不暴露给 native function calling
        """
        tools: list[dict[str, Any]] = []
        for name in sorted(self.definitions):
            definition = self.definitions[name]
            if definition.json_schema is None:
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": definition.name,
                        "description": definition.description,
                        "parameters": definition.json_schema,
                    },
                }
            )
        return tools


def create_default_tool_registry() -> ToolRegistry:
    """构造默认 8 件套工具注册表。

    注册顺序（按规范固定）：
    1. list_context           列目录树
    2. read_csv               预览 CSV
    3. read_doc               预览文本文档
    4. read_json              预览 JSON
    5. inspect_sqlite_schema  sqlite 表结构
    6. execute_context_sql    sqlite 只读 SQL
    7. execute_python         子进程执行任意 Python
    8. answer                 终止 + 提交答案（唯一终止工具）

    `input_schema` 里的示例值是"演示"形式（不是真实 JSON Schema），模型看到
    `"columns": ["column_name"]` 会按模式填自己的数据，实测比严谨 schema 更不易出错。
    `json_schema` 则是给 OpenAI function calling 用的严格 schema。
    """
    definitions: list[ToolDefinition] = [
        ToolDefinition(
            name="list_context",
            description=(
                "Use first to discover which files and directories exist under context "
                "and what file types are available."
            ),
            input_schema=ListContextInput.example_value(),
            json_schema=_LIST_CONTEXT_SCHEMA,
            handler=_list_context,
        ),
        ToolDefinition(
            name="read_csv",
            description=(
                "Use when you need to preview a CSV file, confirm column names, or "
                "inspect example rows before choosing a computation tool."
            ),
            input_schema=ReadCsvInput.example_value(),
            json_schema=_READ_CSV_SCHEMA,
            handler=_read_csv,
        ),
        ToolDefinition(
            name="read_doc",
            description=(
                "Use when you need to preview a text-like document, confirm relevant "
                "text, or search for the first case-insensitive keyword match. Use "
                "`has_more` and the next `start_line` to continue reading."
            ),
            input_schema=ReadDocInput.example_value(),
            json_schema=_READ_DOC_SCHEMA,
            handler=_read_doc,
        ),
        ToolDefinition(
            name="read_json",
            description=(
                "Use when you need to preview a JSON file, confirm top-level fields, "
                "or inspect sample structured records."
            ),
            input_schema=ReadJsonInput.example_value(),
            json_schema=_READ_JSON_SCHEMA,
            handler=_read_json,
        ),
        ToolDefinition(
            name="inspect_sqlite_schema",
            description=(
                "Use first when you find a sqlite/db file and need to know its "
                "tables, columns, and schema before writing SQL."
            ),
            input_schema=InspectSqliteSchemaInput.example_value(),
            json_schema=_INSPECT_SQLITE_SCHEMA,
            handler=_inspect_sqlite_schema,
        ),
        ToolDefinition(
            name="execute_context_sql",
            description=(
                "Use when data in a sqlite/db file can be answered with read-only SQL "
                "queries or aggregations. Call inspect_sqlite_schema first for an "
                "unknown database."
            ),
            input_schema=ExecuteContextSqlInput.example_value(),
            json_schema=_EXECUTE_CONTEXT_SQL_SCHEMA,
            handler=_execute_context_sql,
        ),
        ToolDefinition(
            name="execute_python",
            description=(
                "Use when you need computation, cleaning, aggregation, cross-file "
                "joins, large-file streaming, structured processing, or anything "
                "beyond simple preview tools. Python runs with the task context "
                "directory as the working directory and returns captured stdout as `output`. "
                # 动态把超时秒数写进描述，避免改常量后忘记更新文案
                f"The execution timeout is fixed at {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
            input_schema=ExecutePythonInput.example_value(),
            json_schema=_EXECUTE_PYTHON_SCHEMA,
            handler=_execute_python,
        ),
        ToolDefinition(
            name="answer",
            description=(
                "Use this when the final answer table is determined. This is the only "
                "way to submit the answer to the user/evaluator; plain text answers "
                "will be ignored."
            ),
            input_schema=AnswerInput.example_value(),
            json_schema=_ANSWER_SCHEMA,
            handler=_answer,
            is_terminal=True,  # 唯一终止工具
        ),
    ]
    return ToolRegistry(definitions={d.name: d for d in definitions})
