"""工具注册表（Tool Registry）的公开 API 与构造工厂。

本模块只承担三件事：
- 暴露 `ToolDefinition` / `ToolRegistry` 两个不可变结构 + `ToolExecutionResult`
  返回类型（最后一个从 `handlers.py` 重导出，保持历史导入路径）
- 将工具名映射到 Pydantic 入参模型，校验失败时翻译成单行可操作的 `ValueError`
- `create_default_tool_registry()` 按规范顺序构造默认工具集

实现细节按职责拆到三个邻居模块：
- `tools/schema_normalize.py`   ← `normalize_for_vllm`（vLLM 字节稳定形态）
- `tools/inputs.py`             ← Pydantic 入参模型 + 字段工厂 + `*_SCHEMA` 常量
- `tools/handlers.py`           ← `handle_xxx` handler + `ToolExecutionResult`/`ToolHandler`

`input_schema` vs `json_schema` 的区分：
- `input_schema` 是**给文本 ReAct prompt 看的演示示例**（不是严格 JSON Schema），用于提示模型
  入参长什么样；保持为非严格形式，是为了实测里发现其比严谨 schema 更不易让模型出错
- `json_schema` 是**给 OpenAI 原生 function calling 用的严格 JSON Schema**，由 `to_openai_tools`
  拼进请求体，vLLM 侧可在生成时做 constrained decoding
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.handlers import (
    ToolExecutionResult,
    ToolHandler,
    handle_answer,
    handle_execute_context_sql,
    handle_execute_python,
    handle_inspect_files,
    handle_inspect_sqlite_schema,
    handle_lookup_field,
    handle_read_csv,
    handle_read_doc,
    handle_read_json,
    handle_search_context,
    handle_validate_answer,
)
from data_agent_baseline.tools.inputs import (
    ANSWER_SCHEMA,
    EXECUTE_CONTEXT_SQL_SCHEMA,
    EXECUTE_PYTHON_SCHEMA,
    EXECUTE_PYTHON_TIMEOUT_SECONDS,
    INSPECT_FILES_SCHEMA,
    INSPECT_SQLITE_SCHEMA,
    LOOKUP_FIELD_SCHEMA,
    READ_CSV_SCHEMA,
    READ_DOC_SCHEMA,
    READ_JSON_SCHEMA,
    SEARCH_CONTEXT_SCHEMA,
    VALIDATE_ANSWER_SCHEMA,
    AnswerInput,
    ExecuteContextSqlInput,
    ExecutePythonInput,
    InspectFilesInput,
    InspectSqliteSchemaInput,
    LookupFieldInput,
    ReadCsvInput,
    ReadDocInput,
    ReadJsonInput,
    SearchContextInput,
    ValidateAnswerInput,
)
from data_agent_baseline.tools.schema_normalize import normalize_for_vllm

# 别名：`tests/test_schema_normalizer.py` 历史上 `from ...registry import _normalize_for_vllm`，
# 不想动测试就在此处保留下划线名。新代码请直接用 `normalize_for_vllm`。
_normalize_for_vllm = normalize_for_vllm

__all__ = [
    "EXECUTE_PYTHON_TIMEOUT_SECONDS",
    "ToolDefinition",
    "ToolExecutionResult",
    "ToolHandler",
    "ToolRegistry",
    "_normalize_for_vllm",
    "create_default_tool_registry",
]


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


# 工具名 → 对应入参模型；ToolRegistry.execute 用这张表把原始 dict 翻译成已验证模型实例。
# 仅内置工具登记于此；未登记的（用户自定义）工具走原始 dict 透传，保留向后兼容。
_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "inspect_files": InspectFilesInput,
    "read_csv": ReadCsvInput,
    "read_json": ReadJsonInput,
    "read_doc": ReadDocInput,
    "search_context": SearchContextInput,
    "lookup_field": LookupFieldInput,
    "inspect_sqlite_schema": InspectSqliteSchemaInput,
    "execute_context_sql": ExecuteContextSqlInput,
    "execute_python": ExecutePythonInput,
    "validate_answer": ValidateAnswerInput,
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
    """构造默认工具注册表。

    注册顺序（按规范固定）：
    1. inspect_files          一次性回传所有文件的轻量 schema（首步探查的零决策入口）
    2. read_csv               预览 CSV
    3. read_doc               预览文本文档
    4. search_context         Local Retrieval Index 全文检索
    5. lookup_field           Local Retrieval Index 字段/单位/别名检索
    6. read_json              预览 JSON
    7. inspect_sqlite_schema  sqlite 表结构
    8. execute_context_sql    sqlite 只读 SQL
    9. execute_python         子进程执行任意 Python
    10. validate_answer       提交前 sanity check（非终止）
    11. answer                终止 + 提交答案（唯一终止工具）

    `input_schema` 里的示例值是"演示"形式（不是真实 JSON Schema），模型看到
    `"columns": ["column_name"]` 会按模式填自己的数据，实测比严谨 schema 更不易出错。
    `json_schema` 则是给 OpenAI function calling 用的严格 schema。
    """
    definitions: list[ToolDefinition] = [
        ToolDefinition(
            name="inspect_files",
            description=(
                "Use first to bootstrap context exploration: returns a one-shot summary "
                "of every file under context (columns, dtypes, row counts, table schemas). "
                "This replaces the usual sequence of read_csv / read_json / "
                "inspect_sqlite_schema calls for initial discovery."
            ),
            input_schema=InspectFilesInput.example_value(),
            json_schema=INSPECT_FILES_SCHEMA,
            handler=handle_inspect_files,
        ),
        ToolDefinition(
            name="read_csv",
            description=(
                "Use when you need to inspect a single CSV file: returns columns, "
                "approximate dtypes, row_count, head (first 20 data rows), and tail "
                "(last 5 data rows that are not already in head). This is a SAMPLE "
                "preview, not full data access. For filtering, aggregation, joins, or "
                "any full-file scan, use execute_python instead — the head/tail will "
                "miss rows in between."
            ),
            input_schema=ReadCsvInput.example_value(),
            json_schema=READ_CSV_SCHEMA,
            handler=handle_read_csv,
        ),
        ToolDefinition(
            name="read_doc",
            description=(
                "Use when you need to preview a text-like document (md/txt). Two "
                "modes: (1) paging — without `keyword`, returns a fixed line window; "
                "use `has_more` + next `start_line` to continue. (2) keyword search "
                "— with `keyword`, returns ALL paragraphs containing the keyword "
                "across the whole file (capped at 20 matches / 8KB total). Prefer "
                "keyword mode when looking for an ID/name/term — one call replaces "
                "a sequential scan. This is a SAMPLE preview, not full data access. "
                "For full-document regex extraction, cross-paragraph aggregation, "
                "or structured parsing, use execute_python instead."
            ),
            input_schema=ReadDocInput.example_value(),
            json_schema=READ_DOC_SCHEMA,
            handler=handle_read_doc,
        ),
        ToolDefinition(
            name="search_context",
            description=(
                "Use when the local retrieval index is enabled and you need to find "
                "evidence in long local documents, knowledge files, or schema summaries "
                "by entity name, ID, phrase, or keyword. Returns path, line range, score, "
                "and text preview. Retrieval results are evidence pointers, not final "
                "answers; use execute_python or SQL for full computation."
            ),
            input_schema=SearchContextInput.example_value(),
            json_schema=SEARCH_CONTEXT_SCHEMA,
            handler=handle_search_context,
        ),
        ToolDefinition(
            name="lookup_field",
            description=(
                "Use when the local retrieval index is enabled and you need to resolve "
                "field meanings, units, code labels, aliases, thresholds, table names, "
                "or column candidates from knowledge.md and schema summaries before "
                "filtering or joining data."
            ),
            input_schema=LookupFieldInput.example_value(),
            json_schema=LOOKUP_FIELD_SCHEMA,
            handler=handle_lookup_field,
        ),
        ToolDefinition(
            name="read_json",
            description=(
                "Use when you need to preview a single JSON file: returns a "
                "structured summary keyed by `kind` — `array` (length + first 20 "
                "items), `object` (key_count + first 50 keys + value_preview for "
                "first 20 keys), or `scalar` (the value). This is a SAMPLE "
                "preview, not full data access. For nested field extraction, "
                "filtering, cross-file joins, or deep traversal, use execute_python "
                "instead. Files larger than 100MB return a streamed schema "
                "(`kind` + `keys` + `sample_value_kinds` for object; `length` + "
                "`first_item_kind` for array) without full load."
            ),
            input_schema=ReadJsonInput.example_value(),
            json_schema=READ_JSON_SCHEMA,
            handler=handle_read_json,
        ),
        ToolDefinition(
            name="inspect_sqlite_schema",
            description=(
                "Use first when you find a sqlite/db file (`.db` / `.sqlite` / "
                "`.sqlite3`) and need to know its tables, columns, and schema "
                "before writing SQL. "
                # 回归 run 20260430-013 task_16 类故障：模型对 `csv/trans.csv` 调
                # execute_context_sql。OpenAI tools[].description 在 native function
                # calling 下 attention 高于 system prompt 中的 Decision-tree bullet，
                # 所以 do-NOT 必须落在 description 才能可靠覆盖 routing 偏置。
                # 失败签名 + 备选工具路由不在此处复述：由 sqlite.py 的
                # `_maybe_raise_non_sqlite_path_error` 在违规时翻译成 observation.error
                # 下发——description 只负责"预防"，运行时负责"纠正"，分工互补，避免
                # 在 description 里维护两套独立的扩展名→工具映射（已踩过两段措辞不
                # 一致的坑）。
                "do NOT call on `.csv` / `.json` / `.md` / `.txt` paths."
            ),
            input_schema=InspectSqliteSchemaInput.example_value(),
            json_schema=INSPECT_SQLITE_SCHEMA,
            handler=handle_inspect_sqlite_schema,
        ),
        ToolDefinition(
            name="execute_context_sql",
            description=(
                "Use when data in a sqlite/db file (`.db` / `.sqlite` / `.sqlite3`) "
                "can be answered with read-only SQL queries or aggregations. Call "
                "inspect_sqlite_schema first for an unknown database. "
                # 同 inspect_sqlite_schema：description 仅 do-NOT 预防，
                # 失败签名 + 备选工具路由由 sqlite.py 运行时翻译。
                "do NOT call on `.csv` / `.json` / `.md` / `.txt` paths."
            ),
            input_schema=ExecuteContextSqlInput.example_value(),
            json_schema=EXECUTE_CONTEXT_SQL_SCHEMA,
            handler=handle_execute_context_sql,
        ),
        ToolDefinition(
            name="execute_python",
            description=(
                "Use when you need computation, aggregation, cross-file joins, or "
                "large-file scans beyond the preview tools. Working directory: the "
                "task context dir; stdout returns as `output`. "
                # 动态把超时秒数写进描述，避免改常量后忘记更新文案
                f"Timeout: {EXECUTE_PYTHON_TIMEOUT_SECONDS} s per call. "
                # `_DATA_OUTPUT_RULES` 的 pandas 强制只覆盖 30+rows / 200+cells 的 artifact 写入路径；
                # 小答案与读路径不在它的覆盖里，这里的"default to pandas"是补这个独家信号。
                # WH-target / 三段式投影 / cell-type 保真等细则不再复述（_DATA_OUTPUT_RULES 已写明）。
                "Default to `pandas` for tabular work — projections, filters, and "
                "joins are idiomatic and types are preserved automatically. "
                "Available libs: pandas, numpy, pyarrow, polars, openpyxl. "
                "The `csv` module is fine only for trivial reads (a handful of rows). "
                # 回归 run 20260430-013 task_16：模型先试 pd.read_sql_query(q, "sqlite:///...")
                # 吃 ImportError 再回落到 sqlite3.connect。SQLAlchemy 不在 runtime image 内，
                # URI 形式永远不可达——`do NOT` 框定行为反向，消除"先尝试再 fallback"的浪费步。
                "For `.db` / `.sqlite` files, use `sqlite3.connect(path)` + "
                "`pd.read_sql_query(query, conn)`; do NOT try the pandas URI form "
                '(`"sqlite:///..."`) — SQLAlchemy is not installed and it fails '
                "with `ImportError`."
            ),
            input_schema=ExecutePythonInput.example_value(),
            json_schema=EXECUTE_PYTHON_SCHEMA,
            handler=handle_execute_python,
        ),
        ToolDefinition(
            name="validate_answer",
            description=(
                "Use first to sanity-check the answer table before calling `answer`. "
                "Returns warnings (e.g., leaked ID columns, constant-value columns, "
                "empty table) so you can fix issues before final submission. "
                "Pass the same `columns`/`rows` you plan to submit, or "
                "`from_csv` pointing at the artifact CSV."
            ),
            input_schema=ValidateAnswerInput.example_value(),
            json_schema=VALIDATE_ANSWER_SCHEMA,
            handler=handle_validate_answer,
        ),
        ToolDefinition(
            name="answer",
            description=(
                "Use this when the final answer table is determined. This is the only "
                "way to submit the answer to the user/evaluator; plain text answers "
                "will be ignored. IMPORTANT: for content-entity questions "
                '("what is the comment/post/message/title"), columns must contain '
                "the text body (Text/Body/Title), NOT a surrogate key (Id/Code)."
            ),
            input_schema=AnswerInput.example_value(),
            json_schema=ANSWER_SCHEMA,
            handler=handle_answer,
            is_terminal=True,  # 唯一终止工具
        ),
    ]
    return ToolRegistry(definitions={d.name: d for d in definitions})
