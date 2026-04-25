"""工具注册表（Tool Registry）的公开 API 与构造工厂。

本模块只承担三件事：
- 暴露 `ToolDefinition` / `ToolRegistry` 两个不可变结构 + `ToolExecutionResult`
  返回类型（最后一个从 `handlers.py` 重导出，保持历史导入路径）
- 将工具名映射到 Pydantic 入参模型，校验失败时翻译成单行可操作的 `ValueError`
- `create_default_tool_registry()` 按规范顺序构造默认 8 件套

实现细节按职责拆到三个邻居模块：
- `tools/_schema_normalize.py`  ← `normalize_for_vllm`（vLLM 字节稳定形态）
- `tools/inputs.py`             ← 8 个 Pydantic 入参模型 + 字段工厂 + `*_SCHEMA` 常量
- `tools/handlers.py`           ← 8 个 `handle_xxx` handler + `ToolExecutionResult`/`ToolHandler`

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
from data_agent_baseline.tools._schema_normalize import normalize_for_vllm
from data_agent_baseline.tools.handlers import (
    ToolExecutionResult,
    ToolHandler,
    handle_answer,
    handle_execute_context_sql,
    handle_execute_python,
    handle_inspect_sqlite_schema,
    handle_list_context,
    handle_read_csv,
    handle_read_doc,
    handle_read_json,
)
from data_agent_baseline.tools.inputs import (
    ANSWER_SCHEMA,
    EXECUTE_CONTEXT_SQL_SCHEMA,
    EXECUTE_PYTHON_SCHEMA,
    EXECUTE_PYTHON_TIMEOUT_SECONDS,
    INSPECT_SQLITE_SCHEMA,
    LIST_CONTEXT_SCHEMA,
    READ_CSV_SCHEMA,
    READ_DOC_SCHEMA,
    READ_JSON_SCHEMA,
    AnswerInput,
    ExecuteContextSqlInput,
    ExecutePythonInput,
    InspectSqliteSchemaInput,
    ListContextInput,
    ReadCsvInput,
    ReadDocInput,
    ReadJsonInput,
)

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
            json_schema=LIST_CONTEXT_SCHEMA,
            handler=handle_list_context,
        ),
        ToolDefinition(
            name="read_csv",
            description=(
                "Use when you need to preview a CSV file, confirm column names, or "
                "inspect example rows before choosing a computation tool."
            ),
            input_schema=ReadCsvInput.example_value(),
            json_schema=READ_CSV_SCHEMA,
            handler=handle_read_csv,
        ),
        ToolDefinition(
            name="read_doc",
            description=(
                "Use when you need to preview a text-like document, confirm relevant "
                "text, or search for the first case-insensitive keyword match. Use "
                "`has_more` and the next `start_line` to continue reading."
            ),
            input_schema=ReadDocInput.example_value(),
            json_schema=READ_DOC_SCHEMA,
            handler=handle_read_doc,
        ),
        ToolDefinition(
            name="read_json",
            description=(
                "Use when you need to preview a JSON file, confirm top-level fields, "
                "or inspect sample structured records."
            ),
            input_schema=ReadJsonInput.example_value(),
            json_schema=READ_JSON_SCHEMA,
            handler=handle_read_json,
        ),
        ToolDefinition(
            name="inspect_sqlite_schema",
            description=(
                "Use first when you find a sqlite/db file and need to know its "
                "tables, columns, and schema before writing SQL."
            ),
            input_schema=InspectSqliteSchemaInput.example_value(),
            json_schema=INSPECT_SQLITE_SCHEMA,
            handler=handle_inspect_sqlite_schema,
        ),
        ToolDefinition(
            name="execute_context_sql",
            description=(
                "Use when data in a sqlite/db file can be answered with read-only SQL "
                "queries or aggregations. Call inspect_sqlite_schema first for an "
                "unknown database."
            ),
            input_schema=ExecuteContextSqlInput.example_value(),
            json_schema=EXECUTE_CONTEXT_SQL_SCHEMA,
            handler=handle_execute_context_sql,
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
            json_schema=EXECUTE_PYTHON_SCHEMA,
            handler=handle_execute_python,
        ),
        ToolDefinition(
            name="answer",
            description=(
                "Use this when the final answer table is determined. This is the only "
                "way to submit the answer to the user/evaluator; plain text answers "
                "will be ignored."
            ),
            input_schema=AnswerInput.example_value(),
            json_schema=ANSWER_SCHEMA,
            handler=handle_answer,
            is_terminal=True,  # 唯一终止工具
        ),
    ]
    return ToolRegistry(definitions={d.name: d for d in definitions})
