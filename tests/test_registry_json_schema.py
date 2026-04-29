"""测试 ToolDefinition 的严格 JSON Schema 字段与 ToolRegistry.to_openai_tools 的输出。

覆盖点：
- 所有默认工具都声明了 json_schema（native function calling 必需）
- 每个 json_schema 是合法的 object 结构（type/properties/required/additionalProperties）
- required 字段里的键都真实存在于 properties
- `to_openai_tools` 输出符合 OpenAI tools 请求字段的结构（type=function + function.name/description/parameters）
- `to_openai_tools` 的输出字节稳定（重复调用得到完全相同的 JSON）
"""

from __future__ import annotations

import json

import pytest

from data_agent_baseline.tools.registry import (
    ToolDefinition,
    ToolExecutionResult,
    ToolRegistry,
    create_default_tool_registry,
)

EXPECTED_REQUIRED: dict[str, set[str]] = {
    "inspect_files": set(),
    "read_csv": {"path"},
    "read_json": {"path"},
    "read_doc": {"path"},
    "inspect_sqlite_schema": {"path"},
    "execute_context_sql": {"path", "sql"},
    "execute_python": {"code"},
    # answer 现在三字段都可选：inline columns/rows 与 from_csv 通过 model_validator 形成 XOR。
    # required=[] 在 schema 层把决策权交给 validator，模型仍能拿到结构化错误。
    "answer": set(),
}


@pytest.fixture
def registry():
    return create_default_tool_registry()


def test_every_default_tool_has_json_schema(registry) -> None:
    for name, definition in registry.definitions.items():
        assert definition.json_schema is not None, f"tool {name} is missing json_schema"


def test_json_schema_structure_is_valid(registry) -> None:
    for name, definition in registry.definitions.items():
        schema = definition.json_schema
        assert schema is not None
        assert schema["type"] == "object", f"{name} schema type must be object"
        assert "properties" in schema and isinstance(schema["properties"], dict)
        assert "required" in schema and isinstance(schema["required"], list)
        assert schema.get("additionalProperties") is False, (
            f"{name} schema must set additionalProperties: false for constrained decoding"
        )
        # required 的每个键都应能在 properties 中找到
        for required_field in schema["required"]:
            assert required_field in schema["properties"], (
                f"{name} required field {required_field!r} missing from properties"
            )


def test_json_schema_required_keys_match_expected(registry) -> None:
    for name, expected in EXPECTED_REQUIRED.items():
        schema = registry.definitions[name].json_schema
        assert schema is not None
        assert set(schema["required"]) == expected, (
            f"{name} required fields mismatch: expected {expected}, got {set(schema['required'])}"
        )


def test_default_tool_descriptions_are_use_case_oriented(registry) -> None:
    for name, definition in registry.definitions.items():
        description = definition.description.lower()
        assert any(
            phrase in description for phrase in ("use when", "use first", "use this when")
        ), f"{name} description should explain when to use the tool"


def test_answer_description_explains_submission_contract(registry) -> None:
    description = registry.definitions["answer"].description.lower()

    assert "only" in description
    assert "submit" in description
    assert "plain text" in description
    assert "ignored" in description


def test_execute_python_description_recommends_pandas(registry) -> None:
    """execute_python 描述必须把 pandas 推为 tabular 工作的默认。

    回归 run 20260429T210729531308Z task_8：模型用 `csv.DictReader` + 手写
    `transactions.append({...全 10 列...})` 走"路径最阻力小"，导致 fieldnames
    照搬源 schema。pandas 的 `df[[...]].to_csv(...)` 让 column projection
    变成习惯写法；同时 import pandas 才能让模型看到自动类型保留 / 一行
    verification（`df.head()` / `df.dtypes`）的 ergonomic 优势。
    """
    description = registry.definitions["execute_python"].description.lower()

    # pandas 推荐 + 反对默认 csv stdlib
    assert "prefer `pandas` over the `csv` stdlib" in description
    # 关键 idiom：projection
    assert "df[['target_col']]" in description
    # 类型保留与 verification 的 ergonomic 卖点
    assert "no manual `int()`/`float()` casts" in description
    assert "df.head()" in description and "df.dtypes" in description
    # 显式列出 runtime 已装的库，避免模型怀疑可用性
    assert "pandas, numpy, pyarrow, polars, openpyxl" in description
    # csv stdlib 仍允许小场景，避免一刀切
    assert "trivial reads" in description


def test_to_openai_tools_produces_expected_shape(registry) -> None:
    tools = registry.to_openai_tools()

    assert len(tools) == len(registry.definitions), (
        "Every default tool should be surfaced to OpenAI native function calling"
    )
    # 排序应与 definitions 的字母序一致（便于 prompt cache 命中）
    names = [tool["function"]["name"] for tool in tools]
    assert names == sorted(registry.definitions.keys())

    for tool in tools:
        assert tool["type"] == "function"
        function_payload = tool["function"]
        assert isinstance(function_payload["name"], str) and function_payload["name"]
        assert isinstance(function_payload["description"], str) and function_payload["description"]
        parameters = function_payload["parameters"]
        assert parameters["type"] == "object"
        assert parameters.get("additionalProperties") is False


def test_to_openai_tools_is_byte_stable_across_calls(registry) -> None:
    """重复调用 to_openai_tools 必须返回字节相同的 JSON 序列化结果。"""
    first = json.dumps(registry.to_openai_tools(), sort_keys=True)
    second = json.dumps(registry.to_openai_tools(), sort_keys=True)
    assert first == second


def _noop_handler(_task, _action_input):
    return ToolExecutionResult(ok=True, content={})


def test_to_openai_tools_skips_definitions_without_json_schema() -> None:
    # 构造一个最简 registry，其中一个 definition 没有 json_schema
    registry = ToolRegistry(
        definitions={
            "ok_tool": ToolDefinition(
                name="ok_tool",
                description="ok",
                input_schema={},
                json_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                handler=_noop_handler,
            ),
            "schema_less": ToolDefinition(
                name="schema_less",
                description="no schema",
                input_schema={},
                json_schema=None,
                handler=_noop_handler,
            ),
        }
    )

    tools = registry.to_openai_tools()
    names = [tool["function"]["name"] for tool in tools]
    assert names == ["ok_tool"]


def test_answer_schema_accepts_heterogeneous_row_cells(registry) -> None:
    """answer 的 rows 单元格需允许 string / integer / number / boolean / null 五种标量。

    `integer` 单独枚举（而非靠 `number` 兼容）确保 Pydantic 在 union 中优先匹中 int，
    避免主键 ID 等整数被静默 coerce 成 float（在 prediction.csv 写出 "163109.0"
    导致与 gold 的 "163109" 字面失配）。
    """
    schema = registry.definitions["answer"].json_schema
    assert schema is not None
    row_item_schema = schema["properties"]["rows"]["items"]
    assert row_item_schema["type"] == "array"
    cell_schema = row_item_schema["items"]
    any_of = cell_schema["anyOf"]
    allowed_types = {entry["type"] for entry in any_of}
    assert allowed_types == {"string", "integer", "number", "boolean", "null"}
