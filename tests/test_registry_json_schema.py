"""测试 ToolSpec 的严格 JSON Schema 字段与 ToolRegistry.to_openai_tools 的输出。

覆盖点：
- 所有默认工具都声明了 json_schema（native function calling 必需）
- 每个 json_schema 是合法的 object 结构（type/properties/required/additionalProperties）
- required 字段里的键都真实存在于 properties
- `to_openai_tools` 输出符合 OpenAI tools 请求字段的结构（type=function + function.name/description/parameters）
"""

from __future__ import annotations

import pytest

from data_agent_baseline.tools.registry import create_default_tool_registry

EXPECTED_REQUIRED: dict[str, set[str]] = {
    "list_context": set(),
    "read_csv": {"path"},
    "read_json": {"path"},
    "read_doc": {"path"},
    "inspect_sqlite_schema": {"path"},
    "execute_context_sql": {"path", "sql"},
    "execute_python": {"code"},
    "answer": {"columns", "rows"},
}


@pytest.fixture
def registry():
    return create_default_tool_registry()


def test_every_default_tool_has_json_schema(registry) -> None:
    for name, spec in registry.specs.items():
        assert spec.json_schema is not None, f"tool {name} is missing json_schema"


def test_json_schema_structure_is_valid(registry) -> None:
    for name, spec in registry.specs.items():
        schema = spec.json_schema
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
        schema = registry.specs[name].json_schema
        assert schema is not None
        assert set(schema["required"]) == expected, (
            f"{name} required fields mismatch: expected {expected}, got {set(schema['required'])}"
        )


def test_to_openai_tools_produces_expected_shape(registry) -> None:
    tools = registry.to_openai_tools()

    assert len(tools) == len(registry.specs), (
        "Every default tool should be surfaced to OpenAI native function calling"
    )
    # 排序应与 specs 的排序一致（便于 prompt cache 命中）
    names = [tool["function"]["name"] for tool in tools]
    assert names == sorted(registry.specs.keys())

    for tool in tools:
        assert tool["type"] == "function"
        function_payload = tool["function"]
        assert isinstance(function_payload["name"], str) and function_payload["name"]
        assert isinstance(function_payload["description"], str) and function_payload["description"]
        parameters = function_payload["parameters"]
        assert parameters["type"] == "object"
        assert parameters.get("additionalProperties") is False


def test_to_openai_tools_skips_specs_without_json_schema() -> None:
    # 构造一个最简 registry，其中一个 spec 没有 json_schema
    from data_agent_baseline.tools.registry import ToolRegistry, ToolSpec

    registry = ToolRegistry(
        specs={
            "ok_tool": ToolSpec(
                name="ok_tool",
                description="ok",
                input_schema={},
                json_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            ),
            "schema_less": ToolSpec(
                name="schema_less",
                description="no schema",
                input_schema={},
                json_schema=None,
            ),
        },
        handlers={},
    )

    tools = registry.to_openai_tools()
    names = [tool["function"]["name"] for tool in tools]
    assert names == ["ok_tool"]


def test_answer_schema_accepts_heterogeneous_row_cells(registry) -> None:
    """answer 的 rows 单元格需允许 string / number / boolean / null 四种标量。"""
    schema = registry.specs["answer"].json_schema
    assert schema is not None
    row_item_schema = schema["properties"]["rows"]["items"]
    assert row_item_schema["type"] == "array"
    cell_schema = row_item_schema["items"]
    any_of = cell_schema["anyOf"]
    allowed_types = {entry["type"] for entry in any_of}
    assert allowed_types == {"string", "number", "boolean", "null"}
