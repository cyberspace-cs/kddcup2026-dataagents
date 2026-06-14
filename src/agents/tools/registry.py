"""Tool registry: ``ToolDefinition`` / ``ToolRegistry`` + factory.

Responsibilities:
- ``ToolDefinition`` / ``ToolRegistry``: immutable structures
- ``ToolExecutionResult`` / ``ToolHandler``: canonical return type + handler signature
- ``create_default_tool_registry()``: builds the standard tool set from ``defs/``
- Input validation via ``definition.input_model`` (no global lookup table)

Implementation details live in sibling modules:
- ``tools/contracts.py``        — shared constants
- ``tools/descriptions.py``     — tool description text
- ``tools/schema_normalize.py`` — ``normalize_for_vllm``
- ``tools/defs/``               — self-contained tool definitions
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from agents.benchmark.schema import AnswerTable, PublicTask
from agents.tools.contracts import EXECUTE_PYTHON_TIMEOUT_SECONDS
from agents.tools.schema_normalize import normalize_for_vllm

# Alias kept for ``tests/test_schema_normalizer.py`` historical import path.
_normalize_for_vllm = normalize_for_vllm


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Uniform return structure for tool handlers."""

    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


ToolHandler = Callable[[PublicTask, Any], ToolExecutionResult]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Full definition of a single tool: prompt-side + execution-side."""

    name: str
    description: str
    json_schema: dict[str, Any] | None
    handler: ToolHandler
    is_terminal: bool = False
    input_model: type[BaseModel] | None = None


def _format_validation_error(tool_name: str, error: ValidationError) -> ValueError:
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
        msg = str(first["msg"]).removeprefix("Value error, ")
        if field_path:
            return ValueError(f"{tool_name}: field '{field_path}' {msg}")
        return ValueError(f"{tool_name}: {msg}")
    return ValueError(f"{tool_name}: field '{field_path}' {first['msg']}")


@dataclass(frozen=True, slots=True)
class ToolRegistry:
    """Immutable name → ToolDefinition mapping."""

    definitions: Mapping[str, ToolDefinition]

    def __post_init__(self) -> None:
        object.__setattr__(self, "definitions", MappingProxyType(dict(self.definitions)))

    def execute(
        self, task: PublicTask, action: str, action_input: dict[str, Any]
    ) -> ToolExecutionResult:
        if action not in self.definitions:
            raise KeyError(f"Unknown tool: {action}")
        defn = self.definitions[action]
        model_cls = defn.input_model
        if model_cls is None:
            return defn.handler(task, action_input)
        try:
            validated = model_cls.model_validate(action_input)
        except ValidationError as exc:
            raise _format_validation_error(action, exc) from exc
        return defn.handler(task, validated)

    def to_openai_tools(self) -> list[dict[str, Any]]:
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


def create_default_tool_registry(
    *,
    explore_tool: ToolDefinition | None = None,
) -> ToolRegistry:
    """Build the default tool registry from ``defs/`` specs.

    Registration order (fixed by convention):
    1. explore / inspect_files  — first-step exploration
    2. read_csv                 — CSV preview
    3. read_doc                 — text document preview
    4. search_context           — retrieval index full-text search
    5. lookup_field             — retrieval index field/unit/alias lookup
    6. read_json                — JSON preview
    7. inspect_sqlite_schema    — SQLite table structure
    8. execute_context_sql      — read-only SQL
    9. grep_context             — regex search across files
    10. execute_python          — subprocess Python execution
    11. answer                  — terminal: submit answer (only terminal tool)
    """
    from agents.tools import defs

    first_tool = explore_tool or defs.inspect_files.to_definition()
    definitions: list[ToolDefinition] = [
        first_tool,
        defs.read_csv.to_definition(),
        defs.read_doc.to_definition(),
        defs.search_context.to_definition(),
        defs.lookup_field.to_definition(),
        defs.read_json.to_definition(),
        defs.inspect_sqlite_schema.to_definition(),
        defs.execute_context_sql.to_definition(),
        defs.grep_context.to_definition(),
        defs.execute_python.to_definition(),
        defs.answer.to_definition(),
    ]
    return ToolRegistry(definitions={d.name: d for d in definitions})


__all__ = [
    "EXECUTE_PYTHON_TIMEOUT_SECONDS",
    "ToolDefinition",
    "ToolExecutionResult",
    "ToolHandler",
    "ToolRegistry",
    "_normalize_for_vllm",
    "create_default_tool_registry",
]
