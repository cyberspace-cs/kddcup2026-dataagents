"""``@function_tool`` decorator and ``FunctionToolSpec`` dataclass.

A tool is defined by decorating its handler with ``@function_tool``.  The
decorator captures the Pydantic input model, description, and handler into a
frozen ``FunctionToolSpec`` whose ``to_definition()`` materializes a
``ToolDefinition`` on demand (with optional description override for
sub-agent registries).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agents.tools.schema_normalize import normalize_for_vllm

if TYPE_CHECKING:
    from pydantic import BaseModel

    from agents.tools.registry import ToolDefinition, ToolHandler


@dataclass(frozen=True, slots=True)
class FunctionToolSpec:
    """Immutable spec produced by ``@function_tool``."""

    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler
    is_terminal: bool
    json_schema: dict[str, Any]

    def to_definition(self, *, description: str | None = None) -> ToolDefinition:
        """Materialize into ``ToolDefinition``.

        *description* override lets sub-agent registries (explorer) reuse the
        same handler with shorter context-specific descriptions.
        """
        from agents.tools.registry import ToolDefinition as _ToolDefinition

        return _ToolDefinition(
            name=self.name,
            description=description or self.description,
            json_schema=self.json_schema,
            handler=self.handler,
            is_terminal=self.is_terminal,
            input_model=self.input_model,
        )


def function_tool(
    name: str,
    *,
    description: str,
    input_model: type[BaseModel],
    is_terminal: bool = False,
) -> Any:
    """Decorator: wraps a handler into a ``FunctionToolSpec``.

    ``json_schema`` is computed once at decoration time via
    ``normalize_for_vllm``.
    """

    def decorator(handler: ToolHandler) -> FunctionToolSpec:
        schema = normalize_for_vllm(input_model.model_json_schema())
        return FunctionToolSpec(
            name=name,
            description=description,
            input_model=input_model,
            handler=handler,
            is_terminal=is_terminal,
            json_schema=schema,
        )

    return decorator
