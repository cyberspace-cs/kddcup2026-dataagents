"""inspect_sqlite_schema tool: SQLite table structure inspection."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict

from agents.benchmark.schema import PublicTask
from agents.tools.decorator import function_tool
from agents.tools.defs._fields import path_field
from agents.tools.descriptions import INSPECT_SQLITE_SCHEMA_DESCRIPTION
from agents.tools.filesystem import resolve_context_path
from agents.tools.registry import ToolExecutionResult
from agents.tools.sqlite import inspect_sqlite_schema as _inspect_sqlite_schema


class InspectSqliteSchemaInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Annotated[
        str,
        path_field(
            file_kind="sqlite database file",
            examples=("db/clinical.sqlite", "main.db"),
        ),
    ]


@function_tool(
    name="inspect_sqlite_schema",
    description=INSPECT_SQLITE_SCHEMA_DESCRIPTION,
    input_model=InspectSqliteSchemaInput,
)
def inspect_sqlite_schema(task: PublicTask, args: InspectSqliteSchemaInput) -> ToolExecutionResult:
    path = resolve_context_path(task, args.path)
    return ToolExecutionResult(ok=True, content=_inspect_sqlite_schema(path))
