"""execute_context_sql tool: read-only SQL execution on SQLite databases."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from agents.benchmark.schema import PublicTask
from agents.tools.contracts import PATH_CONVENTION_NOTE
from agents.tools.decorator import function_tool
from agents.tools.defs._fields import positive_int_field
from agents.tools.descriptions import EXECUTE_CONTEXT_SQL_DESCRIPTION
from agents.tools.filesystem import resolve_context_path
from agents.tools.registry import ToolExecutionResult
from agents.tools.sqlite import execute_read_only_sql


class ExecuteContextSqlInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
        positive_int_field(
            default=200,
            description="Maximum number of rows to return.",
        ),
    ]


@function_tool(
    name="execute_context_sql",
    description=EXECUTE_CONTEXT_SQL_DESCRIPTION,
    input_model=ExecuteContextSqlInput,
)
def execute_context_sql(task: PublicTask, args: ExecuteContextSqlInput) -> ToolExecutionResult:
    path = resolve_context_path(task, args.path)
    content = execute_read_only_sql(path, args.sql, limit=args.limit)
    if content.get("row_count", -1) == 0:
        content["warning"] = (
            "Query returned 0 rows. Verify filter values with "
            "SELECT DISTINCT or check for case/format mismatches."
        )
    return ToolExecutionResult(ok=True, content=content)
