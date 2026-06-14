"""execute_context_sql tool: read-only SQL execution on SQLite databases."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from agents.benchmark.schema import PublicTask
from agents.tools.contracts import PATH_CONVENTION_NOTE
from agents.tools.decorator import function_tool
from agents.tools.defs._fields import positive_int_field
from agents.tools.filesystem import resolve_context_path
from agents.tools.registry import ToolExecutionResult
from agents.tools.sqlite import execute_read_only_sql


@function_tool
def execute_context_sql(
    task: PublicTask,
    path: Annotated[
        str,
        Field(
            description=(
                "Relative path to the sqlite/db file. "
                "Examples: 'db/clinical.sqlite', 'main.db'. " + PATH_CONVENTION_NOTE
            ),
        ),
    ],
    sql: Annotated[str, Field(description="Read-only SQL query to execute.")],
    limit: Annotated[
        int,
        positive_int_field(description="Maximum number of rows to return."),
    ] = 200,
) -> ToolExecutionResult:
    """Use when data in a sqlite/db file (`.db` / `.sqlite` / `.sqlite3`)
    can be answered with read-only SQL queries or aggregations. Call
    inspect_sqlite_schema first for an unknown database.
    do NOT call on `.csv` / `.json` / `.md` / `.txt` paths."""
    resolved = resolve_context_path(task, path)
    content = execute_read_only_sql(resolved, sql, limit=limit)
    if content.get("row_count", -1) == 0:
        content["warning"] = (
            "Query returned 0 rows. Verify filter values with "
            "SELECT DISTINCT or check for case/format mismatches."
        )
    return ToolExecutionResult(ok=True, content=content)
