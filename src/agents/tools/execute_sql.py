"""execute_context_sql tool: read-only SQL execution on SQLite databases."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from agents.benchmark.schema import PublicTask
from agents.tools._fields import positive_int_field
from agents.tools._sqlite_common import connect_read_only, maybe_raise_non_sqlite_path_error
from agents.tools.context import resolve_context_path
from agents.tools.contracts import PATH_CONVENTION_NOTE
from agents.tools.decorator import function_tool
from agents.tools.registry import ToolExecutionResult


def execute_read_only_sql(path: Path, sql: str, *, limit: int = 200) -> dict[str, Any]:
    """Execute SELECT/WITH/PRAGMA against a SQLite database with row truncation."""
    normalized_sql = sql.lstrip().lower()
    if not normalized_sql.startswith(("select", "with", "pragma")):
        raise ValueError("Only read-only SQL statements are allowed.")

    try:
        with connect_read_only(path) as conn:
            cursor = conn.execute(sql)
            column_names = [item[0] for item in cursor.description or []]
            rows = cursor.fetchmany(limit + 1)
    except sqlite3.DatabaseError as exc:
        maybe_raise_non_sqlite_path_error(path, exc)
        raise

    truncated = len(rows) > limit
    limited_rows = rows[:limit]
    return {
        "columns": column_names,
        "rows": [list(row) for row in limited_rows],
        "row_count": len(limited_rows),
        "truncated": truncated,
    }


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
    """Run a read-only SQL query (SELECT/WITH/PRAGMA) on a SQLite database.
    Returns {columns, rows, row_count, truncated}; default limit 200 rows.
    Call inspect_sqlite_schema first for an unknown database to learn
    table and column names. For complex multi-table JOINs across
    heterogeneous files or write operations, use execute_python instead.
    Do NOT call on .csv/.json/.md/.txt paths.
    Example: execute_context_sql({"path": "db/sales.db",
    "sql": "SELECT product, SUM(amount) FROM orders GROUP BY product LIMIT 10"})"""
    resolved = resolve_context_path(task, path)
    content = execute_read_only_sql(resolved, sql, limit=limit)
    if content.get("row_count", -1) == 0:
        content["warning"] = (
            "Query returned 0 rows. Verify filter values with "
            "SELECT DISTINCT or check for case/format mismatches."
        )
    return ToolExecutionResult(ok=True, content=content)
