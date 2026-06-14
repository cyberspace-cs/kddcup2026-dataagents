"""execute_context_sql tool: read-only SQL execution on SQLite databases."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from agents.benchmark.schema import PublicTask
from agents.tools._fields import positive_int_field
from agents.tools.contracts import PATH_CONVENTION_NOTE
from agents.tools.decorator import function_tool
from agents.tools.filesystem import resolve_context_path
from agents.tools.registry import ToolExecutionResult

_NON_SQLITE_PATH_REDIRECT: dict[str, str] = {
    ".csv": "`read_csv` (preview) or `execute_python` (full scan with pandas)",
    ".json": "`read_json` (preview) or `execute_python` (nested extraction)",
    ".md": "`read_doc`",
    ".txt": "`read_doc`",
}


def _maybe_raise_non_sqlite_path_error(path: Path, exc: sqlite3.DatabaseError) -> None:
    """Translate sqlite's opaque non-database error into an actionable tool hint."""
    if "file is not a database" not in str(exc).lower():
        return
    ext = path.suffix.lower()
    redirect = _NON_SQLITE_PATH_REDIRECT.get(ext, "`execute_python`")
    raise ValueError(
        f"Path '{path.name}' is not a sqlite database "
        f"(extension: {ext or 'none'}). "
        f"For this file type, use {redirect} instead of "
        f"inspect_sqlite_schema / execute_context_sql."
    ) from exc


def _connect_read_only(path: Path) -> sqlite3.Connection:
    """Connect to a SQLite file in URI read-only mode."""
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def execute_read_only_sql(path: Path, sql: str, *, limit: int = 200) -> dict[str, Any]:
    """Execute SELECT/WITH/PRAGMA against a SQLite database with row truncation."""
    normalized_sql = sql.lstrip().lower()
    if not normalized_sql.startswith(("select", "with", "pragma")):
        raise ValueError("Only read-only SQL statements are allowed.")

    try:
        with _connect_read_only(path) as conn:
            cursor = conn.execute(sql)
            column_names = [item[0] for item in cursor.description or []]
            rows = cursor.fetchmany(limit + 1)
    except sqlite3.DatabaseError as exc:
        _maybe_raise_non_sqlite_path_error(path, exc)
        raise

    truncated = len(rows) > limit
    limited_rows = rows[:limit]
    return {
        "path": str(path),
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
