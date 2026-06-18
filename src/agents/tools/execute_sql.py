"""execute_context_sql tool: SQL execution on SQLite databases."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from agents.benchmark.schema import PublicTask
from agents.tools._fields import positive_int_field
from agents.tools._sqlite_common import (
    connect_read_only,
    connect_writable,
    guard_non_sqlite_path,
    maybe_raise_non_sqlite_path_error,
    validate_subquery_columns,
)
from agents.tools.context import resolve_context_path
from agents.tools.contracts import PATH_CONVENTION_NOTE
from agents.tools.decorator import function_tool
from agents.tools.registry import ToolExecutionResult

_TIMEOUT_HINT = (
    "Query timed out (>30 s). Possible causes: "
    "(1) full table scan on a large table — run EXPLAIN QUERY PLAN and "
    "CREATE INDEX on filter/join columns; "
    "(2) a subquery column that does not exist in the subquery table — "
    "SQLite silently resolves it from the outer query, making IN(...) always "
    "true and producing a cartesian product; "
    "(3) query is too broad — add WHERE filters or split into smaller queries."
)


def _get_schema_map(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Build {table_lower: [col_lower, ...]} from the database."""
    schema: dict[str, list[str]] = {}
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    for (table_name,) in tables:
        cols = conn.execute(f"PRAGMA table_info([{table_name}])").fetchall()
        schema[table_name.lower()] = [row[1].lower() for row in cols]
    return schema


def execute_sql(path: Path, sql: str, *, limit: int = 200) -> dict[str, Any]:
    """Execute SELECT/WITH/PRAGMA/EXPLAIN/CREATE INDEX against a SQLite database."""
    guard_non_sqlite_path(path)
    normalized = sql.strip()
    normalized_lower = normalized.lower()

    if normalized_lower.startswith(("create index", "create unique index")):
        return _handle_create_index(path, normalized)

    if normalized_lower.startswith("explain"):
        return _handle_explain(path, normalized)

    if not normalized_lower.startswith(("select", "with", "pragma")):
        raise ValueError(
            "Only SELECT, WITH, PRAGMA, EXPLAIN, and CREATE INDEX statements are allowed."
        )

    try:
        with connect_read_only(path) as conn:
            warnings = _validate_columns(conn, sql)
            if warnings:
                raise ValueError(
                    "Unsafe SQL subquery column reference: "
                    + " | ".join(warnings)
                    + " Re-check the schema and select a column that exists in the "
                    "subquery table before running this query."
                )
            cursor = conn.execute(sql)
            column_names = [item[0] for item in cursor.description or []]
            rows = cursor.fetchmany(limit + 1)
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise ValueError(_TIMEOUT_HINT) from exc
        maybe_raise_non_sqlite_path_error(path, exc)
        raise
    except sqlite3.DatabaseError as exc:
        maybe_raise_non_sqlite_path_error(path, exc)
        raise

    truncated = len(rows) > limit
    limited_rows = rows[:limit]
    result: dict[str, Any] = {
        "path": str(path),
        "columns": column_names,
        "rows": [list(row) for row in limited_rows],
        "row_count": len(limited_rows),
        "truncated": truncated,
    }
    return result


def _handle_create_index(path: Path, sql: str) -> dict[str, Any]:
    """Execute CREATE INDEX on a writable connection."""
    try:
        with connect_writable(path) as conn:
            conn.execute(sql)
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise ValueError(_TIMEOUT_HINT) from exc
        maybe_raise_non_sqlite_path_error(path, exc)
        raise
    except sqlite3.DatabaseError as exc:
        maybe_raise_non_sqlite_path_error(path, exc)
        raise
    return {"ok": True, "message": "Index created."}


def _handle_explain(path: Path, sql: str) -> dict[str, Any]:
    """Execute EXPLAIN QUERY PLAN on a read-only connection."""
    try:
        with connect_read_only(path) as conn:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise ValueError(_TIMEOUT_HINT) from exc
        maybe_raise_non_sqlite_path_error(path, exc)
        raise
    except sqlite3.DatabaseError as exc:
        maybe_raise_non_sqlite_path_error(path, exc)
        raise
    lines = [" | ".join(str(c) for c in row) for row in rows]
    return {"ok": True, "plan": "\n".join(lines)}


def _validate_columns(conn: sqlite3.Connection, sql: str) -> list[str]:
    """Run subquery column validation and return warnings."""
    try:
        schema = _get_schema_map(conn)
        return validate_subquery_columns(sql, schema)
    except Exception:
        return []


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
    sql: Annotated[
        str,
        Field(
            description="SQL statement to execute. Supports SELECT, WITH, PRAGMA, EXPLAIN QUERY PLAN, and CREATE INDEX."
        ),
    ],
    limit: Annotated[
        int,
        positive_int_field(description="Maximum number of rows to return."),
    ] = 200,
) -> ToolExecutionResult:
    """Use when data in a sqlite/db file (`.db` / `.sqlite` / `.sqlite3`)
    can be answered with SQL queries or aggregations. Call
    inspect_sqlite_schema first for an unknown database.
    For large tables (>5000 rows), run EXPLAIN QUERY PLAN to check for full
    scans, then CREATE INDEX IF NOT EXISTS idx_<table>_<col> on filter/join
    columns before querying.
    do NOT call on `.csv` / `.json` / `.md` / `.txt` paths."""
    resolved = resolve_context_path(task, path)
    content = execute_sql(resolved, sql, limit=limit)
    if content.get("row_count", -1) == 0:
        content.setdefault(
            "warning",
            "Query returned 0 rows. Verify filter values with "
            "SELECT DISTINCT or check for case/format mismatches.",
        )
    return ToolExecutionResult(ok=True, content=content)
