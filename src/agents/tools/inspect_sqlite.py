"""inspect_sqlite_schema tool: SQLite table structure inspection."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Annotated, Any

from agents.benchmark.schema import PublicTask
from agents.tools._fields import path_field
from agents.tools._sqlite_common import connect_read_only, maybe_raise_non_sqlite_path_error
from agents.tools.context import resolve_context_path
from agents.tools.decorator import function_tool
from agents.tools.registry import ToolExecutionResult


def inspect_sqlite_database(path: Path) -> dict[str, Any]:
    """List user tables and their CREATE statements."""
    try:
        with connect_read_only(path) as conn:
            rows = conn.execute(
                """
                SELECT name, sql
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()
    except sqlite3.DatabaseError as exc:
        maybe_raise_non_sqlite_path_error(path, exc)
        raise
    tables: list[dict[str, Any]] = []
    for name, create_sql in rows:
        tables.append(
            {
                "name": name,
                "create_sql": create_sql,
            }
        )
    return {
        "path": str(path),
        "tables": tables,
    }


@function_tool
def inspect_sqlite_schema(
    task: PublicTask,
    path: Annotated[
        str,
        path_field(
            file_kind="sqlite database file",
            examples=("db/clinical.sqlite", "main.db"),
        ),
    ],
) -> ToolExecutionResult:
    """Use first when you find a sqlite/db file (`.db` / `.sqlite` /
    `.sqlite3`) and need to know its tables, columns, and schema
    before writing SQL. do NOT call on `.csv` / `.json` / `.md` / `.txt` paths."""
    return ToolExecutionResult(
        ok=True, content=inspect_sqlite_database(resolve_context_path(task, path))
    )
