"""inspect_sqlite_schema tool: SQLite table structure inspection."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Annotated, Any

from agents.benchmark.schema import PublicTask
from agents.tools._fields import path_field
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


def inspect_sqlite_database(path: Path) -> dict[str, Any]:
    """List user tables and their CREATE statements."""
    try:
        with _connect_read_only(path) as conn:
            rows = conn.execute(
                """
                SELECT name, sql
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()
    except sqlite3.DatabaseError as exc:
        _maybe_raise_non_sqlite_path_error(path, exc)
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
