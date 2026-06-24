"""Shared SQLite helpers for tool modules."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

_SPECIAL_CHAR_RE = re.compile(r"[() /\-]")

_NON_SQLITE_PATH_REDIRECT: dict[str, str] = {
    ".csv": "`preview_file` or `execute_python` (full scan with pandas)",
    ".json": "`preview_file` or `execute_python` (nested extraction)",
    ".md": "`preview_file`",
    ".txt": "`preview_file`",
}


def maybe_raise_non_sqlite_path_error(path: Path, exc: sqlite3.DatabaseError) -> None:
    """Translate sqlite's opaque non-database error into an actionable tool hint."""
    if "file is not a database" not in str(exc).lower():
        return
    ext = path.suffix.lower()
    redirect = _NON_SQLITE_PATH_REDIRECT.get(ext, "`execute_python`")
    raise ValueError(
        f"Path '{path.name}' is not a sqlite database "
        f"(extension: {ext or 'none'}). "
        f"For this file type, use {redirect} instead of "
        f"preview_file / execute_context_sql."
    ) from exc


def auto_quote_identifiers(sql: str, conn: sqlite3.Connection) -> str:
    """Auto-quote table/column names containing special characters like ``() / -``."""
    tables = [
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]

    identifiers: list[str] = []
    for tbl in tables:
        if tbl in sql or f'"{tbl}"' in sql:
            cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{tbl}")').fetchall()]
            identifiers.extend(cols)
        identifiers.append(tbl)

    need_quote = sorted(
        {name for name in identifiers if _SPECIAL_CHAR_RE.search(name)},
        key=len,
        reverse=True,
    )

    for name in need_quote:
        pattern = r'(?<!")' + re.escape(name) + r'(?!")'
        sql = re.sub(pattern, f'"{name}"', sql)

    return sql


def connect_read_only(path: Path) -> sqlite3.Connection:
    """Connect to a SQLite file in URI read-only mode."""
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)
