"""Shared SQLite helpers for tool modules."""

from __future__ import annotations

import sqlite3
from pathlib import Path

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


def connect_read_only(path: Path) -> sqlite3.Connection:
    """Connect to a SQLite file in URI read-only mode."""
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)
