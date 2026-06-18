"""Shared SQLite helpers for tool modules."""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

from agents.tools.contracts import SQLITE_QUERY_TIMEOUT_SECONDS

_NON_SQLITE_PATH_REDIRECT: dict[str, str] = {
    ".csv": "`read_csv` (preview) or `execute_python` (full scan with pandas)",
    ".json": "`read_json` (preview) or `execute_python` (nested extraction)",
    ".md": "`read_doc`",
    ".txt": "`read_doc`",
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
        f"inspect_sqlite_schema / execute_context_sql."
    ) from exc


def guard_non_sqlite_path(path: Path) -> None:
    """Reject known non-SQLite extensions before opening a connection."""
    ext = path.suffix.lower()
    redirect = _NON_SQLITE_PATH_REDIRECT.get(ext)
    if redirect is not None:
        raise ValueError(
            f"Path '{path.name}' is not a sqlite database "
            f"(extension: {ext}). "
            f"For this file type, use {redirect} instead of "
            f"inspect_sqlite_schema / execute_context_sql."
        ) from None


def _install_query_timeout(
    conn: sqlite3.Connection, timeout: float = SQLITE_QUERY_TIMEOUT_SECONDS
) -> None:
    """Install a progress handler that raises OperationalError on timeout."""
    deadline = time.monotonic() + timeout

    def _check() -> int:
        if time.monotonic() >= deadline:
            return 1  # non-zero → interrupt
        return 0

    conn.set_progress_handler(_check, 10_000)


def connect_read_only(path: Path) -> sqlite3.Connection:
    """Connect to a SQLite file in URI read-only mode with query timeout."""
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    _install_query_timeout(conn)
    return conn


def connect_writable(path: Path) -> sqlite3.Connection:
    """Connect to a SQLite file in read-write mode (for DDL like CREATE INDEX)."""
    conn = sqlite3.connect(str(path.resolve()))
    _install_query_timeout(conn)
    return conn


# ---------------------------------------------------------------------------
# Subquery column validation
# ---------------------------------------------------------------------------

_SUBQUERY_RE = re.compile(
    r"""\(\s*SELECT\s+                 # opening paren + SELECT
    (?:DISTINCT\s+)?                   # optional DISTINCT
    (\w+)                              # captured column name
    \s+FROM\s+                         # FROM keyword
    (\w+)                              # captured table name
    """,
    re.IGNORECASE | re.VERBOSE,
)


def validate_subquery_columns(sql: str, schema: dict[str, list[str]]) -> list[str]:
    """Check that columns in (SELECT col FROM table) exist in the named table.

    *schema* maps table names (lower-cased) to lists of column names (lower-cased).
    Returns a list of warning strings (empty if everything checks out).
    """
    warnings: list[str] = []
    for match in _SUBQUERY_RE.finditer(sql):
        col = match.group(1).lower()
        table = match.group(2).lower()
        if table not in schema:
            continue
        if col in schema[table]:
            continue
        present_in = [t for t, cols in schema.items() if col in cols]
        if present_in:
            warnings.append(
                f"Column '{match.group(1)}' does not exist in table "
                f"'{match.group(2)}' (found in: {', '.join(present_in)}). "
                f"SQLite silently resolves this to an outer-scope column, "
                f"which almost certainly produces wrong results."
            )
    return warnings
