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


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _rewrite_code_span(span: str, identifiers: list[str]) -> str:
    pattern = re.compile("|".join(re.escape(name) for name in identifiers))
    return pattern.sub(lambda m: _quote_identifier(m.group(0)), span)


def _auto_quote_code_spans(sql: str, identifiers: list[str]) -> str:
    """Rewrite only executable SQL text, leaving literals/comments/quoted names intact."""
    out: list[str] = []
    start = 0
    i = 0
    n = len(sql)

    def flush_code(until: int) -> None:
        nonlocal start
        if until > start:
            out.append(_rewrite_code_span(sql[start:until], identifiers))
        start = until

    while i < n:
        ch = sql[i]

        if ch == "'":
            flush_code(i)
            j = i + 1
            while j < n:
                if sql[j] == "'":
                    if j + 1 < n and sql[j + 1] == "'":
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            out.append(sql[i:j])
            i = j
            start = i
            continue

        if ch in {'"', "`", "["}:
            flush_code(i)
            close = "]" if ch == "[" else ch
            j = i + 1
            while j < n:
                if sql[j] == close:
                    if close in {'"', "`"} and j + 1 < n and sql[j + 1] == close:
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            out.append(sql[i:j])
            i = j
            start = i
            continue

        if sql.startswith("--", i):
            flush_code(i)
            j = sql.find("\n", i + 2)
            j = n if j == -1 else j + 1
            out.append(sql[i:j])
            i = j
            start = i
            continue

        if sql.startswith("/*", i):
            flush_code(i)
            j = sql.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append(sql[i:j])
            i = j
            start = i
            continue

        i += 1

    flush_code(n)
    return "".join(out)


def auto_quote_identifiers(sql: str, conn: sqlite3.Connection) -> str:
    """Auto-quote table/column names containing special characters like ``() / -``."""
    tables = [
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    ]

    identifiers: list[str] = []
    for tbl in tables:
        if tbl in sql or _quote_identifier(tbl) in sql:
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({_quote_identifier(tbl)})")]
            identifiers.extend(cols)
        identifiers.append(tbl)

    need_quote = sorted(
        {name for name in identifiers if _SPECIAL_CHAR_RE.search(name)},
        key=len,
        reverse=True,
    )
    if not need_quote:
        return sql
    return _auto_quote_code_spans(sql, need_quote)


def connect_read_only(path: Path) -> sqlite3.Connection:
    """Connect to a SQLite file in URI read-only mode."""
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)
