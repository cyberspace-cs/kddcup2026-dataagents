"""preview_file tool: unified single-file preview dispatched by extension."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from agents.benchmark.schema import PublicTask
from agents.tools import constants
from agents.tools._fields import path_field
from agents.tools.context import resolve_context_path
from agents.tools.decorator import function_tool
from agents.tools.registry import ToolExecutionResult

PREVIEW_DOC_HEAD_LINES = 100


def _preview_doc(path: Path) -> dict[str, Any]:
    """Head-only preview for text documents."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(lines)
    head = lines[:PREVIEW_DOC_HEAD_LINES]
    return {
        "format": "text",
        "total_lines": total,
        "head": head,
        "has_more": total > PREVIEW_DOC_HEAD_LINES,
    }


@function_tool
def preview_file(
    task: PublicTask,
    path: Annotated[
        str,
        path_field(
            file_kind="any context file",
            examples=("csv/member.csv", "db/sales.db", "knowledge.md", "json/Patient.json"),
        ),
    ],
) -> ToolExecutionResult:
    """Preview a single context file. Automatically detects the format by
    extension and returns a structured summary. Supported formats:
    CSV → {columns, dtypes, row_count, head (20 rows), tail (5 rows)}.
    JSON → structured summary by kind: array (length + head), object
    (keys + value_preview), scalar (value). Files >100MB streamed.
    SQLite (.db/.sqlite) → {tables: [{name, create_sql}]} showing
    CREATE TABLE statements for all tables.
    Text (md/txt) → {total_lines, head (first 100 lines), has_more}.
    Use when you need to inspect one file's structure or sample values
    after inspect_files. This is a SAMPLE preview, not full data access
    — for filtering, aggregation, joins, or full-file scans, use
    execute_python. For SQL queries on SQLite, use execute_context_sql.
    Example: preview_file({"path": "csv/member.csv"})"""
    resolved = resolve_context_path(task, path)
    suffix = resolved.suffix.lower()

    if suffix in constants.CSV_EXTS:
        from agents.tools.read_csv import read_csv_preview

        return ToolExecutionResult(ok=True, content=read_csv_preview(task, path))

    if suffix in constants.JSON_EXTS:
        from agents.tools.read_json import read_json_preview

        return ToolExecutionResult(ok=True, content=read_json_preview(task, path))

    if suffix in constants.SQLITE_EXTS:
        from agents.tools.inspect_sqlite import inspect_sqlite_database

        return ToolExecutionResult(ok=True, content=inspect_sqlite_database(resolved))

    if suffix in constants.DOC_EXTS:
        return ToolExecutionResult(ok=True, content=_preview_doc(resolved))

    raise ValueError(
        f"Unsupported format '{suffix}' for preview_file. "
        "Use execute_python to read this file type."
    )
