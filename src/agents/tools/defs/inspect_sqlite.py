"""inspect_sqlite_schema tool: SQLite table structure inspection."""

from __future__ import annotations

from typing import Annotated

from agents.benchmark.schema import PublicTask
from agents.tools.decorator import function_tool
from agents.tools.defs._fields import path_field
from agents.tools.filesystem import resolve_context_path
from agents.tools.registry import ToolExecutionResult
from agents.tools.sqlite import inspect_sqlite_schema as _inspect_sqlite_schema


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
        ok=True, content=_inspect_sqlite_schema(resolve_context_path(task, path))
    )
