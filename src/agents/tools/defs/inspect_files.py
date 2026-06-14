"""inspect_files tool: one-shot context file inventory."""

from __future__ import annotations

from agents.benchmark.schema import PublicTask
from agents.tools.decorator import function_tool
from agents.tools.filesystem import inspect_context_files
from agents.tools.registry import ToolExecutionResult


@function_tool
def inspect_files(task: PublicTask) -> ToolExecutionResult:
    """Use first to bootstrap context exploration: returns a one-shot summary
    of every file under context (columns, dtypes, row counts, table schemas).
    This replaces the usual sequence of read_csv / read_json /
    inspect_sqlite_schema calls for initial discovery."""
    return ToolExecutionResult(ok=True, content=inspect_context_files(task))
