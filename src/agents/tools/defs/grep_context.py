"""grep_context tool: regex search across all context files."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from agents.benchmark.schema import PublicTask
from agents.tools.decorator import function_tool
from agents.tools.filesystem.grep import grep_context as _grep_context
from agents.tools.registry import ToolExecutionResult


@function_tool
def grep_context(
    task: PublicTask,
    pattern: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Regex pattern to search for across all context files. "
                "Case-insensitive. Supports full Python regex syntax."
            ),
        ),
    ],
    path: Annotated[
        str | None,
        Field(
            description=(
                "Optional path substring filter. When set, only files whose "
                "path contains this string are searched."
            ),
        ),
    ] = None,
) -> ToolExecutionResult:
    """Use this when you need to search for a regex pattern across ALL context
    files at once: CSV column values, JSON fields, Markdown/text content,
    and SQLite text columns. Returns matching lines/rows with file path
    and location. Checks whether a specific value exists in the data
    (e.g., does province column contain '全国'?) without needing to
    read each file individually."""
    content = _grep_context(task, pattern=pattern, path_filter=path)
    return ToolExecutionResult(ok=True, content=content)
