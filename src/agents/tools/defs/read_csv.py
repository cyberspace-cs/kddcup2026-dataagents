"""read_csv tool: CSV schema + head/tail preview."""

from __future__ import annotations

from typing import Annotated

from agents.benchmark.schema import PublicTask
from agents.tools.decorator import function_tool
from agents.tools.defs._fields import path_field
from agents.tools.filesystem import read_csv_preview
from agents.tools.registry import ToolExecutionResult


@function_tool
def read_csv(
    task: PublicTask,
    path: Annotated[
        str,
        path_field(file_kind="CSV file", examples=("csv/member.csv", "data/train.csv")),
    ],
) -> ToolExecutionResult:
    """Use when you need to inspect a single CSV file: returns columns,
    approximate dtypes, row_count, head (first 20 data rows), and tail
    (last 5 data rows that are not already in head). This is a SAMPLE
    preview, not full data access. For filtering, aggregation, joins, or
    any full-file scan, use execute_python instead — the head/tail will
    miss rows in between."""
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, path))
