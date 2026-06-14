"""execute_python tool: arbitrary Python execution in a subprocess."""

from __future__ import annotations

import pathlib
from typing import Annotated

from pydantic import Field

from agents.benchmark.schema import PublicTask
from agents.tools.contracts import EXECUTE_PYTHON_TIMEOUT_SECONDS
from agents.tools.decorator import function_tool
from agents.tools.python_exec import execute_python_code
from agents.tools.registry import ToolExecutionResult

_ANSWER_SCRATCH_ROOT = pathlib.Path("/tmp/dabench")


def _answer_dir_for(task: PublicTask) -> pathlib.Path:
    return _ANSWER_SCRATCH_ROOT / task.task_id / "_answer"


@function_tool(
    description=(
        "Use when you need computation, aggregation, cross-file joins, or "
        "large-file scans beyond the preview tools. Working directory: the "
        "task context dir; stdout returns as `output`. "
        f"Timeout: {EXECUTE_PYTHON_TIMEOUT_SECONDS} s per call. "
        "Default to `pandas` for tabular work — projections, filters, and "
        "joins are idiomatic and types are preserved automatically. "
        "Available libs: pandas, numpy, pyarrow, polars, openpyxl, sqlalchemy. "
        "The `csv` module is fine only for trivial reads (a handful of rows). "
        "For `.db` / `.sqlite` files, either use the pandas URI form "
        '(`pd.read_sql_query(query, "sqlite:///path")`) or '
        "`sqlite3.connect(path)` + `pd.read_sql_query(query, conn)`."
    ),
)
def execute_python(
    task: PublicTask,
    code: Annotated[
        str,
        Field(
            description=(
                "Python source to execute with the task context directory as CWD. "
                f"Execution timeout is {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
        ),
    ],
) -> ToolExecutionResult:
    content = execute_python_code(
        context_root=task.context_dir,
        code=code,
        answer_dir=_answer_dir_for(task),
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)
