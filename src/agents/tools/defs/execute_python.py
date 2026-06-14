"""execute_python tool: arbitrary Python execution in a subprocess."""

from __future__ import annotations

import pathlib
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from agents.benchmark.schema import PublicTask
from agents.tools.contracts import EXECUTE_PYTHON_TIMEOUT_SECONDS
from agents.tools.decorator import function_tool
from agents.tools.descriptions import EXECUTE_PYTHON_DESCRIPTION
from agents.tools.python_exec import execute_python_code
from agents.tools.registry import ToolExecutionResult

_ANSWER_SCRATCH_ROOT = pathlib.Path("/tmp/dabench")


def _answer_dir_for(task: PublicTask) -> pathlib.Path:
    return _ANSWER_SCRATCH_ROOT / task.task_id / "_answer"


class ExecutePythonInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: Annotated[
        str,
        Field(
            description=(
                "Python source to execute with the task context directory as CWD. "
                f"Execution timeout is {EXECUTE_PYTHON_TIMEOUT_SECONDS} seconds."
            ),
        ),
    ]


@function_tool(
    name="execute_python",
    description=EXECUTE_PYTHON_DESCRIPTION,
    input_model=ExecutePythonInput,
)
def execute_python(task: PublicTask, args: ExecutePythonInput) -> ToolExecutionResult:
    content = execute_python_code(
        context_root=task.context_dir,
        code=args.code,
        answer_dir=_answer_dir_for(task),
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)
