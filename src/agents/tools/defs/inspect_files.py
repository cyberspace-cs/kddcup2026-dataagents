"""inspect_files tool: one-shot context file inventory."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from agents.benchmark.schema import PublicTask
from agents.tools.decorator import function_tool
from agents.tools.descriptions import INSPECT_FILES_DESCRIPTION
from agents.tools.filesystem import inspect_context_files
from agents.tools.registry import ToolExecutionResult


class InspectFilesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


@function_tool(
    name="inspect_files",
    description=INSPECT_FILES_DESCRIPTION,
    input_model=InspectFilesInput,
)
def inspect_files(task: PublicTask, _args: InspectFilesInput) -> ToolExecutionResult:
    return ToolExecutionResult(ok=True, content=inspect_context_files(task))
