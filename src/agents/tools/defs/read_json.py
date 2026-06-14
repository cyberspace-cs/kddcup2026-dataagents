"""read_json tool: structured JSON preview."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict

from agents.benchmark.schema import PublicTask
from agents.tools.decorator import function_tool
from agents.tools.defs._fields import path_field
from agents.tools.descriptions import READ_JSON_DESCRIPTION
from agents.tools.filesystem import read_json_preview
from agents.tools.registry import ToolExecutionResult


class ReadJsonInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Annotated[
        str,
        path_field(file_kind="JSON file", examples=("json/Patient.json", "zip_code.json")),
    ]


@function_tool(
    name="read_json",
    description=READ_JSON_DESCRIPTION,
    input_model=ReadJsonInput,
)
def read_json(task: PublicTask, args: ReadJsonInput) -> ToolExecutionResult:
    return ToolExecutionResult(ok=True, content=read_json_preview(task, args.path))
