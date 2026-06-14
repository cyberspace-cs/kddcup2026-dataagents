"""read_csv tool: CSV schema + head/tail preview."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict

from agents.benchmark.schema import PublicTask
from agents.tools.decorator import function_tool
from agents.tools.defs._fields import path_field
from agents.tools.descriptions import READ_CSV_DESCRIPTION
from agents.tools.filesystem import read_csv_preview
from agents.tools.registry import ToolExecutionResult


class ReadCsvInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Annotated[
        str,
        path_field(file_kind="CSV file", examples=("csv/member.csv", "data/train.csv")),
    ]


@function_tool(
    name="read_csv",
    description=READ_CSV_DESCRIPTION,
    input_model=ReadCsvInput,
)
def read_csv(task: PublicTask, args: ReadCsvInput) -> ToolExecutionResult:
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, args.path))
