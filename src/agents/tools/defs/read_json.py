"""read_json tool: structured JSON preview."""

from __future__ import annotations

from typing import Annotated

from agents.benchmark.schema import PublicTask
from agents.tools.decorator import function_tool
from agents.tools.defs._fields import path_field
from agents.tools.filesystem import read_json_preview
from agents.tools.registry import ToolExecutionResult


@function_tool
def read_json(
    task: PublicTask,
    path: Annotated[
        str,
        path_field(file_kind="JSON file", examples=("json/Patient.json", "zip_code.json")),
    ],
) -> ToolExecutionResult:
    """Use when you need to preview a single JSON file: returns a
    structured summary keyed by `kind` — `array` (length + first 20
    items), `object` (key_count + first 50 keys + value_preview for
    first 20 keys), or `scalar` (the value). This is a SAMPLE
    preview, not full data access. For nested field extraction,
    filtering, cross-file joins, or deep traversal, use execute_python
    instead. Files larger than 100MB return a streamed schema
    (`kind` + `keys` + `sample_value_kinds` for object; `length` +
    `first_item_kind` for array) without full load."""
    return ToolExecutionResult(ok=True, content=read_json_preview(task, path))
