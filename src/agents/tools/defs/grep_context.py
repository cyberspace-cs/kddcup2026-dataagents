"""grep_context tool: regex search across all context files."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from agents.benchmark.schema import PublicTask
from agents.tools.decorator import function_tool
from agents.tools.descriptions import GREP_CONTEXT_DESCRIPTION
from agents.tools.filesystem.grep import grep_context as _grep_context
from agents.tools.registry import ToolExecutionResult


class GrepContextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Regex pattern to search for across all context files. "
                "Case-insensitive. Supports full Python regex syntax."
            ),
        ),
    ]
    path: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Optional path substring filter. When set, only files whose "
                "path contains this string are searched."
            ),
        ),
    ] = None


@function_tool(
    name="grep_context",
    description=GREP_CONTEXT_DESCRIPTION,
    input_model=GrepContextInput,
)
def grep_context(task: PublicTask, args: GrepContextInput) -> ToolExecutionResult:
    content = _grep_context(task, pattern=args.pattern, path_filter=args.path)
    return ToolExecutionResult(ok=True, content=content)
