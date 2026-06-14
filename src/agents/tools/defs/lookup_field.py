"""lookup_field tool: field/unit/code alias resolution from local index."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from agents.benchmark.schema import PublicTask
from agents.retrieval.index import lookup_field as _lookup_field
from agents.tools.decorator import function_tool
from agents.tools.descriptions import LOOKUP_FIELD_DESCRIPTION
from agents.tools.registry import ToolExecutionResult


class LookupFieldInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    term: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Field, unit, code, alias, threshold, table, or column term to look up "
                "in knowledge.md and schema summaries."
            ),
        ),
    ]
    limit: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description=(
                "Optional result cap. The runtime also caps this at the configured "
                "run.retrieval_index.max_results."
            ),
        ),
    ] = None


@function_tool(
    name="lookup_field",
    description=LOOKUP_FIELD_DESCRIPTION,
    input_model=LookupFieldInput,
)
def lookup_field(task: PublicTask, args: LookupFieldInput) -> ToolExecutionResult:
    content = _lookup_field(task, term=args.term, limit=args.limit)
    return ToolExecutionResult(
        ok=bool(content.get("enabled")) and "error" not in content, content=content
    )
