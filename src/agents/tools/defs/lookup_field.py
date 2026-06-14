"""lookup_field tool: field/unit/code alias resolution from local index."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from agents.benchmark.schema import PublicTask
from agents.retrieval.index import lookup_field as _lookup_field
from agents.tools.decorator import function_tool
from agents.tools.registry import ToolExecutionResult


@function_tool
def lookup_field(
    task: PublicTask,
    term: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Field, unit, code, alias, threshold, table, or column term to look up "
                "in knowledge.md and schema summaries."
            ),
        ),
    ],
    limit: Annotated[
        int | None,
        Field(
            ge=1,
            description=(
                "Optional result cap. The runtime also caps this at the configured "
                "run.retrieval_index.max_results."
            ),
        ),
    ] = None,
) -> ToolExecutionResult:
    """Use when the local retrieval index is enabled and you need to resolve
    field meanings, units, code labels, aliases, thresholds, table names,
    or column candidates from knowledge.md and schema summaries before
    filtering or joining data."""
    content = _lookup_field(task, term=term, limit=limit)
    return ToolExecutionResult(
        ok=bool(content.get("enabled")) and "error" not in content, content=content
    )
