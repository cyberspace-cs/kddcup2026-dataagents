"""search_context tool: local retrieval index full-text search."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from agents.benchmark.schema import PublicTask
from agents.retrieval.index import search_context as _search_context
from agents.tools.decorator import function_tool
from agents.tools.descriptions import SEARCH_CONTEXT_DESCRIPTION
from agents.tools.registry import ToolExecutionResult


class SearchContextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Search query for the task-local retrieval index. Use this for "
                "entity names, IDs, phrases, or evidence in long documents."
            ),
        ),
    ]
    path: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Optional exact path filter, relative to context for source files "
                "or the absolute ETL CSV path shown by inspect_files."
            ),
        ),
    ] = None
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
    name="search_context",
    description=SEARCH_CONTEXT_DESCRIPTION,
    input_model=SearchContextInput,
)
def search_context(task: PublicTask, args: SearchContextInput) -> ToolExecutionResult:
    content = _search_context(task, query=args.query, path=args.path, limit=args.limit)
    return ToolExecutionResult(
        ok=bool(content.get("enabled")) and "error" not in content, content=content
    )
