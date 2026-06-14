"""search_context tool: local retrieval index full-text search."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from agents.benchmark.schema import PublicTask
from agents.retrieval.index import search_context as _search_context
from agents.tools.decorator import function_tool
from agents.tools.registry import ToolExecutionResult


@function_tool
def search_context(
    task: PublicTask,
    query: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Search query for the task-local retrieval index. Use this for "
                "entity names, IDs, phrases, or evidence in long documents."
            ),
        ),
    ],
    path: Annotated[
        str | None,
        Field(
            description=(
                "Optional exact path filter, relative to context for source files "
                "or the absolute ETL CSV path shown by inspect_files."
            ),
        ),
    ] = None,
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
    """Use when the local retrieval index is enabled and you need to find
    evidence in long local documents, knowledge files, or schema summaries
    by entity name, ID, phrase, or keyword. Returns path, line range, score,
    and text preview. Retrieval results are evidence pointers, not final
    answers; use execute_python or SQL for full computation."""
    content = _search_context(task, query=query, path=path, limit=limit)
    return ToolExecutionResult(
        ok=bool(content.get("enabled")) and "error" not in content, content=content
    )
