"""read_doc tool: paged text preview with keyword search."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from agents.benchmark.schema import PublicTask
from agents.tools.decorator import function_tool
from agents.tools.defs._fields import path_field, positive_int_field
from agents.tools.filesystem import read_doc_preview
from agents.tools.registry import ToolExecutionResult


@function_tool
def read_doc(
    task: PublicTask,
    path: Annotated[
        str,
        path_field(
            file_kind="text document (md/txt/...)",
            examples=("knowledge.md", "docs/readme.txt"),
        ),
    ],
    start_line: Annotated[
        int,
        positive_int_field(description="1-based line number to start reading from."),
    ] = 1,
    line_count: Annotated[
        int,
        positive_int_field(description="Number of lines to return in this page."),
    ] = 100,
    keyword: Annotated[
        str | None,
        Field(
            description=(
                "Optional case-insensitive keyword. When set, returns ALL paragraphs "
                "(blank-line-separated) containing the keyword across the whole file "
                "(start_line / line_count are ignored), capped at 20 matches / 8KB total. "
                "Prefer this over manual paging when searching for an ID/name/term — "
                "one call replaces a sequential scan."
            ),
        ),
    ] = None,
) -> ToolExecutionResult:
    """Use when you need to preview a text-like document (md/txt). Two
    modes: (1) paging — without `keyword`, returns a fixed line window;
    use `has_more` + next `start_line` to continue. (2) keyword search
    — with `keyword`, returns ALL paragraphs containing the keyword
    across the whole file (capped at 20 matches / 8KB total). Prefer
    keyword mode when looking for an ID/name/term — one call replaces
    a sequential scan. This is a SAMPLE preview, not full data access.
    For full-document regex extraction, cross-paragraph aggregation,
    or structured parsing, use execute_python instead."""
    kw = keyword if keyword else None
    return ToolExecutionResult(
        ok=True,
        content=read_doc_preview(
            task,
            path,
            start_line=start_line,
            line_count=line_count,
            keyword=kw,
        ),
    )
