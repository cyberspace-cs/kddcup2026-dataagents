"""read_doc tool: paged text preview with keyword search."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from agents.benchmark.schema import PublicTask
from agents.tools.decorator import function_tool
from agents.tools.defs._fields import path_field, positive_int_field
from agents.tools.descriptions import READ_DOC_DESCRIPTION
from agents.tools.filesystem import read_doc_preview
from agents.tools.registry import ToolExecutionResult


class ReadDocInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Annotated[
        str,
        path_field(
            file_kind="text document (md/txt/...)",
            examples=("knowledge.md", "docs/readme.txt"),
        ),
    ]
    start_line: Annotated[
        int,
        positive_int_field(
            default=1,
            description="1-based line number to start reading from.",
        ),
    ]
    line_count: Annotated[
        int,
        positive_int_field(
            default=100,
            description="Number of lines to return in this page.",
        ),
    ]
    keyword: str | None = Field(
        default=None,
        description=(
            "Optional case-insensitive keyword. When set, returns ALL paragraphs "
            "(blank-line-separated) containing the keyword across the whole file "
            "(start_line / line_count are ignored), capped at 20 matches / 8KB total. "
            "Prefer this over manual paging when searching for an ID/name/term — "
            "one call replaces a sequential scan."
        ),
    )


@function_tool(
    name="read_doc",
    description=READ_DOC_DESCRIPTION,
    input_model=ReadDocInput,
)
def read_doc(task: PublicTask, args: ReadDocInput) -> ToolExecutionResult:
    keyword = None if not args.keyword else args.keyword
    return ToolExecutionResult(
        ok=True,
        content=read_doc_preview(
            task,
            args.path,
            start_line=args.start_line,
            line_count=args.line_count,
            keyword=keyword,
        ),
    )
