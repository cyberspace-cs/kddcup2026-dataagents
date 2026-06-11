"""Explorer sub-agent tool registry: read-only tools + report terminal tool."""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from agents.benchmark.schema import PublicTask
from agents.tools.handlers import (
    ToolExecutionResult,
    handle_execute_context_sql,
    handle_grep_context,
    handle_inspect_files,
    handle_inspect_sqlite_schema,
    handle_lookup_field,
    handle_read_csv,
    handle_read_doc,
    handle_read_json,
    handle_search_context,
)
from agents.tools.inputs import (
    EXECUTE_CONTEXT_SQL_SCHEMA,
    GREP_CONTEXT_SCHEMA,
    INSPECT_FILES_SCHEMA,
    INSPECT_SQLITE_SCHEMA,
    LOOKUP_FIELD_SCHEMA,
    READ_CSV_SCHEMA,
    READ_DOC_SCHEMA,
    READ_JSON_SCHEMA,
    SEARCH_CONTEXT_SCHEMA,
)
from agents.tools.registry import ToolDefinition, ToolRegistry
from agents.tools.schema_normalize import normalize_for_vllm


class ReportInput(BaseModel):
    """Explorer report terminal tool input — structured fields."""

    model_config = ConfigDict(extra="forbid")

    files: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Flat list of file entries, one per file.",
    )
    schema_map: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Per-table/file schema: columns with dtypes, semantics, "
            "and column_conversions if inspect_files provided them."
        ),
    )
    knowledge: dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted mappings, formulas, units, thresholds from docs.",
    )
    join_paths: list[dict[str, str]] = Field(  # pyright: ignore[reportUnknownVariableType]
        default_factory=list,
        description='Discovered relationships, e.g. [{"left": "a.col", "right": "b.col"}].',
    )
    value_samples: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Sampled distinct values for filter-relevant columns.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Objective data-quality observations (one atomic fact each).",
    )


REPORT_SCHEMA = normalize_for_vllm(ReportInput.model_json_schema())


def handle_report(_task: PublicTask, args: Any) -> ToolExecutionResult:
    import json as _json

    _KEYS = ("files", "schema_map", "knowledge", "join_paths", "value_samples", "warnings")
    d: dict[str, Any] = cast(
        dict[str, Any],
        args if isinstance(args, dict) else {k: getattr(args, k, None) for k in _KEYS},
    )
    findings = _json.dumps(
        {k: d.get(k, {} if k in ("schema_map", "knowledge") else []) for k in _KEYS},
        ensure_ascii=False,
    )
    return ToolExecutionResult(
        ok=True,
        content={"status": "reported", "findings": findings},
        is_terminal=True,
    )


def create_explorer_tool_registry(
    *,
    video_tool: ToolDefinition | None = None,
) -> ToolRegistry:
    """Read-only tool registry for the Explorer sub-agent.

    Includes all inspection/preview tools from the main registry, plus `report`
    as the only terminal tool. Excludes `execute_python` and `answer`.
    ``video_tool``（`create_explore_video_tool_definition` 的产物）按任务条件注入：
    无视频任务传 None，模型完全看不到该工具（schema 仅经 tools=[...] 下推）。
    """
    definitions: list[ToolDefinition] = [
        ToolDefinition(
            name="inspect_files",
            description=(
                "One-shot summary of every file under context "
                "(columns, dtypes, row counts, table schemas, column_conversions)."
            ),
            json_schema=INSPECT_FILES_SCHEMA,
            handler=handle_inspect_files,
        ),
        ToolDefinition(
            name="read_csv",
            description=(
                "Preview a CSV file: columns, dtypes, row_count, "
                "head (20 rows), tail (5 rows). Sample only."
            ),
            json_schema=READ_CSV_SCHEMA,
            handler=handle_read_csv,
        ),
        ToolDefinition(
            name="read_doc",
            description=(
                "Preview a text document (md/txt). Without keyword: paging. "
                "With keyword: returns ALL matching paragraphs across the file."
            ),
            json_schema=READ_DOC_SCHEMA,
            handler=handle_read_doc,
        ),
        ToolDefinition(
            name="search_context",
            description=(
                "Search the local retrieval index by entity, ID, phrase, or keyword. "
                "Returns path, line range, score, text preview."
            ),
            json_schema=SEARCH_CONTEXT_SCHEMA,
            handler=handle_search_context,
        ),
        ToolDefinition(
            name="lookup_field",
            description=(
                "Resolve field meanings, units, code labels, aliases, thresholds "
                "from knowledge.md and schema summaries."
            ),
            json_schema=LOOKUP_FIELD_SCHEMA,
            handler=handle_lookup_field,
        ),
        ToolDefinition(
            name="read_json",
            description=(
                "Preview a JSON file: kind (array/object/scalar), length/keys, "
                "sample values. Sample only."
            ),
            json_schema=READ_JSON_SCHEMA,
            handler=handle_read_json,
        ),
        ToolDefinition(
            name="inspect_sqlite_schema",
            description=(
                "View SQLite database table structures (columns, row counts). "
                "Do NOT call on .csv/.json/.md/.txt paths."
            ),
            json_schema=INSPECT_SQLITE_SCHEMA,
            handler=handle_inspect_sqlite_schema,
        ),
        ToolDefinition(
            name="execute_context_sql",
            description=(
                "Run read-only SQL on a SQLite database. Use for SELECT DISTINCT, "
                "MIN/MAX, sample queries. Do NOT call on .csv/.json/.md/.txt paths."
            ),
            json_schema=EXECUTE_CONTEXT_SQL_SCHEMA,
            handler=handle_execute_context_sql,
        ),
        ToolDefinition(
            name="grep_context",
            description=(
                "Search for a regex pattern across ALL context files at once: "
                "CSV values, JSON fields, Markdown text, and SQLite text columns. "
                "Use this to check whether a value exists in any file."
            ),
            json_schema=GREP_CONTEXT_SCHEMA,
            handler=handle_grep_context,
        ),
    ]
    if video_tool is not None:
        definitions.append(video_tool)
    definitions.append(
        ToolDefinition(
            name="report",
            description=(
                "Submit your exploration findings. This is the ONLY way to complete "
                "the exploration — call it when you have mapped the data landscape. "
                "Pass each field (files, schema_map, knowledge, join_paths, "
                "value_samples, warnings) as a separate parameter."
            ),
            json_schema=REPORT_SCHEMA,
            handler=handle_report,
            is_terminal=True,
            input_model=ReportInput,
        ),
    )
    return ToolRegistry(definitions={d.name: d for d in definitions})
