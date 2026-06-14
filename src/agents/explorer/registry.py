"""Explorer sub-agent tool registry: read-only tools + report terminal tool."""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from agents.benchmark.schema import PublicTask
from agents.tools import defs
from agents.tools.registry import ToolDefinition, ToolExecutionResult, ToolRegistry
from agents.tools.schema_normalize import normalize_for_vllm


class ReportInput(BaseModel):
    """Explorer report terminal tool input — structured fields."""

    model_config = ConfigDict(extra="forbid")

    files: list[dict[str, Any]] = Field(  # pyright: ignore[reportUnknownVariableType]
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

    Includes inspection/preview tools from ``defs/``, plus ``report``
    as the only terminal tool. Excludes ``execute_python`` and ``answer``.
    """
    definitions: list[ToolDefinition] = [
        defs.inspect_files.to_definition(
            description=(
                "One-shot summary of every file under context "
                "(columns, dtypes, row counts, table schemas, column_conversions)."
            ),
        ),
        defs.read_csv.to_definition(
            description=(
                "Preview a CSV file: columns, dtypes, row_count, "
                "head (20 rows), tail (5 rows). Sample only."
            ),
        ),
        defs.read_doc.to_definition(
            description=(
                "Preview a text document (md/txt). Without keyword: paging. "
                "With keyword: returns ALL matching paragraphs across the file."
            ),
        ),
        defs.search_context.to_definition(
            description=(
                "Search the local retrieval index by entity, ID, phrase, or keyword. "
                "Returns path, line range, score, text preview."
            ),
        ),
        defs.lookup_field.to_definition(
            description=(
                "Resolve field meanings, units, code labels, aliases, thresholds "
                "from knowledge.md and schema summaries."
            ),
        ),
        defs.read_json.to_definition(
            description=(
                "Preview a JSON file: kind (array/object/scalar), length/keys, "
                "sample values. Sample only."
            ),
        ),
        defs.inspect_sqlite_schema.to_definition(
            description=(
                "View SQLite database table structures (columns, row counts). "
                "Do NOT call on .csv/.json/.md/.txt paths."
            ),
        ),
        defs.execute_context_sql.to_definition(
            description=(
                "Run read-only SQL on a SQLite database. Use for SELECT DISTINCT, "
                "MIN/MAX, sample queries. Do NOT call on .csv/.json/.md/.txt paths."
            ),
        ),
        defs.grep_context.to_definition(
            description=(
                "Search for a regex pattern across ALL context files at once: "
                "CSV values, JSON fields, Markdown text, and SQLite text columns. "
                "Use this to check whether a value exists in any file."
            ),
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
