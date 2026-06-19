"""Explorer sub-agent tool registry: read-only tools + report terminal tool."""

from __future__ import annotations

import json
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.benchmark.schema import PublicTask
from agents.tools.execute_sql import execute_context_sql
from agents.tools.grep_context import grep_context
from agents.tools.inspect_files import inspect_files
from agents.tools.inspect_sqlite import inspect_sqlite_schema
from agents.tools.read_csv import read_csv
from agents.tools.read_doc import read_doc
from agents.tools.read_json import read_json
from agents.tools.registry import FunctionTool, ToolExecutionResult, ToolRegistry
from agents.tools.schema_normalize import normalize_for_vllm


def coerce_dict_field(val: Any) -> Any:
    """Convert a list-of-objects to a keyed dict if the model sent the wrong shape."""
    if isinstance(val, dict):
        return cast(dict[str, Any], val)
    if not isinstance(val, list):
        return val
    raw_items = cast(list[Any], val)
    if any(not isinstance(item, dict) for item in raw_items):
        return raw_items
    items = cast(list[dict[str, Any]], raw_items)
    out: dict[str, Any] = {}
    for item in items:
        key: str | None = item.get("name") or item.get("table") or item.get("file")
        if key:
            out[str(key)] = {k: v for k, v in item.items() if k not in ("name", "table", "file")}
            continue
        out[f"_item_{len(out)}"] = item
    return out


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
    value_samples: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Sampled distinct values for filter-relevant columns.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Objective data-quality observations (one atomic fact each).",
    )

    @field_validator("schema_map", "knowledge", "value_samples", mode="before")
    @classmethod
    def _coerce_list_to_dict(cls, v: Any) -> Any:
        return coerce_dict_field(v)


REPORT_SCHEMA = normalize_for_vllm(ReportInput.model_json_schema())

REPORT_KEYS = ("files", "schema_map", "knowledge", "join_paths", "value_samples", "warnings")


def handle_report(_task: PublicTask, args: Any) -> ToolExecutionResult:
    d: dict[str, Any] = cast(
        dict[str, Any],
        args if isinstance(args, dict) else {k: getattr(args, k, None) for k in REPORT_KEYS},
    )
    report = ReportInput.model_validate(d)
    findings = json.dumps(
        {
            k: getattr(report, k, {} if k in ("schema_map", "knowledge") else [])
            for k in REPORT_KEYS
        },
        ensure_ascii=False,
    )
    return ToolExecutionResult(
        ok=True,
        content={"status": "reported", "findings": findings},
        is_terminal=True,
    )


def create_explorer_tool_registry(
    *,
    video_tool: FunctionTool | None = None,
) -> ToolRegistry:
    """Read-only tool registry for the Explorer sub-agent.

    Includes inspection/preview tools from top-level tool modules, plus ``report``
    as the only terminal tool. Excludes ``execute_python`` and ``answer``.
    """
    definitions: list[FunctionTool] = [
        inspect_files.with_description(
            "One-shot summary of every file under context "
            "(columns, dtypes, row counts, table schemas, column_conversions).",
        ),
        read_csv.with_description(
            "Preview a CSV file: columns, dtypes, row_count, "
            "head (20 rows), tail (5 rows). Sample only.",
        ),
        read_doc.with_description(
            "Preview a text document (md/txt). Without keyword: paging. "
            "With keyword: returns ALL matching paragraphs across the file.",
        ),
        read_json.with_description(
            "Preview a JSON file: kind (array/object/scalar), length/keys, "
            "sample values. Sample only.",
        ),
        inspect_sqlite_schema.with_description(
            "View SQLite database table structures (columns, row counts). "
            "Do NOT call on .csv/.json/.md/.txt paths.",
        ),
        execute_context_sql.with_description(
            "Run read-only SQL on a SQLite database. Use for SELECT DISTINCT, "
            "MIN/MAX, sample queries. Do NOT call on .csv/.json/.md/.txt paths.",
        ),
        grep_context.with_description(
            "Search for a regex pattern across ALL context files at once: "
            "CSV values, JSON fields, Markdown text, and SQLite text columns. "
            "Use this to check whether a value exists in any file.",
        ),
    ]
    if video_tool is not None:
        definitions.append(video_tool)
    definitions.append(
        FunctionTool(
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
