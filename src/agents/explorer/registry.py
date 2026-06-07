"""Explorer sub-agent tool registry: read-only tools + report terminal tool."""

from __future__ import annotations

from typing import Any

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
    ExecuteContextSqlInput,
    GrepContextInput,
    InspectFilesInput,
    InspectSqliteSchemaInput,
    LookupFieldInput,
    ReadCsvInput,
    ReadDocInput,
    ReadJsonInput,
    SearchContextInput,
)
from agents.tools.registry import ToolDefinition, ToolRegistry
from agents.tools.schema_normalize import normalize_for_vllm


class ReportInput(BaseModel):
    """Explorer report terminal tool input."""

    model_config = ConfigDict(extra="forbid")

    findings: str = Field(
        description=(
            "Structured JSON string containing exploration findings: "
            "files, schema_map, knowledge, join_paths, value_samples, "
            "warnings."
        ),
    )

    @classmethod
    def example_value(cls) -> dict[str, Any]:
        return {"findings": '{"files": {...}, "schema_map": {...}, ...}'}


REPORT_SCHEMA = normalize_for_vllm(ReportInput.model_json_schema())


def handle_report(_task: PublicTask, args: Any) -> ToolExecutionResult:
    from typing import cast

    if isinstance(args, dict):
        d = cast(dict[str, str], args)
        findings = d.get("findings", "")
    else:
        findings = str(getattr(args, "findings", ""))
    return ToolExecutionResult(
        ok=True,
        content={"status": "reported", "findings": findings},
        is_terminal=True,
    )


def create_explorer_tool_registry() -> ToolRegistry:
    """Read-only tool registry for the Explorer sub-agent.

    Includes all inspection/preview tools from the main registry,
    plus `report` as the only terminal tool. Excludes `execute_python`
    and `answer`.
    """
    definitions: list[ToolDefinition] = [
        ToolDefinition(
            name="inspect_files",
            description=(
                "One-shot summary of every file under context "
                "(columns, dtypes, row counts, table schemas, unit_hint)."
            ),
            input_schema=InspectFilesInput.example_value(),
            json_schema=INSPECT_FILES_SCHEMA,
            handler=handle_inspect_files,
        ),
        ToolDefinition(
            name="read_csv",
            description=(
                "Preview a CSV file: columns, dtypes, row_count, "
                "head (20 rows), tail (5 rows). Sample only."
            ),
            input_schema=ReadCsvInput.example_value(),
            json_schema=READ_CSV_SCHEMA,
            handler=handle_read_csv,
        ),
        ToolDefinition(
            name="read_doc",
            description=(
                "Preview a text document (md/txt). Without keyword: paging. "
                "With keyword: returns ALL matching paragraphs across the file."
            ),
            input_schema=ReadDocInput.example_value(),
            json_schema=READ_DOC_SCHEMA,
            handler=handle_read_doc,
        ),
        ToolDefinition(
            name="search_context",
            description=(
                "Search the local retrieval index by entity, ID, phrase, or keyword. "
                "Returns path, line range, score, text preview."
            ),
            input_schema=SearchContextInput.example_value(),
            json_schema=SEARCH_CONTEXT_SCHEMA,
            handler=handle_search_context,
        ),
        ToolDefinition(
            name="lookup_field",
            description=(
                "Resolve field meanings, units, code labels, aliases, thresholds "
                "from knowledge.md and schema summaries."
            ),
            input_schema=LookupFieldInput.example_value(),
            json_schema=LOOKUP_FIELD_SCHEMA,
            handler=handle_lookup_field,
        ),
        ToolDefinition(
            name="read_json",
            description=(
                "Preview a JSON file: kind (array/object/scalar), length/keys, "
                "sample values. Sample only."
            ),
            input_schema=ReadJsonInput.example_value(),
            json_schema=READ_JSON_SCHEMA,
            handler=handle_read_json,
        ),
        ToolDefinition(
            name="inspect_sqlite_schema",
            description=(
                "View SQLite database table structures (columns, row counts). "
                "Do NOT call on .csv/.json/.md/.txt paths."
            ),
            input_schema=InspectSqliteSchemaInput.example_value(),
            json_schema=INSPECT_SQLITE_SCHEMA,
            handler=handle_inspect_sqlite_schema,
        ),
        ToolDefinition(
            name="execute_context_sql",
            description=(
                "Run read-only SQL on a SQLite database. Use for SELECT DISTINCT, "
                "MIN/MAX, sample queries. Do NOT call on .csv/.json/.md/.txt paths."
            ),
            input_schema=ExecuteContextSqlInput.example_value(),
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
            input_schema=GrepContextInput.example_value(),
            json_schema=GREP_CONTEXT_SCHEMA,
            handler=handle_grep_context,
        ),
        ToolDefinition(
            name="report",
            description=(
                "Submit your exploration findings. This is the ONLY way to complete "
                "the exploration — call it when you have mapped the data landscape. "
                "Pass a structured JSON string with keys: files, schema_map, knowledge, "
                "join_paths, value_samples, warnings."
            ),
            input_schema=ReportInput.example_value(),
            json_schema=REPORT_SCHEMA,
            handler=handle_report,
            is_terminal=True,
        ),
    ]
    return ToolRegistry(definitions={d.name: d for d in definitions})
