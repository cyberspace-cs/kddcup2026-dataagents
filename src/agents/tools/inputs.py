"""Backward-compatibility shim: re-exports input models and schemas from ``defs/``.

``ExploreInput``, ``ExploreVideoInput``, and ``EXPLORE_VIDEO_SCHEMA`` remain
defined here (they are not part of the default registry and stay local to
their respective factory modules).
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from agents.tools.defs import (
    AnswerInput,
    path_field,
)
from agents.tools.defs import (
    answer as _answer_tool,
)
from agents.tools.defs import (
    execute_context_sql as _execute_context_sql_tool,
)
from agents.tools.defs import (
    execute_python as _execute_python_tool,
)
from agents.tools.defs import (
    grep_context as _grep_context_tool,
)
from agents.tools.defs import (
    inspect_files as _inspect_files_tool,
)
from agents.tools.defs import (
    inspect_sqlite_schema as _inspect_sqlite_schema_tool,
)
from agents.tools.defs import (
    lookup_field as _lookup_field_tool,
)
from agents.tools.defs import (
    read_csv as _read_csv_tool,
)
from agents.tools.defs import (
    read_doc as _read_doc_tool,
)
from agents.tools.defs import (
    read_json as _read_json_tool,
)
from agents.tools.defs import (
    search_context as _search_context_tool,
)
from agents.tools.schema_normalize import normalize_for_vllm

# Input models re-exported from their respective FunctionTool.input_model
InspectFilesInput = _inspect_files_tool.input_model
ReadCsvInput = _read_csv_tool.input_model
ReadJsonInput = _read_json_tool.input_model
ReadDocInput = _read_doc_tool.input_model
SearchContextInput = _search_context_tool.input_model
LookupFieldInput = _lookup_field_tool.input_model
InspectSqliteSchemaInput = _inspect_sqlite_schema_tool.input_model
ExecuteContextSqlInput = _execute_context_sql_tool.input_model
GrepContextInput = _grep_context_tool.input_model
ExecutePythonInput = _execute_python_tool.input_model

# Schemas re-exported from their respective FunctionTool.json_schema
INSPECT_FILES_SCHEMA = _inspect_files_tool.json_schema
READ_CSV_SCHEMA = _read_csv_tool.json_schema
READ_JSON_SCHEMA = _read_json_tool.json_schema
READ_DOC_SCHEMA = _read_doc_tool.json_schema
SEARCH_CONTEXT_SCHEMA = _search_context_tool.json_schema
LOOKUP_FIELD_SCHEMA = _lookup_field_tool.json_schema
INSPECT_SQLITE_SCHEMA = _inspect_sqlite_schema_tool.json_schema
EXECUTE_CONTEXT_SQL_SCHEMA = _execute_context_sql_tool.json_schema
GREP_CONTEXT_SCHEMA = _grep_context_tool.json_schema
EXECUTE_PYTHON_SCHEMA = _execute_python_tool.json_schema
ANSWER_SCHEMA = _answer_tool.json_schema


class ExploreInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExploreVideoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Annotated[
        str,
        path_field(file_kind="video file", examples=("video.mp4", "media/demo.mov")),
    ]
    instructions: str = Field(
        description=(
            "Analysis brief for the video agent: what the task needs from this video — "
            "which charts, tables, values, or text to transcribe, plus any IDs or labels "
            "to look for."
        ),
    )


EXPLORE_VIDEO_SCHEMA = normalize_for_vllm(ExploreVideoInput.model_json_schema())

__all__ = [
    "ANSWER_SCHEMA",
    "EXECUTE_CONTEXT_SQL_SCHEMA",
    "EXECUTE_PYTHON_SCHEMA",
    "EXPLORE_VIDEO_SCHEMA",
    "GREP_CONTEXT_SCHEMA",
    "INSPECT_FILES_SCHEMA",
    "INSPECT_SQLITE_SCHEMA",
    "LOOKUP_FIELD_SCHEMA",
    "READ_CSV_SCHEMA",
    "READ_DOC_SCHEMA",
    "READ_JSON_SCHEMA",
    "SEARCH_CONTEXT_SCHEMA",
    "AnswerInput",
    "ExecuteContextSqlInput",
    "ExecutePythonInput",
    "ExploreInput",
    "ExploreVideoInput",
    "GrepContextInput",
    "InspectFilesInput",
    "InspectSqliteSchemaInput",
    "LookupFieldInput",
    "ReadCsvInput",
    "ReadDocInput",
    "ReadJsonInput",
    "SearchContextInput",
]
