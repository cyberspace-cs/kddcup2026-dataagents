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
    ExecuteContextSqlInput,
    ExecutePythonInput,
    GrepContextInput,
    InspectFilesInput,
    InspectSqliteSchemaInput,
    LookupFieldInput,
    ReadCsvInput,
    ReadDocInput,
    ReadJsonInput,
    SearchContextInput,
    path_field,
)
from agents.tools.defs.answer import answer as _answer_spec
from agents.tools.defs.execute_python import execute_python as _execute_python_spec
from agents.tools.defs.execute_sql import execute_context_sql as _execute_context_sql_spec
from agents.tools.defs.grep_context import grep_context as _grep_context_spec
from agents.tools.defs.inspect_files import inspect_files as _inspect_files_spec
from agents.tools.defs.inspect_sqlite import inspect_sqlite_schema as _inspect_sqlite_schema_spec
from agents.tools.defs.lookup_field import lookup_field as _lookup_field_spec
from agents.tools.defs.read_csv import read_csv as _read_csv_spec
from agents.tools.defs.read_doc import read_doc as _read_doc_spec
from agents.tools.defs.read_json import read_json as _read_json_spec
from agents.tools.defs.search_context import search_context as _search_context_spec
from agents.tools.schema_normalize import normalize_for_vllm

# Schemas re-exported from their respective FunctionToolSpec.json_schema
INSPECT_FILES_SCHEMA = _inspect_files_spec.json_schema
READ_CSV_SCHEMA = _read_csv_spec.json_schema
READ_JSON_SCHEMA = _read_json_spec.json_schema
READ_DOC_SCHEMA = _read_doc_spec.json_schema
SEARCH_CONTEXT_SCHEMA = _search_context_spec.json_schema
LOOKUP_FIELD_SCHEMA = _lookup_field_spec.json_schema
INSPECT_SQLITE_SCHEMA = _inspect_sqlite_schema_spec.json_schema
EXECUTE_CONTEXT_SQL_SCHEMA = _execute_context_sql_spec.json_schema
GREP_CONTEXT_SCHEMA = _grep_context_spec.json_schema
EXECUTE_PYTHON_SCHEMA = _execute_python_spec.json_schema
ANSWER_SCHEMA = _answer_spec.json_schema


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
