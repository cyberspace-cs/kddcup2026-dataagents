"""Self-contained tool definitions: each module co-locates input model + handler.

Import ``FunctionToolSpec`` instances by tool name for use in registries.
"""

from agents.tools.defs._fields import path_field, positive_int_field
from agents.tools.defs.answer import (
    AnswerInput,
    answer,
    build_answer_table_without_side_effects,
    cleanup_answer_artifacts,
    parse_answer_csv,
)
from agents.tools.defs.execute_python import ExecutePythonInput, execute_python
from agents.tools.defs.execute_sql import ExecuteContextSqlInput, execute_context_sql
from agents.tools.defs.grep_context import GrepContextInput, grep_context
from agents.tools.defs.inspect_files import InspectFilesInput, inspect_files
from agents.tools.defs.inspect_sqlite import InspectSqliteSchemaInput, inspect_sqlite_schema
from agents.tools.defs.lookup_field import LookupFieldInput, lookup_field
from agents.tools.defs.read_csv import ReadCsvInput, read_csv
from agents.tools.defs.read_doc import ReadDocInput, read_doc
from agents.tools.defs.read_json import ReadJsonInput, read_json
from agents.tools.defs.search_context import SearchContextInput, search_context

__all__ = [
    "AnswerInput",
    "ExecuteContextSqlInput",
    "ExecutePythonInput",
    "GrepContextInput",
    "InspectFilesInput",
    "InspectSqliteSchemaInput",
    "LookupFieldInput",
    "ReadCsvInput",
    "ReadDocInput",
    "ReadJsonInput",
    "SearchContextInput",
    "answer",
    "build_answer_table_without_side_effects",
    "cleanup_answer_artifacts",
    "execute_context_sql",
    "execute_python",
    "grep_context",
    "inspect_files",
    "inspect_sqlite_schema",
    "lookup_field",
    "parse_answer_csv",
    "path_field",
    "positive_int_field",
    "read_csv",
    "read_doc",
    "read_json",
    "search_context",
]
