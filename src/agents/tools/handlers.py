"""Backward-compatibility shim: re-exports from canonical locations.

Handler functions now live in ``tools/defs/*.py``; ``ToolExecutionResult``
and ``ToolHandler`` moved to ``tools/registry.py``.
"""

from __future__ import annotations

from agents.tools.defs.answer import (
    answer as _answer_spec,
)
from agents.tools.defs.answer import (
    build_answer_table_without_side_effects,
    cleanup_answer_artifacts,
    parse_answer_csv,
)
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
from agents.tools.registry import ToolExecutionResult, ToolHandler

handle_inspect_files = _inspect_files_spec.handler
handle_read_csv = _read_csv_spec.handler
handle_read_json = _read_json_spec.handler
handle_read_doc = _read_doc_spec.handler
handle_search_context = _search_context_spec.handler
handle_lookup_field = _lookup_field_spec.handler
handle_inspect_sqlite_schema = _inspect_sqlite_schema_spec.handler
handle_execute_context_sql = _execute_context_sql_spec.handler
handle_grep_context = _grep_context_spec.handler
handle_execute_python = _execute_python_spec.handler
handle_answer = _answer_spec.handler

__all__ = [
    "ToolExecutionResult",
    "ToolHandler",
    "build_answer_table_without_side_effects",
    "cleanup_answer_artifacts",
    "handle_answer",
    "handle_execute_context_sql",
    "handle_execute_python",
    "handle_grep_context",
    "handle_inspect_files",
    "handle_inspect_sqlite_schema",
    "handle_lookup_field",
    "handle_read_csv",
    "handle_read_doc",
    "handle_read_json",
    "handle_search_context",
    "parse_answer_csv",
]
