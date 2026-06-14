"""Tools sub-package: ReAct agent tool registry and default tool set.

``create_default_tool_registry()`` returns the default tools; extend with
custom tools by building ``FunctionTool`` instances or using ``@function_tool``.
"""

from agents.tools._fields import path_field, positive_int_field
from agents.tools.answer import (
    AnswerInput,
    answer,
    build_answer_table_without_side_effects,
    cleanup_answer_artifacts,
    parse_answer_csv,
)
from agents.tools.decorator import function_tool
from agents.tools.dispatcher import dispatch_tool_call
from agents.tools.execute_python import execute_python
from agents.tools.execute_sql import execute_context_sql
from agents.tools.filesystem import list_context_tree
from agents.tools.grep_context import grep_context
from agents.tools.inspect_files import inspect_files
from agents.tools.inspect_sqlite import inspect_sqlite_schema
from agents.tools.lookup_field import lookup_field
from agents.tools.read_csv import read_csv
from agents.tools.read_doc import read_doc
from agents.tools.read_json import read_json
from agents.tools.registry import (
    FunctionTool,
    ToolExecutionResult,
    ToolRegistry,
    create_default_tool_registry,
)
from agents.tools.search_context import search_context

__all__ = [
    "AnswerInput",
    "FunctionTool",
    "ToolExecutionResult",
    "ToolRegistry",
    "answer",
    "build_answer_table_without_side_effects",
    "cleanup_answer_artifacts",
    "create_default_tool_registry",
    "dispatch_tool_call",
    "execute_context_sql",
    "execute_python",
    "function_tool",
    "grep_context",
    "inspect_files",
    "inspect_sqlite_schema",
    "list_context_tree",
    "lookup_field",
    "parse_answer_csv",
    "path_field",
    "positive_int_field",
    "read_csv",
    "read_doc",
    "read_json",
    "search_context",
]
