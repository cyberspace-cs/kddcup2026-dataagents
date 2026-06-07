"""Shared schema-type constants and validation helpers for ETL stages."""

from __future__ import annotations

import re

RECORD_ID_FIELDS = frozenset(
    {"record_id", "archive_id", "file_id", "case_id", "entry_id", "local_id", "unit_id"}
)

RECORD_ID_KEYWORDS = r"档案|记录|条目|战略单元|Record|File|Case"

NUMERIC_SCALAR = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$")
DATE_SCALAR = re.compile(r"^\d{4}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?$")

PLACEHOLDER_VALS = frozenset(
    {
        "",
        "-",
        "none",
        "nan",
        "n/a",
        "placeholder",
        "none (placeholder)",
        "nan (placeholder)",
        "- (placeholder)",
        "0.0 (placeholder)",
        "redacted",
    }
)

_BOOL_TRUE = frozenset(("true", "yes", "1", "是"))
_BOOL_FALSE = frozenset(("false", "no", "0", "否"))


def is_placeholder(val: str) -> bool:
    return val.lower().strip() in PLACEHOLDER_VALS or "(placeholder)" in val.lower()


def normalize_boolean(val: str) -> str | None:
    """Normalize a boolean-ish string to 'true'/'false', or None if unrecognizable."""
    low = val.strip().lower()
    if low in _BOOL_TRUE:
        return "true"
    if low in _BOOL_FALSE:
        return "false"
    return None
