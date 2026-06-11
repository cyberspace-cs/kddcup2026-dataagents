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


def is_embedded_digit_run(text: str, start: int, end: int) -> bool:
    """True when ``text[start:end]`` is a digit fragment of a larger token.

    Record IDs are standalone digit tokens.  Captures inside comma/dot
    formatted numbers ("8,408,793" → "408", "671,408.57" → "408"), longer
    digit runs ("2003" → "200"), or digit+letter words ("30th" → "30") are
    false IDs: they spawn spurious entity groups and hijack paragraphs away
    from the entity they belong to.
    """
    before = text[start - 1] if start > 0 else ""
    before2 = text[start - 2] if start > 1 else ""
    after = text[end] if end < len(text) else ""
    after2 = text[end + 1] if end + 1 < len(text) else ""
    if before.isdigit() or after.isdigit():
        return True
    if before in ",." and before2.isdigit():
        return True
    if after in ",." and after2.isdigit():
        return True
    # ASCII letters only: a CJK char right after the digits ("档案286的…")
    # is ordinary prose, not a fragment.
    return after.isascii() and after.isalpha()


def has_standalone_id(text: str, rid: str) -> bool:
    """``rid`` occurs in ``text`` as a standalone digit token (anchor verification)."""
    for m in re.finditer(re.escape(rid), text):
        if not is_embedded_digit_run(text, m.start(), m.end()):
            return True
    return False


APPROX_TAG = "~"
_APPROX_PREFIXES = ("~", "～")


def split_approx_tag(val: str) -> tuple[str, bool]:
    """Split a leading approximate-value marker off ``val``.

    Compression tags values stated approximately in the source ("约六百九十七万"
    → ``~6970000``) so the deterministic identity-repair pass can recompute
    them; everything downstream of that pass must never see the tag.
    """
    stripped = val.strip()
    for prefix in _APPROX_PREFIXES:
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip(), True
    return stripped, False


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
