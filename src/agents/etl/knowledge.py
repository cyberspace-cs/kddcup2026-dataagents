"""Shared helpers for parsing ETL governance metadata from ``knowledge.md``."""

from __future__ import annotations

import re

_KM_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_KM_TABLE_SEPARATOR = re.compile(r":?-{3,}:?")
# Column-name cells: ASCII identifiers or CJK names, optionally carrying a
# unit suffix like "(百万元)"; anything with spaces/backticks inside is prose.
_KM_FIELD_CELL = re.compile(r"^[\w一-鿿][\w一-鿿()（）%/·\-]{0,39}$")


def _km_section_table_fields(lines: list[str], start: int, end: int) -> dict[str, str]:
    """Parse ``| column | definition |`` table rows inside one km section."""
    fields: dict[str, str] = {}
    in_table = False
    header_seen = False
    col_idx = 0
    for ln in lines[start:end]:
        stripped = ln.strip()
        if not stripped.startswith("|"):
            in_table, header_seen = False, False
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if cells and all(_KM_TABLE_SEPARATOR.fullmatch(c) for c in cells if c):
            in_table = True
            continue
        if not in_table and not header_seen:
            header_seen = True
            col_idx = next(
                (k for k, c in enumerate(cells) if re.search(r"column|field|字段|列", c, re.I)),
                0,
            )
            continue
        if not in_table or col_idx >= len(cells):
            continue
        cell = cells[col_idx].strip().strip("`").strip()
        if _KM_FIELD_CELL.fullmatch(cell):
            fields[cell] = cells[col_idx + 1].strip() if col_idx + 1 < len(cells) else ""
    return fields


def km_table_fields(km_text: str, prose_stem: str) -> dict[str, str]:
    """Deterministically parse the governance field table for ``prose_stem``.

    Locates heading lines mentioning the stem (backticked or word-bounded -
    ``\b`` keeps "mf_netvalue" from anchoring inside
    "mf_netvalueperformancehis"), bounds the section at the next heading of
    the same or higher level, and parses its markdown column table. The first
    matching heading whose section yields fields wins. Returns {} when nothing
    parses so callers can fall back to the LLM-only schema unchanged.
    """
    lines = km_text.splitlines()
    starts: list[tuple[int, int]] = []
    for i, ln in enumerate(lines):
        m = _KM_HEADING.match(ln)
        if m and re.search(rf"`{re.escape(prose_stem)}`|\b{re.escape(prose_stem)}\b", m.group(2)):
            starts.append((i, len(m.group(1))))
    for start, level in starts:
        end = len(lines)
        for j in range(start + 1, len(lines)):
            m = _KM_HEADING.match(lines[j])
            if m and len(m.group(1)) <= level:
                end = j
                break
        fields = _km_section_table_fields(lines, start, end)
        if fields:
            return fields
    return {}
