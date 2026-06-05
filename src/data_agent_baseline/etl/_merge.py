"""ID conflict resolution, pre-merge, normalization, and deterministic KV→CSV."""

from __future__ import annotations

import csv
import io
import logging
import re
from pathlib import Path

from data_agent_baseline.etl._detect import ENTITY_LINE

logger = logging.getLogger(__name__)

KV_LINE = re.compile(r"^ID:\s*(\d+)\s*\|\s*entity_name:\s*(.+?)\s*\|", re.MULTILINE)

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
UNIT_SUFFIX = re.compile(r"^([\d.]+)\s*(?:cm|kg|U/L|mg/dL|mmHg|%)\s*$")


def resolve_id_conflicts(compressed: str) -> str:
    """Detect and fix systematic ID-shift errors in compressed prose.

    Pattern: the prose defines entity names in the first section (identity).
    Later sections sometimes shift IDs by a fixed offset — e.g. entity "X"
    is ID 365 in section 1, but sections 2-5 assign ID 369 to "X" instead
    (a chain shift where every ID is off by the same amount).

    Strategy:
      1. Build the canonical name→ID mapping from the FIRST occurrence of each name.
      2. Scan all lines — if a line's (ID, name) pair disagrees with the canonical
         mapping AND the canonical ID exists for that name, drop the line.
    """
    _placeholder_set = {"-", "none", "nan", "placeholder", "- (placeholder)", ""}

    name_to_first_id: dict[str, int] = {}
    id_to_first_name: dict[int, str] = {}
    for m in KV_LINE.finditer(compressed):
        eid = int(m.group(1))
        name = m.group(2).strip()
        if name.lower() in _placeholder_set:
            continue
        name_to_first_id.setdefault(name, eid)
        id_to_first_name.setdefault(eid, name)

    fixed_lines: list[str] = []
    fixed_count = 0
    for line in compressed.splitlines():
        m = KV_LINE.match(line)
        if m:
            eid = int(m.group(1))
            name = m.group(2).strip()
            if name.lower() not in _placeholder_set:
                canonical_id = name_to_first_id.get(name)
                if canonical_id is not None and canonical_id != eid:
                    line = line.replace(f"ID: {eid}", f"ID: {canonical_id}", 1)
                    if fixed_count < 10:
                        logger.info(
                            "ETL ID-fix: '%s' ID %d → %d",
                            name,
                            eid,
                            canonical_id,
                        )
                    fixed_count += 1
        fixed_lines.append(line)
    if fixed_count:
        logger.info("ETL ID-fix: corrected %d lines total", fixed_count)
    return "\n".join(fixed_lines)


def pre_merge_entity_lines(
    text: str,
    primary_key: str | None = None,
) -> str:
    """Merge multi-line entity records into one line per unique entity.

    Compression can produce several lines for the same entity (each carrying
    a different attribute subset).  This collapses them so downstream
    extraction sees one complete record per entity.
    """
    pk_token = f"{primary_key}:" if primary_key else None

    entity_lines: list[str] = []
    header_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_entity = (pk_token in stripped) if pk_token else ENTITY_LINE.match(stripped)
        if is_entity:
            entity_lines.append(stripped)
        elif not entity_lines:
            header_lines.append(line)

    if len(entity_lines) <= 1:
        return text

    def _is_placeholder(val: str) -> bool:
        return val.lower().strip() in PLACEHOLDER_VALS or "(placeholder)" in val.lower()

    def _parse_kv(line: str) -> tuple[str | None, list[tuple[str, str]]]:
        parts = line.split("|")
        pk_val: str | None = None
        pairs: list[tuple[str, str]] = []
        for part in parts:
            if ":" not in part:
                continue
            key, _, val = part.partition(":")
            key = key.strip()
            val = val.strip()
            if (pk_token and key == primary_key) or (not pk_token and key.upper() == "ID"):
                pk_val = val
            pairs.append((key, val))
        return pk_val, pairs

    groups: dict[str, dict[str, str]] = {}
    order: list[str] = []

    for line in entity_lines:
        pk_val, pairs = _parse_kv(line)
        if pk_val is None:
            continue
        if pk_val not in groups:
            groups[pk_val] = {}
            order.append(pk_val)
        rec = groups[pk_val]
        for k, v in pairs:
            if k not in rec or (_is_placeholder(rec[k]) and not _is_placeholder(v)):
                rec[k] = v

    merged: list[str] = []
    for pk_val in order:
        parts = [f"{k}: {v}" for k, v in groups[pk_val].items()]
        merged.append(" | ".join(parts))

    before = len(entity_lines)
    after = len(merged)
    if before != after:
        logger.info("ETL pre-merge: %d entity lines → %d unique entities", before, after)

    header = "\n".join(header_lines).strip()
    body = "\n".join(merged)
    return f"{header}\n\n{body}" if header else body


def normalize_entity_lines(
    text: str,
    schema_columns: list[str],
    primary_key: str | None = None,
) -> str:
    """Reorder fields in every entity line to match schema_columns order.

    Drops fields not in the schema and fills missing fields with empty
    values, so downstream CSV extraction sees a uniform tabular layout.
    """
    pk_token = f"{primary_key}:" if primary_key else None
    col_set = set(schema_columns)

    normalized: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_entity = (pk_token in stripped) if pk_token else ENTITY_LINE.match(stripped)
        if not is_entity:
            normalized.append(line)
            continue

        fields: dict[str, str] = {}
        for part in stripped.split("|"):
            if ":" not in part:
                continue
            key, _, val = part.partition(":")
            key = key.strip()
            if key in col_set:
                fields[key] = val.strip()

        parts = [f"{c}: {fields.get(c, '')}" for c in schema_columns]
        normalized.append(" | ".join(parts))

    return "\n".join(normalized)


def clean_cell(val: str) -> str:
    """Normalize a single KV value into a CSV-ready cell."""
    if not val or val.lower().strip() in PLACEHOLDER_VALS or "(placeholder)" in val.lower():
        return ""
    m = UNIT_SUFFIX.match(val)
    if m:
        return m.group(1)
    return val


def deterministic_kv_to_csv(
    text: str,
    schema_columns: list[str],
    primary_key: str | None = None,
) -> str | None:
    """Parse normalized KV entity lines directly into CSV without LLM.

    Each entity line is expected to be in the form:
      col_a: val | col_b: val | ...
    with columns in schema_columns order (produced by normalize_entity_lines).
    """
    pk_token = f"{primary_key}:" if primary_key else None
    col_set = set(schema_columns)

    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_entity = (pk_token in stripped) if pk_token else ENTITY_LINE.match(stripped)
        if not is_entity:
            continue

        fields: dict[str, str] = {}
        for part in stripped.split("|"):
            if ":" not in part:
                continue
            key, _, val = part.partition(":")
            key = key.strip()
            if key in col_set:
                fields[key] = clean_cell(val.strip())

        row = [fields.get(c, "") for c in schema_columns]
        if any(row):
            rows.append(row)

    if not rows:
        return None

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(schema_columns)
    writer.writerows(rows)
    return buf.getvalue()


def validate_csv(csv_path: Path) -> tuple[list[str], int] | None:
    """Read a CSV and return (header, row_count), or None if invalid."""
    if not csv_path.is_file():
        return None
    try:
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return None
            row_count = sum(1 for _ in reader)
        return (header, row_count) if row_count > 0 else None
    except csv.Error:
        return None
