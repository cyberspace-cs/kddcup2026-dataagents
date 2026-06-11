"""ID conflict resolution, pre-merge, normalization, and deterministic KV→CSV."""

from __future__ import annotations

import csv
import io
import logging
import re
from pathlib import Path

from agents.etl._detect import ENTITY_LINE
from agents.etl._types import (
    DATE_SCALAR,
    NUMERIC_SCALAR,
    is_placeholder,
    normalize_boolean,
)

logger = logging.getLogger(__name__)

KV_LINE = re.compile(r"^ID:\s*(\d+)\s*\|\s*entity_name:\s*(.+?)\s*\|", re.MULTILINE)
TEMP_ANCHOR_PREFIX = "__anchor__:"

UNIT_SUFFIX = re.compile(r"^([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)\s+\S+\s*$")


def _canonical_key_map(keys: list[str]) -> dict[str, str]:
    """Map lower-cased field names to their canonical schema spelling."""
    return {key.lower(): key for key in keys}


def _canonicalize_key(key: str, key_map: dict[str, str]) -> str:
    return key_map.get(key.strip().lower(), key.strip())


def _line_has_kv_key(line: str, key: str) -> bool:
    """Return whether a pipe-separated KV line contains ``key`` as a full field."""
    target = key.lower()
    for part in line.split("|"):
        raw_key, sep, _value = part.partition(":")
        if sep and raw_key.strip().lower() == target:
            return True
    return False


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


def _repair_fabricated_pks(
    parsed: list[tuple[str | None, list[tuple[str, str]]]],
    primary_key: str | None,
    anchor_keys: list[str] | None,
) -> list[tuple[str | None, list[tuple[str, str]]]]:
    """Rewrite sequentially fabricated primary keys from an alias column.

    When compression renumbers paragraphs (1, 2, 3, ...) instead of copying
    the stated record label, the true id usually still lands in a sibling
    identity column (e.g. "Strategic Unit 300" → strategic_unit_id).  Two
    deterministic signatures gate the repair:

    - alias column: pk == column on at least 3 lines and on more lines than
      it differs (true dual-id documents, e.g. record_id vs personalcode,
      agree on ~0 lines and never qualify);
    - fabricated run: the disagreeing pk values are integers forming a dense
      run starting at 1 (legitimate disagreeing id pairs such as secucode
      vs companycode are large sparse codes and never qualify).

    Lines whose pk falls inside the run get pk := alias value, so round-1
    grouping folds each fragment into its real entity row.
    """
    if not primary_key:
        return parsed

    # Candidate columns: declared anchors first, then remaining columns in
    # first-seen order — the alias column is not always a declared anchor.
    candidates: list[str] = [a for a in (anchor_keys or []) if a != primary_key]
    for _, pairs in parsed:
        for key, _val in pairs:
            if key != primary_key and key not in candidates:
                candidates.append(key)

    for column in candidates:
        agree = 0
        disagree_pks: list[str] = []
        for pk_val, pairs in parsed:
            if not pk_val or is_placeholder(pk_val):
                continue
            col_val = next((v for k, v in pairs if k == column and v.strip()), "")
            if not col_val or is_placeholder(col_val):
                continue
            if pk_val == col_val:
                agree += 1
            else:
                disagree_pks.append(pk_val)
        if agree < 3 or len(disagree_pks) >= agree or not disagree_pks:
            continue
        if not all(pk.isdigit() for pk in disagree_pks):
            continue
        run = {int(pk) for pk in disagree_pks}
        if min(run) != 1 or max(run) > len(run) + max(2, len(run) // 5):
            continue

        repaired = 0
        out: list[tuple[str | None, list[tuple[str, str]]]] = []
        for pk_val, pairs in parsed:
            col_val = next((v for k, v in pairs if k == column and v.strip()), "")
            if (
                pk_val
                and pk_val.isdigit()
                and int(pk_val) in run
                and col_val
                and not is_placeholder(col_val)
                and col_val != pk_val
            ):
                pairs = [(k, col_val if k == primary_key else v) for k, v in pairs]
                pk_val = col_val
                repaired += 1
            out.append((pk_val, pairs))
        logger.info(
            "ETL pre-merge: rewrote %d fabricated sequential PKs (run 1..%d) from alias column %r",
            repaired,
            max(run),
            column,
        )
        return out
    return parsed


def pre_merge_entity_lines(
    text: str,
    primary_key: str | None = None,
    anchor_keys: list[str] | None = None,
) -> str:
    """Merge multi-line entity records into one line per unique entity.

    Compression can produce several lines for the same entity (each carrying
    a different attribute subset).  This collapses them so downstream
    extraction sees one complete record per entity.

    Round 1: group by primary key (existing behavior).
    Round 2: merge groups that share a secondary anchor value (e.g. same
    ``chinesename`` but different ``personalcode`` across sections).
    """
    pk_token = f"{primary_key}:" if primary_key else None
    key_map = _canonical_key_map([k for k in [primary_key, *(anchor_keys or [])] if k])

    entity_lines: list[str] = []
    header_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_entity = (
            _line_has_kv_key(stripped, primary_key) if primary_key else ENTITY_LINE.match(stripped)
        )
        if is_entity:
            entity_lines.append(stripped)
        elif not entity_lines:
            header_lines.append(line)

    if len(entity_lines) <= 1:
        return text

    _is_placeholder = is_placeholder

    def _parse_kv(line: str) -> tuple[str | None, list[tuple[str, str]]]:
        parts = line.split("|")
        pk_val: str | None = None
        pairs: list[tuple[str, str]] = []
        for part in parts:
            if ":" not in part:
                continue
            key, _, val = part.partition(":")
            key = _canonicalize_key(key, key_map)
            val = val.strip()
            if (primary_key and key == primary_key) or (not pk_token and key.upper() == "ID"):
                pk_val = val
            pairs.append((key, val))
        return pk_val, pairs

    def _group_key(pk_val: str | None, pairs: list[tuple[str, str]]) -> str | None:
        """Pick a stable grouping key for one entity line.

        LLM compression sometimes emits the primary key label with an empty
        value while still providing a secondary identity anchor such as a
        person's name.  Grouping all blank-PK lines together loses fields across
        entities; use the first available secondary anchor as a temporary key
        and let round-2 anchor merge attach it to the canonical primary-key row.
        """
        if pk_val is not None and not _is_placeholder(pk_val):
            return pk_val
        if anchor_keys and len(anchor_keys) > 1:
            pair_map = dict(pairs)
            for anchor in anchor_keys[1:]:
                anchor_val = pair_map.get(anchor, "").strip()
                if anchor_val and not _is_placeholder(anchor_val):
                    return f"{TEMP_ANCHOR_PREFIX}{anchor}:{anchor_val}"
        return None

    parsed = [_parse_kv(line) for line in entity_lines]
    parsed = _repair_fabricated_pks(parsed, primary_key, anchor_keys)

    # Round 1: group by primary key
    groups: dict[str, dict[str, str]] = {}
    order: list[str] = []

    for pk_val, pairs in parsed:
        group_key = _group_key(pk_val, pairs)
        if group_key is None:
            continue
        if group_key not in groups:
            groups[group_key] = {}
            order.append(group_key)
        rec = groups[group_key]
        for k, v in pairs:
            if k not in rec or (_is_placeholder(rec[k]) and not _is_placeholder(v)):
                rec[k] = v

    # Round 2: merge groups that share secondary anchor values
    if anchor_keys and len(anchor_keys) > 1:
        groups_before_r2 = len(groups)
        for anchor in anchor_keys[1:]:
            anchor_to_pks: dict[str, list[str]] = {}
            for pk_val in list(groups.keys()):
                a_val = groups[pk_val].get(anchor, "").strip()
                if a_val and not _is_placeholder(a_val):
                    anchor_to_pks.setdefault(a_val, []).append(pk_val)

            for _a_val, pk_list in anchor_to_pks.items():
                if len(pk_list) <= 1:
                    continue
                temp_pks = [pk for pk in pk_list if pk.startswith(TEMP_ANCHOR_PREFIX)]
                real_pks = [pk for pk in pk_list if not pk.startswith(TEMP_ANCHOR_PREFIX)]
                if not temp_pks or len(real_pks) != 1:
                    continue

                canonical = real_pks[0]
                for pk in temp_pks:
                    conflict = False
                    for a in anchor_keys[1:]:
                        t_val = groups[pk].get(a, "").strip()
                        c_val = groups[canonical].get(a, "").strip()
                        if (
                            t_val
                            and c_val
                            and not _is_placeholder(t_val)
                            and not _is_placeholder(c_val)
                            and t_val != c_val
                        ):
                            conflict = True
                            break
                    if conflict:
                        continue
                    for k, v in groups[pk].items():
                        if (
                            k not in groups[canonical] or _is_placeholder(groups[canonical][k])
                        ) and not _is_placeholder(v):
                            groups[canonical][k] = v
                    del groups[pk]

        groups_after_r2 = len(groups)
        if groups_before_r2 != groups_after_r2:
            logger.info(
                "ETL anchor-merge: %d groups → %d unique entities (anchors: %s)",
                groups_before_r2,
                groups_after_r2,
                anchor_keys,
            )

    merged: list[str] = []
    for pk_val in order:
        if pk_val not in groups:
            continue
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
    field_map = _canonical_key_map(schema_columns)
    col_set = set(schema_columns)

    normalized: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_entity = (
            _line_has_kv_key(stripped, primary_key) if primary_key else ENTITY_LINE.match(stripped)
        )
        if not is_entity:
            normalized.append(line)
            continue

        fields: dict[str, str] = {}
        for part in stripped.split("|"):
            if ":" not in part:
                continue
            key, _, val = part.partition(":")
            key = _canonicalize_key(key, field_map)
            if key in col_set:
                fields[key] = val.strip()

        parts = [f"{c}: {fields.get(c, '')}" for c in schema_columns]
        normalized.append(" | ".join(parts))

    return "\n".join(normalized)


def clean_cell(val: str) -> str:
    """Normalize a single KV value into a CSV-ready cell."""
    if not val or is_placeholder(val):
        return ""
    m = UNIT_SUFFIX.match(val)
    if m:
        return m.group(1)
    return val


def clean_cell_for_type(val: str, field_type: str | None = None) -> str:
    """Normalize a KV value and enforce lightweight schema field types."""
    cleaned = clean_cell(val)
    if not cleaned:
        return ""
    type_text = (field_type or "").lower()
    if "number" in type_text or "integer" in type_text:
        normalized = cleaned.replace(",", "")
        if normalized.endswith("%"):
            normalized = normalized[:-1].strip()
        return normalized if NUMERIC_SCALAR.match(normalized) else ""
    if "date" in type_text:
        return cleaned if DATE_SCALAR.match(cleaned) else ""
    if "boolean" in type_text:
        return normalize_boolean(cleaned) or ""
    return cleaned


def deterministic_kv_to_csv(
    text: str,
    schema_columns: list[str],
    primary_key: str | None = None,
    schema_field_types: dict[str, str] | None = None,
) -> str | None:
    """Parse normalized KV entity lines directly into CSV without LLM.

    Each entity line is expected to be in the form:
      col_a: val | col_b: val | ...
    with columns in schema_columns order (produced by normalize_entity_lines).
    """
    field_map = _canonical_key_map(schema_columns)
    col_set = set(schema_columns)

    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_entity = (
            _line_has_kv_key(stripped, primary_key) if primary_key else ENTITY_LINE.match(stripped)
        )
        if not is_entity:
            continue

        fields: dict[str, str] = {}
        for part in stripped.split("|"):
            if ":" not in part:
                continue
            key, _, val = part.partition(":")
            key = _canonicalize_key(key, field_map)
            if key in col_set:
                fields[key] = clean_cell_for_type(
                    val.strip(),
                    schema_field_types.get(key) if schema_field_types else None,
                )

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


def _cells_agree(a: str, b: str) -> bool:
    """Two non-empty cells agree when equal as strings or as parsed numbers."""
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def column_fill_counts(header: list[str], rows: list[list[str]]) -> dict[str, int]:
    """Count non-empty cells per column."""
    counts = dict.fromkeys(header, 0)
    for row in rows:
        for col, val in zip(header, row, strict=False):
            if val.strip():
                counts[col] += 1
    return counts


def apply_synonym_merges(
    header: list[str],
    rows: list[list[str]],
    merges: list[tuple[str, str]],
    protected: set[str] | None = None,
) -> tuple[list[str], list[list[str]], list[tuple[str, str]]]:
    """Fold synonym source columns into their canonical target columns.

    ``merges`` holds adjudicated ``(source, target)`` pairs — typically a
    populated extraction-discovered column folding into an empty
    governance-canonical column (the value-split failure mode).  The
    adjudication says the two columns *mean* the same thing; this function
    re-checks, per pair and against the CURRENT table state, that the data
    *agrees* with that claim:

    - identity columns named in ``protected`` are never touched;
    - a pair with any row-level conflict (both cells non-empty and not equal
      as strings or numbers) is skipped — semantically-approved but
      data-contradicted merges must not corrupt either column;
    - an all-empty source is a no-op and is skipped.

    Pairs are processed in deterministic order (descending source fill, then
    name), so when two sources target one canonical column the second is
    conflict-checked against the already-merged result.

    Returns ``(header, rows, applied)`` — new copies; inputs are not mutated.
    """
    protected_lower = {p.lower() for p in (protected or set())}
    header = list(header)
    rows = [list(row) for row in rows]
    fill = column_fill_counts(header, rows)

    applied: list[tuple[str, str]] = []
    for src, dst in sorted(merges, key=lambda m: (-fill.get(m[0], 0), m[0])):
        if src == dst or src.lower() in protected_lower or dst.lower() in protected_lower:
            continue
        if src not in header or dst not in header:
            continue
        si, di = header.index(src), header.index(dst)
        if not any(row[si].strip() for row in rows):
            continue
        conflicts = sum(
            1
            for row in rows
            if row[si].strip()
            and row[di].strip()
            and not _cells_agree(row[si].strip(), row[di].strip())
        )
        if conflicts:
            logger.warning(
                "ETL synonym-unify: %d row conflicts between '%s' and '%s' — not merged",
                conflicts,
                src,
                dst,
            )
            continue
        for row in rows:
            if row[si].strip() and not row[di].strip():
                row[di] = row[si]
        header.pop(si)
        for row in rows:
            row.pop(si)
        applied.append((src, dst))
    return header, rows, applied


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
