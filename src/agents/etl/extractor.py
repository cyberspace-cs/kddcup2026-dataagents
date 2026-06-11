"""Core ETL: detect large prose files, compress noise, extract to CSV.

Pipeline (when knowledge.md schema is available):
  1. **Compress** — LLM strips narrative filler from prose chunks, producing
     pipe-separated key-value entity lines.
  2. **Pre-merge** — Python collapses multi-line entity records into one line
     per unique primary key.
  3. **Normalize** — reorder fields to match schema, drop unknowns.
  4. **Deterministic CSV** — Python parses normalized KV lines into CSV
     (no LLM needed).

Fallback (no schema): LLM direct extraction → code-generation.
"""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.config import ETL_SCRATCH_ROOT
from agents.etl._codegen import (
    MAX_RETRIES,
    build_codegen_prompt,
    build_retry_prompt,
    call_llm,
    exec_code,
    extract_code_block,
)
from agents.etl._compress import (
    compress_prose,
    direct_extract_csv,
    sample_paragraphs_by_section,
)
from agents.etl._detect import (
    ETLResult,
    detect_prose_files,
)
from agents.etl._merge import (
    apply_synonym_merges,
    column_fill_counts,
    deterministic_kv_to_csv,
    normalize_entity_lines,
    pre_merge_entity_lines,
    repair_numeric_identities,
    resolve_id_conflicts,
    validate_csv,
)
from agents.etl._pdf import pdf_to_markdown
from agents.etl._types import RECORD_ID_FIELDS, RECORD_ID_KEYWORDS
from agents.llm import ModelAdapter, ModelMessage, ModelResponse, OpenAIModelAdapter

if TYPE_CHECKING:
    from agents.benchmark.schema import PublicTask

logger = logging.getLogger(__name__)

_MAX_WORKERS = 4
_LLM_CALL_TIMEOUT = 180
_ETL_HTTP_TIMEOUT = 300.0
_DIRECT_EXTRACT_CHAR_LIMIT = 60_000
_LOCAL_RECORD_ID_RE = re.compile(
    rf"(?:{RECORD_ID_KEYWORDS})\s*[：:#-]?\s*(\d{{1,8}})",
    re.IGNORECASE,
)


def _has_local_record_ids(prose_text: str) -> bool:
    """Return whether prose uses a repeated document-local record identifier."""
    matches = _LOCAL_RECORD_ID_RE.findall(prose_text)
    return len(set(matches)) >= 3


def _ensure_local_record_anchor(
    prose_text: str,
    columns: list[str],
    anchor_keys: list[str],
    explicit_types: dict[str, str] | None = None,
) -> tuple[list[str], list[str]]:
    """Add a document-local record ID anchor when prose is sectioned by one.

    Some generated prose files describe the same row across multiple sections
    using labels like "档案 353" or "战略单元 353".  Those labels are not domain
    keys such as ``personalcode``; they are a local row key needed to merge the
    partial facts before downstream analysis.
    """
    known_fields = {c.lower() for c in columns} | {a.lower() for a in anchor_keys}

    def _promote_record_id(cols: list[str], ancs: list[str]) -> tuple[list[str], list[str]]:
        """Ensure record_id is first in both columns and anchor_keys."""
        rid = next((c for c in cols if c.lower() in RECORD_ID_FIELDS), "record_id")
        if cols[0].lower() not in RECORD_ID_FIELDS:
            cols = [rid] + [c for c in cols if c != rid]
        if not ancs or ancs[0].lower() not in RECORD_ID_FIELDS:
            ancs = [rid] + [a for a in ancs if a != rid]
        return cols, ancs

    if known_fields & RECORD_ID_FIELDS:
        return _promote_record_id(columns, anchor_keys)
    if explicit_types:
        for field, ftype in explicit_types.items():
            if field.lower() in known_fields and "local_record_id" in ftype.lower():
                return _promote_record_id(columns, anchor_keys)
    if not _has_local_record_ids(prose_text):
        return columns, anchor_keys

    columns = ["record_id", *columns]
    anchor_keys = ["record_id", *anchor_keys]
    logger.info("ETL schema: added local record_id anchor from prose record labels")
    return columns, anchor_keys


def _parse_schema_types_line(line: str) -> dict[str, str]:
    """Parse TYPES: field=type, field=type output from schema extraction."""
    if not line.upper().startswith("TYPES"):
        return {}
    raw = line.split(":", 1)[-1]
    parsed: dict[str, str] = {}
    for item in re.split(r"[,;]", raw):
        item = item.strip()
        if not item:
            continue
        field, sep, field_type = item.partition("=")
        if not sep:
            field, sep, field_type = item.partition(":")
        if sep and field.strip() and field_type.strip():
            parsed[field.strip()] = field_type.strip()
    return parsed


def _parse_kv_line(line: str) -> dict[str, str]:
    """Parse ``field=value, field=value`` from any prefixed line."""
    raw = line.split(":", 1)[-1]
    parsed: dict[str, str] = {}
    for item in re.split(r"[,;]", raw):
        item = item.strip()
        if not item:
            continue
        field, sep, val = item.partition("=")
        if not sep:
            field, sep, val = item.partition(":")
        if sep and field.strip() and val.strip():
            parsed[field.strip()] = val.strip()
    return parsed


def _parse_defs_line(line: str) -> dict[str, str]:
    """Parse ``DEFS: field=definition; field=definition`` output.

    Splits on semicolons only (definitions may contain commas).
    """
    if not line.upper().startswith("DEF"):
        return {}
    raw = line.split(":", 1)[-1]
    parsed: dict[str, str] = {}
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        field, sep, defn = item.partition("=")
        if sep and field.strip() and defn.strip():
            parsed[field.strip()] = defn.strip()
    return parsed


def _parse_schema_response(
    text: str,
) -> tuple[str, list[str], dict[str, str], list[str], dict[str, str], dict[str, str]] | None:
    """Parse LLM schema output by line prefix, not position.

    Looks for PK:, ANCHORS:, TYPES:, UNITS:, DEFS: prefixes in any order.
    The first non-prefixed line with commas is treated as the column list.

    Returns (pk, anchor_keys, explicit_types, columns, field_units, field_defs)
    or None.
    """
    lines = [line.strip() for line in text.strip().strip("`").splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    pk: str | None = None
    anchor_keys: list[str] = []
    explicit_types: dict[str, str] = {}
    field_units: dict[str, str] = {}
    field_defs: dict[str, str] = {}
    columns: list[str] = []

    for line in lines:
        upper = line.upper()
        if upper.startswith("PK"):
            pk = line.split(":", 1)[-1].strip()
        elif upper.startswith("ANCHOR"):
            raw = [a.strip() for a in line.split(":", 1)[-1].split(",") if a.strip()]
            if raw:
                anchor_keys = raw
        elif upper.startswith("TYPE"):
            explicit_types = _parse_schema_types_line(line)
        elif upper.startswith("UNIT"):
            field_units = _parse_kv_line(line)
        elif upper.startswith("DEF"):
            field_defs = _parse_defs_line(line)
        elif "," in line and not columns and "=" not in line:
            tokens = [c.strip() for c in line.split(",") if c.strip()]
            if all(len(t.split()) <= 3 for t in tokens):
                columns = tokens

    if not pk or len(columns) < 2:
        return None

    if not anchor_keys:
        anchor_keys = [pk]
    elif pk not in anchor_keys:
        anchor_keys.insert(0, pk)

    for ak in anchor_keys:
        if ak not in columns:
            columns.append(ak)

    return pk, anchor_keys, explicit_types, columns, field_units, field_defs


def _complete_schema_field_defs(
    columns: list[str],
    field_defs: dict[str, str] | None = None,
) -> dict[str, str]:
    """Canonicalize LLM-provided field definitions to match schema column spelling."""
    if not field_defs:
        return {}
    canonical = {column.lower(): column for column in columns}
    result: dict[str, str] = {}
    for field, defn in field_defs.items():
        column = canonical.get(field.lower())
        if column:
            result[column] = defn
    return result


def _complete_schema_field_types(
    columns: list[str],
    explicit_types: dict[str, str] | None = None,
) -> dict[str, str]:
    """Canonicalize LLM-provided field types to match schema column spelling."""
    if not explicit_types:
        return {}
    canonical = {column.lower(): column for column in columns}
    result: dict[str, str] = {}
    for field, field_type in explicit_types.items():
        column = canonical.get(field.lower())
        if column:
            result[column] = field_type
    return result


# Magnitude scales for unit families whose conversion factors are exactly
# computable. Within a family, factor(source→target) = scale[source] / scale[target].
# Families must never be mixed (a 元→% "conversion" is meaningless).
_UNIT_SCALE_FAMILIES: tuple[dict[str, float], ...] = (
    {  # CNY monetary magnitudes
        "元": 1.0,
        "千元": 1e3,
        "万元": 1e4,
        "十万元": 1e5,
        "百万元": 1e6,
        "千万元": 1e7,
        "亿元": 1e8,
        "十亿元": 1e9,
        "百亿元": 1e10,
        "千亿元": 1e11,
        "万亿元": 1e12,
    },
    {  # bare count magnitudes (人/吨/股 suffixes share these prefixes)
        "万": 1e4,
        "亿": 1e8,
        "万亿": 1e12,
    },
    {  # percentage ↔ decimal ratio
        "%": 0.01,
        "ratio": 1.0,
    },
)


def _deterministic_factor(source_unit: str, target_unit: str) -> float | None:
    """Compute the exact source→target multiplier when both units share a scale family.

    Returns None when the pair is not exactly computable (unknown unit, or the
    units belong to different families).
    """
    src = source_unit.strip()
    tgt = target_unit.strip()
    for family in _UNIT_SCALE_FAMILIES:
        if src in family and tgt in family:
            return family[src] / family[tgt]
    return None


def _vote_target_units(
    adapter: ModelAdapter,
    km_text: str,
    field_units: dict[str, str],
    question: str = "",
) -> dict[str, tuple[str, float]]:
    """Ask the LLM for each field's target unit and conversion factor.

    Returns {field: (target_unit, factor)} where factor is the multiplier
    to convert source values to target values (e.g. 亿元→万元 = 10000).
    Fields whose source unit already matches the target are excluded.
    """
    source_fields = {
        f: u
        for f, u in field_units.items()
        if not f.startswith("_") and u and not u.startswith("_")
    }
    if not source_fields:
        return {}

    field_list = ", ".join(f"{f}={u}" for f, u in source_fields.items())
    prompt = (
        "Below is knowledge.md and a list of fields with their "
        "current units as found in the source data.\n\n"
        f"Fields: {field_list}\n\n"
        "For each field, determine the TARGET unit that knowledge.md "
        "says answers should use. If the document declares a default unit "
        "(e.g. '万元'), apply it to all monetary fields. If a field's source "
        "unit already matches the target, repeat it unchanged. "
        "Dimensionless fields (price indices, ranks) must keep their "
        "source unit as-is — do NOT replace with '_default' or any placeholder.\n\n"
        "PERCENTAGE RULE:\n"
        "1. If the QUESTION asks for a '比例' (proportion/ratio) and the "
        "field's source unit is '%', convert to 'ratio' (factor 0.01). "
        "In Chinese financial data, '比例' means a decimal fraction "
        "(e.g. 0.0155 = 1.55%), matching the canonical database storage.\n"
        "2. Otherwise, only convert to 'ratio' when knowledge.md "
        "EXPLICITLY uses the word 'ratio' or 'decimal ratio' "
        "in that field's definition. If knowledge.md merely says '%', "
        "'percentage', '百分比', or the field is just a return/rate expressed "
        "in percent, keep the source unit as '%' — do NOT convert.\n\n"
        "For each field where source ≠ target, also output the numeric "
        "multiplication factor: target_value = source_value × factor. "
        "Mind the direction — converting to a LARGER unit needs a factor "
        "BELOW 1.\n\n"
        "Output format — one field per entry, comma-separated:\n"
        "  field=target_unit*factor\n"
        "If source == target (no conversion needed), omit the *factor part:\n"
        "  field=target_unit\n\n"
        "Examples:\n"
        "  gdp=万元*10000, deposits=万元*0.0001, rate=%\n"
        "  (gdp source was 亿元, target is 万元, so factor=10000)\n"
        "  (deposits source was 元, target is 万元 — larger unit, "
        "so factor=0.0001)\n"
        "  (rate source was %, target is %, so no factor)\n\n"
        f"knowledge.md:\n{km_text}"
    )
    if question:
        prompt += f"\n\nQUESTION (for context): {question}"
    messages = [
        ModelMessage(
            role="system",
            content=(
                "Map each field to its target unit per knowledge.md and the question. "
                "Output field=unit*factor pairs. Omit *factor when source==target."
            ),
        ),
        ModelMessage(role="user", content=prompt),
    ]

    try:
        response: ModelResponse = adapter.complete(messages)
    except Exception:
        return {}

    return _parse_unit_factor_response(response.content.strip(), source_fields)


def _parse_unit_factor_response(
    text: str,
    source_fields: dict[str, str],
) -> dict[str, tuple[str, float]]:
    """Parse ``field=unit*factor`` response into {field: (unit, factor)}.

    Only returns entries where source_unit != target_unit and factor is valid.
    For unit pairs within a known scale family the factor is computed
    deterministically, overriding whatever the LLM produced — the LLM is
    trusted only for choosing the target unit, not the arithmetic.
    """
    result: dict[str, tuple[str, float]] = {}
    raw = text.split(":", 1)[-1] if ":" in text else text
    for item in re.split(r"[,;\n]", raw):
        item = item.strip()
        if not item:
            continue
        field, sep, rest = item.partition("=")
        if not sep:
            continue
        field = field.strip()
        if field not in source_fields:
            continue
        rest = rest.strip()
        unit_part, star, factor_part = rest.partition("*")
        unit_part = unit_part.strip()
        if not unit_part or unit_part.startswith("_"):
            continue
        if unit_part == source_fields[field]:
            continue
        exact = _deterministic_factor(source_fields[field], unit_part)
        if exact is not None:
            if exact == 1:
                continue
            if star:
                with contextlib.suppress(ValueError, TypeError):
                    if float(factor_part.strip()) != exact:
                        logger.warning(
                            "Unit factor for %s (%s→%s) corrected: LLM gave %s, exact is %s",
                            field,
                            source_fields[field],
                            unit_part,
                            factor_part.strip(),
                            exact,
                        )
            result[field] = (unit_part, exact)
            continue
        if not star:
            continue
        try:
            factor = float(factor_part.strip())
        except (ValueError, TypeError):
            continue
        if factor == 0 or factor == 1:
            continue
        result[field] = (unit_part, factor)
    return result


def _write_units_sidecar(
    task: PublicTask,
    prose_stem: str,
    field_units: dict[str, str],
) -> None:
    """Write per-field unit metadata to a sidecar JSON in _cache/."""
    if not field_units:
        return
    cache_dir = ETL_SCRATCH_ROOT / task.task_id / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    units_path = cache_dir / f"{prose_stem}_units.json"
    units_path.write_text(json.dumps(field_units, ensure_ascii=False), encoding="utf-8")


def _parse_alias_response(
    text: str,
    discovered: list[str],
    canonicals: list[str],
) -> dict[str, str]:
    """Parse ``discovered=canonical`` / ``discovered=DISTINCT`` adjudication lines.

    Only entries naming a known discovered column on the left and a known
    canonical column (or DISTINCT) on the right are kept; everything else is
    ignored, so an unlisted discovered column defaults to DISTINCT.
    """
    canon_by_lower = {c.lower(): c for c in canonicals}
    disc_by_lower = {d.lower(): d for d in discovered}
    mapping: dict[str, str] = {}
    raw = text.strip().strip("`")
    for item in re.split(r"[,;\n]", raw):
        item = item.strip()
        if not item or "=" not in item:
            continue
        left, _, right = item.partition("=")
        left = left.split(":")[-1].strip()  # tolerate an echoed "ALIASES:" prefix
        disc = disc_by_lower.get(left.lower())
        if not disc:
            continue
        target = right.strip().strip(".").lower()
        if target == "distinct":
            continue
        canon = canon_by_lower.get(target)
        if canon:
            mapping[disc] = canon
    return mapping


def _adjudicate_synonym_columns(
    adapter: ModelAdapter,
    prose_stem: str,
    empty_canonicals: dict[str, str],
    discovered: list[tuple[str, str, list[str]]],
    question: str,
) -> dict[str, str]:
    """Ask the LLM which populated discovered columns are exact synonyms of
    empty canonical columns.

    The LLM is trusted only for the semantic judgment; whether a merge
    actually happens is re-gated deterministically against the data
    (`apply_synonym_merges`).  Returns {discovered: canonical}.
    """
    canon_block = "\n".join(
        f"- {name}: {defn or '(no definition available)'}"
        for name, defn in empty_canonicals.items()
    )
    disc_block = "\n".join(
        f"- {name}: {defn or '(no definition available)'} | samples: "
        f"{', '.join(samples) if samples else '(none)'}"
        for name, defn, samples in discovered
    )
    prompt = textwrap.dedent(f"""\
        A table extracted from the prose file "{prose_stem}" shows a value-split
        signature: the CANONICAL columns below (defined by the data governance
        document) contain NO values at all, while the DISCOVERED columns below
        (added during extraction) are populated.

        For each DISCOVERED column, decide whether it denotes EXACTLY the same
        quantity as ONE of the empty canonical columns — same concept AND every
        binding qualifier: entity scope (whose quantity it is), before/after
        state, time window, numerator/denominator, unit family, aggregation
        level.
        - Full match → the values belong under the canonical name (the
          extraction accidentally split one concept into two columns).
        - Any qualifier differs, no canonical fits, or you are unsure →
          DISTINCT.
        Each canonical column can absorb at most ONE discovered column; when
        several discovered columns seem close, pick the single best match and
        mark the rest DISTINCT.

        EMPTY CANONICAL COLUMNS:
        {canon_block}

        POPULATED DISCOVERED COLUMNS (with sample values):
        {disc_block}

        Task question (context for how the table is used):
        {question}

        Output one line per discovered column, nothing else:
          <discovered_column>=<canonical_column>
          <discovered_column>=DISTINCT""")
    messages = [
        ModelMessage(
            role="system",
            content=(
                "You adjudicate whether extracted table columns are exact "
                "synonyms of governance schema columns. Output only the "
                "mapping lines."
            ),
        ),
        ModelMessage(role="user", content=prompt),
    ]
    try:
        response: ModelResponse = adapter.complete(messages)
    except Exception as exc:
        logger.warning("ETL synonym adjudication failed for '%s': %s", prose_stem, exc)
        return {}
    return _parse_alias_response(
        response.content,
        [name for name, _defn, _samples in discovered],
        list(empty_canonicals),
    )


def _unify_csv_synonyms(
    adapter: ModelAdapter,
    csv_path: Path,
    prose_stem: str,
    governance_columns: list[str],
    anchor_keys: list[str],
    field_defs: dict[str, str],
    question: str,
    units_path: Path | None,
) -> list[tuple[str, str]]:
    """Detect and repair value-split columns in a freshly materialized CSV.

    Value split: the schema keeps a governance-canonical column AND an
    extraction-discovered column for the same concept; values land in the
    discovered one, leaving the canonical column 100% empty (and any unit
    conversion bound to the canonical name pointing at dead cells).

    Trigger is deliberately narrow — only canonical columns with ZERO fill
    are offered for adjudication; partially-filled canonicals are left alone.
    The merge itself is deterministic (`apply_synonym_merges`): identity
    columns are protected and any row-level disagreement vetoes the pair.
    On merge, unit-sidecar metadata follows the values to the canonical name.

    Returns the applied (source, canonical) pairs; [] when nothing changed.
    """
    try:
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = list(reader)
    except (OSError, csv.Error):
        return []
    if not header or not rows:
        return []
    width = len(header)
    rows = [row + [""] * (width - len(row)) if len(row) < width else row[:width] for row in rows]

    fill = column_fill_counts(header, rows)
    gov_lower = {g.lower() for g in governance_columns}
    anchor_lower = {a.lower() for a in anchor_keys}
    empty_canonicals = {
        col: field_defs.get(col, "")
        for col in header
        if col.lower() in gov_lower and col.lower() not in anchor_lower and fill[col] == 0
    }
    if not empty_canonicals:
        return []
    discovered_cols = [
        col
        for col in header
        if col.lower() not in gov_lower and col.lower() not in anchor_lower and fill[col] > 0
    ]
    if not discovered_cols:
        return []

    sample_idx = {col: header.index(col) for col in discovered_cols}
    samples: dict[str, list[str]] = {col: [] for col in discovered_cols}
    for row in rows:
        for col in discovered_cols:
            val = row[sample_idx[col]].strip()
            if val and val not in samples[col] and len(samples[col]) < 3:
                samples[col].append(val)

    mapping = _adjudicate_synonym_columns(
        adapter,
        prose_stem,
        empty_canonicals,
        [(col, field_defs.get(col, ""), samples[col]) for col in discovered_cols],
        question,
    )
    if not mapping:
        return []

    # Collision guard: two discovered columns claiming one canonical column is
    # either a misjudgment or naming drift — trust neither, corrupt nothing.
    by_canon: dict[str, list[str]] = {}
    for disc, canon in mapping.items():
        by_canon.setdefault(canon, []).append(disc)
    merges: list[tuple[str, str]] = []
    for canon, discs in sorted(by_canon.items()):
        if len(discs) > 1:
            logger.warning(
                "ETL synonym-unify: columns %s all claim canonical '%s' — skipped as ambiguous",
                sorted(discs),
                canon,
            )
            continue
        merges.append((discs[0], canon))

    # Unit-family guard: never merge values recorded under a different source
    # unit than the canonical column's — that would mix magnitudes in one column.
    units_meta: dict[str, Any] = {}
    if units_path is not None and units_path.is_file():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            units_meta = json.loads(units_path.read_text(encoding="utf-8"))
    guarded: list[tuple[str, str]] = []
    for src, dst in merges:
        src_unit, dst_unit = units_meta.get(src), units_meta.get(dst)
        if src_unit and dst_unit and str(src_unit) != str(dst_unit):
            logger.warning(
                "ETL synonym-unify: unit mismatch '%s'(%s) vs '%s'(%s) — not merged",
                src,
                src_unit,
                dst,
                dst_unit,
            )
            continue
        guarded.append((src, dst))
    if not guarded:
        return []

    new_header, new_rows, applied = apply_synonym_merges(
        header,
        rows,
        guarded,
        protected=set(anchor_keys),
    )
    if not applied:
        return []

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(new_rows)

    if units_meta:
        changed = False
        for src, dst in applied:
            for prefix in ("", "_target_", "_factor_"):
                src_key, dst_key = f"{prefix}{src}", f"{prefix}{dst}"
                if src_key in units_meta:
                    units_meta.setdefault(dst_key, units_meta[src_key])
                    del units_meta[src_key]
                    changed = True
        if changed and units_path is not None:
            with contextlib.suppress(OSError):
                units_path.write_text(
                    json.dumps(units_meta, ensure_ascii=False),
                    encoding="utf-8",
                )

    for src, dst in applied:
        logger.info(
            "ETL synonym-unify: merged column '%s' → '%s' in %s",
            src,
            dst,
            csv_path.name,
        )
    return applied


def _repair_csv_numeric_identities(csv_path: Path) -> list[tuple[int, str, str, str]]:
    """Run additive-identity repair over a materialized CSV, in place.

    Recomputes approximate numerals (tagged ``~`` by compression, or untagged
    coarse values) from table invariants mined deterministically from the
    exact rows, and strips all approx tags. Rewrites the file only when a
    cell changed. Returns the applied repairs.
    """
    try:
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = list(reader)
    except (OSError, csv.Error):
        return []
    if not header or not rows:
        return []
    width = len(header)
    rows = [row + [""] * (width - len(row)) if len(row) < width else row[:width] for row in rows]

    repaired_rows, repairs = repair_numeric_identities(header, rows)
    if repaired_rows == rows:
        return repairs

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(repaired_rows)
    for row_idx, column, old, new in repairs:
        logger.info(
            "ETL identity-repair: %s row %d '%s': %s → %s",
            csv_path.name,
            row_idx,
            column,
            old,
            new,
        )
    return repairs


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
            # Header row: locate the column-name column, default first.
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


def _km_table_fields(km_text: str, prose_stem: str) -> dict[str, str]:
    """Deterministically parse the governance field table for ``prose_stem``.

    Locates heading lines mentioning the stem (backticked or word-bounded —
    ``\\b`` keeps "mf_netvalue" from anchoring inside
    "mf_netvalueperformancehis"), bounds the section at the next heading of
    the same or higher level, and parses its markdown column table.  The
    first matching heading whose section yields fields wins.  Returns {} when
    nothing parses — callers fall back to the LLM-only schema unchanged.
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


def _apply_km_schema_floor(
    columns: list[str],
    raw_defs: dict[str, str],
    km_fields: dict[str, str],
) -> list[str]:
    """Re-add governance-listed columns the LLM omitted from the schema.

    The governance document's field table is authoritative: a column listed
    there must survive into the schema even when the prose sample happens to
    show no values for it (task_53 lost ``totalliabilities`` exactly this
    way).  Mutates ``columns``/``raw_defs`` in place; returns the re-added
    column names.
    """
    existing = {c.lower() for c in columns}
    readded = [c for c in km_fields if c.lower() not in existing]
    for c in readded:
        columns.append(c)
        raw_defs.setdefault(c, km_fields[c])
    return readded


def _extract_relevant_entity(
    adapter: ModelAdapter,
    km_text: str,
    prose_stem: str,
    prose_sample: str,
) -> str | None:
    """Pre-filter knowledge.md to only the entity section matching the prose file.

    Sends a lightweight LLM call to identify which entity in the governance
    document corresponds to the prose file, and returns only that section.
    This prevents cross-table column contamination in downstream schema inference.
    """
    prompt = textwrap.dedent(f"""\
        Below is a data governance document that describes MULTIPLE database \
        tables / entities. I also provide a prose data file name and a short \
        sample from it.

        Prose file: "{prose_stem}"
        Prose sample (first few lines):
        ---
        {prose_sample[:1500]}
        ---

        Task: find the ONE entity / table section in the governance document \
        that best matches this prose file. Copy that section VERBATIM — \
        include the heading, all column definitions, all footnotes and units \
        that belong to it. Do NOT include sections for other entities.

        If no entity matches, output exactly: NONE

        Governance document:
        {km_text}""")
    messages = [
        ModelMessage(
            role="system",
            content="You extract the relevant section from a governance document. Output only the matched section verbatim, nothing else.",
        ),
        ModelMessage(role="user", content=prompt),
    ]
    try:
        response: ModelResponse = adapter.complete(messages)
    except Exception as exc:
        logger.warning("LLM entity pre-filter failed: %s", exc)
        return None

    raw = response.content.strip()
    if not raw or raw.strip().strip("`").upper() == "NONE":
        return None
    if len(raw) < 20:
        return None
    logger.info(
        "Knowledge pre-filter for '%s': %d → %d chars",
        prose_stem,
        len(km_text),
        len(raw),
    )
    return raw


def _parse_knowledge_schema(
    task: PublicTask, prose_path: Path, adapter: ModelAdapter, prose_text: str
) -> tuple[list[str], list[str], dict[str, str], dict[str, str], dict[str, str]] | None:
    """Use LLM to extract column names, anchor keys, units, and definitions from knowledge.md.

    Samples the prose file so the LLM can discover fields present in the data
    but missing from the governance document (e.g. record IDs).

    Returns (columns, anchor_keys, field_types, field_units, field_defs) or None.
    ``anchor_keys[0]`` is the canonical primary key; subsequent entries are
    secondary identity fields (name, label, secondary ID) that can uniquely
    identify an entity when the PK takes different forms across sections.
    """
    km_path = task.context_dir / "knowledge.md"
    if not km_path.is_file():
        return None
    try:
        km_text = km_path.read_text(encoding="utf-8")
    except OSError:
        return None

    prose_stem = prose_path.stem

    # Deterministic gate: a governance document that never mentions this
    # table cannot define its schema.  Skipping the knowledge path outright
    # keeps the LLM entity matcher from force-matching a DIFFERENT table's
    # section (task_59 ed_consumerpriceindex inherited retailvalueofscgoods
    # columns that way) and saves two LLM calls.
    if prose_stem.lower() not in km_text.lower():
        logger.info(
            "ETL schema: '%s' not mentioned in knowledge.md — skipping knowledge path",
            prose_stem,
        )
        return None

    prose_sample = sample_paragraphs_by_section(prose_text) if prose_text else ""

    # Deterministic field floor, parsed from the ORIGINAL governance text —
    # before the LLM pre-filter gets a chance to drop table rows.
    km_fields = _km_table_fields(km_text, prose_stem)

    filtered_km = _extract_relevant_entity(adapter, km_text, prose_stem, prose_sample or "")
    if filtered_km and prose_stem.lower() in filtered_km.lower():
        km_text = filtered_km
    elif filtered_km:
        # The filter returned a section that never mentions this table — it
        # matched some other entity.  A real section would carry the table
        # name in its heading; treat as no-match instead of letting the
        # schema call inherit a wrong table's columns.
        logger.warning(
            "ETL schema: knowledge pre-filter for '%s' returned a section "
            "not mentioning it — treating as no-match",
            prose_stem,
        )
        return None

    prose_section = ""
    if prose_sample:
        prose_section = (
            f"\n\nSample paragraphs from EVERY SECTION of the prose file "
            f'"{prose_stem}.md" (use these to discover ALL numeric/categorical '
            f"data fields present in the prose, even if the governance document "
            f"does not list them):\n"
            f"---\n{prose_sample}\n---"
        )

    prompt = textwrap.dedent(f"""\
        Below is a data governance document AND a sample from a prose data file.
        Find the entity in the governance document that best matches \
        the prose file "{prose_stem}".

        If NO entity in the governance document matches, output exactly: NONE

        Otherwise output exactly SIX lines:
        Line 1: PK: <primary_key_field_name>
        Line 2: ANCHORS: <comma-separated identity fields that can uniquely \
        identify an entity — includes PK plus name/label/secondary ID fields>
        Line 3: TYPES: <field=type, field=type, ... for every field>
        Line 4: UNITS: <field=unit for each numeric field whose values carry \
        a unit in the prose sample or governance document. Use _default=<unit> \
        for the governance document's declared default monetary unit.>
        Line 5: DEFS: <field=one-sentence semantic definition; field=definition; \
        ... for every field. Use semicolons between entries. Copy the exact \
        semantic definition from the governance document when available; for \
        fields discovered only in the prose sample, write a concise definition \
        that distinguishes this field from similarly-named fields.>
        Line 6: all field names as a comma-separated list (must include all ANCHORS)

        IMPORTANT:
        - CRITICAL: columns listed in the governance document already have \
        canonical database names (e.g. "sumbeforetran", "pctbeforetran", \
        "companycode"). You MUST use these EXACT names verbatim in PK, \
        ANCHORS, TYPES, UNITS, DEFS, and the column list. NEVER rename, \
        translate, expand, or paraphrase them — "sumbeforetran" stays \
        "sumbeforetran", NOT "shares_held_before" or "sum_before_tran".
        - Only invent new field names for fields discovered in the prose sample \
        that are not fully covered by any governance column's definition. \
        New names must be SHORT GENERIC FIELD NAMES (like "event_id", \
        "tran_date"), NEVER a specific data value.
        - ANCHORS should include the PK and any name/label/secondary identifier \
        field that could be used to recognize the same entity across different \
        sections of the document (e.g. a person's name, a record label).
        - MANDATORY: include EVERY column listed in the governance document's \
        table for the matched entity — even when the prose sample shows no \
        values for it (the sample covers a fraction of the document; absence \
        from the sample is NOT absence from the data). Omitting a governance \
        column is an error.
        - Use simple type labels such as integer_scalar_id, scalar_id, date, time, \
        number, integer_count, integer_rank, string, category, boolean, url.
        - For document-local record_id fields use type "integer_scalar local_record_id"; \
        record_id values must be a single scalar number, not a range or list.
        - MANDATORY: scan the prose sample for numeric, date, and \
        categorical data fields that have NO matching governance column. \
        For each prose field, check every governance column's \
        definition / description. A governance column covers a prose field \
        ONLY when it matches the same concept AND all binding qualifiers: \
        time window, before/after state, numerator/denominator, entity scope, \
        category, unit, and aggregation level. Opaque database names are \
        canonical when their definitions match exactly; do NOT add translated \
        aliases for them. If a prose field is only related to a governance \
        column but differs in any binding qualifier, treat it as a distinct \
        discovered field and add a short descriptive name. \
        Example — WRONG: governance has "rrintenyear" defined as \
        "近十年累计回报率", prose shows "近十年累计回报率: 166.09" → adding \
        "ten_year_return" creates a duplicate column and splits values. \
        CORRECT: recognise "近十年累计回报率" matches "rrintenyear" → use \
        "rrintenyear", do NOT add "ten_year_return". \
        Example — CORRECT addition: governance has "sumbeforetran" \
        ("转让前持股") but prose ALSO has "转让数量" which no governance \
        column covers → add "transfer_shares" as a new discovered field. \
        Two columns for the same concept causes value split (some rows \
        fill column A, others fill column B) and is a critical schema error.
        - If paragraphs are keyed by repeated document-local labels such as \
        "档案 353", "记录 353", "条目 353", "战略单元 353", "Record 353", \
        "File 353", or "Case 353", include a field named "record_id" and \
        put it FIRST in PK and ANCHORS. This local record ID is the row key \
        for merging sections; it is NOT the same as domain IDs such as \
        personalcode, innercode, fundcode, or secucode.
        - Do NOT add narrative/filler fields — only structured data fields.
        - For the UNITS line: scan the prose sample for unit suffixes attached \
        to numeric values (e.g. "2478.76亿元" → gdp=亿元, "15.3%" → rate=%). \
        Output a field=unit pair when you find a CONCRETE unit suffix \
        (e.g. 元, 万元, 亿元, %, 万吨, 万人, 公斤) in the actual data values \
        or governance document. Fields whose values appear with "%" in the \
        prose or are described as "percentage" / "比例" in the governance \
        document MUST be tagged with unit=%. \
        A currency denomination statement (e.g. "denominated in Chinese \
        Yuan", "currency: CNY", "以人民币计价") declares the CURRENCY, not \
        the magnitude — it is NOT a unit. NEVER emit field=元 from such a \
        statement; only tag a field 元 when its numeric values carry an \
        explicit 元 suffix. Monetary fields with no visible magnitude suffix \
        are already in the governance document's default unit and must be \
        OMITTED. \
        Dimensionless fields (price indices, ranks, counts without unit) and \
        fields with no visible unit suffix \
        must be OMITTED entirely — do NOT write them with "_default", \
        "currency", "units", "amount", "count", or any placeholder.

        Governance document:
        {km_text}{prose_section}""")
    messages = [
        ModelMessage(
            role="system",
            content=(
                "You extract database schema. "
                "Output PK, ANCHORS, TYPES, UNITS, DEFS, then comma-separated columns."
            ),
        ),
        ModelMessage(role="user", content=prompt),
    ]
    try:
        response: ModelResponse = adapter.complete(messages)
    except Exception as exc:
        logger.warning("LLM schema extraction failed: %s", exc)
        return None

    raw = response.content.strip()
    if not raw or raw.strip().strip("`").upper() == "NONE":
        return None

    parsed = _parse_schema_response(raw)
    if not parsed:
        return None
    _, anchor_keys, explicit_types, columns, field_units, raw_defs = parsed

    if km_fields:
        readded = _apply_km_schema_floor(columns, raw_defs, km_fields)
        if readded:
            logger.warning(
                "ETL schema: LLM omitted %d columns listed in knowledge.md for '%s': %s — re-added",
                len(readded),
                prose_stem,
                readded,
            )

    columns, anchor_keys = _ensure_local_record_anchor(
        prose_text,
        columns,
        anchor_keys,
        explicit_types,
    )
    field_types = _complete_schema_field_types(columns, explicit_types)
    field_defs = _complete_schema_field_defs(columns, raw_defs)

    logger.info(
        "Schema from knowledge.md (LLM) for '%s': pk=%s, anchors=%s, types=%s, units=%s, defs=%s, cols=%s",
        prose_stem,
        anchor_keys[0],
        anchor_keys,
        field_types,
        field_units,
        field_defs,
        columns,
    )
    return columns, anchor_keys, field_types, field_units, field_defs


def _infer_schema_from_sample(
    adapter: ModelAdapter,
    prose_text: str,
    question: str,
) -> tuple[list[str], list[str], dict[str, str], dict[str, str], dict[str, str]] | None:
    """Infer a unified schema by sampling the raw prose text before compression.

    Sends a representative sample to the LLM to discover structured data fields
    and produce a canonical column list + anchor keys.

    Returns (columns, anchor_keys, field_types, field_units, field_defs) or None.
    """
    sample = sample_paragraphs_by_section(prose_text)
    if not sample or len(sample) < 50:
        return None

    prompt = textwrap.dedent(f"""\
        Below is a sample from a prose data document. Each paragraph describes
        one entity and mixes structured data with narrative filler.

        Task question for context:
        {question}

        Sample (from multiple sections):
        ---
        {sample}
        ---

        Your job:
        1. Identify ALL structured data fields present in the sample
           (numeric values, dates, categories, IDs, codes, statuses).
        2. Ignore narrative filler (descriptions, opinions, context sentences
           that don't carry a reusable data value).
        3. Assign each field a SHORT canonical name (snake_case, e.g.
           "effective_date", "reserve_ratio", "target_entity").
        4. Identify the primary key field (the field that uniquely identifies
           each entity — usually an ID number).
        5. Identify ANCHOR fields — identity fields (including PK) that can
           uniquely identify an entity even when the PK is missing or takes
           a different form (e.g. a name, label, or secondary ID).

        Output exactly SIX lines:
        Line 1: PK: <primary_key_field_name>
        Line 2: ANCHORS: <comma-separated identity fields including PK>
        Line 3: TYPES: <field=type, field=type, ... for every field>
        Line 4: UNITS: <field=unit for each numeric field whose values carry \
        a CONCRETE unit suffix in the prose sample (e.g. gdp=亿元, rate=%). \
        Only use real unit suffixes visible in the data values. Dimensionless \
        fields (price indices, ratios, ranks, counts without unit) and fields \
        with no visible unit suffix must be OMITTED — do NOT write them with \
        "_default", "currency", "units", "amount", or any placeholder.>
        Line 5: DEFS: <field=one-sentence semantic definition; field=definition; \
        ... for every field. Semicolons between entries. Write a concise \
        definition that distinguishes this field from similarly-named fields.>
        Line 6: all field names as a comma-separated list (must include all ANCHORS)

        Rules:
        - Column names must be SHORT GENERIC FIELD NAMES, NEVER specific data values.
        - Only include fields that carry actual structured data values.
        - Use simple type labels such as integer_scalar_id, scalar_id, date, time,
          number, integer_count, integer_rank, string, category, boolean, url.
        - If paragraphs are keyed by repeated document-local labels such as
          "档案 353", "记录 353", "条目 353", "战略单元 353", "Record 353",
          "File 353", or "Case 353", include a field named "record_id" and
          put it FIRST in PK and ANCHORS. This local record ID is the row key
          for merging sections; it is NOT the same as domain IDs such as
          personalcode, innercode, fundcode, or secucode.
          Give record_id type "integer_scalar local_record_id"; record_id values
          must be a single scalar number, not a range or list.
        - Do NOT include narrative/context/filler fields.""")

    messages = [
        ModelMessage(
            role="system",
            content="You extract database schema from prose samples. Output PK, ANCHORS, TYPES, UNITS, DEFS, then columns.",
        ),
        ModelMessage(role="user", content=prompt),
    ]
    try:
        response: ModelResponse = adapter.complete(messages)
    except Exception as exc:
        logger.warning("LLM schema inference from sample failed: %s", exc)
        return None

    parsed = _parse_schema_response(response.content)
    if not parsed:
        return None
    _, anchor_keys, explicit_types, columns, field_units, raw_defs = parsed

    columns, anchor_keys = _ensure_local_record_anchor(
        prose_text,
        columns,
        anchor_keys,
        explicit_types,
    )
    field_types = _complete_schema_field_types(columns, explicit_types)
    field_defs = _complete_schema_field_defs(columns, raw_defs)

    logger.info(
        "Schema inferred from sample: pk=%s, anchors=%s, types=%s, defs=%s, cols=%s",
        anchor_keys[0],
        anchor_keys,
        field_types,
        field_defs,
        columns,
    )
    return columns, anchor_keys, field_types, field_units, field_defs


def _make_text_adapter(model_adapter: Any) -> ModelAdapter:
    """Build a plain-text adapter (no tools) from the existing adapter's credentials.

    When *model_adapter* is a ``RateLimitedAdapter``, the returned adapter is
    also rate-limited — sharing the same token-bucket state files so ETL and
    agent calls compete for the same RPM/TPM budget.  Likewise, when the
    incoming adapter is wrapped with ``TracedModelAdapter``, the returned
    adapter is re-wrapped so ETL LLM calls produce generation spans.
    """
    from agents.llm.rate_limit import RateLimitedAdapter
    from agents.tracing.instrument import TracedModelAdapter

    source = model_adapter
    traced = False
    rate_limit_cfg: Any = None
    rate_limit_key: str | None = None

    if isinstance(source, TracedModelAdapter):
        traced = True
        source = source.inner

    if isinstance(source, RateLimitedAdapter):
        rate_limit_cfg = source.rate_limit_config
        rate_limit_key = source.api_key
        source = source.inner

    model: str | None = getattr(source, "model", None)
    api_base: str | None = getattr(source, "api_base", None)
    api_key: str | None = getattr(source, "api_key", None)
    backend_kind: Any = getattr(source, "backend_kind", None)

    if not model or not api_base or not api_key:
        raise RuntimeError("Cannot extract model credentials from adapter for ETL.")

    has_rate_limiter = rate_limit_cfg is not None and rate_limit_key
    seed: int | None = getattr(source, "seed", None)
    top_p: float | None = getattr(source, "top_p", None)
    top_k: int | None = getattr(source, "top_k", None)
    is_deepseek = backend_kind == "deepseek" or getattr(source, "thinking_mode", None) is not None

    inner: ModelAdapter = OpenAIModelAdapter(
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=0.1,
        enable_thinking=None if is_deepseek else False,
        thinking_mode="disabled" if is_deepseek else None,
        backend_kind=backend_kind,
        seed=seed,
        top_p=top_p,
        top_k=top_k,
        max_retries=0 if has_rate_limiter else 2,
        timeout=_ETL_HTTP_TIMEOUT,
    )

    if rate_limit_cfg is not None and rate_limit_key:
        inner = RateLimitedAdapter(
            inner=inner,
            api_key=rate_limit_key,
            rate_limit=rate_limit_cfg,
        )

    if traced:
        inner = TracedModelAdapter(inner)

    return inner


def extract_prose_file(
    adapter: ModelAdapter,
    path: Path,
    task: PublicTask,
    prose_text: str,
    schema_columns: list[str] | None = None,
    anchor_keys: list[str] | None = None,
    schema_field_types: dict[str, str] | None = None,
    field_units: dict[str, str] | None = None,
    schema_field_defs: dict[str, str] | None = None,
    governance_columns: list[str] | None = None,
) -> ETLResult | None:
    """Extract structured data from a prose file.

    Pipeline: compress → pre-merge → normalize → deterministic CSV.
    Fallback: LLM direct extraction → code-gen.
    """
    primary_key = anchor_keys[0] if anchor_keys else None

    etl_dir = ETL_SCRATCH_ROOT / task.task_id / "_etl"
    etl_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ETL_SCRATCH_ROOT / task.task_id / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = etl_dir / f"{path.stem}.csv"
    full_text = prose_text

    def _finalize(phase: str) -> ETLResult | None:
        """Validate the materialized CSV, unify value-split synonym columns,
        repair approximate numerals from mined identities, build result."""
        result = validate_csv(csv_path)
        if not result:
            return None
        changed = False
        if governance_columns:
            applied = _unify_csv_synonyms(
                adapter,
                csv_path,
                path.stem,
                governance_columns,
                anchor_keys or [],
                schema_field_defs or {},
                task.question,
                cache_dir / f"{path.stem}_units.json",
            )
            changed = bool(applied)
        if _repair_csv_numeric_identities(csv_path):
            changed = True
        if changed:
            result = validate_csv(csv_path) or result
        header, row_count = result
        logger.info(
            "ETL %s: %s → %s (%d cols, %d rows)",
            phase,
            path.name,
            csv_path,
            len(header),
            row_count,
        )
        return ETLResult(
            source_file=path.name,
            csv_path=str(csv_path),
            columns=header,
            row_count=row_count,
        )

    if path.suffix.lower() == ".pdf":
        md_path = cache_dir / f"{path.stem}.md"
        md_path.write_text(full_text, encoding="utf-8")

    # ---- Cache: reuse existing CSV if present and valid ----
    cached = validate_csv(csv_path)
    if cached:
        header, row_count = cached
        logger.info("ETL cache hit: %s (%d cols, %d rows)", csv_path, len(header), row_count)
        return ETLResult(
            source_file=path.name,
            csv_path=str(csv_path),
            columns=header,
            row_count=row_count,
        )

    if not full_text.strip():
        logger.info("ETL skip: no extractable text in %s", path.name)
        return None

    # ---- Phase 0: infer schema from prose sample (when no schema from knowledge.md) ----
    if not schema_columns:
        inferred = _infer_schema_from_sample(adapter, full_text, task.question)
        if inferred:
            schema_columns, anchor_keys, schema_field_types, field_units, schema_field_defs = (
                inferred
            )
            primary_key = anchor_keys[0] if anchor_keys else None

    # ---- Phase 1: compress ----
    logger.info("ETL compress: %s (%d chars)", path.name, len(full_text))
    compressed = compress_prose(
        adapter,
        full_text,
        schema_columns=schema_columns,
        primary_key=primary_key,
        anchor_keys=anchor_keys,
        schema_field_types=schema_field_types,
        schema_field_defs=schema_field_defs,
    )
    compressed_path = cache_dir / f"{path.stem}_clean.md"
    compressed_path.write_text(compressed, encoding="utf-8")
    logger.info(
        "ETL compressed: %d → %d chars (%.0f%%)",
        len(full_text),
        len(compressed),
        len(compressed) / len(full_text) * 100,
    )

    if not primary_key:
        compressed = resolve_id_conflicts(compressed)

    compressed = pre_merge_entity_lines(
        compressed,
        primary_key=primary_key,
        anchor_keys=anchor_keys,
    )
    if schema_columns:
        compressed = normalize_entity_lines(compressed, schema_columns, primary_key)
    compressed_path.write_text(compressed, encoding="utf-8")

    if field_units:
        _write_units_sidecar(task, path.stem, field_units)

    # ---- Phase 2a: deterministic KV→CSV (when normalized) ----
    if schema_columns:
        csv_text = deterministic_kv_to_csv(
            compressed,
            schema_columns,
            primary_key,
            schema_field_types,
        )
        if csv_text:
            csv_path.write_text(csv_text, encoding="utf-8")
            result = _finalize("deterministic")
            if result:
                return result
            logger.warning("ETL deterministic extraction produced invalid CSV, falling back")

    # ---- Phase 2b: LLM extraction (if compressed text is small enough) ----
    if len(compressed) <= _DIRECT_EXTRACT_CHAR_LIMIT:
        logger.info("ETL: attempting LLM CSV extraction from compressed text")
        csv_text = direct_extract_csv(
            adapter,
            compressed,
            schema_columns,
            task.question,
            primary_key=primary_key,
            schema_field_types=schema_field_types,
        )
        if csv_text:
            csv_path.write_text(csv_text, encoding="utf-8")
            result = _finalize("direct")
            if result:
                return result
        logger.warning("ETL direct extraction failed, falling back to code-gen")

    # ---- Phase 2c: code-generation fallback (on compressed file) ----
    entity_lines = [
        ln.strip() for ln in compressed.splitlines() if ln.strip() and not ln.startswith("#")
    ]
    sample_text = "\n".join(entity_lines[:5] + (["..."] if len(entity_lines) > 5 else []))

    prompt = build_codegen_prompt(
        compressed_path,
        csv_path,
        sample_text,
        len(entity_lines),
        schema_columns,
        task.question,
    )

    code = ""
    for attempt in range(MAX_RETRIES + 1):
        raw = call_llm(adapter, prompt)
        code = extract_code_block(raw)
        ok, output = exec_code(code)
        if ok:
            logger.info("ETL code exec succeeded (attempt %d): %s", attempt + 1, output[:200])
            break
        logger.warning("ETL code exec failed (attempt %d): %s", attempt + 1, output[:300])
        if attempt < MAX_RETRIES:
            prompt = build_retry_prompt(code, output)
    else:
        logger.error("ETL code generation failed after %d attempts", MAX_RETRIES + 1)
        return None

    result = _finalize("codegen")
    if not result:
        logger.error("ETL code ran but CSV is empty or missing: %s", csv_path)
        return None
    return result


def run_etl_for_task(
    task: PublicTask,
    model_adapter: Any,
) -> list[ETLResult] | None:
    """Orchestrator: detect large prose files, extract each, return metadata list.

    Returns None if no large prose files found. Non-fatal: callers should
    catch exceptions and fall back to manual parsing.
    """
    force_etl = task.difficulty.lower() in {"hard", "extreme"}
    prose_files = detect_prose_files(task, force=force_etl)
    if not prose_files:
        return None

    adapter = _make_text_adapter(model_adapter)

    # Read knowledge.md once for target-unit voting (shared across all files).
    km_path = task.context_dir / "knowledge.md"
    km_text = ""
    if km_path.is_file():
        with contextlib.suppress(OSError):
            km_text = km_path.read_text(encoding="utf-8")

    def _process(pf: Path) -> ETLResult | None:
        # Early cache check: skip file reading and all LLM calls if CSV exists.
        etl_dir = ETL_SCRATCH_ROOT / task.task_id / "_etl"
        csv_path = etl_dir / f"{pf.stem}.csv"
        cached = validate_csv(csv_path)
        if cached:
            header, row_count = cached
            logger.info(
                "ETL cache hit (early): %s (%d cols, %d rows)", csv_path, len(header), row_count
            )
            return ETLResult(
                source_file=pf.name,
                csv_path=str(csv_path),
                columns=header,
                row_count=row_count,
            )

        if pf.suffix.lower() == ".pdf":
            logger.info("ETL PDF→markdown: %s", pf.name)
            text: str = pdf_to_markdown(pf)  # pyright: ignore[reportUnknownVariableType]
        else:
            text = pf.read_text(encoding="utf-8")
        parsed = _parse_knowledge_schema(task, pf, adapter, text)
        if parsed:
            schema, anchors, field_types, units, defs = parsed
            units.pop("_default", None)
            if km_text and units:
                targets = _vote_target_units(adapter, km_text, units, question=task.question)
                if targets:
                    for field, (target_unit, factor) in targets.items():
                        units[f"_target_{field}"] = target_unit
                        units[f"_factor_{field}"] = str(factor)
                    logger.info("ETL: voted target units for %s: %s", pf.stem, targets)
            governance_cols = sorted(_km_table_fields(km_text, pf.stem)) if km_text else []
            return extract_prose_file(
                adapter,
                pf,
                task,
                text,
                schema_columns=schema,
                anchor_keys=anchors,
                schema_field_types=field_types,
                field_units=units,
                schema_field_defs=defs,
                governance_columns=governance_cols or None,
            )
        logger.info("ETL: no knowledge.md schema for %s, will infer from sample", pf.name)
        return extract_prose_file(adapter, pf, task, text)

    if len(prose_files) == 1:
        result = _process(prose_files[0])
        return [result] if result else None

    results: list[ETLResult] = []
    orch_timeout = _LLM_CALL_TIMEOUT * max(4, len(prose_files))
    with ThreadPoolExecutor(max_workers=min(len(prose_files), _MAX_WORKERS)) as pool:
        futs = {pool.submit(_process, pf): pf for pf in prose_files}
        try:
            for fut in as_completed(futs, timeout=orch_timeout):
                try:
                    result = fut.result(timeout=orch_timeout)
                    if result is not None:
                        results.append(result)
                except TimeoutError:
                    logger.error("ETL timed out for %s", futs[fut].name)
                except Exception as exc:
                    logger.error("ETL failed for %s: %s", futs[fut].name, exc)
        except TimeoutError:
            logger.error("ETL orchestrator timed out, proceeding with partial results")

    return results if results else None
