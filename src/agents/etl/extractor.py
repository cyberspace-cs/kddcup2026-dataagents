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
    deterministic_kv_to_csv,
    normalize_entity_lines,
    pre_merge_entity_lines,
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
    if known_fields & RECORD_ID_FIELDS:
        return columns, anchor_keys
    if explicit_types:
        for field, ftype in explicit_types.items():
            if field.lower() in known_fields and "local_record_id" in ftype.lower():
                return columns, anchor_keys
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


def _parse_schema_response(
    text: str,
) -> tuple[str, list[str], dict[str, str], list[str], dict[str, str]] | None:
    """Parse LLM schema output by line prefix, not position.

    Looks for PK:, ANCHORS:, TYPES:, UNITS: prefixes in any order.
    The first non-prefixed line with commas is treated as the column list.

    Returns (pk, anchor_keys, explicit_types, columns, field_units) or None.
    """
    lines = [line.strip() for line in text.strip().strip("`").splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    pk: str | None = None
    anchor_keys: list[str] = []
    explicit_types: dict[str, str] = {}
    field_units: dict[str, str] = {}
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

    return pk, anchor_keys, explicit_types, columns, field_units


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


def _vote_target_units(
    adapter: ModelAdapter,
    km_text: str,
    field_units: dict[str, str],
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
        "Dimensionless fields (price indices, ratios, ranks) must keep their "
        "source unit as-is — do NOT replace with '_default' or any placeholder.\n\n"
        "For each field where source ≠ target, also output the numeric "
        "multiplication factor to convert from source to target.\n\n"
        "Output format — one field per entry, comma-separated:\n"
        "  field=target_unit*factor\n"
        "If source == target (no conversion needed), omit the *factor part:\n"
        "  field=target_unit\n\n"
        "Examples:\n"
        "  gdp=万元*10000, revenue=万元*10000, rate=%\n"
        "  (gdp source was 亿元, target is 万元, so factor=10000)\n"
        "  (rate source was %, target is %, so no factor)\n\n"
        f"knowledge.md:\n{km_text}"
    )
    messages = [
        ModelMessage(
            role="system",
            content=(
                "Map each field to its target unit per knowledge.md. "
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


def _parse_knowledge_schema(
    task: PublicTask, prose_path: Path, adapter: ModelAdapter, prose_text: str
) -> tuple[list[str], list[str], dict[str, str], dict[str, str]] | None:
    """Use LLM to extract column names, anchor keys, and units from knowledge.md.

    Samples the prose file so the LLM can discover fields present in the data
    but missing from the governance document (e.g. record IDs).

    Returns (columns, anchor_keys, field_types, field_units) or None.
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

    prose_sample = sample_paragraphs_by_section(prose_text, per_section=1) if prose_text else ""

    prose_stem = prose_path.stem
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

        Otherwise output exactly FIVE lines:
        Line 1: PK: <primary_key_field_name>
        Line 2: ANCHORS: <comma-separated identity fields that can uniquely \
        identify an entity — includes PK plus name/label/secondary ID fields>
        Line 3: TYPES: <field=type, field=type, ... for every field>
        Line 4: UNITS: <field=unit for each numeric field whose values carry \
        a unit in the prose sample or governance document. Use _default=<unit> \
        for the governance document's declared default monetary unit.>
        Line 5: all field names as a comma-separated list (must include all ANCHORS)

        IMPORTANT:
        - PK and column names must be SHORT GENERIC FIELD NAMES (like "event_id", \
        "molecule_id", "patient_id"), NEVER a specific data value.
        - ANCHORS should include the PK and any name/label/secondary identifier \
        field that could be used to recognize the same entity across different \
        sections of the document (e.g. a person's name, a record label).
        - Start with fields from the governance document.
        - Use simple type labels such as integer_scalar_id, scalar_id, date, time, \
        number, integer_count, integer_rank, string, category, boolean, url.
        - For document-local record_id fields use type "integer_scalar local_record_id"; \
        record_id values must be a single scalar number, not a range or list.
        - ALSO scan the prose sample for ADDITIONAL structured data fields \
        that are present in the data but MISSING from the governance document. \
        Add these to the list with descriptive short names matching the \
        abbreviations used in the prose.
        - If paragraphs are keyed by repeated document-local labels such as \
        "档案 353", "记录 353", "条目 353", "战略单元 353", "Record 353", \
        "File 353", or "Case 353", include a field named "record_id" and \
        put it FIRST in PK and ANCHORS. This local record ID is the row key \
        for merging sections; it is NOT the same as domain IDs such as \
        personalcode, innercode, fundcode, or secucode.
        - Do NOT add narrative/filler fields — only structured data fields.
        - For the UNITS line: scan the prose sample for unit suffixes attached \
        to numeric values (e.g. "2478.76亿元" → gdp=亿元, "15.3%" → rate=%). \
        Only output a field=unit pair when you find a CONCRETE unit suffix \
        (e.g. 元, 万元, 亿元, %, 万吨, 万人, 公斤) in the actual data values \
        or governance document. Dimensionless fields (price indices, ratios, \
        ranks, counts without unit) and fields with no visible unit suffix \
        must be OMITTED entirely — do NOT write them with "_default", \
        "currency", "units", "amount", "count", or any placeholder.

        Governance document:
        {km_text}{prose_section}""")
    messages = [
        ModelMessage(
            role="system",
            content=(
                "You extract database schema. "
                "Output PK, ANCHORS, TYPES, UNITS, then comma-separated columns."
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
    _, anchor_keys, explicit_types, columns, field_units = parsed

    columns, anchor_keys = _ensure_local_record_anchor(
        prose_text,
        columns,
        anchor_keys,
        explicit_types,
    )
    field_types = _complete_schema_field_types(columns, explicit_types)

    logger.info(
        "Schema from knowledge.md (LLM) for '%s': pk=%s, anchors=%s, types=%s, units=%s, cols=%s",
        prose_stem,
        anchor_keys[0],
        anchor_keys,
        field_types,
        field_units,
        columns,
    )
    return columns, anchor_keys, field_types, field_units


def _infer_schema_from_sample(
    adapter: ModelAdapter,
    prose_text: str,
    question: str,
) -> tuple[list[str], list[str], dict[str, str], dict[str, str]] | None:
    """Infer a unified schema by sampling the raw prose text before compression.

    Sends a representative sample to the LLM to discover structured data fields
    and produce a canonical column list + anchor keys.

    Returns (columns, anchor_keys, field_types, field_units) or None.
    """
    sample = sample_paragraphs_by_section(prose_text, per_section=2)
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

        Output exactly FIVE lines:
        Line 1: PK: <primary_key_field_name>
        Line 2: ANCHORS: <comma-separated identity fields including PK>
        Line 3: TYPES: <field=type, field=type, ... for every field>
        Line 4: UNITS: <field=unit for each numeric field whose values carry \
        a CONCRETE unit suffix in the prose sample (e.g. gdp=亿元, rate=%). \
        Only use real unit suffixes visible in the data values. Dimensionless \
        fields (price indices, ratios, ranks, counts without unit) and fields \
        with no visible unit suffix must be OMITTED — do NOT write them with \
        "_default", "currency", "units", "amount", or any placeholder.>
        Line 5: all field names as a comma-separated list (must include all ANCHORS)

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
            content="You extract database schema from prose samples. Output PK, ANCHORS, TYPES, UNITS, then columns.",
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
    _, anchor_keys, explicit_types, columns, field_units = parsed

    columns, anchor_keys = _ensure_local_record_anchor(
        prose_text,
        columns,
        anchor_keys,
        explicit_types,
    )
    field_types = _complete_schema_field_types(columns, explicit_types)

    logger.info(
        "Schema inferred from sample: pk=%s, anchors=%s, types=%s, cols=%s",
        anchor_keys[0],
        anchor_keys,
        field_types,
        columns,
    )
    return columns, anchor_keys, field_types, field_units


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
    inner: ModelAdapter = OpenAIModelAdapter(
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=0.1,
        enable_thinking=False,
        backend_kind=backend_kind,
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
            schema_columns, anchor_keys, schema_field_types, field_units = inferred
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
            result = validate_csv(csv_path)
            if result:
                header, row_count = result
                logger.info(
                    "ETL deterministic: %s → %s (%d cols, %d rows)",
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
            result = validate_csv(csv_path)
            if result:
                header, row_count = result
                logger.info(
                    "ETL direct: %s → %s (%d cols, %d rows)",
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

    result = validate_csv(csv_path)
    if not result:
        logger.error("ETL code ran but CSV is empty or missing: %s", csv_path)
        return None

    header, row_count = result
    logger.info(
        "ETL codegen: %s → %s (%d cols, %d rows)",
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
        if pf.suffix.lower() == ".pdf":
            logger.info("ETL PDF→markdown: %s", pf.name)
            text: str = pdf_to_markdown(pf)  # pyright: ignore[reportUnknownVariableType]
        else:
            text = pf.read_text(encoding="utf-8")
        parsed = _parse_knowledge_schema(task, pf, adapter, text)
        if parsed:
            schema, anchors, field_types, units = parsed
            units.pop("_default", None)
            if km_text and units:
                targets = _vote_target_units(adapter, km_text, units)
                if targets:
                    for field, (target_unit, factor) in targets.items():
                        units[f"_target_{field}"] = target_unit
                        units[f"_factor_{field}"] = str(factor)
                    logger.info("ETL: voted target units for %s: %s", pf.stem, targets)
            return extract_prose_file(
                adapter,
                pf,
                task,
                text,
                schema_columns=schema,
                anchor_keys=anchors,
                schema_field_types=field_types,
                field_units=units,
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
