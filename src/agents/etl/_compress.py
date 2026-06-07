"""LLM compression, paragraph sampling, and direct CSV extraction."""

from __future__ import annotations

import csv
import io
import logging
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from agents.etl._detect import ENTITY_LINE, detect_sections
from agents.etl._types import (
    DATE_SCALAR,
    NUMERIC_SCALAR,
    RECORD_ID_FIELDS,
    RECORD_ID_KEYWORDS,
    is_placeholder,
    normalize_boolean,
)

if TYPE_CHECKING:
    from agents.llm.types import ModelAdapter

logger = logging.getLogger(__name__)

COMPRESS_CHUNK_SIZE = 30
EXTRACT_CHUNK_LINES = 30

MAX_WORKERS = 4
LLM_CALL_TIMEOUT = 180
RECORD_ID_LABEL = re.compile(
    rf"^(?:{RECORD_ID_KEYWORDS})\s*[：:#-]?\s*(\d{{1,8}})$",
    re.IGNORECASE,
)
RECORD_ID_RANGE = re.compile(
    rf"(?:{RECORD_ID_KEYWORDS})?\s*\d{{1,8}}\s*(?:至|到|[-~–—])\s*\d{{1,8}}",
    re.IGNORECASE,
)


def _clean_numeric_scalar(val: str) -> str | None:
    stripped = val.strip()
    if is_placeholder(stripped):
        return ""
    normalized = stripped.replace(",", "")
    if normalized.endswith("%"):
        normalized = normalized[:-1].strip()
    unit_match = re.match(r"^([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)\s+\S+$", normalized)
    if unit_match:
        normalized = unit_match.group(1)
    return normalized if NUMERIC_SCALAR.match(normalized) else None


def _clean_typed_scalar(val: str, field_type: str) -> str | None:
    type_text = field_type.lower()
    if "number" in type_text or "integer" in type_text:
        return _clean_numeric_scalar(val)

    stripped = val.strip()
    if is_placeholder(stripped):
        return ""
    if "date" in type_text:
        return stripped if DATE_SCALAR.match(stripped) else None
    if "boolean" in type_text:
        return normalize_boolean(stripped)
    return stripped


def _field_type_instruction(field: str, field_type: str) -> str:
    type_text = field_type.lower()
    if "local_record_id" in type_text or "integer_scalar" in type_text:
        return (
            f"- {field}: {field_type}. Extract one scalar integer only. "
            'Convert labels like "档案 286" or "Record 286" to "286". '
            'Reject ranges/groups/lists such as "197至217"; if the paragraph only '
            "states missing values for such a range/group, skip the paragraph."
        )
    if "number" in type_text or "integer" in type_text:
        return (
            f"- {field}: {field_type}. Extract only an explicit numeric scalar that "
            "matches this exact field; remove unit suffixes."
        )
    if "date" in type_text:
        return (
            f"- {field}: {field_type}. Extract only an explicit reporting date/period; "
            "use the final corrected value when corrections are described."
        )
    if "time" in type_text:
        return f"- {field}: {field_type}. Extract only an explicit clock/timestamp value."
    if "id" in type_text or "identity" in type_text:
        return (
            f"- {field}: {field_type}. Extract one scalar identity value only; "
            "do not use ranges, grouped labels, or narrative descriptions."
        )
    return f"- {field}: {field_type}. Extract only explicit values of this type."


def _format_schema_type_hint(
    schema_columns: list[str] | None,
    schema_field_types: dict[str, str] | None,
) -> str:
    if not schema_columns or not schema_field_types:
        return ""
    lines = [
        _field_type_instruction(field, schema_field_types[field])
        for field in schema_columns
        if field in schema_field_types
    ]
    if not lines:
        return ""
    return (
        "\nField type constraints (obey strictly; omit values that violate the type):\n"
        + "\n".join(lines)
        + "\n"
    )


def _format_hard_type_rules(
    schema_columns: list[str] | None,
    schema_field_types: dict[str, str] | None,
) -> str:
    if not schema_columns or not schema_field_types:
        return ""

    number_fields = [
        field
        for field in schema_columns
        if "number" in schema_field_types.get(field, "").lower()
        or "integer" in schema_field_types.get(field, "").lower()
    ]
    date_fields = [
        field for field in schema_columns if "date" in schema_field_types.get(field, "").lower()
    ]
    boolean_fields = [
        field for field in schema_columns if "boolean" in schema_field_types.get(field, "").lower()
    ]
    if not number_fields and not date_fields and not boolean_fields:
        return ""

    lines = [
        "HARD TYPE RULES:",
        "- If a value has the wrong type for a field, leave the value blank.",
        "- Do NOT put names, descriptions, or labels into number/date/boolean fields.",
    ]
    if number_fields:
        lines.append(
            "- number/integer fields: numeric values only (integers or decimals); "
            "strip unit suffixes and thousand separators. "
            f"Fields: {', '.join(number_fields)}."
        )
    if date_fields:
        lines.append(
            "- date fields: convert to YYYY-MM-DD (or YYYY-MM / YYYY if day/month unknown); "
            "leave blank if not a recognizable date. "
            f"Fields: {', '.join(date_fields)}."
        )
    if boolean_fields:
        lines.append(
            "- boolean fields: normalize to true or false; "
            "leave blank if the value is not boolean. "
            f"Fields: {', '.join(boolean_fields)}."
        )
    return "\n" + "\n".join(lines) + "\n"


def _clean_compressed_entity_lines(
    text: str,
    schema_columns: list[str] | None = None,
    schema_field_types: dict[str, str] | None = None,
) -> str:
    """Clean schema-specific mistakes while still in the compression stage."""
    if not schema_columns:
        return text

    field_map = {field.lower(): field for field in schema_columns}
    field_types = schema_field_types or {}
    cleaned: list[str] = []
    dropped = 0
    normalized = 0

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            cleaned.append(line)
            continue

        parts: list[str] = []
        drop_line = False
        changed = False
        for part in stripped.split("|"):
            if ":" not in part:
                parts.append(part.strip())
                continue
            key, _, val = part.partition(":")
            key = field_map.get(key.strip().lower(), key.strip())
            val = val.strip()
            field_type = field_types.get(key, "").lower()
            if key.lower() in RECORD_ID_FIELDS or "local_record_id" in field_type:
                if RECORD_ID_RANGE.search(val):
                    drop_line = True
                    break
                if m := RECORD_ID_LABEL.match(val):
                    val = m.group(1)
                    changed = True
            else:
                typed_val = _clean_typed_scalar(val, field_type)
                if typed_val is None:
                    val = ""
                    changed = True
                elif typed_val != val:
                    val = typed_val
                    changed = True
            parts.append(f"{key}: {val}")

        if drop_line:
            dropped += 1
            continue
        if changed:
            normalized += 1
        cleaned.append(" | ".join(parts))

    if dropped:
        logger.info("ETL compress-clean: dropped %d grouped/range record-id lines", dropped)
    if normalized:
        logger.info("ETL compress-clean: normalized %d local record labels", normalized)
    return "\n".join(cleaned)


def sample_paragraphs(text: str, first_n: int = 3, stride_n: int = 10) -> str:
    """Sample paragraphs: first few + evenly spaced from the rest of the file.

    This ensures the LLM sees all paragraph types (identity, physical stats,
    demographics, classification, etc.) rather than only the first section.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) <= first_n + stride_n:
        return "\n\n".join(paras)

    samples = paras[:first_n]
    remaining = paras[first_n:]
    step = max(1, len(remaining) // stride_n)
    for i in range(0, len(remaining), step):
        samples.append(remaining[i])
        if len(samples) >= first_n + stride_n:
            break

    return "\n\n".join(samples)


def sample_paragraphs_by_section(text: str, per_section: int = 1) -> str:
    """Sample data-bearing paragraphs for field discovery.

    For multi-section files (>=3 sections), picks the longest paragraph from
    each section so every field subset is represented.  For single/few-section
    files, falls back to stride sampling for broader coverage.
    """
    sections = detect_sections(text)
    if not sections or len(sections) < 3:
        return sample_paragraphs(text, first_n=3, stride_n=5)

    samples: list[str] = []
    for sec in sections:
        samples.append(f"### {sec['title']}")
        paras = [
            p.strip() for p in sec["content"].split("\n\n") if p.strip() and len(p.strip()) > 50
        ]
        if not paras:
            continue
        paras.sort(key=len, reverse=True)
        for p in paras[:per_section]:
            samples.append(p[:1000])
    return "\n\n".join(samples)


def compress_chunk(
    adapter: ModelAdapter,
    chunk_text: str,
    schema_columns: list[str] | None = None,
    primary_key: str | None = None,
    anchor_keys: list[str] | None = None,
    schema_field_types: dict[str, str] | None = None,
) -> str:
    """Ask LLM to extract labeled key-value pairs from prose paragraphs."""
    if schema_columns and primary_key:
        cols = ", ".join(schema_columns)
        schema_hint = (
            f"\nUse EXACTLY these field names: {cols}\n"
            "Map a value to a field only when it explicitly matches that field's "
            "meaning. Do NOT force narrative text into the closest field, and "
            "do NOT invent synonyms or alternative names.\n"
        )
        type_hint = _format_schema_type_hint(schema_columns, schema_field_types)
        hard_type_rules = _format_hard_type_rules(schema_columns, schema_field_types)
        format_hint = (
            f"Format — one line per entity-paragraph, pipe-separated key-value pairs:\n"
            f"  {primary_key}: <value> | field_a: <value> | field_b: <value> | ...\n"
            "ONLY include fields that have actual values in the paragraph. "
            "OMIT non-identity fields with no suitable value; downstream "
            "normalization will fill them as empty cells.\n"
        )
        if anchor_keys and len(anchor_keys) > 1:
            anchor_names = ", ".join(anchor_keys)
            anchor_rule = (
                f"- Start each line with these identity fields: {anchor_names}. "
                "Include ALL of them when mentioned in the paragraph. "
                "If a field's value is not stated in the paragraph, include its "
                "key with an empty value."
            )
        else:
            anchor_rule = (
                f"- Start each line with {primary_key} as stated in the paragraph. "
                "This field anchors every line."
            )
    else:
        schema_hint = ""
        type_hint = ""
        hard_type_rules = ""
        format_hint = (
            "Format — one line per entity-paragraph:\n"
            "  ID: <number> | entity_name: <name> | <field>: <value> | ...\n"
        )
        anchor_rule = (
            "- Start each line with the entity's numeric ID AND its name/codename/label "
            "as stated in the paragraph. These two fields anchor every line."
        )

    id_field_label = primary_key or "ID"

    sparse = bool(schema_columns and len(schema_columns) > 10)
    if sparse:
        placeholder_rule = (
            "- If a field is not mentioned or its value is None/NaN/placeholder/redacted, "
            "SKIP IT entirely — do NOT include it in the output line."
        )
    else:
        placeholder_rule = (
            "- If a field's ENTIRE value is 0.0, None, NaN, placeholder, or "
            'redacted (with no real data), output it with a "(placeholder)" marker — '
            'e.g. "height_cm: 0.0 (placeholder)". Do NOT mark corrected values as '
            'placeholder — "initially X, corrected to Y" means Y is the real value.'
        )

    prompt = textwrap.dedent(f"""\
        Below are paragraphs from a data document. Each paragraph describes one entity \
        and mixes data with narrative filler.

        Task: for each paragraph, extract data as labeled key-value pairs.
        {schema_hint}
        {type_hint}
        {hard_type_rules}
        {format_hint}
        Rules:
        {anchor_rule}
        - Every value MUST have a descriptive field label before it.
        - If a schema field has no explicit, suitable value in this paragraph, \
        leave it empty by omitting that non-identity field. Do NOT substitute \
        background descriptions, business strengths, culture, plans, or prose \
        summaries just because they are loosely related to the field name.
        - If a paragraph only says values are missing/not provided/unavailable \
        for an entity or a range/group of entities, SKIP that paragraph entirely. \
        Do NOT output a row that contains only identity fields and empty data fields.
        - Field-name qualifiers are binding. Do NOT map an overall/global \
        total, count, rank, or average into a qualified/category-specific \
        field unless the paragraph explicitly states the same qualifier for \
        that value; omit the field instead.
        - Local document labels such as "档案 353", "记录 353", "条目 353", \
        "战略单元 353", "Record 353", "File 353", or "Case 353" are \
        record identifiers. Put only the scalar numeric portion (for example, \
        "353") in the {id_field_label} field. \
        NEVER output ranges like "197至217" or grouped labels as a single \
        {id_field_label} value, and never put a local record label into domain ID \
        fields such as personalcode, innercode, fundcode, or secucode unless \
        the paragraph explicitly labels it as that domain ID.
        - personalcode/PersonalCode fields are person identifiers. Do NOT \
        fill them with a manager name or with a local document record number; \
        leave personalcode empty unless an explicit personal/manager code is \
        stated.
        - Output numeric values WITHOUT unit suffixes (e.g. 193.0, not 193.0 cm).
        {placeholder_rule}
        - For corrections: output the FINAL corrected value directly — e.g. \
        "label: positive", not "label: positive (placeholder)".
        - Preserve URLs, names, dates, times, categories, classifications.
        - Strip only narrative filler that contains no structured field, schema field, \
        ID, value, date, unit, label, status, or question-relevant fact.
        - Separate entries with ONE blank line. No commentary, no markdown.

        ---
        {chunk_text}
        ---""")
    from agents.llm.types import ModelMessage

    messages = [
        ModelMessage(role="system", content="You are a data extraction preprocessor."),
        ModelMessage(role="user", content=prompt),
    ]
    return adapter.complete(messages).content.strip()


def compress_prose(
    adapter: ModelAdapter,
    text: str,
    schema_columns: list[str] | None = None,
    primary_key: str | None = None,
    anchor_keys: list[str] | None = None,
    schema_field_types: dict[str, str] | None = None,
) -> str:
    """Compress prose by stripping noise. All chunks across all sections run in one pool."""
    sections = detect_sections(text)

    work: list[tuple[str | None, str]] = []
    if not sections:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i in range(0, len(paras), COMPRESS_CHUNK_SIZE):
            work.append((None, "\n\n".join(paras[i : i + COMPRESS_CHUNK_SIZE])))
    else:
        for sec in sections:
            paras = [
                p.strip() for p in sec["content"].split("\n\n") if p.strip() and len(p.strip()) > 30
            ]
            first = True
            for i in range(0, len(paras), COMPRESS_CHUNK_SIZE):
                title = sec["title"] if first else None
                work.append((title, "\n\n".join(paras[i : i + COMPRESS_CHUNK_SIZE])))
                first = False

    if not work:
        return text

    compressed: list[str | None] = [None] * len(work)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {
            pool.submit(
                compress_chunk,
                adapter,
                item[1],
                schema_columns,
                primary_key,
                anchor_keys,
                schema_field_types,
            ): idx
            for idx, item in enumerate(work)
        }
        try:
            for fut in as_completed(futs, timeout=LLM_CALL_TIMEOUT * len(work)):
                idx = futs[fut]
                try:
                    compressed[idx] = fut.result(timeout=LLM_CALL_TIMEOUT)
                except TimeoutError:
                    logger.warning("Compress chunk %d timed out", idx)
                except Exception as exc:
                    logger.warning("Compress chunk %d failed: %s", idx, exc)
        except TimeoutError:
            logger.warning("Compress pool timed out, proceeding with partial results")

    parts: list[str] = []
    for idx, (title, _) in enumerate(work):
        if title:
            parts.append(f"### {title}")
        chunk_result = compressed[idx]
        if chunk_result:
            parts.append(chunk_result)
    return _clean_compressed_entity_lines(
        "\n\n".join(parts),
        schema_columns,
        schema_field_types,
    )


def extract_csv_from_text(
    adapter: ModelAdapter,
    text: str,
    schema_columns: list[str] | None,
    question: str,
    schema_field_types: dict[str, str] | None = None,
) -> str | None:
    """Single LLM call: text → CSV.  Used as fallback when no schema is available."""
    schema_line = (
        f"\nColumns: {','.join(schema_columns)}\n"
        if schema_columns
        else "\nInfer column names from the data. First column must be the entity's "
        "numeric ID named 'id'.\n"
    )
    hard_type_rules = _format_hard_type_rules(schema_columns, schema_field_types)

    prompt = textwrap.dedent(f"""\
        Below is a compressed data document. Each line describes one entity \
        as pipe-separated key-value pairs (e.g. "key: value | key: value").

        Output a CSV with one row per entity.
        {schema_line}
        {hard_type_rules}
        Rules:
        - Output ONLY valid CSV. First line is the header.
        - Placeholder/None/NaN/redacted values → leave the cell empty.
        - Strip unit suffixes (cm, kg, etc.) from numeric fields.
        - For genuinely missing values: leave empty.
        - No markdown fences, no explanation.

        Question context: {question}

        Document:
        {text}

        Output CSV now:""")
    from agents.llm.types import ModelMessage

    messages = [
        ModelMessage(role="system", content="You are a CSV data extractor. Output only CSV."),
        ModelMessage(role="user", content=prompt),
    ]
    csv_text = adapter.complete(messages).content.strip()
    if csv_text.startswith("```"):
        csv_text = csv_text.split("\n", 1)[1]
        csv_text = csv_text.rsplit("```", 1)[0].strip()
    try:
        reader = csv.reader(io.StringIO(csv_text))
        header = next(reader, None)
        if not header or len(header) < 2:
            return None
        ncols = len(header)
        clean_rows = [header]
        for row in reader:
            if len(row) > ncols:
                row = row[:ncols]
            clean_rows.append(row)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerows(clean_rows)
        return buf.getvalue()
    except csv.Error:
        return None


def split_entity_chunks(
    text: str,
    chunk_size: int = EXTRACT_CHUNK_LINES,
    primary_key: str | None = None,
) -> list[str]:
    """Split compressed text into chunks of ~chunk_size entity lines each."""
    pk_token = f"{primary_key}:" if primary_key else None
    entity_lines: list[str] = []
    non_entity_header: list[str] = []
    for line in text.splitlines():
        is_entity = (pk_token and pk_token in line) if pk_token else ENTITY_LINE.match(line)
        if is_entity:
            entity_lines.append(line)
        elif not entity_lines:
            non_entity_header.append(line)

    if len(entity_lines) <= chunk_size:
        return [text]

    header = "\n".join(non_entity_header).strip()
    chunks: list[str] = []
    for i in range(0, len(entity_lines), chunk_size):
        batch = "\n".join(entity_lines[i : i + chunk_size])
        chunks.append(f"{header}\n\n{batch}" if header else batch)
    return chunks


def direct_extract_csv(
    adapter: ModelAdapter,
    compressed_text: str,
    schema_columns: list[str] | None,
    question: str,
    primary_key: str | None = None,
    schema_field_types: dict[str, str] | None = None,
) -> str | None:
    """Extract CSV from compressed text. Splits into entity-line chunks for parallel extraction."""
    chunks = split_entity_chunks(compressed_text, primary_key=primary_key)

    if len(chunks) == 1:
        return extract_csv_from_text(
            adapter,
            compressed_text,
            schema_columns,
            question,
            schema_field_types,
        )

    import pandas as pd

    logger.info("ETL extract: splitting into %d chunks for parallel extraction", len(chunks))
    partial: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=min(len(chunks), MAX_WORKERS)) as pool:
        futs = {
            pool.submit(
                extract_csv_from_text,
                adapter,
                chunk,
                schema_columns,
                question,
                schema_field_types,
            ): idx
            for idx, chunk in enumerate(chunks)
        }
        try:
            for fut in as_completed(futs, timeout=LLM_CALL_TIMEOUT * len(chunks)):
                idx = futs[fut]
                try:
                    result = fut.result(timeout=LLM_CALL_TIMEOUT)
                    if result:
                        partial[idx] = result
                except TimeoutError:
                    logger.warning("Extract chunk %d timed out", idx)
                except Exception as exc:
                    logger.warning("Extract chunk %d failed: %s", idx, exc)
        except TimeoutError:
            logger.warning("Extract pool timed out, proceeding with partial results")

    if not partial:
        return None

    dfs: list[pd.DataFrame] = []
    for idx in sorted(partial):
        try:
            df = pd.read_csv(io.StringIO(partial[idx]))
            if len(df) == 0:
                continue
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return None

    merged = pd.concat(dfs, ignore_index=True)
    dedup_col = (
        primary_key
        if primary_key and primary_key in merged.columns
        else next((c for c in ("entity_name", "id") if c in merged.columns), None)
    )
    if dedup_col:
        merged = merged.groupby(dedup_col, sort=False).first().reset_index()
    buf = io.StringIO()
    merged.to_csv(buf, index=False)
    return buf.getvalue()
