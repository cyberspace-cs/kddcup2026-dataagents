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
ENTITY_CHUNK_TOKEN_BUDGET = 2500

RECORD_ID_INLINE = re.compile(
    rf"(?:{RECORD_ID_KEYWORDS})\s*[：:#\-]?\s*(\d{{1,8}})",
    re.IGNORECASE,
)

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


def _normalize_kv_separators(line: str) -> str:
    """Rewrite ``key=value`` parts to ``key: value`` so downstream code sees ``:`` only."""
    if "=" not in line:
        return line
    parts: list[str] = []
    for part in line.split("|"):
        stripped = part.strip()
        if "=" in stripped and ":" not in stripped:
            key, _, val = stripped.partition("=")
            parts.append(f"{key.strip()}: {val.strip()}")
        else:
            parts.append(stripped)
    return " | ".join(parts)


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
        stripped = _normalize_kv_separators(line.strip())
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


def _group_paragraphs_by_entity(
    text: str,
) -> dict[str, list[str]] | None:
    """Group paragraphs by record_id extracted via regex.

    Returns {record_id: [paragraph, ...]} preserving paragraph order within
    each entity, or None if the text lacks record_id patterns.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 30]
    if not paras:
        return None

    groups: dict[str, list[str]] = {}
    tagged = 0
    for para in paras:
        match = RECORD_ID_INLINE.search(para)
        if match:
            rid = match.group(1)
            groups.setdefault(rid, []).append(para)
            tagged += 1

    if tagged < 3 or not groups:
        return None
    return groups


def _entity_chunks_by_token_budget(
    entity_groups: dict[str, list[str]],
    token_budget: int = ENTITY_CHUNK_TOKEN_BUDGET,
) -> list[str]:
    """Pack complete entity contexts into chunks that fit a token budget.

    Each entity's paragraphs are joined and never split across chunks.
    Entities larger than the budget get their own chunk.
    """
    from agents.llm.tokenizer import count_qwen_tokens

    entity_texts: list[tuple[str, int]] = []
    for _rid, paras in entity_groups.items():
        full = "\n\n".join(paras)
        toks = count_qwen_tokens(full)
        entity_texts.append((full, toks))

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for full, toks in entity_texts:
        if toks >= token_budget:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0
            chunks.append(full)
            continue

        if current_tokens + toks > token_budget and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_tokens = 0

        current_parts.append(full)
        current_tokens += toks

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def _generate_compress_guide(
    adapter: ModelAdapter,
    prose_text: str,
    schema_columns: list[str] | None = None,
    schema_field_defs: dict[str, str] | None = None,
    schema_field_types: dict[str, str] | None = None,
) -> str | None:
    """Analyze document structure once, produce a reusable extraction system prompt."""
    if not schema_columns:
        return None

    paras = [p.strip() for p in prose_text.split("\n\n") if p.strip() and len(p.strip()) > 50]
    if not paras:
        return None
    if len(paras) <= 20:
        sample = "\n\n".join(paras)
    else:
        step = max(1, len(paras) // 20)
        sample = "\n\n".join(p[:600] for p in paras[::step][:20])

    defs_lines: list[str] = []
    for f in schema_columns:
        d = (schema_field_defs or {}).get(f, "")
        t = (schema_field_types or {}).get(f, "")
        parts = [f]
        if t:
            parts.append(f"type={t}")
        if d:
            parts.append(d)
        defs_lines.append("  - " + " | ".join(parts))
    defs_block = "\n".join(defs_lines)
    cols = ", ".join(schema_columns)

    prompt = textwrap.dedent(f"""\
        Below is a representative sample from a large prose data document.
        Each entity has a numeric record identifier (e.g. "档案 29", "记录 29").

        Target schema fields:
        {defs_block}

        DOCUMENT SAMPLE (paragraphs from evenly-spaced positions):
        ---
        {sample}
        ---

        Analyze this document thoroughly and produce a SYSTEM PROMPT that will
        guide another LLM to extract the correct field values from any
        paragraph of this document into pipe-separated key-value lines like:
          {cols}

        Your system prompt MUST cover:

        1. DOCUMENT LAYOUT: describe how the document is organized — are all
           fields for one record in a single paragraph, or does the same record
           appear across multiple paragraphs/sections with different fields
           each time?

        2. FIELD-TO-PROSE MAPPING: for EACH schema field, state the EXACT
           prose phrase template that provides its value. Quote the Chinese or
           English template with a placeholder X. Example:
             field_a ← "在本次交易前，该实体持有 X 股" → X
             field_b ← "此次股权变动涉及的股份数量为 X 股" → X

        3. CRITICAL DISAMBIGUATION: list pairs of fields that look similar
           in the prose and explain precisely how to tell them apart. This is
           the most important part — mistakes here cause wrong data.
           Pay special attention to percentage/ratio fields: a ratio stated
           alongside a QUANTITY (e.g. "涉及 X 股…占转让方持股的 Y") is NOT
           the same as a ratio stated alongside a HOLDING (e.g. "持有 X 股，
           占总股本的 Y%"). Map each ratio to the field whose SHARES COUNT
           appears in the same sentence.
           CRITICAL: when a record spans multiple paragraphs — one describing
           the transaction (transferred shares + ratio to total) and another
           describing the transferor's holding state (held shares + ratio to
           total) — explicitly instruct which paragraph supplies which field.
           "Before/after holding pct" comes ONLY from the holding-state
           paragraph (large share count, large pct ~1-50%), NEVER from the
           transaction paragraph (small transfer count, tiny ratio ~0.001-0.01%).

        4. FUZZY NUMBER RESOLUTION: if any values use approximate language
           ("约", "大约", "近", Chinese number words like "六百九十七万"),
           instruct the extractor to compute the exact value from other
           precise numbers in the same paragraph (e.g. pre = post + transfer).
           Always output computed precise values, never fuzzy approximations.

        Output ONLY the system prompt text (under 1200 words), no wrapping
        or commentary.""")
    from agents.llm.types import ModelMessage

    messages = [
        ModelMessage(
            role="system",
            content="You analyze document structures and write precise data extraction instructions.",
        ),
        ModelMessage(role="user", content=prompt),
    ]
    try:
        response = adapter.complete(messages)
        guide = response.content.strip()
        if len(guide) < 100:
            return None
        logger.info("ETL: generated compress guide (%d chars)", len(guide))
        return guide
    except Exception as exc:
        logger.warning("Failed to generate compress guide: %s", exc)
        return None


def _format_schema_def_hint(
    schema_columns: list[str] | None,
    schema_field_defs: dict[str, str] | None,
) -> str:
    if not schema_columns or not schema_field_defs:
        return ""
    lines = [
        f"  - {field}: {schema_field_defs[field]}"
        for field in schema_columns
        if field in schema_field_defs
    ]
    if not lines:
        return ""
    return (
        "\nField semantic definitions — match values strictly by these meanings, "
        "NOT by proximity or order of appearance in the text:\n" + "\n".join(lines) + "\n"
    )


def compress_chunk(
    adapter: ModelAdapter,
    chunk_text: str,
    schema_columns: list[str] | None = None,
    primary_key: str | None = None,
    anchor_keys: list[str] | None = None,
    schema_field_types: dict[str, str] | None = None,
    schema_field_defs: dict[str, str] | None = None,
    compress_guide: str | None = None,
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
            "Use COLON (:) to separate key from value. NEVER use = as separator.\n"
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
            "Use COLON (:) to separate key from value. NEVER use = as separator.\n"
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

    def_hint = _format_schema_def_hint(schema_columns, schema_field_defs)

    prompt = textwrap.dedent(f"""\
        Below are paragraphs from a data document. Each paragraph describes one entity \
        and mixes data with narrative filler.

        Task: for each paragraph, extract data as labeled key-value pairs.
        {schema_hint}
        {def_hint}
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
        - PERCENTAGES: the source text may express proportions in TWO forms: \
        percent form ("1.55%") or ratio form ("0.001910" meaning 0.191%). \
        Normalize ALL proportion/percentage values to PERCENT form: \
        "1.55%" → 1.55 (strip %); bare ratio like "0.001910" that \
        describes a proportion (e.g. "占总股本的 0.001910") → multiply \
        by 100 → 0.191. This ensures consistent units within each column.
        - NUMERIC CONSISTENCY (mandatory cross-check after extraction):
          (a) For paired shares+percentage fields representing the SAME state \
        (e.g. pre-transfer shares and pre-transfer pct), verify: \
        pct_value ≈ shares_value / total_capital. If your extracted pct is \
        orders of magnitude off from this ratio, you picked the WRONG number.
          (b) A "before" percentage must be LARGER than the corresponding \
        "after" percentage (transferor lost shares, so pct dropped).
          (c) When a paragraph contains MULTIPLE ratio/decimal values, the one \
        stated in the SAME sentence as the shares count (e.g. "持有 X 股，\
        占总股本的 Y%") is the matching percentage. Ratios stated alongside \
        the TRANSFER QUANTITY (e.g. "涉及 X 股…占转让方持股的 Y") belong \
        to a DIFFERENT field or should be discarded.
          (d) HOLDING vs TRANSACTION disambiguation — this is the #1 error \
        source: documents often describe one record across TWO sections: \
        (i) a transaction section stating the transferred quantity and its \
        ratio to total capital (e.g. "转让 30万股…占总股本 0.00191"); \
        (ii) a holding section stating the pre/post-transfer HOLDING and \
        its ratio to total capital (e.g. "持有 244万股，占总股本 1.55%"). \
        The "before/after holding pct" field MUST come from section (ii) — \
        the ratio stated in the same clause as the HOLDING shares count. \
        The ratio from section (i) describes what fraction of total capital \
        was TRANSFERRED, which is a different semantic and must NEVER fill \
        holding percentage fields. \
        Key signal: match the percentage to its CO-OCCURRING shares figure. \
        "持有 X 股，占总股本的 Y" → Y is the holding pct for X shares. \
        "涉及/转让 X 股…占总股本的 Y" → Y is the transfer-volume ratio, \
        discard for holding fields. \
        Cross-check: holding_pct ≈ holding_shares / total_capital. If the \
        ratio you picked doesn't satisfy this, you grabbed the wrong one.
          If any check fails, re-read the paragraph and pick the value that \
        satisfies all three constraints.
        - FUZZY NUMBERS: when a value is approximate (marked with "约", \
        "大约", "近", "roughly", Chinese number words like "六百九十七万"), \
        compute the exact value from other precise numbers in the SAME \
        paragraph if possible (e.g. pre = post + transferred, or \
        post = pre - transferred). Output the computed precise value, \
        not the fuzzy approximation.
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

    system_content = compress_guide or "You are a data extraction preprocessor."
    messages = [
        ModelMessage(role="system", content=system_content),
        ModelMessage(role="user", content=prompt),
    ]
    return adapter.complete(messages).content.strip()


def _positional_chunks(text: str) -> list[tuple[str | None, str]]:
    """Original positional chunking: split by section, then by paragraph count."""
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
    return work


def compress_prose(
    adapter: ModelAdapter,
    text: str,
    schema_columns: list[str] | None = None,
    primary_key: str | None = None,
    anchor_keys: list[str] | None = None,
    schema_field_types: dict[str, str] | None = None,
    schema_field_defs: dict[str, str] | None = None,
) -> str:
    """Compress prose by stripping noise. All chunks across all sections run in one pool."""

    # Try entity-aware chunking: group paragraphs by record_id so each
    # entity's full context is in one chunk, never split across chunks.
    entity_groups = _group_paragraphs_by_entity(text)
    if entity_groups:
        entity_chunks = _entity_chunks_by_token_budget(entity_groups)
        work: list[tuple[str | None, str]] = [(None, c) for c in entity_chunks]
        logger.info(
            "ETL compress: entity-aware chunking: %d entities → %d chunks",
            len(entity_groups),
            len(work),
        )
    else:
        work = _positional_chunks(text)

    if not work:
        return text

    guide = _generate_compress_guide(
        adapter,
        text,
        schema_columns,
        schema_field_defs,
        schema_field_types,
    )

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
                schema_field_defs,
                guide,
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
