"""Value-level retry, cross-field dedup, and outlier verification."""

from __future__ import annotations

import logging
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from agents.etl._constants import (
    DEDUP_CONTEXT_WINDOW,
    ENTITY_LINE,
    FILL_RATE_THRESHOLD,
    LLM_CALL_TIMEOUT,
    MAX_RETRY_ENTITIES,
    MAX_VERIFY_ENTITIES,
    MAX_WORKERS,
)

if TYPE_CHECKING:
    from agents.llm.types import ModelAdapter

logger = logging.getLogger(__name__)


def _parse_kv_lines(
    text: str,
    schema_columns: list[str],
    primary_key: str | None,
) -> list[dict[str, str]]:
    """Parse normalized KV text into list of {col: val} dicts."""
    from agents.etl._merge import (
        canonical_key_map,
        canonicalize_key,
        line_has_kv_key,
    )

    field_map = canonical_key_map(schema_columns)
    col_set = set(schema_columns)
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_entity = (
            line_has_kv_key(stripped, primary_key) if primary_key else ENTITY_LINE.match(stripped)
        )
        if not is_entity:
            continue
        fields: dict[str, str] = {}
        for part in stripped.split("|"):
            if ":" not in part:
                continue
            key, _, val = part.partition(":")
            key = canonicalize_key(key, field_map)
            if key in col_set:
                fields[key] = val.strip()
        rows.append(fields)
    return rows


def _detect_gap_cells(
    rows: list[dict[str, str]],
    schema_columns: list[str],
    primary_key: str | None,
    anchor_keys: list[str] | None,
) -> dict[str, list[str]]:
    """Return {record_id: [missing_col, ...]} for high-fill columns with gaps."""
    if not rows or not primary_key:
        return {}

    skip = set(anchor_keys or [])
    candidate_cols = [c for c in schema_columns if c not in skip]
    if not candidate_cols:
        return {}

    n = len(rows)
    fill_counts: dict[str, int] = dict.fromkeys(candidate_cols, 0)
    for row in rows:
        for c in candidate_cols:
            if row.get(c, "").strip():
                fill_counts[c] += 1

    gap_cols = [
        c
        for c in candidate_cols
        if fill_counts[c] / n >= FILL_RATE_THRESHOLD and fill_counts[c] < n
    ]
    if not gap_cols:
        return {}

    gaps: dict[str, list[str]] = {}
    for row in rows:
        pk_val = row.get(primary_key, "").strip()
        if not pk_val:
            continue
        missing = [c for c in gap_cols if not row.get(c, "").strip()]
        if missing:
            gaps[pk_val] = missing
    return gaps


def _value_search_needles(value_str: str) -> list[str]:
    """Return plain and comma-separated forms of a numeric value string."""
    plain = value_str.replace(",", "")
    if not plain:
        return []
    int_part = plain.split(".")[0].lstrip("-")
    comma_form = ""
    if int_part.isdigit() and len(int_part) > 3:
        chunks: list[str] = []
        while int_part:
            chunks.append(int_part[-3:])
            int_part = int_part[:-3]
        comma_form = ",".join(reversed(chunks))
    needles = [plain]
    if comma_form and comma_form != plain:
        needles.append(comma_form)
    return needles


def _extract_value_context(value_str: str, source_text: str) -> str:
    """Extract the first source context snippet surrounding *value_str*."""
    for needle in _value_search_needles(value_str):
        m = re.search(re.escape(needle), source_text)
        if m:
            start = max(0, m.start() - DEDUP_CONTEXT_WINDOW)
            end = min(len(source_text), m.end() + DEDUP_CONTEXT_WINDOW)
            return source_text[start:end]
    return ""


def dedup_cross_field_copies(
    adapter: ModelAdapter,
    text: str,
    schema_columns: list[str],
    primary_key: str | None,
    anchor_keys: list[str] | None,
    source_text: str | None,
    schema_field_types: dict[str, str] | None = None,
) -> tuple[str, dict[str, list[str]]]:
    """Clear values duplicated across distinct fields by the same entity.

    When compress puts the same numeric value into two columns, one is
    almost certainly wrong (e.g. reserveassets value copied into
    totalassets).  Uses one LLM call to resolve which column each
    duplicate value actually belongs to, then clears the wrong column
    so retry can re-extract it.

    Returns (text, cleared) where cleared maps pk_val → [col, ...].
    """
    if not primary_key or not source_text:
        return text, {}

    rows = _parse_kv_lines(text, schema_columns, primary_key)
    if not rows:
        return text, {}

    skip = set(anchor_keys or [])
    data_cols = [c for c in schema_columns if c not in skip]

    dupes: list[tuple[str, str, list[str], str]] = []
    for row in rows:
        pk_val = row.get(primary_key, "").strip()
        if not pk_val:
            continue

        val_to_cols: dict[str, list[str]] = {}
        for c in data_cols:
            v = row.get(c, "").strip().replace(",", "")
            if not v:
                continue
            try:
                float(v)
            except ValueError:
                continue
            val_to_cols.setdefault(v, []).append(c)

        for val, cols in val_to_cols.items():
            if len(cols) < 2:
                continue
            ctx = _extract_value_context(val, source_text)
            dupes.append((pk_val, val, cols, ctx))

    if not dupes:
        return text, {}

    from agents.llm.types import ModelMessage

    items: list[str] = []
    for i, (pk_val, val, cols, ctx) in enumerate(dupes):
        items.append(
            f"{i + 1}. {primary_key}={pk_val}, value={val}, "
            f"columns={cols}\n   Source context: ...{ctx}..."
        )

    prompt = textwrap.dedent(f"""\
        The same numeric value appears in multiple columns for each entity
        below. Based on the source context, determine which ONE column
        the value actually belongs to. The other column(s) received it
        by mistake.

        {chr(10).join(items)}

        For each item output EXACTLY one line:
        <item_number>:<column_to_keep>

        Output nothing else.""")

    messages: list[ModelMessage] = [
        ModelMessage(
            role="system",
            content="Resolve cross-field value duplicates. "
            "Output item_number:column_to_keep lines only.",
        ),
        ModelMessage(role="user", content=prompt),
    ]

    try:
        response = adapter.complete(messages)
    except Exception as exc:
        logger.warning("ETL dedup LLM resolution failed: %s", exc)
        return text, {}

    raw = response.content.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    to_clear: dict[str, list[str]] = {}
    for line in raw.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        idx_str, _, keep_col = line.partition(":")
        try:
            idx = int(idx_str.strip()) - 1
        except ValueError:
            continue
        if idx < 0 or idx >= len(dupes):
            continue
        keep_col = keep_col.strip().lower()
        pk_val, _val, cols, _ctx = dupes[idx]
        matched = [c for c in cols if c.lower() == keep_col]
        if len(matched) != 1:
            continue
        for c in cols:
            if c != matched[0]:
                to_clear.setdefault(pk_val, []).append(c)

    if not to_clear:
        return text, {}

    cleared = 0
    for pk_val, cols in to_clear.items():
        for col in cols:
            pk_pat = re.compile(
                rf"^(.*?\b{re.escape(primary_key)}\s*:\s*{re.escape(pk_val)}\b.*)$",
                re.MULTILINE | re.IGNORECASE,
            )

            def _clear(m: re.Match[str], _col: str = col) -> str:
                line = m.group(1)
                field_pat = re.compile(
                    rf"(\b{re.escape(_col)}\s*:)\s*[^|]+(\||$)",
                    re.IGNORECASE,
                )
                return field_pat.sub(r"\1 \2", line, count=1)

            new_text, n = pk_pat.subn(_clear, text)
            if n:
                text = new_text
                cleared += 1

    if cleared:
        logger.info(
            "ETL dedup: cleared %d cross-field copy values across %d entities",
            cleared,
            len(to_clear),
        )
    return text, to_clear


def retry_missing_values(
    adapter: ModelAdapter,
    text: str,
    schema_columns: list[str],
    primary_key: str | None,
    anchor_keys: list[str] | None,
    entity_groups: dict[str, list[str]] | None,
    schema_field_types: dict[str, str] | None = None,
    schema_field_defs: dict[str, str] | None = None,
    forced_gaps: dict[str, list[str]] | None = None,
) -> str:
    """Re-extract missing values for high-fill columns from source paragraphs."""
    if not primary_key or not entity_groups or not schema_columns:
        return text

    rows = _parse_kv_lines(text, schema_columns, primary_key)
    gaps = _detect_gap_cells(rows, schema_columns, primary_key, anchor_keys)
    if forced_gaps:
        for pk_val, cols in forced_gaps.items():
            existing = gaps.get(pk_val, [])
            gaps[pk_val] = sorted(set(existing + cols))
    if not gaps:
        return text

    retry_ids = sorted(gaps.keys())[:MAX_RETRY_ENTITIES]
    all_missing_cols = sorted({c for cols in gaps.values() for c in cols})

    logger.info(
        "ETL value retry: %d entities missing values in columns %s",
        len(retry_ids),
        all_missing_cols,
    )

    patched = 0
    for rid in retry_ids:
        source_paras = entity_groups.get(rid)
        if not source_paras:
            continue
        missing_cols = gaps[rid]
        missing_col_by_lower = {c.lower(): c for c in missing_cols}
        source_text = "\n\n".join(source_paras)

        col_list = ", ".join(missing_cols)
        defs_hint = ""
        if schema_field_defs:
            parts: list[str] = []
            for c in missing_cols:
                d = schema_field_defs.get(c, "")
                if d:
                    parts.append(f"  {c}: {d}")
            if parts:
                defs_hint = "\nField definitions:\n" + "\n".join(parts) + "\n"

        prompt = (
            f"Extract ONLY these fields from the text below:\n"
            f"Fields: {col_list}\n"
            f"{defs_hint}"
            f"Output format — one line, pipe-separated:\n"
            f"  {' | '.join(f'{c}: <value>' for c in missing_cols)}\n"
            f"If a field's value is not in the text, leave it empty: {missing_cols[0]}: \n"
            f"Output ONLY the single key-value line, nothing else.\n\n"
            f"TEXT:\n{source_text}"
        )

        from agents.llm.types import ModelMessage as _MM

        messages = [_MM(role="user", content=prompt)]
        try:
            response = adapter.complete(messages)
        except Exception:
            logger.debug("ETL value retry: LLM call failed for %s", rid, exc_info=True)
            continue

        reply = response.content.strip()
        if reply.startswith("```"):
            reply = reply.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        extracted: dict[str, str] = {}
        for part in reply.split("|"):
            if ":" not in part:
                continue
            key, _, val = part.partition(":")
            key = missing_col_by_lower.get(key.strip().lower())
            val = val.strip()
            if key and val:
                extracted[key] = val

        if not extracted:
            continue

        pk_pattern = re.compile(
            rf"^(.*?\b{re.escape(primary_key)}\s*:\s*{re.escape(rid)}\b.*)$",
            re.MULTILINE | re.IGNORECASE,
        )

        def _patch_line(m: re.Match[str], _extracted: dict[str, str] = extracted) -> str:
            line = m.group(1)
            changed = False
            for col, val in _extracted.items():
                empty_pat = re.compile(
                    rf"(\b{re.escape(col)}\s*:)\s*(\||$)",
                    re.IGNORECASE,
                )
                new_line, n = empty_pat.subn(rf"\1 {val} \2", line, count=1)
                if n:
                    line = new_line
                    changed = True
            if changed:
                return line
            return m.group(1)

        new_text, n = pk_pattern.subn(_patch_line, text)
        if n:
            text = new_text
            patched += len(extracted)
            logger.info(
                "ETL value retry: patched %d fields for entity %s: %s",
                len(extracted),
                rid,
                list(extracted.keys()),
            )

    if patched:
        logger.info(
            "ETL value retry: total %d values patched across %d entities", patched, len(retry_ids)
        )
    return text


def _sample_verify_entities(
    rows: list[dict[str, str]],
    primary_key: str,
    entity_groups: dict[str, list[str]],
) -> list[str]:
    """Select a spread of entity IDs for verification, prioritizing entities
    with more source paragraphs (higher chance of cross-paragraph confusion)."""
    candidates: list[tuple[str, int]] = []
    for row in rows:
        pk = row.get(primary_key, "").strip()
        if pk and pk in entity_groups:
            candidates.append((pk, len(entity_groups[pk])))
    candidates.sort(key=lambda x: -x[1])
    return [pk for pk, _ in candidates[:MAX_VERIFY_ENTITIES]]


def verify_field_values(
    adapter: ModelAdapter,
    text: str,
    schema_columns: list[str],
    primary_key: str | None,
    anchor_keys: list[str] | None,
    entity_groups: dict[str, list[str]] | None,
    schema_field_types: dict[str, str] | None = None,
    schema_field_defs: dict[str, str] | None = None,
) -> str:
    """Verify extracted field values against source paragraphs using LLM."""
    if not primary_key or not entity_groups or not schema_columns:
        return text

    rows = _parse_kv_lines(text, schema_columns, primary_key)
    if not rows:
        return text

    verify_ids = _sample_verify_entities(rows, primary_key, entity_groups)
    if not verify_ids:
        return text

    skip = set(anchor_keys or [])
    types = schema_field_types or {}
    verify_cols = [
        c for c in schema_columns if c not in skip and "date" not in types.get(c, "").lower()
    ]
    if not verify_cols:
        return text

    logger.info(
        "ETL field verify: checking %d entities (%d numeric fields each)",
        len(verify_ids),
        len(verify_cols),
    )

    from agents.llm.types import ModelMessage as _MM

    col_list = ", ".join(verify_cols)
    defs_block = ""
    if schema_field_defs:
        parts = [f"  {c}: {schema_field_defs[c]}" for c in verify_cols if c in schema_field_defs]
        if parts:
            defs_block = "\nField definitions:\n" + "\n".join(parts) + "\n"

    def _verify_one(rid: str) -> tuple[str, dict[str, str]]:
        source_paras = entity_groups.get(rid)
        if not source_paras:
            return rid, {}
        row = next((r for r in rows if r.get(primary_key, "").strip() == rid), None)
        if not row:
            return rid, {}

        current_kv = " | ".join(f"{c}: {row.get(c, '')}" for c in verify_cols)
        source_text = "\n\n".join(source_paras)

        prompt = (
            f"Verify each field value against the source text.\n"
            f"Fields to verify: {col_list}\n"
            f"{defs_block}\n"
            f"CURRENT EXTRACTION:\n{current_kv}\n\n"
            f"SOURCE TEXT:\n{source_text}\n\n"
            f"IMPORTANT: Match each value to the EXACT field label in the "
            f"source text. A value labeled 'Reserve Assets' is NOT the same "
            f"as 'Total Assets'. A sub-total or component value must NOT be "
            f"assigned to an aggregate/total field.\n\n"
            f"For each field whose extracted value is WRONG, output:\n"
            f"  field_name: correct_value\n"
            f"If the source text does not mention a field at all, output:\n"
            f"  field_name: NONE\n"
            f"Output ONLY the corrected fields (one per line). "
            f"If all values are correct, output: ALL_CORRECT"
        )

        try:
            response = adapter.complete([_MM(role="user", content=prompt)])
        except Exception:
            logger.debug("ETL field verify failed for %s", rid, exc_info=True)
            return rid, {}

        reply = response.content.strip()
        if reply.startswith("```"):
            reply = reply.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        if "ALL_CORRECT" in reply.upper():
            return rid, {}

        corrections: dict[str, str] = {}
        col_by_lower = {c.lower(): c for c in verify_cols}
        for line in reply.splitlines():
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = col_by_lower.get(key.strip().lower())
            val = val.strip().replace(",", "")
            if key and val:
                try:
                    float(val)
                    current = row.get(key, "").strip().replace(",", "")
                    if val != current:
                        corrections[key] = val
                except ValueError:
                    continue
        return rid, corrections

    all_corrections: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(_verify_one, rid): rid for rid in verify_ids}
        try:
            for fut in as_completed(futs, timeout=LLM_CALL_TIMEOUT * len(verify_ids)):
                try:
                    rid, corr = fut.result(timeout=LLM_CALL_TIMEOUT)
                    if corr:
                        all_corrections[rid] = corr
                except Exception:
                    pass
        except TimeoutError:
            logger.warning("ETL field verify: pool timed out")

    if not all_corrections:
        logger.info("ETL field verify: all values confirmed correct")
        return text

    patched = 0
    for rid, corrections in all_corrections.items():
        pk_pattern = re.compile(
            rf"^(.*?\b{re.escape(primary_key)}\s*:\s*{re.escape(rid)}\b.*)$",
            re.MULTILINE | re.IGNORECASE,
        )

        def _patch_line(m: re.Match[str], _corr: dict[str, str] = corrections) -> str:
            line = m.group(1)
            for col, val in _corr.items():
                pat = re.compile(
                    rf"(\b{re.escape(col)}\s*:)\s*[^|]*(\||$)",
                    re.IGNORECASE,
                )
                new_line, n = pat.subn(rf"\1 {val} \2", line, count=1)
                if n:
                    line = new_line
            return line

        new_text, n = pk_pattern.subn(_patch_line, text)
        if n:
            text = new_text
            patched += len(corrections)
            logger.info(
                "ETL field verify: corrected entity %s: %s",
                rid,
                dict(corrections),
            )

    logger.info(
        "ETL field verify: corrected %d values across %d entities",
        patched,
        len(all_corrections),
    )
    return text
