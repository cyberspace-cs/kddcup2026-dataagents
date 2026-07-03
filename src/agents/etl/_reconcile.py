"""Synonym adjudication, field-name reconciliation, and malformed-line repair."""

from __future__ import annotations

import contextlib
import json
import logging
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.etl._columns import metadata_by_name
from agents.etl._constants import FIX_FORMAT_MAX_LINES, FIX_FORMAT_MAX_RETRIES
from agents.etl._merge import (
    apply_synonym_merges,
    cells_agree,
    column_fill_counts,
)
from agents.llm import ModelMessage, ModelResponse

if TYPE_CHECKING:
    from agents.llm import ModelAdapter

logger = logging.getLogger(__name__)


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
    canonicals: list[tuple[str, str, str]],
    discovered: list[tuple[str, str, str, list[str]]],
    question: str,
) -> dict[str, str]:
    """Ask the LLM which populated discovered columns are exact synonyms of
    incompletely-filled canonical columns.

    Both lists carry ``(name, definition, fill_note)`` — fill notes like
    "7/50 rows filled" let the adjudicator see the split signature. The LLM
    is trusted only for the semantic judgment; whether a merge actually
    happens is re-gated deterministically against the data
    (`apply_synonym_merges`).  Returns {discovered: canonical}.
    """
    canon_block = "\n".join(
        f"- {name} [{fill_note}]: {defn or '(no definition available)'}"
        for name, defn, fill_note in canonicals
    )
    disc_block = "\n".join(
        f"- {name} [{fill_note}]: {defn or '(no definition available)'} | samples: "
        f"{', '.join(samples) if samples else '(none)'}"
        for name, defn, fill_note, samples in discovered
    )
    prompt = textwrap.dedent(f"""\
        A table extracted from the prose file "{prose_stem}" shows a value-split
        signature: the CANONICAL columns below (defined by the data governance
        document) are missing values on rows where the DISCOVERED columns below
        (added during extraction) hold them — typically because extraction
        routed one concept's values under two names.

        For each DISCOVERED column, decide whether it denotes EXACTLY the same
        quantity as ONE of the canonical columns — same concept AND every
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

        CANONICAL COLUMNS WITH MISSING VALUES:
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
        [name for name, _defn, _note, _samples in discovered],
        [name for name, _defn, _note in canonicals],
    )


def fix_malformed_lines(
    adapter: ModelAdapter,
    compressed: str,
    schema_columns: list[str],
    schema_field_defs: dict[str, str] | None,
) -> str:
    """Detect and LLM-rewrite malformed lines, up to 3 retries.

    After compression the output should be ``key: val | key: val | ...``
    with keys from the schema.  Any line that looks like data but does
    not follow this format is sent back to the LLM for rewriting.
    """
    from agents.etl._merge import detect_malformed_lines

    defs = metadata_by_name(schema_columns, schema_field_defs)
    schema_block = ", ".join(schema_columns)
    defs_block = "\n".join(
        f"- {col}: {defs.get(col, '')}" for col in schema_columns if defs.get(col)
    )

    for attempt in range(1, FIX_FORMAT_MAX_RETRIES + 1):
        bad = detect_malformed_lines(compressed, schema_columns)
        if not bad:
            return compressed

        batch = bad[:FIX_FORMAT_MAX_LINES]
        logger.info(
            "ETL fix-format attempt %d: %d malformed lines detected",
            attempt,
            len(bad),
        )

        prompt = f"Target schema columns: {schema_block}\n"
        if defs_block:
            prompt += f"Column definitions:\n{defs_block}\n"
        prompt += (
            "\nThe lines below are MALFORMED — they lack proper key: value format. "
            "Rewrite EVERY line into pipe-separated KEY-VALUE pairs using ONLY "
            "the schema columns above.\n"
            "Required format: col1: val1 | col2: val2 | col3: val3 | ...\n"
            "Use COLON (:) between key and value. Use PIPE (|) between fields.\n"
            "NEVER drop, merge, or reorder lines. Output the SAME number of lines.\n\n"
            + "\n".join(batch)
        )

        messages = [
            ModelMessage(
                role="system",
                content="Rewrite malformed lines into key-value format. Output only rewritten lines.",
            ),
            ModelMessage(role="user", content=prompt),
        ]
        try:
            response: ModelResponse = adapter.complete(messages)
        except Exception as exc:
            logger.warning("ETL fix-format attempt %d failed: %s", attempt, exc)
            break

        content = response.content.strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        rewritten = [ln.strip() for ln in content.splitlines() if ln.strip() and ":" in ln]

        if not rewritten:
            break

        batch_set = set(batch)
        lines = compressed.splitlines()
        out: list[str] = []
        ri = 0
        for line in lines:
            if line.strip() in batch_set and ri < len(rewritten):
                out.append(rewritten[ri])
                ri += 1
            else:
                out.append(line)

        compressed = "\n".join(out)
        logger.info(
            "ETL fix-format attempt %d: rewrote %d/%d lines",
            attempt,
            ri,
            len(batch),
        )

    return compressed


def reconcile_field_names(
    adapter: ModelAdapter,
    compressed: str,
    schema_columns: list[str],
    schema_field_defs: dict[str, str] | None,
) -> str:
    """Map variant field names in compressed output back to schema columns.

    Compression can emit different field names for the same concept across
    chunks (e.g. ``patient_id`` vs ``uniquepid``, ``height_cm`` vs
    ``admissionheight_cm``). This function collects all unmatched keys,
    asks the LLM to map them to schema columns, and rewrites the compressed
    text with canonical names — before normalize_entity_lines drops them.
    """
    from agents.etl._merge import apply_field_name_map, collect_unmatched_keys

    unmatched = collect_unmatched_keys(compressed, schema_columns)
    if not unmatched:
        return compressed

    defs = metadata_by_name(schema_columns, schema_field_defs)
    schema_block = "\n".join(f"- {col}: {defs.get(col, '')}" for col in schema_columns)
    unmatched_block = "\n".join(f"- {k}" for k in unmatched)

    prompt = textwrap.dedent(f"""\
        An ETL pipeline extracted entity data into pipe-separated KV lines.
        The target schema columns are:
        {schema_block}

        The following field names appeared in the extraction output but do NOT
        match any schema column (case-insensitive):
        {unmatched_block}

        For each unmatched field, decide whether it is an EXACT synonym of one
        schema column — same concept, same entity scope, same semantics.
        - Match → output: <unmatched>=<schema_column>
        - No match or unsure → output: <unmatched>=NONE

        Output one line per unmatched field, nothing else.""")

    messages = [
        ModelMessage(
            role="system",
            content="Map extraction field names to schema columns. Output only mapping lines.",
        ),
        ModelMessage(role="user", content=prompt),
    ]
    try:
        response: ModelResponse = adapter.complete(messages)
    except Exception as exc:
        logger.warning("Field name reconciliation failed: %s", exc)
        return compressed

    content = response.content.strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    col_lower = {c.lower(): c for c in schema_columns}
    field_map: dict[str, str] = {}
    for line in content.splitlines():
        line = line.strip().lstrip("- ")
        if "=" not in line:
            continue
        src, _, dst = line.partition("=")
        src = src.strip()
        dst = dst.strip().lower()
        if dst == "none" or not dst:
            continue
        canon = col_lower.get(dst)
        if canon and src.lower() != canon.lower():
            field_map[src] = canon

    if not field_map:
        return compressed

    logger.info(
        "ETL field-name reconciliation: %d variant(s) mapped → %s",
        len(field_map),
        dict(list(field_map.items())[:10]),
    )
    return apply_field_name_map(compressed, field_map)


def _case_only_header_merges(
    header: list[str],
    preferred_columns: list[str],
) -> list[tuple[str, str]]:
    """Return deterministic merges for header names that differ only by case."""
    preferred = {c.lower(): c for c in preferred_columns}
    groups: dict[str, list[str]] = {}
    for col in header:
        groups.setdefault(col.lower(), []).append(col)

    merges: list[tuple[str, str]] = []
    for lower, cols in groups.items():
        unique_cols = list(dict.fromkeys(cols))
        if len(unique_cols) <= 1:
            continue
        preferred_col = preferred.get(lower)
        target = preferred_col if preferred_col in unique_cols else unique_cols[0]
        merges.extend((col, target) for col in unique_cols if col != target)
    return merges


def _unit_meta_get(units_meta: dict[str, Any], field: str) -> Any:
    lower = field.lower()
    for key, value in units_meta.items():
        if not key.startswith("_") and key.lower() == lower:
            return value
    return None


def _move_unit_meta(units_meta: dict[str, Any], src: str, dst: str) -> bool:
    changed = False
    for prefix in ("", "_target_", "_factor_"):
        src_lower = f"{prefix}{src}".lower()
        dst_key = f"{prefix}{dst}"
        src_key = next((key for key in list(units_meta) if key.lower() == src_lower), None)
        if src_key is None:
            continue
        if src_key.lower() == dst_key.lower():
            if src_key != dst_key and dst_key not in units_meta:
                units_meta[dst_key] = units_meta.pop(src_key)
                changed = True
            continue
        dst_existing = next((key for key in units_meta if key.lower() == dst_key.lower()), None)
        units_meta.setdefault(dst_existing or dst_key, units_meta[src_key])
        del units_meta[src_key]
        changed = True
    return changed


def unify_table_synonyms(
    adapter: ModelAdapter,
    header: list[str],
    rows: list[list[str]],
    prose_stem: str,
    governance_columns: list[str],
    anchor_keys: list[str],
    field_defs: dict[str, str],
    question: str,
    units_path: Path | None,
) -> tuple[list[str], list[list[str]], list[tuple[str, str]]]:
    """Detect and repair value-split columns in an in-memory table.

    Value split: the schema keeps a governance-canonical column AND an
    extraction-discovered column for the same concept; values land partly or
    wholly under the discovered name, leaving the canonical column gapped
    (and any unit conversion bound to the canonical name pointing at dead
    cells).

    Trigger: a canonical column with missing cells PLUS at least one
    structurally mergeable discovered column — zero row-level disagreements
    AND either new cells to contribute or full overlap-agreement (a
    redundant duplicate). Only then is one adjudication LLM call spent; the
    merge itself is deterministic (`apply_synonym_merges`): identity columns
    are protected and any row-level disagreement vetoes the pair. On merge,
    unit-sidecar metadata follows the values to the canonical name.

    Returns ``(header, rows, applied)``; ``applied`` is [] when nothing
    changed. CSV I/O belongs to the caller — only the unit sidecar is
    read/written here.
    """
    if not header or not rows:
        return header, rows, []
    current_header = list(header)
    width = len(current_header)
    current_rows = [
        row + [""] * (width - len(row)) if len(row) < width else row[:width] for row in rows
    ]

    pending_applied: list[tuple[str, str]] = []

    def _commit(
        applied: list[tuple[str, str]],
    ) -> tuple[list[str], list[list[str]], list[tuple[str, str]]]:
        if not applied:
            return current_header, current_rows, []

        units_meta: dict[str, Any] = {}
        if units_path is not None and units_path.is_file():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                units_meta = json.loads(units_path.read_text(encoding="utf-8"))
        if units_meta:
            changed = False
            for src, dst in applied:
                changed = _move_unit_meta(units_meta, src, dst) or changed
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
                prose_stem,
            )
        return current_header, current_rows, applied

    case_merges = _case_only_header_merges(current_header, governance_columns)
    if case_merges:
        current_header, current_rows, case_applied = apply_synonym_merges(
            current_header,
            current_rows,
            case_merges,
            protected=set(anchor_keys),
        )
        pending_applied.extend(case_applied)

    fill = column_fill_counts(current_header, current_rows)
    n_rows = len(current_rows)
    gov_lower = {g.lower() for g in governance_columns}
    anchor_lower = {a.lower() for a in anchor_keys}
    gapped_canonicals = [
        col
        for col in current_header
        if col.lower() in gov_lower and col.lower() not in anchor_lower and fill[col] < n_rows
    ]
    discovered_cols = [
        col
        for col in current_header
        if col.lower() not in gov_lower and col.lower() not in anchor_lower and fill[col] > 0
    ]
    if not gapped_canonicals or not discovered_cols:
        return _commit(pending_applied)

    col_idx = {col: current_header.index(col) for col in current_header}

    def _pair_viable(canon: str, disc: str) -> bool:
        """Zero conflicts AND (contributes new cells OR fully redundant duplicate)."""
        ci, di = col_idx[canon], col_idx[disc]
        contributes = 0
        overlap = 0
        disc_nonempty = 0
        for row in current_rows:
            src, dst = row[di].strip(), row[ci].strip()
            if not src:
                continue
            disc_nonempty += 1
            if not dst:
                contributes += 1
            elif cells_agree(src, dst):
                overlap += 1
            else:
                return False
        return disc_nonempty > 0 and (contributes > 0 or overlap == disc_nonempty)

    viable_canonicals = [
        canon
        for canon in gapped_canonicals
        if any(_pair_viable(canon, disc) for disc in discovered_cols)
    ]
    if not viable_canonicals:
        return _commit(pending_applied)

    samples: dict[str, list[str]] = {col: [] for col in discovered_cols}
    for row in current_rows:
        for col in discovered_cols:
            val = row[col_idx[col]].strip()
            if val and val not in samples[col] and len(samples[col]) < 3:
                samples[col].append(val)

    field_defs = metadata_by_name(current_header, field_defs)
    mapping = _adjudicate_synonym_columns(
        adapter,
        prose_stem,
        [
            (col, field_defs.get(col, ""), f"{fill[col]}/{n_rows} rows filled")
            for col in viable_canonicals
        ],
        [
            (col, field_defs.get(col, ""), f"{fill[col]}/{n_rows} rows filled", samples[col])
            for col in discovered_cols
        ],
        question,
    )
    if not mapping:
        return _commit(pending_applied)

    mapping = {disc: canon for disc, canon in mapping.items() if _pair_viable(canon, disc)}

    if not mapping:
        return _commit(pending_applied)

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
        src_unit, dst_unit = _unit_meta_get(units_meta, src), _unit_meta_get(units_meta, dst)
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
        return _commit(pending_applied)

    new_header, new_rows, applied = apply_synonym_merges(
        current_header,
        current_rows,
        guarded,
        protected=set(anchor_keys),
    )
    if not applied:
        return _commit(pending_applied)

    current_header, current_rows = new_header, new_rows
    pending_applied.extend(applied)
    return _commit(pending_applied)
