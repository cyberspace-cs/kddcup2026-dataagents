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
    has_standalone_id,
    is_embedded_digit_run,
    is_placeholder,
    normalize_boolean,
)

if TYPE_CHECKING:
    from agents.llm.types import ModelAdapter, ModelMessage

logger = logging.getLogger(__name__)

COMPRESS_CHUNK_SIZE = 30
EXTRACT_CHUNK_LINES = 30

MAX_WORKERS = 4
LLM_CALL_TIMEOUT = 180
ENTITY_CHUNK_TOKEN_BUDGET = 2500

_ID_SEP = r"\s*(?:\(?(?:ID|id|Id)[：:]\s*)?[：:#\-]?\s*"

RECORD_ID_INLINE = re.compile(
    rf"(?:{RECORD_ID_KEYWORDS}){_ID_SEP}(\d{{1,8}})",
    re.IGNORECASE,
)

RECORD_ID_LABEL = re.compile(
    rf"^(?:{RECORD_ID_KEYWORDS}){_ID_SEP}(\d{{1,8}})$",
    re.IGNORECASE,
)
RECORD_ID_RANGE = re.compile(
    rf"(?:{RECORD_ID_KEYWORDS})?\s*\d{{1,8}}\s*(?:至|到|[-~–—])\s*\d{{1,8}}",
    re.IGNORECASE,
)

_DIGIT_RUN = re.compile(r"\d{1,8}")


# Anchor-verification helpers live in _types (shared with the merge layer);
# private aliases keep this module's established import surface stable.
_is_embedded_digit_run = is_embedded_digit_run
_has_standalone_id = has_standalone_id


def _first_standalone_match(regex: re.Pattern[str], text: str) -> re.Match[str] | None:
    """First match whose captured digits form a standalone token."""
    group = 1 if regex.groups else 0
    for m in regex.finditer(text):
        if m.group(group) is None:
            continue
        if not _is_embedded_digit_run(text, m.start(group), m.end(group)):
            return m
    return None


def _contains_standalone_number(text: str) -> bool:
    return any(
        not _is_embedded_digit_run(text, m.start(), m.end()) for m in _DIGIT_RUN.finditer(text)
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
            'Convert labels like "档案 <N>" or "Record <N>" to "<N>". '
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
            "strip unit suffixes and thousand separators. A leading ~ marking an "
            "approximate value is allowed (see FUZZY NUMBERS). "
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


SCHEMA_SAMPLE_TOKEN_BUDGET = 8000


def sample_paragraphs_by_section(text: str, token_limit: int = SCHEMA_SAMPLE_TOKEN_BUDGET) -> str:
    """Sample whole paragraphs for schema field discovery.

    Small documents pass through IN FULL — complete evidence beats any
    sample.  Larger documents get a token budget spread across sections in
    proportion to their size, with CENTERED systematic sampling inside each
    section (k=1 picks the middle paragraph): section-boundary paragraphs
    are intro/outro narrative where field mentions are weakest, while the
    interior carries the value-bearing paragraphs (regression: task_59
    ed_consumerpriceindex — boundary-biased samples fed the schema call
    date narrative only and the CPI fields never entered the schema).
    Documents with <3 sections are sampled the same way as one big section.
    """
    from agents.llm.tokenizer import count_qwen_tokens

    sections = detect_sections(text)
    if not sections or len(sections) < 3:
        sections = [{"title": "", "content": text}]

    sec_paras: list[tuple[str, list[str]]] = []
    for sec in sections:
        paras = [
            p.strip() for p in sec["content"].split("\n\n") if p.strip() and len(p.strip()) > 50
        ]
        if paras:
            sec_paras.append((str(sec["title"]), paras))
    if not sec_paras:
        return ""

    def build(scale: float) -> str:
        parts: list[str] = []
        for title, paras in sec_paras:
            if title:
                parts.append(f"### {title}")
            k = min(len(paras), max(1, round(len(paras) * scale)))
            indices = sorted({round((j + 0.5) * len(paras) / k - 0.5) for j in range(k)})
            parts.extend(paras[i] for i in indices)
        return "\n\n".join(parts)

    sample = build(1.0)
    total_tokens = count_qwen_tokens(sample)
    if total_tokens <= token_limit:
        return sample

    scale = token_limit / total_tokens
    for _ in range(8):
        sample = build(scale)
        if count_qwen_tokens(sample) <= token_limit:
            return sample
        scale *= 0.8

    # One paragraph per section still exceeds the budget: flatten the whole
    # document and stride evenly across it.
    flat = [p for _, paras in sec_paras for p in paras]
    flat_tokens = count_qwen_tokens("\n\n".join(flat))
    keep = max(1, int(len(flat) * token_limit / max(1, flat_tokens)))
    indices = sorted({round((j + 0.5) * len(flat) / keep - 0.5) for j in range(keep)})
    return "\n\n".join(flat[i] for i in indices)


GROUPING_BATCH_SIZE = 40
GROUPING_TRUNCATE_CHARS = 300

# Models sometimes echo the input's paragraph numbering as a line prefix
# ("1) 9: 1" = paragraph 1 → record 9): the optional enumeration prefix is
# stripped.  It requires ")"/"."/"）" after the digits, so a plain
# "<rid>: ..." line can never lose its rid to it.
_ENUM_PREFIX = r"(?:\d{1,8}\s*[).．）]\s*)?"
_GROUP_LINE = re.compile(rf"^[-*\s]*{_ENUM_PREFIX}(\d{{1,8}})\s*[:：]\s*(.+)$")
_NONE_LINE = re.compile(rf"^[-*\s]*{_ENUM_PREFIX}NONE\s*[:：]\s*(.+)$", re.IGNORECASE)
_GROUPS_NONE = re.compile(r"GROUPS\s*[:：]\s*NONE", re.IGNORECASE)
_INDEX_TOKEN = re.compile(r"\d+")

# Instructions live in the SYSTEM message (stronger adherence on instruct
# models); the user message carries only the data block.
GROUPING_SYSTEM_PROMPT = """\
You group data-document paragraphs by entity.

The user supplies numbered paragraph openings from a data document. Most \
paragraphs describe exactly ONE record/entity, introduced somewhere in the \
text by a numeric record ID (phrasings vary: "Record <N>", "档案 <N>", \
"registered as <N>", "the strategic unit identified as <N>", ...). Some \
paragraphs are narrative filler describing no record, and some CONTINUE the \
previous record without restating its ID.

Task: group the paragraph numbers by entity.

Output format — one line per entity, nothing else:
<record_id>: <paragraph numbers, comma-separated>

Rules:
- <record_id> is the numeric ID stated IN the paragraph text. COPY it exactly. \
NEVER invent IDs and NEVER renumber paragraphs sequentially.
- A paragraph usually states OTHER numbers besides its record ID: domain \
codes (identification/company/security codes), dates, years, amounts. The \
record ID is the document-local label that INTRODUCES the record, using the \
same labeling scheme across the whole document ("Record <N>", \
"档案条目 <N>", "备案号 <N>"); the other numbers are that record's DATA \
values. When a paragraph states both, group by the record label — NEVER by \
a data value, even when that value is a unique code (e.g. \
"档案条目 <N>，识别码为 <M>" groups as <N>, not <M>).
- A paragraph that continues an earlier entity without restating the ID goes \
on that entity's line.
- A GROUP paragraph that explicitly states the record IDs of SEVERAL entities \
(e.g. "for units <A>, <B> and <C>, the index ...") goes on EACH stated \
entity's line.
- Narrative/filler paragraphs that describe no record: put their numbers on a \
single line `NONE: <numbers>`.
- Every paragraph number must appear on at least one line; only group \
paragraphs stating several record IDs may appear on more than one line.
- If the paragraphs are NOT organized by per-record numeric IDs (e.g. a pure \
time series keyed by dates), output exactly `GROUPS: NONE` instead."""

GROUPING_USER_TEMPLATE = """\
Below are numbered paragraph openings from a data document. Group them by \
entity.

{paragraphs}"""


def _grouping_messages(numbered_block: str) -> list[ModelMessage]:
    """Build the grouping chat messages (shared with the validation script)."""
    from agents.llm.types import ModelMessage

    return [
        ModelMessage(role="system", content=GROUPING_SYSTEM_PROMPT),
        ModelMessage(role="user", content=GROUPING_USER_TEMPLATE.format(paragraphs=numbered_block)),
    ]


def _split_prose_paragraphs(text: str) -> list[str]:
    """Split prose into paragraphs — THE shared 1-based index space.

    Every component of the grouping pipeline (batching, LLM responses,
    verification, leftover routing) refers to paragraphs by their 1-based
    position in this list.
    """
    return [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 30]


_SENTENCE_END = re.compile(r"[。！？]|[.!?](?=\s|$)")


def _leading_sentences(flat: str, count: int) -> str:
    """First *count* sentences of whitespace-collapsed text.

    A sentence ends at a CJK terminator or at ``.!?`` followed by whitespace
    (or end of text) — the lookahead keeps decimal points inside numerals
    ("141,389,573.43") from terminating the sentence early. Falls back to the
    whole text when fewer terminators exist.
    """
    end = 0
    for _ in range(count):
        m = _SENTENCE_END.search(flat, end)
        if not m:
            return flat
        end = m.end()
    return flat[:end]


def _grouping_head(para: str) -> str:
    """Paragraph opening rendered for grouping prompts.

    First sentence by default — record IDs introduce their paragraph, so that
    is where the grouping signal lives. Extends to the second sentence only
    when the first carries no standalone number while the paragraph does:
    rhetorical openers ("The investigation now turns to ...") defer the ID to
    sentence two, and a head without the ID turns a data paragraph into
    apparent filler.
    """
    flat = " ".join(para.split())
    head = _leading_sentences(flat, 1)
    if _contains_standalone_number(head) or not _contains_standalone_number(flat):
        return head
    return _leading_sentences(flat, 2)


def _make_grouping_batches(
    paras: list[str],
    batch_size: int = GROUPING_BATCH_SIZE,
    truncate: int = GROUPING_TRUNCATE_CHARS,
) -> list[tuple[list[int], str]]:
    """Contiguous balanced batches of (global 1-based indices, rendered numbered block).

    Batch count is ceil(n / batch_size) but sizes are balanced (they differ by
    at most one) instead of fixed-stride: a stride remainder produces a
    sub-handful tail batch whose numbered-list format anchor is too weak — the
    model drifts to prose, every claim falls to unparsed, and the batch's
    paragraphs all drop to the leftover path (task_53 ed_otherdepositorycorpbs:
    606 paras → tail batch of 6 → the document's last five entities lost their
    Section XI fields). Balancing keeps every multi-batch slice at
    >= batch_size // 2 lines with no extra parameter.
    """
    if not paras:
        return []
    n_batches = -(-len(paras) // batch_size)
    base, extra = divmod(len(paras), n_batches)
    batches: list[tuple[list[int], str]] = []
    start = 0
    for i in range(n_batches):
        size = base + (1 if i < extra else 0)
        chunk = paras[start : start + size]
        indices = list(range(start + 1, start + 1 + size))
        lines: list[str] = []
        for idx, para in zip(indices, chunk, strict=True):
            # head = " ".join(para[:truncate].split())  # 300-char opening (sentence-head A/B)
            head = _grouping_head(para)
            lines.append(f"{idx}) {head}")
        batches.append((indices, "\n".join(lines)))
        start += size
    return batches


def _parse_grouping_response(raw: str) -> tuple[dict[str, list[int]], list[int], bool, int]:
    """Parse one grouping response.

    Returns (groups rid->indices, none_indices, groups_none, unparsed_lines).
    Prose lines the model adds around the groups are counted, never fatal.
    """
    raw = raw.replace("```", "")
    groups: dict[str, list[int]] = {}
    none_indices: list[int] = []
    unparsed = 0
    if _GROUPS_NONE.search(raw):
        return {}, [], True, 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if m := _NONE_LINE.match(line):
            none_indices.extend(int(t) for t in _INDEX_TOKEN.findall(m.group(1)))
            continue
        if m := _GROUP_LINE.match(line):
            idxs = [int(t) for t in _INDEX_TOKEN.findall(m.group(2))]
            if idxs:
                groups.setdefault(m.group(1), []).extend(idxs)
            continue
        unparsed += 1
    return groups, none_indices, False, unparsed


def _verify_paragraph_groups(
    groups: dict[str, list[int]],
    paras: list[str],
) -> dict[str, list[int]] | None:
    """Deterministically verify LLM-proposed paragraph groups.

    Three layers:

    1. Structure — out-of-range indices are dropped.  An index claimed by
       several groups is kept for every claimant whose ID the paragraph
       states as a standalone token: explicit multi-entity GROUP paragraphs
       ("for units 206, 211 and 216, ...") legitimately belong to each
       stated entity.  Claims without such an anchor are dropped, and an
       index where NO claimant anchors is invalidated entirely (it routes
       to the leftover/positional path instead of being arbitrarily
       assigned).
    2. Member — a group survives only with >= 1 anchor paragraph: its ID
       occurs as a standalone digit token in the FULL paragraph text.
       Anchorless groups are hallucinated IDs and are dropped.  An attached
       (non-anchor) member is rejected when it contains ANOTHER verified
       group's ID standalone — the LLM mis-attached it (hijack guard).
    3. Global — the verified result must look like per-record grouping at
       all: >= 3 distinct entities whose IDs are not a tiny dense integer
       run.  Evidence from the corpus: legit record IDs are sparse (16,
       20, 21, ... 532) while categorical period keys form tiny dense runs
       ({1..4} quarters, {1..12} months).  An anchor-ratio gate does NOT
       work here: legit entities restate their ID in every paragraph.

    Returns rid -> sorted paragraph indices, or None when the global gate
    rejects the grouping as not-per-record data.
    """
    valid_indices = set(range(1, len(paras) + 1))

    # Layer 1 — structure: out-of-range indices dropped; contested indices
    # resolved per claimant by anchor presence (multi-entity group
    # paragraphs anchor every claimant and keep them all).
    claims: dict[int, list[str]] = {}
    out_of_range = 0
    for rid, idxs in groups.items():
        for idx in idxs:
            if idx not in valid_indices:
                out_of_range += 1
                continue
            rids = claims.setdefault(idx, [])
            if rid not in rids:
                rids.append(rid)

    invalidated: set[int] = set()
    multi_member: list[int] = []
    for idx, rids in claims.items():
        if len(rids) == 1:
            continue
        anchored = [rid for rid in rids if _has_standalone_id(paras[idx - 1], rid)]
        if not anchored:
            invalidated.add(idx)
            continue
        claims[idx] = anchored
        if len(anchored) > 1:
            multi_member.append(idx)

    cleaned: dict[str, list[int]] = {}
    for rid, idxs in groups.items():
        kept = [
            idx
            for idx in dict.fromkeys(idxs)  # dedupe, preserve order
            if idx in claims and idx not in invalidated and rid in claims[idx]
        ]
        if kept:
            cleaned[rid] = kept

    # Layer 2 — member level: anchors and dropped (hallucinated-ID) groups.
    anchors: dict[str, list[int]] = {}
    verified: dict[str, list[int]] = {}
    dropped_groups: list[str] = []
    for rid, idxs in cleaned.items():
        anch = [i for i in idxs if _has_standalone_id(paras[i - 1], rid)]
        if anch:
            anchors[rid] = anch
            verified[rid] = idxs
        else:
            dropped_groups.append(rid)

    # Hijack guard on attached (non-anchor) members.
    hijack_rejected: list[tuple[int, str, str]] = []  # (idx, claimed, other)
    for rid, idxs in verified.items():
        for idx in idxs:
            if idx in anchors[rid]:
                continue
            other = next(
                (g for g in verified if g != rid and _has_standalone_id(paras[idx - 1], g)),
                None,
            )
            if other is not None:
                hijack_rejected.append((idx, rid, other))
    rejected_idx = {i for i, _, _ in hijack_rejected}
    final = {
        rid: sorted(i for i in idxs if i not in rejected_idx) for rid, idxs in verified.items()
    }
    final = {rid: idxs for rid, idxs in final.items() if idxs}

    if out_of_range:
        logger.warning(
            "ETL compress: grouping dropped %d out-of-range paragraph indices", out_of_range
        )
    if multi_member:
        logger.info(
            "ETL compress: %d group paragraphs granted multi-entity membership: %s",
            len(multi_member),
            sorted(multi_member)[:10],
        )
    if invalidated:
        logger.warning(
            "ETL compress: grouping invalidated indices claimed by multiple groups "
            "with no anchoring claimant: %s",
            sorted(invalidated)[:10],
        )
    if dropped_groups:
        logger.warning(
            "ETL compress: grouping dropped %d hallucinated groups (no anchor paragraph): %s",
            len(dropped_groups),
            sorted(dropped_groups)[:10],
        )
    if hijack_rejected:
        logger.warning(
            "ETL compress: grouping hijack guard rejected %d attached members: %s",
            len(hijack_rejected),
            hijack_rejected[:10],
        )

    # Layer 3 — global distinctness gate (anti time-series collapse).
    distinct = len(final)
    ids = [int(r) for r in final if r.isdigit()]
    tiny_dense_run = (
        bool(ids)
        and min(ids) <= 2
        and max(ids) <= 12
        and len(set(ids)) > (max(ids) - min(ids)) * 0.8
    )
    if distinct < 3 or tiny_dense_run:
        logger.info(
            "ETL compress: grouping gate rejected result (%d distinct entities, tiny_dense_run=%s)",
            distinct,
            tiny_dense_run,
        )
        return None
    logger.info(
        "ETL compress: LLM grouping verified %d entities over %d paragraphs "
        "(%d anchor-verified members)",
        distinct,
        sum(len(v) for v in final.values()),
        sum(len(v) for v in anchors.values()),
    )
    return final


def _group_paragraphs_by_llm(
    adapter: ModelAdapter,
    paras: list[str],
) -> tuple[dict[str, list[int]] | None, bool]:
    """Number paragraphs, batch them to the LLM, verify the claimed groups.

    Returns ``(verified_groups, llm_responded)``:

    - ``(groups, True)`` — verified rid -> sorted paragraph indices;
    - ``(None, True)`` — the LLM responded but the result was semantically
      rejected (GROUPS:NONE everywhere, or the verification gate failed);
      callers must NOT override this verdict with the builtin regex;
    - ``(None, False)`` — every batch call failed (infrastructure), so no
      verdict exists; callers may fall back to the builtin regex.
    """
    if len(paras) < 3:
        return None, True

    batches = _make_grouping_batches(paras)

    def call(batch: tuple[list[int], str]) -> str:
        return adapter.complete(_grouping_messages(batch[1])).content

    raw_results: list[str | None] = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(call, batch): idx for idx, batch in enumerate(batches)}
        try:
            for fut in as_completed(futs, timeout=LLM_CALL_TIMEOUT * len(batches)):
                idx = futs[fut]
                try:
                    raw_results[idx] = fut.result(timeout=LLM_CALL_TIMEOUT)
                except TimeoutError:
                    logger.warning("ETL compress: grouping batch %d timed out", idx)
                except Exception as exc:
                    logger.warning("ETL compress: grouping batch %d failed: %s", idx, exc)
        except TimeoutError:
            logger.warning("ETL compress: grouping pool timed out; using partial batches")

    if all(raw is None for raw in raw_results):
        return None, False

    merged: dict[str, list[int]] = {}
    unparsed_total = 0
    for (indices, _), raw in zip(batches, raw_results, strict=True):
        if raw is None:
            continue
        groups, _nones, groups_none, unparsed = _parse_grouping_response(raw)
        unparsed_total += unparsed
        if groups_none:
            continue
        batch_indices = set(indices)
        for rid, idxs in groups.items():
            kept = [i for i in idxs if i in batch_indices]
            if kept:
                merged.setdefault(rid, []).extend(kept)
    if unparsed_total:
        logger.info("ETL compress: grouping responses had %d unparsed lines", unparsed_total)

    return _verify_paragraph_groups(merged, paras), True


def _group_paragraphs_by_entity(
    text: str,
    adapter: ModelAdapter | None = None,
) -> tuple[dict[str, list[str]], list[str]] | None:
    """Group paragraphs by record ID — LLM grouping verified deterministically.

    Returns ``(groups, leftover)``: groups maps record_id → [paragraph, ...]
    preserving paragraph order within each entity and ordering entities by
    first appearance; leftover lists paragraphs in no verified group
    (NONE-classified or unassigned) that still carry standalone numbers —
    potential entities the grouping missed.  Callers must process leftovers
    too: silently dropping them loses whole records.  Returns None when the
    document should be processed positionally instead.

    With an adapter, paragraphs are numbered and batched to the LLM, which
    emits ``<record_id>: <paragraph indices>`` lines; the claims are then
    verified deterministically (see _verify_paragraph_groups).  A semantic
    reject — GROUPS:NONE or a failed verification gate — returns None
    WITHOUT consulting the builtin regex: the gate exists precisely because
    the regex happily tags quarter/month period keys as record IDs.  Only
    when every grouping call fails on infrastructure (or no adapter is
    given) does the builtin record-ID regex take over as fallback.

    Regex-path matches whose captured digits are fragments of larger
    formatted numbers ("8,408,793" → "408") are ignored — they used to
    spawn spurious entity groups and hijack paragraphs away from the
    entity they belong to.
    """
    paras = _split_prose_paragraphs(text)
    if not paras:
        return None

    if adapter is not None:
        verified, llm_responded = _group_paragraphs_by_llm(adapter, paras)
        if verified:
            ordered = sorted(verified.items(), key=lambda kv: kv[1][0])
            grouped = {rid: [paras[i - 1] for i in idxs] for rid, idxs in ordered}
            assigned = {i for idxs in verified.values() for i in idxs}
            unassigned = [
                para
                for i, para in enumerate(paras, start=1)
                if i not in assigned and _contains_standalone_number(para)
            ]
            return grouped, unassigned
        if llm_responded:
            return None
        logger.warning("ETL compress: all grouping batches failed; falling back to record-ID regex")

    groups: dict[str, list[str]] = {}
    leftover: list[str] = []
    tagged = 0
    for para in paras:
        match = _first_standalone_match(RECORD_ID_INLINE, para)
        if match:
            groups.setdefault(match.group(1), []).append(para)
            tagged += 1
        elif _contains_standalone_number(para):
            leftover.append(para)

    if tagged < 3 or not groups:
        return None
    return groups, leftover


ENTITY_SEPARATOR = "\n\n---\n\n"


def _entity_chunks_by_token_budget(
    entity_groups: dict[str, list[str]],
    token_budget: int = ENTITY_CHUNK_TOKEN_BUDGET,
) -> list[tuple[str, list[str]]]:
    """Pack complete entity contexts into chunks that fit a token budget.

    Each entity's paragraphs are joined and never split across chunks.
    Entities are separated by ``---`` and each begins with a
    ``[RECORD_ID: <rid>]`` marker so the LLM copies the authoritative ID
    instead of inventing sequential numbers (which would collide across
    chunks and collapse entities during the downstream PK merge).
    Entities larger than the budget get their own chunk.

    Returns a list of ``(chunk_text, rids_in_chunk)`` tuples.
    """
    from agents.llm.tokenizer import count_qwen_tokens

    entity_texts: list[tuple[str, str, int]] = []
    for rid, paras in entity_groups.items():
        full = f"[RECORD_ID: {rid}]\n" + "\n\n".join(paras)
        toks = count_qwen_tokens(full)
        entity_texts.append((rid, full, toks))

    chunks: list[tuple[str, list[str]]] = []
    current_parts: list[str] = []
    current_rids: list[str] = []
    current_tokens = 0

    def _flush() -> None:
        nonlocal current_parts, current_rids, current_tokens
        if current_parts:
            chunks.append((ENTITY_SEPARATOR.join(current_parts), current_rids))
            current_parts = []
            current_rids = []
            current_tokens = 0

    for rid, full, toks in entity_texts:
        if toks >= token_budget:
            _flush()
            chunks.append((full, [rid]))
            continue

        if current_tokens + toks > token_budget and current_parts:
            _flush()

        current_parts.append(full)
        current_rids.append(rid)
        current_tokens += toks

    _flush()
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
        sample = "\n\n".join(paras[::step][:20])

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

        4. FUZZY NUMBER MARKING: if any values use approximate language
           ("约", "大约", "近", Chinese number words like "六百九十七万"),
           instruct the extractor to convert the numeral mechanically and
           prefix it with ~ (e.g. "约六百九十七万" → ~6970000) — never to
           attempt arithmetic on it; a downstream deterministic pass
           recomputes exact values from sibling fields.

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


# Quoted example list like '"档案 <N>", "记录 <N>", ...' for prompts —
# derived from RECORD_ID_KEYWORDS so the label set has one source of truth.
# Placeholder <N> instead of a concrete number: idle chunks were observed
# echoing concrete example values into output as fabricated rows.
_RECORD_LABEL_EXAMPLES = ", ".join(f'"{lbl} <N>"' for lbl in RECORD_ID_KEYWORDS.split("|"))


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
    entity_aware: bool = False,
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
        if entity_aware:
            format_hint = (
                f"Format — ONE line per entity, pipe-separated key-value pairs:\n"
                f"  {primary_key}: <value> | field_a: <value> | field_b: <value> | ...\n"
                "Use COLON (:) to separate key from value. NEVER use = as separator.\n"
                "Each entity's paragraphs are grouped between '---' delimiters and "
                "begin with a [RECORD_ID: N] marker. Copy that exact N as the "
                f"{primary_key} value of the output line — NEVER renumber entities "
                "sequentially (1, 2, 3, ...) and NEVER invent IDs. "
                "Combine information from ALL paragraphs of the same entity into "
                "ONE output line. OMIT fields with no suitable value in any of "
                "the entity's paragraphs.\n"
            )
        else:
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
            if entity_aware:
                anchor_rule = (
                    f"- Each output line MUST include these identity fields: {anchor_names}. "
                    f"Take {primary_key} from the [RECORD_ID: N] marker; collect the "
                    "other identity values from any paragraph in the entity group."
                )
            else:
                anchor_rule = (
                    f"- Start each line with these identity fields: {anchor_names}. "
                    "Include ALL of them when mentioned in the paragraph. "
                    "If a field's value is not stated in the paragraph, include its "
                    f"key with an empty value. Take {primary_key} from the "
                    "paragraph's own record label; NEVER renumber paragraphs "
                    "sequentially (1, 2, 3, ...) and NEVER invent IDs."
                )
        else:
            anchor_rule = (
                f"- Start each line with {primary_key} as stated in the paragraph. "
                "This field anchors every line."
            )
            if not entity_aware:
                anchor_rule += (
                    " Copy the number from the paragraph's own record label "
                    f'(e.g. "Record <N>" → {primary_key}: <N>). NEVER renumber '
                    "paragraphs sequentially (1, 2, 3, ...) and NEVER invent IDs — "
                    "if a paragraph states no record identifier at all, SKIP that "
                    "paragraph."
                )
    else:
        schema_hint = ""
        type_hint = ""
        hard_type_rules = ""
        if entity_aware:
            format_hint = (
                "Format — ONE line per entity, pipe-separated key-value pairs:\n"
                "  ID: <number> | entity_name: <name> | <field>: <value> | ...\n"
                "Use COLON (:) to separate key from value. NEVER use = as separator.\n"
                "Each entity's paragraphs are grouped between '---' delimiters and "
                "begin with a [RECORD_ID: N] marker. Copy that exact N as the ID "
                "value — NEVER renumber entities sequentially and NEVER invent IDs. "
                "Combine all paragraphs of the same entity into ONE line.\n"
            )
        else:
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

    if entity_aware:
        opening = (
            "Below are groups of paragraphs from a data document, separated by '---'. "
            "Each group contains ALL paragraphs about ONE entity, mixing data with "
            "narrative filler, and starts with its authoritative [RECORD_ID: N] marker."
        )
        task_line = (
            "Task: for each entity group, combine information from all its "
            "paragraphs and extract ONE line of labeled key-value pairs."
        )
        skip_rule = (
            f"- If an entire entity group provides NO extractable field values "
            f"beyond the {id_field_label}, SKIP it."
        )
        shared_para_rule = (
            "- A paragraph may be SHARED by several entities and enumerate "
            'per-entity values (e.g. "... stood at <X> for unit <A>, <Y> for '
            "unit <B>\"): take ONLY the value stated for this group's "
            "[RECORD_ID: N]; NEVER borrow a value stated for another entity. "
            "If such a paragraph only declares values missing/unavailable for "
            "a set of entities, leave those fields empty — that is not a "
            "reason to skip the rest of the entity group."
        )
    else:
        opening = (
            "Below are paragraphs from a data document. Each paragraph describes "
            "one entity and mixes data with narrative filler."
        )
        task_line = "Task: for each paragraph, extract data as labeled key-value pairs."
        shared_para_rule = ""
        skip_rule = (
            "- If a paragraph provides NO extractable field values beyond the "
            "record_id (e.g. it only says data is missing/not provided/unavailable), "
            "SKIP that paragraph entirely — do NOT output a row with only identity fields and empty data fields. "
            "However, if the paragraph carries identity/anchor values such as "
            "secucode, companycode, or entity name, DO output a row even when no "
            "other data fields are present — downstream merge needs those identity "
            "values to complete the record."
        )

    prompt = textwrap.dedent(f"""\
        {opening}

        {task_line}
        {schema_hint}
        {def_hint}
        {type_hint}
        {hard_type_rules}
        {format_hint}
        Rules:
        {anchor_rule}
        {shared_para_rule}
        - Every value MUST have a descriptive field label before it.
        - If a schema field has no explicit, suitable value in this {"entity group" if entity_aware else "paragraph"}, \
        leave it empty by omitting that non-identity field. Do NOT substitute \
        background descriptions, business strengths, culture, plans, or prose \
        summaries just because they are loosely related to the field name.
        {skip_rule}
        - Field-name qualifiers are binding. Do NOT map an overall/global \
        total, count, rank, or average into a qualified/category-specific \
        field unless the paragraph explicitly states the same qualifier for \
        that value; omit the field instead.
        - Local document labels such as {_RECORD_LABEL_EXAMPLES} are \
        record identifiers. Put only the scalar numeric portion (for example, \
        "<N>") in the {id_field_label} field. \
        NEVER output ranges like "197至217" or grouped labels as a single \
        {id_field_label} value, and never put a local record label into domain ID \
        fields such as personalcode, innercode, or fundcode unless \
        the paragraph explicitly labels it as that domain ID.
        - personalcode/PersonalCode fields are person identifiers. Do NOT \
        fill them with a manager name or with a local document record number; \
        leave personalcode empty unless an explicit personal/manager code is \
        stated.
        - Output numeric values WITHOUT unit suffixes (e.g. "<value> cm" → <value>).
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
        - FUZZY NUMBERS: when a value is approximate — marked with "约", \
        "大约", "近", "左右", "roughly", "approximately", or written in \
        coarse Chinese number words like "六百九十七万" — convert the \
        numeral mechanically and prefix it with ~ (e.g. "约六百九十七万股" \
        → ~6970000). Do NOT attempt arithmetic to sharpen it, and NEVER \
        omit a value merely because it is approximate — always output it \
        with the ~ prefix: a downstream deterministic pass recomputes exact \
        values from sibling fields. Values stated as exact digits get NO \
        prefix.
        {placeholder_rule}
        - For corrections: output the FINAL corrected value directly — e.g. \
        "label: positive", not "label: positive (placeholder)".
        - Preserve URLs, names, dates, times, categories, classifications.
        - Strip only narrative filler that contains no structured field, schema field, \
        ID, value, date, unit, label, status, or question-relevant fact.
        - If NOTHING in this input qualifies for extraction under these rules, \
        return an EMPTY response. NEVER fabricate demonstration rows and NEVER \
        copy placeholder examples (such as "<N>" or "<value>") from these \
        instructions into the output.
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

    # Try entity-aware chunking: group paragraphs by entity (LLM grouping
    # with deterministic verification) so each entity's full context is in
    # one chunk, never split across chunks.
    grouping = _group_paragraphs_by_entity(text, adapter=adapter)
    entity_groups: dict[str, list[str]] | None = None
    work: list[tuple[str | None, str]] = []
    chunk_rids: list[list[str]] = []
    if grouping:
        entity_groups, leftover = grouping
        chunk_infos = _entity_chunks_by_token_budget(entity_groups)
        work = [(None, c) for c, _ in chunk_infos]
        chunk_rids = [rids for _, rids in chunk_infos]
        # Paragraphs in no verified group (NONE-classified or unassigned)
        # still carry numbers: compress them positionally instead of
        # dropping them — the LLM reads the record id straight from the
        # paragraph text and the downstream PK merge unifies those lines
        # with the entity lines.
        if leftover:
            logger.info(
                "ETL compress: %d numeric paragraphs matched no entity; "
                "processing them positionally",
                len(leftover),
            )
            for i in range(0, len(leftover), COMPRESS_CHUNK_SIZE):
                work.append((None, "\n\n".join(leftover[i : i + COMPRESS_CHUNK_SIZE])))
                chunk_rids.append([])
        logger.info(
            "ETL compress: entity-aware chunking: %d entities → %d chunks",
            len(entity_groups),
            len(work),
        )
    else:
        work = _positional_chunks(text)
        chunk_rids = [[] for _ in work]

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
                entity_aware=bool(chunk_rids[idx]),
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

    # Entity-aware integrity check: every output line's PK must echo one of
    # the [RECORD_ID: N] markers from its own chunk.  A fabricated sequential
    # PK (1, 2, 3, ...) collides across chunks and silently collapses
    # entities in the downstream PK merge — surface it loudly, then retry
    # the genuinely missing entities in a targeted recovery pass.
    recovered_parts: list[str] = []
    if entity_groups and primary_key:
        # Scan the way the final cleaning pass normalizes lines (case,
        # ``=`` separators) so casing quirks don't read as missing entities;
        # ``(?<!\[)`` keeps echoed [RECORD_ID: N] markers from counting as
        # output lines.
        pk_re = re.compile(
            rf"(?<!\[)\b{re.escape(primary_key)}\s*:\s*([^|\n]+)",
            re.IGNORECASE,
        )

        def _extracted_pks(chunk_text: str) -> set[str]:
            got: set[str] = set()
            for line in chunk_text.splitlines():
                normalized = _normalize_kv_separators(line.strip())
                for m in pk_re.finditer(normalized):
                    got.add(m.group(1).strip())
            return got

        to_recover: set[str] = set()
        for idx, expected_rids in enumerate(chunk_rids):
            if not expected_rids:
                continue
            chunk_result = compressed[idx]
            if not chunk_result:
                # Whole chunk failed or timed out — recover all its entities.
                to_recover.update(expected_rids)
                continue
            expected = set(expected_rids)
            got = _extracted_pks(chunk_result)
            bogus = got - expected
            missing = expected - got
            if bogus:
                logger.warning(
                    "ETL compress: chunk %d PK values %s not in expected markers %s "
                    "— model ignored [RECORD_ID] markers; downstream merge may "
                    "collapse entities",
                    idx,
                    sorted(bogus),
                    sorted(expected),
                )
            if missing:
                logger.warning(
                    "ETL compress: chunk %d missing entities %s (%d/%d extracted)",
                    idx,
                    sorted(missing),
                    len(expected - missing),
                    len(expected),
                )
                to_recover.update(missing)

        if to_recover:
            subset = {rid: entity_groups[rid] for rid in sorted(to_recover) if rid in entity_groups}
            if subset:
                logger.info(
                    "ETL compress: targeted recovery pass for %d missing entities",
                    len(subset),
                )
                for retry_text, _ in _entity_chunks_by_token_budget(subset):
                    try:
                        result = compress_chunk(
                            adapter,
                            retry_text,
                            schema_columns,
                            primary_key,
                            anchor_keys,
                            schema_field_types,
                            schema_field_defs,
                            guide,
                            entity_aware=True,
                        )
                    except Exception as exc:
                        logger.warning("ETL compress: recovery chunk failed: %s", exc)
                        continue
                    if result:
                        recovered_parts.append(result)
                got_all: set[str] = set()
                for part in recovered_parts:
                    got_all |= _extracted_pks(part)
                recovered = sorted(set(subset) & got_all)
                still_missing = sorted(set(subset) - got_all)
                if recovered:
                    logger.info("ETL compress: recovery pass extracted %s", recovered)
                if still_missing:
                    logger.warning(
                        "ETL compress: entities still missing after recovery "
                        "(likely no extractable data): %s",
                        still_missing,
                    )

    parts: list[str] = []
    for idx, (title, _) in enumerate(work):
        if title:
            parts.append(f"### {title}")
        chunk_result = compressed[idx]
        if chunk_result:
            parts.append(chunk_result)
    parts.extend(recovered_parts)
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
