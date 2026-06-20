"""Shared ETL constants.

This module is intentionally logic-free. It centralizes values that are used
by multiple ETL stages so module splits do not create circular imports between
compression, grouping, schema parsing, and verification.
"""

from __future__ import annotations

import re

# LLM work-shaping constants. Compression and verification share these so
# grouped-entity extraction, positional fallback chunks, and retry/verify calls
# use the same concurrency and timeout envelope.
COMPRESS_CHUNK_SIZE = 30
MAX_WORKERS = 4
LLM_CALL_TIMEOUT = 180
ENTITY_CHUNK_TOKEN_BUDGET = 2500

# ETL orchestrator adapter and file-level parallelism settings. These are kept
# separate from compression batch settings so future tuning can distinguish
# API call limits from per-file task scheduling.
ETL_ORCHESTRATOR_MAX_WORKERS = 4
ETL_ORCHESTRATOR_LLM_CALL_TIMEOUT = 180
ETL_HTTP_TIMEOUT = 300.0
ETL_MAX_OUTPUT_TOKENS = 4096
ETL_RESERVE_OUTPUT_TOKENS = 2048

# Prose-file detection suffix allow-list.
PROSE_EXTS = frozenset({".md", ".txt", ".pdf"})

# Document-local record-id recognition. These labels describe generated prose
# row keys such as "档案 29" or "Record 29"; they are not domain identifiers
# like company codes or personal codes. Grouping, compression cleanup, and
# schema anchoring must agree on this vocabulary.
RECORD_ID_FIELDS = frozenset(
    {"record_id", "archive_id", "file_id", "case_id", "entry_id", "local_id", "unit_id"}
)
RECORD_ID_KEYWORDS = r"档案|记录|条目|战略单元|Record|File|Case"
RECORD_ID_SEPARATOR = r"\s*(?:\(?(?:ID|id|Id)[：:]\s*)?[：:#\-]?\s*"
RECORD_ID_INLINE = re.compile(
    rf"(?:{RECORD_ID_KEYWORDS}){RECORD_ID_SEPARATOR}(\d{{1,8}})",
    re.IGNORECASE,
)
RECORD_ID_LABEL = re.compile(
    rf"^(?:{RECORD_ID_KEYWORDS}){RECORD_ID_SEPARATOR}(\d{{1,8}})$",
    re.IGNORECASE,
)
RECORD_ID_RANGE = re.compile(
    rf"(?:{RECORD_ID_KEYWORDS})?\s*\d{{1,8}}\s*(?:至|到|[-~–—])\s*\d{{1,8}}",
    re.IGNORECASE,
)
DIGIT_RUN = re.compile(r"\d{1,8}")
LOCAL_RECORD_ID_RE = re.compile(
    rf"(?:{RECORD_ID_KEYWORDS})\s*[：:#-]?\s*(\d{{1,8}})",
    re.IGNORECASE,
)
# Quoted example list for extraction prompts. It uses <N> rather than a
# concrete number so prompt examples cannot be echoed as fabricated records.
RECORD_LABEL_EXAMPLES = ", ".join(f'"{lbl} <N>"' for lbl in RECORD_ID_KEYWORDS.split("|"))

# Entity-line and markdown-heading patterns shared by detection, merge, and
# verification. Keeping them centralized prevents import cycles between
# `_detect`, `_merge`, and `_verify`.
ENTITY_LINE = re.compile(r"^ID:\s*", re.MULTILINE)
SECTION_HEADING = re.compile(r"^(#{2,4})\s+(.+)$", re.MULTILINE)

# Scalar validators used while enforcing schema types. They intentionally
# accept only simple scalar forms so narrative labels, ranges, and mixed values
# are blanked before deterministic CSV materialization.
NUMERIC_SCALAR = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$")
DATE_SCALAR = re.compile(r"^\d{4}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?$")

# Placeholder and approximation markers. Compression uses the approximate tag
# for fuzzy source numerals; identity repair consumes it later. Placeholder
# tokens normalize common generated "no value" spellings across ETL stages.
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
APPROX_TAG = "~"
APPROX_PREFIXES = ("~", "～")

# Boolean normalization vocabulary for schema-enforced boolean columns.
BOOL_TRUE = frozenset(("true", "yes", "1", "是"))
BOOL_FALSE = frozenset(("false", "no", "0", "否"))

# Schema-sampling limits and parser sentinels. Schema inference uses a larger
# sample budget than routing because it must discover fields across sections;
# multi-round inference hedges against one brittle LLM schema response.
SCHEMA_SAMPLE_TOKEN_BUDGET = 8000
SCHEMA_SAMPLE_ROUNDS = 3
SCHEMA_RESPONSE_PREFIX = re.compile(r"^(PK|ANCHORS?|TYPES?|UNITS?|DEFS?)\s*:", re.IGNORECASE)

# Router prompt compaction bounds. They keep multi-file selection cheap while
# preserving enough schema and sample text to avoid routing away relevant prose.
ROUTER_SAMPLE_TOKEN_BUDGET = 2000
ROUTER_MAX_FIELDS_PER_FILE = 15
ROUTER_MAX_SAMPLE_CHARS = 500
ROUTER_MAX_DESC_CHARS = 200

# PDF text cleanup punctuation and structural-line recognizers. These control
# whether extracted layout lines are stitched as prose or preserved as markdown
# structure such as headings, lists, code fences, or tables.
PDF_CJK_PUNCT = frozenset("，。！？；：、）】》”’")
PDF_STRUCTURAL_LINE = re.compile(r"^(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s+|`{3,}|~~~|\|.*\|$)")
PDF_SENTENCE_END = re.compile(r"[。！？.!?」）)\"']\s*$")

# knowledge.md markdown-table parsing. Field cells are deliberately restrictive
# so prose descriptions are not mistaken for column names.
KM_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
KM_TABLE_SEPARATOR = re.compile(r":?-{3,}:?")
KM_FIELD_CELL = re.compile(r"^[\w一-鿿][\w一-鿿()（）%/·\-]{0,39}$")

# KV-line normalization and pre-merge sentinels. Temporary anchors are internal
# keys used while merging records that lack a primary-key value but carry a
# secondary identity anchor.
KV_LINE = re.compile(r"^ID:\s*(\d+)\s*\|\s*entity_name:\s*(.+?)\s*\|", re.MULTILINE)
TEMP_ANCHOR_PREFIX = "__anchor__:"
UNIT_SUFFIX = re.compile(r"^([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)\s+\S+\s*$")

# Deterministic unit-conversion families. A source and target unit convert
# exactly only when both belong to the same scale family.
UNIT_SCALE_FAMILIES: tuple[dict[str, float], ...] = (
    {
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
    {
        "万": 1e4,
        "亿": 1e8,
        "万亿": 1e12,
    },
    {
        "%": 0.01,
        "ratio": 1.0,
    },
)

# Sibling-source sampling for percentage/proportion convention inference. The
# subfamilies separate names like `pct`, `ratio`, and Chinese `比例`, whose
# storage conventions may differ across datasets.
PROPORTION_SAMPLE_LIMIT = 500
PROPORTION_SUBFAMILIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("pct", re.compile(r"pct|percent", re.IGNORECASE)),
    ("proportion", re.compile(r"proportion", re.IGNORECASE)),
    ("ratio", re.compile(r"ratio", re.IGNORECASE)),
    ("比例", re.compile(r"比例")),
)

# Numeric identity mining gates. The support floor avoids accepting accidental
# arithmetic coincidences; float tolerances only absorb binary floating noise.
MIN_IDENTITY_SUPPORT = 5
NUMERIC_EQ_REL = 1e-9
NUMERIC_EQ_ABS = 1e-9

# Value retry, de-duplication, and verification limits. These bound expensive
# LLM calls to high-signal entities and local source snippets.
FILL_RATE_THRESHOLD = 0.70
MAX_RETRY_ENTITIES = 20
DEDUP_CONTEXT_WINDOW = 300
MAX_VERIFY_ENTITIES = 50

# Malformed-line repair limits. Retries are capped because format repair is a
# cleanup pass, not a replacement for compression/schema correctness.
FIX_FORMAT_MAX_RETRIES = 3
FIX_FORMAT_MAX_LINES = 80

# LLM paragraph-grouping prompt parsing and batching constants. The response
# regexes accept common list prefixes while still requiring an explicit entity
# id or NONE marker.
GROUPING_BATCH_SIZE = 40
GROUPING_TRUNCATE_CHARS = 300
GROUPING_ENUM_PREFIX = r"(?:\d{1,8}\s*[).．）]\s*)?"
GROUP_LINE = re.compile(rf"^[-*\s]*{GROUPING_ENUM_PREFIX}(\d{{1,8}})\s*[:：]\s*(.+)$")
NONE_LINE = re.compile(
    rf"^[-*\s]*{GROUPING_ENUM_PREFIX}NONE\s*[:：]\s*(.+)$",
    re.IGNORECASE,
)
GROUPS_NONE = re.compile(r"GROUPS\s*[:：]\s*NONE", re.IGNORECASE)
INDEX_TOKEN = re.compile(r"\d+")
GROUPING_SENTENCE_END = re.compile(r"[。！？]|[.!?](?=\s|$)")
ENTITY_SEPARATOR = "\n\n---\n\n"

# Prompt templates for the LLM paragraph-grouping stage. Validation scripts use
# the same templates as production so prompt regressions are testable.
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
