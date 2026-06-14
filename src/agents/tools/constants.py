"""filesystem 子包共享的常量与扩展名集合。"""

# ---------------------------------------------------------------------------
# inspect_files
# ---------------------------------------------------------------------------
INSPECT_FILES_MAX_FILES = 100
INSPECT_FILES_JSON_SIZE_LIMIT = 100 * 1024 * 1024
INSPECT_FILES_DTYPE_SAMPLE = 5
INSPECT_FILES_PROFILE_SAMPLE = 100
INSPECT_FILES_DISTINCT_VALUES_CAP = 20
INSPECT_FILES_DOC_HEAD_LINES = 3
INSPECT_FILES_PDF_HEAD_PAGES = 2
INSPECT_FILES_PDF_HEAD_CHARS = 500
KNOWLEDGE_MD_PRELOAD_LIMIT = 32 * 1024
SQLITE_PROFILE_MAX_COLUMNS = 50

# ---------------------------------------------------------------------------
# read_csv
# ---------------------------------------------------------------------------
READ_CSV_HEAD_ROWS = 20
READ_CSV_TAIL_ROWS = 5

# ---------------------------------------------------------------------------
# read_json
# ---------------------------------------------------------------------------
READ_JSON_SIZE_LIMIT = 100 * 1024 * 1024
READ_JSON_OBJECT_KEYS_CAP = 50
READ_JSON_OBJECT_VALUE_PREVIEW_CAP = 20
READ_JSON_ARRAY_HEAD_CAP = 20
READ_JSON_ARRAY_HEAD_BYTES = 6 * 1024
READ_JSON_VALUE_PREVIEW_HEAD_CAP = 3
READ_JSON_VALUE_PREVIEW_HEAD_BYTES = 4 * 1024
READ_JSON_ITEM_BYTE_LIMIT = 2 * 1024
READ_JSON_STRING_VALUE_CAP = 200

# ---------------------------------------------------------------------------
# read_doc
# ---------------------------------------------------------------------------
READ_DOC_KEYWORD_MATCH_CAP = 20
READ_DOC_KEYWORD_PARAGRAPH_BYTES = 2 * 1024
READ_DOC_KEYWORD_TOTAL_BYTES = 8 * 1024

# ---------------------------------------------------------------------------
# 扩展名集合
# ---------------------------------------------------------------------------
SQLITE_EXTS = frozenset({".sqlite", ".sqlite3", ".db"})
CSV_EXTS = frozenset({".csv"})
JSON_EXTS = frozenset({".json"})
DOC_EXTS = frozenset({".md", ".txt"})
PDF_EXTS = frozenset({".pdf"})

UNSUPPORTED_HINT_BY_EXT: dict[str, str] = {
    ".xlsx": "Use execute_python with pandas/openpyxl to read .xlsx.",
    ".xls": "Use execute_python with pandas/xlrd to read legacy .xls.",
    ".parquet": "Use execute_python with pandas/pyarrow to read .parquet.",
    ".feather": "Use execute_python with pandas/pyarrow to read .feather.",
    ".jsonl": "Use execute_python or read_doc to stream JSON Lines.",
    ".ndjson": "Use execute_python or read_doc to stream NDJSON.",
    ".tsv": "Use execute_python with pandas to read .tsv.",
}
