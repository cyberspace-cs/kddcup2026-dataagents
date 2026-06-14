"""Tool-facing descriptions used in native function-calling schemas."""

from __future__ import annotations

from agents.tools.contracts import (
    EXECUTE_PYTHON_TIMEOUT_SECONDS,
    NON_SQLITE_PATH_WARNING,
    artifact_answer_size_phrase,
    inline_answer_size_phrase,
)

INSPECT_FILES_DESCRIPTION = (
    "Use first to bootstrap context exploration: returns a one-shot summary "
    "of every file under context (columns, dtypes, row counts, table schemas). "
    "This replaces the usual sequence of read_csv / read_json / "
    "inspect_sqlite_schema calls for initial discovery."
)

EXPLORE_DESCRIPTION = (
    "Use first to bootstrap context exploration: launches a sub-agent that "
    "inspects every file under context (columns, dtypes, row counts, table "
    "schemas, unit_hint), reads knowledge.md, samples key values, and "
    "identifies join paths — all in one call. Returns a complete data map "
    "so you can skip manual multi-step discovery."
)

EXPLORE_VIDEO_DESCRIPTION = (
    "Launch the video analysis sub-agent on one video file from context. Use "
    "ONLY on video files listed by `inspect_files`. Write `instructions` as "
    "an analysis brief tied to the task question: which charts, tables, "
    "values, or text to transcribe. Call once per video. The findings return "
    "to you AND pass through to the main agent verbatim — reference key facts "
    "in your report's knowledge/value_samples but do not re-transcribe "
    "numbers."
)

READ_CSV_DESCRIPTION = (
    "Use when you need to inspect a single CSV file: returns columns, "
    "approximate dtypes, row_count, head (first 20 data rows), and tail "
    "(last 5 data rows that are not already in head). This is a SAMPLE "
    "preview, not full data access. For filtering, aggregation, joins, or "
    "any full-file scan, use execute_python instead — the head/tail will "
    "miss rows in between."
)

READ_DOC_DESCRIPTION = (
    "Use when you need to preview a text-like document (md/txt). Two "
    "modes: (1) paging — without `keyword`, returns a fixed line window; "
    "use `has_more` + next `start_line` to continue. (2) keyword search "
    "— with `keyword`, returns ALL paragraphs containing the keyword "
    "across the whole file (capped at 20 matches / 8KB total). Prefer "
    "keyword mode when looking for an ID/name/term — one call replaces "
    "a sequential scan. This is a SAMPLE preview, not full data access. "
    "For full-document regex extraction, cross-paragraph aggregation, "
    "or structured parsing, use execute_python instead."
)

SEARCH_CONTEXT_DESCRIPTION = (
    "Use when the local retrieval index is enabled and you need to find "
    "evidence in long local documents, knowledge files, or schema summaries "
    "by entity name, ID, phrase, or keyword. Returns path, line range, score, "
    "and text preview. Retrieval results are evidence pointers, not final "
    "answers; use execute_python or SQL for full computation."
)

LOOKUP_FIELD_DESCRIPTION = (
    "Use when the local retrieval index is enabled and you need to resolve "
    "field meanings, units, code labels, aliases, thresholds, table names, "
    "or column candidates from knowledge.md and schema summaries before "
    "filtering or joining data."
)

READ_JSON_DESCRIPTION = (
    "Use when you need to preview a single JSON file: returns a "
    "structured summary keyed by `kind` — `array` (length + first 20 "
    "items), `object` (key_count + first 50 keys + value_preview for "
    "first 20 keys), or `scalar` (the value). This is a SAMPLE "
    "preview, not full data access. For nested field extraction, "
    "filtering, cross-file joins, or deep traversal, use execute_python "
    "instead. Files larger than 100MB return a streamed schema "
    "(`kind` + `keys` + `sample_value_kinds` for object; `length` + "
    "`first_item_kind` for array) without full load."
)

INSPECT_SQLITE_SCHEMA_DESCRIPTION = (
    "Use first when you find a sqlite/db file (`.db` / `.sqlite` / "
    "`.sqlite3`) and need to know its tables, columns, and schema "
    "before writing SQL. "
    f"{NON_SQLITE_PATH_WARNING}"
)

EXECUTE_CONTEXT_SQL_DESCRIPTION = (
    "Use when data in a sqlite/db file (`.db` / `.sqlite` / `.sqlite3`) "
    "can be answered with read-only SQL queries or aggregations. Call "
    "inspect_sqlite_schema first for an unknown database. "
    f"{NON_SQLITE_PATH_WARNING}"
)

GREP_CONTEXT_DESCRIPTION = (
    "Use this when you need to search for a regex pattern across ALL context "
    "files at once: CSV column values, JSON fields, Markdown/text content, "
    "and SQLite text columns. Returns matching lines/rows with file path "
    "and location. Checks whether a specific value exists in the data "
    "(e.g., does province column contain '全国'?) without needing to "
    "read each file individually."
)

EXECUTE_PYTHON_DESCRIPTION = (
    "Use when you need computation, aggregation, cross-file joins, or "
    "large-file scans beyond the preview tools. Working directory: the "
    "task context dir; stdout returns as `output`. "
    f"Timeout: {EXECUTE_PYTHON_TIMEOUT_SECONDS} s per call. "
    "Default to `pandas` for tabular work — projections, filters, and "
    "joins are idiomatic and types are preserved automatically. "
    "Available libs: pandas, numpy, pyarrow, polars, openpyxl, sqlalchemy. "
    "The `csv` module is fine only for trivial reads (a handful of rows). "
    "For `.db` / `.sqlite` files, either use the pandas URI form "
    '(`pd.read_sql_query(query, "sqlite:///path")`) or '
    "`sqlite3.connect(path)` + `pd.read_sql_query(query, conn)`."
)

ANSWER_DESCRIPTION = (
    "Use this when the final answer table is determined. This is the only "
    "way to submit; plain text answers are ignored. "
    f"For small answers ({inline_answer_size_phrase()}): pass inline columns/rows. "
    f"For {artifact_answer_size_phrase()}: write CSV in execute_python first, "
    "then pass from_csv (inline is rejected at this threshold). "
    "IMPORTANT: for content-entity questions "
    '("what is the comment/post/message/title"), columns must contain '
    "the text body (Text/Body/Title), NOT a surrogate key (Id/Code)."
)
