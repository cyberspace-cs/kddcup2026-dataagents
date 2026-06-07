"""grep_context：跨 context 目录的纯文本 + SQLite 正则搜索。

文本文件（csv/json/md/txt）按行 regex 匹配，类似 ripgrep；
SQLite（db/sqlite）按表×文本列做 LIKE '%pattern%' 查询。
PDF 等二进制格式跳过（ETL 已提取为 CSV）。
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from agents.benchmark.schema import PublicTask
from agents.config import ETL_SCRATCH_ROOT
from agents.tools.filesystem.constants import (
    CSV_EXTS,
    DOC_EXTS,
    JSON_EXTS,
    SQLITE_EXTS,
)

TEXT_EXTS = CSV_EXTS | JSON_EXTS | DOC_EXTS
ALL_SEARCHABLE_EXTS = TEXT_EXTS | SQLITE_EXTS

GREP_MAX_RESULTS = 30
GREP_MAX_LINE_CHARS = 500
GREP_MAX_FILE_BYTES = 50 * 1024 * 1024


def _grep_text_file(
    path: Path,
    rel_path: str,
    compiled: re.Pattern[str],
    remaining: int,
) -> list[dict[str, Any]]:
    """Search a text file line-by-line, return up to `remaining` matches."""
    if path.stat().st_size > GREP_MAX_FILE_BYTES:
        return []
    hits: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line_no, line in enumerate(fh, 1):
                if compiled.search(line):
                    text = line.rstrip("\n\r")
                    if len(text) > GREP_MAX_LINE_CHARS:
                        text = text[:GREP_MAX_LINE_CHARS] + "…"
                    hits.append({"file": rel_path, "line": line_no, "text": text})
                    if len(hits) >= remaining:
                        break
    except OSError:
        pass
    return hits


SQLITE_SAMPLE_PER_TABLE = 3


def _grep_sqlite(
    path: Path,
    rel_path: str,
    pattern: str,
    remaining: int,
) -> list[dict[str, Any]]:
    """Search text columns across all tables; deduplicate by table×column."""
    hits: list[dict[str, Any]] = []
    like_pattern = f"%{pattern}%"
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        return []
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        for (table_name,) in tables:
            if not isinstance(table_name, str):
                continue
            quoted = table_name.replace('"', '""')
            cols_info = conn.execute(f'PRAGMA table_info("{quoted}")').fetchall()
            text_cols = [
                row[1] for row in cols_info if isinstance(row[2], str) and "text" in row[2].lower()
            ]
            if not text_cols:
                text_cols = [row[1] for row in cols_info if isinstance(row[1], str)]
            if not text_cols:
                continue
            for col in text_cols:
                col_quoted = col.replace('"', '""')
                sql = f'SELECT COUNT(*) FROM "{quoted}" WHERE "{col_quoted}" LIKE ?'
                try:
                    total = conn.execute(sql, [like_pattern]).fetchone()[0]
                except sqlite3.Error:
                    continue
                if total == 0:
                    continue
                sample_sql = (
                    f'SELECT rowid, "{col_quoted}" FROM "{quoted}" '
                    f'WHERE "{col_quoted}" LIKE ? LIMIT {SQLITE_SAMPLE_PER_TABLE}'
                )
                try:
                    sample_rows = conn.execute(sample_sql, [like_pattern]).fetchall()
                except sqlite3.Error:
                    sample_rows = []
                samples: list[str] = []
                for row in sample_rows:
                    val = str(row[1]) if row[1] is not None else ""
                    if len(val) > GREP_MAX_LINE_CHARS:
                        val = val[:GREP_MAX_LINE_CHARS] + "…"
                    samples.append(val)
                hits.append(
                    {
                        "file": rel_path,
                        "table": table_name,
                        "column": col,
                        "total_matches": total,
                        "samples": samples,
                    }
                )
                if len(hits) >= remaining:
                    break
            if len(hits) >= remaining:
                break
    finally:
        conn.close()
    return hits


def grep_context(
    task: PublicTask,
    pattern: str,
    *,
    path_filter: str | None = None,
    max_results: int = GREP_MAX_RESULTS,
) -> dict[str, Any]:
    """Search for a regex pattern across all context files.

    Text files (csv/json/md/txt) are searched line-by-line.
    SQLite databases are searched via LIKE across text columns.
    Returns matching lines/rows with file path and location.
    """
    context_root = task.context_dir.resolve()
    etl_dir = ETL_SCRATCH_ROOT / task.task_id / "_etl"

    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return {"error": f"Invalid regex: {exc}", "pattern": pattern}

    candidates: list[tuple[str, Path]] = []

    etl_csv: dict[str, Path] = {}
    if etl_dir.is_dir():
        for f in etl_dir.iterdir():
            if f.is_file() and f.suffix == ".csv":
                etl_csv[f.stem] = f

    for child in context_root.rglob("*"):
        if not child.is_file():
            continue
        if any(part.startswith(".") for part in child.relative_to(context_root).parts):
            continue
        suffix = child.suffix.lower()
        if suffix not in ALL_SEARCHABLE_EXTS:
            continue
        if suffix in {".md", ".pdf"} and child.stem in etl_csv:
            csv_path = etl_csv[child.stem]
            candidates.append((str(csv_path), csv_path))
            continue
        rel = child.relative_to(context_root).as_posix()
        candidates.append((rel, child))

    if etl_dir.is_dir():
        for _rel_stem, csv_path in etl_csv.items():
            if not any(p == csv_path for _, p in candidates):
                candidates.append((str(csv_path), csv_path))

    candidates.sort(key=lambda item: item[0])

    if path_filter:
        candidates = [(r, p) for r, p in candidates if path_filter in r]

    all_hits: list[dict[str, Any]] = []
    files_searched = 0

    for rel_path, abs_path in candidates:
        if len(all_hits) >= max_results:
            break
        remaining = max_results - len(all_hits)
        suffix = abs_path.suffix.lower()

        if suffix in TEXT_EXTS:
            all_hits.extend(_grep_text_file(abs_path, rel_path, compiled, remaining))
        elif suffix in SQLITE_EXTS:
            all_hits.extend(_grep_sqlite(abs_path, rel_path, pattern, remaining))
        files_searched += 1

    return {
        "pattern": pattern,
        "files_searched": files_searched,
        "match_count": len(all_hits),
        "truncated": len(all_hits) >= max_results,
        "results": all_hits,
    }
