"""inspect_context_files：一次性扫描 context_dir 下所有文件的轻量 schema。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from agents.benchmark.schema import PublicTask
from agents.config import ETL_SCRATCH_ROOT
from agents.tools.filesystem import constants
from agents.tools.filesystem.csv_preview import profile_columns, summarize_csv
from agents.tools.filesystem.doc_preview import summarize_doc
from agents.tools.filesystem.json_preview import summarize_json
from agents.tools.filesystem.pdf_preview import summarize_pdf


def _summarize_sqlite(path: Path) -> dict[str, Any]:
    """sqlite/db：返回每张表的 columns + row_count + 列级 profile。"""
    tables: dict[str, dict[str, Any]] = {}
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        for (name,) in table_rows:
            if not isinstance(name, str):
                continue
            quoted = name.replace('"', '""')
            cursor = conn.execute(f'PRAGMA table_info("{quoted}")')
            columns = [row[1] for row in cursor.fetchall()]
            count_cursor = conn.execute(f'SELECT COUNT(*) FROM "{quoted}"')
            row_count = count_cursor.fetchone()[0]
            table_info: dict[str, Any] = {"columns": columns, "row_count": row_count}

            if row_count == 0 or len(columns) > constants.SQLITE_PROFILE_MAX_COLUMNS:
                if len(columns) > constants.SQLITE_PROFILE_MAX_COLUMNS:
                    table_info["profile_skipped"] = "too_many_columns"
                tables[name] = table_info
                continue

            sample_cursor = conn.execute(
                f'SELECT * FROM "{quoted}" LIMIT {constants.INSPECT_FILES_PROFILE_SAMPLE}'
            )
            raw_rows = sample_cursor.fetchall()
            sample_rows = [[str(v) if v is not None else "" for v in row] for row in raw_rows]
            dtypes, profile = profile_columns(columns, sample_rows)
            table_info["dtypes"] = dtypes
            table_info["profile"] = profile
            table_info["profile_sample_rows"] = len(sample_rows)
            tables[name] = table_info
    finally:
        conn.close()
    return {"tables": tables}


def _format_for_inspect(rel_path: str, abs_path: Path) -> dict[str, Any]:
    """单文件 introspection 的统一入口：按扩展名分流到具体 summarizer。"""
    suffix = abs_path.suffix.lower()
    size = abs_path.stat().st_size
    base: dict[str, Any] = {
        "path": rel_path,
        "size": size,
    }
    if suffix in constants.CSV_EXTS:
        base["format"] = "csv"
        base["supported"] = True
        base["schema"] = summarize_csv(abs_path)
        return base
    if suffix in constants.JSON_EXTS:
        base["format"] = "json"
        base["supported"] = True
        base["schema"] = summarize_json(abs_path)
        return base
    if suffix in constants.SQLITE_EXTS:
        base["format"] = "sqlite"
        base["supported"] = True
        base["schema"] = _summarize_sqlite(abs_path)
        return base
    if suffix in constants.PDF_EXTS:
        base["format"] = "pdf"
        base["supported"] = True
        base["schema"] = summarize_pdf(abs_path)
        return base
    if suffix in constants.DOC_EXTS:
        base["format"] = "markdown" if suffix == ".md" else "text"
        base["supported"] = True
        base["schema"] = summarize_doc(abs_path)
        if abs_path.name.lower() == "knowledge.md" and size <= constants.KNOWLEDGE_MD_PRELOAD_LIMIT:
            try:
                base["schema"]["full_text"] = abs_path.read_text(errors="replace")
                base["schema"]["knowledge_preloaded"] = True
            except OSError:
                pass
        return base
    base["format"] = suffix.lstrip(".") or "unknown"
    base["supported"] = False
    base["reason"] = constants.UNSUPPORTED_HINT_BY_EXT.get(
        suffix,
        "Unsupported by inspect_files; use execute_python or another preview tool.",
    )
    return base


def inspect_context_files(task: PublicTask) -> dict[str, Any]:
    """一次性扫描 context_dir 下所有文件的轻量 schema。"""
    context_root = task.context_dir.resolve()
    files_summary: list[dict[str, Any]] = []
    unsupported_formats: set[str] = set()

    candidates: list[tuple[str, Path]] = []
    for child in context_root.rglob("*"):
        if not child.is_file():
            continue
        if any(part.startswith(".") for part in child.relative_to(context_root).parts):
            continue
        rel = child.relative_to(context_root).as_posix()
        candidates.append((rel, child))
    candidates.sort(key=lambda item: item[0])

    etl_dir = ETL_SCRATCH_ROOT / task.task_id / "_etl"
    cache_dir = ETL_SCRATCH_ROOT / task.task_id / "_cache"

    truncated = len(candidates) > constants.INSPECT_FILES_MAX_FILES
    for rel_path, abs_path in candidates[: constants.INSPECT_FILES_MAX_FILES]:
        entry = _format_for_inspect(rel_path, abs_path)
        if not entry["supported"]:
            unsupported_formats.add(str(entry["format"]))
        real_path = abs_path.resolve()
        if real_path.is_relative_to(etl_dir) and cache_dir.is_dir():
            units_path = cache_dir / f"{abs_path.stem}_units.json"
            if units_path.is_file():
                try:
                    units_meta = json.loads(units_path.read_text(encoding="utf-8"))
                    schema: dict[str, Any] | None = entry.get("schema")
                    # Build per-column conversion notes like:
                    # {"pctbeforetran": "CONVERT: ×0.01 (% → ratio)"}
                    col_conversions: dict[str, str] = {}
                    for field, source_unit in units_meta.items():
                        if field.startswith("_"):
                            continue
                        target = units_meta.get(f"_target_{field}")
                        factor = units_meta.get(f"_factor_{field}")
                        if target and factor and str(source_unit) != str(target):
                            col_conversions[field] = (
                                f"CONVERT: ×{factor} ({source_unit} → {target})"
                            )
                    if isinstance(schema, dict) and "columns" in schema:
                        reordered: dict[str, Any] = {}
                        for key in schema:
                            reordered[key] = schema[key]
                            if key == "columns" and col_conversions:
                                reordered["column_conversions"] = col_conversions
                        entry["schema"] = reordered
                    elif isinstance(schema, dict):
                        if col_conversions:
                            schema["column_conversions"] = col_conversions
                except (OSError, json.JSONDecodeError):
                    pass
        files_summary.append(entry)

    result: dict[str, Any] = {
        "context_root": str(context_root),
        "files": files_summary,
        "unsupported_formats": sorted(unsupported_formats),
        "file_count": len(files_summary),
        "truncated": truncated,
    }
    try:
        from agents.retrieval.index import load_task_index_metadata

        retrieval_meta = load_task_index_metadata(task)
    except Exception:
        retrieval_meta = None
    if isinstance(retrieval_meta, dict) and retrieval_meta.get("enabled"):
        result["retrieval_index"] = {
            "built": bool(retrieval_meta.get("built")),
            "cache_hit": bool(retrieval_meta.get("cache_hit")),
            "chunk_count": retrieval_meta.get("chunk_count"),
            "field_count": retrieval_meta.get("field_count"),
            "index_path": retrieval_meta.get("index_path"),
            "hint": (
                "For long documents, field meanings, units, thresholds, or entity "
                "mentions, prefer lookup_field/search_context before paging read_doc."
            ),
        }
        if retrieval_meta.get("error"):
            result["retrieval_index"]["error"] = retrieval_meta.get("error")
    return result
