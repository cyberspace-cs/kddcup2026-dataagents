"""Task-scoped Local Retrieval Index.

The index is intentionally offline and deterministic: SQLite FTS5 for text chunks and
schema/field lookup, no embeddings and no network access. Runner code builds the index
before the agent loop and writes a small sidecar under `/tmp/dabench/<task_id>/_retrieval/`
so stateless tool handlers can locate it later.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from filelock import FileLock

from agents.benchmark.schema import PublicTask
from agents.config import ETL_SCRATCH_ROOT, RetrievalIndexConfig

INDEXER_VERSION = "local-retrieval-v1"
_META_DIRNAME = "_retrieval"
_META_FILENAME = "index_meta.json"
_BUILD_META_FILENAME = "build_meta.json"
_SQLITE_FILENAME = "index.sqlite"
_TEXT_EXTS = frozenset({".md", ".txt"})
_SCHEMA_FORMATS = frozenset({"csv", "json", "sqlite"})
_QUERY_TOKEN = re.compile(r"[\w]+", re.UNICODE)
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FIELD_COLON = re.compile(r"^\s*(?:[-*]\s*)?`?([^:`|]{1,80})`?\s*[:：]\s*(.+)$")
_MARKDOWN_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_TEXT_PREVIEW_CHARS = 700


@dataclass(frozen=True, slots=True)
class _SourceFile:
    label: str
    path: Path
    kind: str


def _metadata_dir_for_task_id(task_id: str) -> Path:
    return ETL_SCRATCH_ROOT / task_id / _META_DIRNAME


def _metadata_path_for_task_id(task_id: str) -> Path:
    return _metadata_dir_for_task_id(task_id) / _META_FILENAME


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload: Any = json.loads(path.read_text())
    except Exception:
        return None
    return cast(dict[str, Any], payload) if isinstance(payload, dict) else None


def load_task_index_metadata_by_id(task_id: str) -> dict[str, Any] | None:
    """Return the retrieval sidecar metadata for a task, if present and readable."""
    return _read_json(_metadata_path_for_task_id(task_id))


def load_task_index_metadata(task: PublicTask) -> dict[str, Any] | None:
    """Return retrieval sidecar metadata for `task`."""
    return load_task_index_metadata_by_id(task.task_id)


def _write_disabled_metadata(task: PublicTask) -> None:
    _write_json(
        _metadata_path_for_task_id(task.task_id),
        {
            "enabled": False,
            "task_id": task.task_id,
            "indexer_version": INDEXER_VERSION,
            "updated_at": datetime.now(UTC).isoformat(),
        },
    )


def _write_enabled_metadata(task: PublicTask, payload: dict[str, Any]) -> None:
    sidecar = {
        "enabled": True,
        "task_id": task.task_id,
        "indexer_version": INDEXER_VERSION,
        "updated_at": datetime.now(UTC).isoformat(),
        **payload,
    }
    _write_json(_metadata_path_for_task_id(task.task_id), sidecar)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(task: PublicTask) -> list[_SourceFile]:
    sources: list[_SourceFile] = []
    context_root = task.context_dir.resolve()
    for path in sorted(context_root.rglob("*")):
        if path.is_file():
            sources.append(
                _SourceFile(
                    label=path.relative_to(context_root).as_posix(),
                    path=path,
                    kind="context",
                )
            )

    etl_dir = ETL_SCRATCH_ROOT / task.task_id / "_etl"
    if etl_dir.is_dir():
        for path in sorted(etl_dir.glob("*.csv")):
            sources.append(_SourceFile(label=path.as_posix(), path=path, kind="etl_csv"))
    return sources


def _manifest_for_sources(task: PublicTask, sources: list[_SourceFile]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for source in sources:
        stat = source.path.stat()
        entries.append(
            {
                "label": source.label,
                "kind": source.kind,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _file_sha256(source.path),
            }
        )
    return {"task_id": task.task_id, "indexer_version": INDEXER_VERSION, "sources": entries}


def _cache_key_for_manifest(manifest: dict[str, Any]) -> str:
    raw = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _preview_text(text: str, limit: int = _TEXT_PREVIEW_CHARS) -> str:
    normalized = _normalize_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _insert_chunk(
    conn: sqlite3.Connection,
    *,
    path: str,
    start_line: int,
    end_line: int,
    heading: str,
    text: str,
    chunk_kind: str,
) -> None:
    stripped = text.strip()
    if not stripped:
        return
    conn.execute(
        """
        INSERT INTO chunks_fts(path, start_line, end_line, heading, text, chunk_kind)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (path, start_line, end_line, heading, stripped, chunk_kind),
    )


def _split_long_paragraph(
    lines: list[str],
    *,
    path: str,
    start_line: int,
    heading: str,
    chunk_kind: str,
    max_chunk_chars: int,
    overlap_lines: int,
    conn: sqlite3.Connection,
) -> int:
    emitted = 0
    cursor = 0
    while cursor < len(lines):
        window: list[str] = []
        char_count = 0
        line_index = cursor
        while line_index < len(lines):
            next_line = lines[line_index]
            if window and char_count + len(next_line) + 1 > max_chunk_chars:
                break
            window.append(next_line)
            char_count += len(next_line) + 1
            line_index += 1
        if not window:
            window = [lines[cursor]]
            line_index = cursor + 1
        window_start = start_line + cursor
        window_end = start_line + line_index - 1
        _insert_chunk(
            conn,
            path=path,
            start_line=window_start,
            end_line=window_end,
            heading=heading,
            text="\n".join(window),
            chunk_kind=chunk_kind,
        )
        emitted += 1
        if line_index >= len(lines):
            break
        cursor = max(line_index - overlap_lines, cursor + 1)
    return emitted


def _index_text_file(
    conn: sqlite3.Connection,
    *,
    source: _SourceFile,
    max_chunk_chars: int,
    overlap_lines: int,
) -> int:
    try:
        text = source.path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = source.path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    heading = ""
    paragraph: list[str] = []
    paragraph_start = 1
    emitted = 0
    chunk_kind = "knowledge" if source.path.name.lower() == "knowledge.md" else "document"

    def flush(end_line: int) -> None:
        nonlocal emitted, paragraph
        if not paragraph:
            return
        paragraph_text = "\n".join(paragraph).strip()
        if not paragraph_text:
            paragraph = []
            return
        if len(paragraph_text) <= max_chunk_chars:
            _insert_chunk(
                conn,
                path=source.label,
                start_line=paragraph_start,
                end_line=end_line,
                heading=heading,
                text=paragraph_text,
                chunk_kind=chunk_kind,
            )
            emitted += 1
        else:
            emitted += _split_long_paragraph(
                paragraph,
                path=source.label,
                start_line=paragraph_start,
                heading=heading,
                chunk_kind=chunk_kind,
                max_chunk_chars=max_chunk_chars,
                overlap_lines=overlap_lines,
                conn=conn,
            )
        paragraph = []

    for index, line in enumerate(lines, start=1):
        heading_match = _HEADING.match(line)
        if heading_match:
            flush(index - 1)
            heading = heading_match.group(2).strip()
            paragraph_start = index
            paragraph = [line]
            flush(index)
            paragraph_start = index + 1
            continue
        if not line.strip():
            flush(index - 1)
            paragraph_start = index + 1
            continue
        if not paragraph:
            paragraph_start = index
        paragraph.append(line)
    flush(len(lines))
    return emitted


def _insert_field(
    conn: sqlite3.Connection,
    *,
    path: str,
    source_kind: str,
    canonical: str,
    alias: str,
    text: str,
) -> None:
    if not text.strip():
        return
    conn.execute(
        """
        INSERT INTO fields_fts(path, source_kind, canonical, alias, text)
        VALUES (?, ?, ?, ?, ?)
        """,
        (path, source_kind, canonical.strip(), alias.strip(), text.strip()),
    )


def _clean_markdown_cell(text: str) -> str:
    return text.strip().strip("|").strip().strip("`").strip()


def _index_knowledge_fields(conn: sqlite3.Connection, source: _SourceFile) -> int:
    try:
        lines = source.path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = source.path.read_text(encoding="utf-8", errors="replace").splitlines()
    inserted = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or _MARKDOWN_TABLE_SEPARATOR.match(stripped):
            continue
        if "|" in stripped:
            cells = [_clean_markdown_cell(cell) for cell in stripped.strip("|").split("|")]
            cells = [cell for cell in cells if cell]
            if len(cells) >= 2:
                canonical = cells[0][:80]
                _insert_field(
                    conn,
                    path=source.label,
                    source_kind="knowledge",
                    canonical=canonical,
                    alias=canonical,
                    text=" | ".join(cells),
                )
                inserted += 1
            continue
        match = _FIELD_COLON.match(stripped)
        if match:
            canonical = _clean_markdown_cell(match.group(1))[:80]
            _insert_field(
                conn,
                path=source.label,
                source_kind="knowledge",
                canonical=canonical,
                alias=canonical,
                text=stripped,
            )
            inserted += 1
        elif any(token in stripped.lower() for token in ("normal", "abnormal", "unit", "code")):
            canonical = " ".join(stripped.split()[:8])[:80]
            _insert_field(
                conn,
                path=source.label,
                source_kind="knowledge",
                canonical=canonical,
                alias=canonical,
                text=stripped,
            )
            inserted += 1
    return inserted


def _schema_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _index_csv_schema(conn: sqlite3.Connection, path: str, schema: dict[str, Any]) -> int:
    columns = schema.get("columns")
    dtypes = schema.get("dtypes")
    profile = schema.get("profile")
    if not isinstance(columns, list):
        return 0
    column_values = cast(list[Any], columns)
    dtype_list = cast(list[Any], dtypes) if isinstance(dtypes, list) else []
    profile_dict = cast(dict[str, Any], profile) if isinstance(profile, dict) else {}
    inserted = 0
    for index, column in enumerate(column_values):
        if not isinstance(column, str):
            continue
        dtype = str(dtype_list[index]) if index < len(dtype_list) else "unknown"
        column_profile = profile_dict.get(column, {})
        text = f"CSV {path} column {column} dtype {dtype}. {_schema_text(column_profile)}"
        _insert_field(
            conn,
            path=path,
            source_kind="csv_column",
            canonical=column,
            alias=column,
            text=text,
        )
        inserted += 1
    return inserted


def _iter_sqlite_columns(schema: dict[str, Any]) -> list[tuple[str, str]]:
    tables = schema.get("tables")
    pairs: list[tuple[str, str]] = []
    if isinstance(tables, dict):
        table_mapping = cast(dict[Any, Any], tables)
        for table_name, table_schema in table_mapping.items():
            if not isinstance(table_name, str) or not isinstance(table_schema, dict):
                continue
            table_schema_dict = cast(dict[str, Any], table_schema)
            columns = table_schema_dict.get("columns")
            if isinstance(columns, list):
                for column in cast(list[Any], columns):
                    if isinstance(column, str):
                        pairs.append((table_name, column))
    elif isinstance(tables, list):
        table_list = cast(list[Any], tables)
        for table in table_list:
            if not isinstance(table, dict):
                continue
            table_dict = cast(dict[str, Any], table)
            table_name = table_dict.get("name")
            columns = table_dict.get("columns")
            if isinstance(table_name, str) and isinstance(columns, list):
                for column in cast(list[Any], columns):
                    if isinstance(column, str):
                        pairs.append((table_name, column))
    return pairs


def _index_sqlite_schema(conn: sqlite3.Connection, path: str, schema: dict[str, Any]) -> int:
    inserted = 0
    for table, column in _iter_sqlite_columns(schema):
        text = f"SQLite {path} table {table} column {column}. Full name {table}.{column}."
        _insert_field(
            conn,
            path=path,
            source_kind="sqlite_column",
            canonical=f"{table}.{column}",
            alias=column,
            text=text,
        )
        inserted += 1
    return inserted


def _index_json_schema(conn: sqlite3.Connection, path: str, schema: dict[str, Any]) -> int:
    keys = schema.get("keys")
    if not isinstance(keys, list):
        return 0
    inserted = 0
    for key in cast(list[Any], keys):
        if not isinstance(key, str):
            continue
        text = f"JSON {path} key {key}. Schema: {_schema_text(schema.get('value_preview', {}))}"
        _insert_field(
            conn,
            path=path,
            source_kind="json_key",
            canonical=key,
            alias=key,
            text=text,
        )
        inserted += 1
    return inserted


def _index_schema_summaries(conn: sqlite3.Connection, task: PublicTask) -> tuple[int, int]:
    from agents.tools.filesystem import inspect_context_files

    summary = inspect_context_files(task)
    files = summary.get("files", [])
    chunk_count = 0
    field_count = 0
    if not isinstance(files, list):
        return 0, 0
    for item in cast(list[Any], files):
        if not isinstance(item, dict):
            continue
        item_dict = cast(dict[str, Any], item)
        path = item_dict.get("path")
        fmt = item_dict.get("format")
        schema = item_dict.get("schema")
        if not isinstance(path, str) or not isinstance(fmt, str) or not isinstance(schema, dict):
            continue
        schema_dict = cast(dict[str, Any], schema)
        if fmt not in _SCHEMA_FORMATS:
            continue
        _insert_chunk(
            conn,
            path=path,
            start_line=1,
            end_line=1,
            heading=f"{fmt} schema",
            text=f"{path} {fmt} schema: {_schema_text(schema_dict)[:4000]}",
            chunk_kind="schema_summary",
        )
        chunk_count += 1
        if fmt == "csv":
            field_count += _index_csv_schema(conn, path, schema_dict)
        elif fmt == "sqlite":
            field_count += _index_sqlite_schema(conn, path, schema_dict)
        elif fmt == "json":
            field_count += _index_json_schema(conn, path, schema_dict)
    return chunk_count, field_count


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE documents(
            path TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            size INTEGER NOT NULL,
            sha256 TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            path UNINDEXED,
            start_line UNINDEXED,
            end_line UNINDEXED,
            heading,
            text,
            chunk_kind UNINDEXED
        );
        CREATE VIRTUAL TABLE fields_fts USING fts5(
            path UNINDEXED,
            source_kind UNINDEXED,
            canonical,
            alias,
            text
        );
        """
    )


def _build_index(
    *,
    index_path: Path,
    task: PublicTask,
    sources: list[_SourceFile],
    manifest: dict[str, Any],
    config: RetrievalIndexConfig,
) -> dict[str, Any]:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        index_path.unlink()
    conn = sqlite3.connect(index_path)
    try:
        _create_schema(conn)
        doc_count = 0
        chunk_count = 0
        field_count = 0
        source_by_label = {source.label: source for source in sources}
        manifest_sources = cast(list[dict[str, Any]], manifest["sources"])
        for item in manifest_sources:
            label = str(item["label"])
            source = source_by_label[label]
            conn.execute(
                "INSERT INTO documents(path, kind, size, sha256) VALUES (?, ?, ?, ?)",
                (label, source.kind, int(item["size"]), str(item["sha256"])),
            )
            doc_count += 1
            if source.path.suffix.lower() in _TEXT_EXTS:
                chunk_count += _index_text_file(
                    conn,
                    source=source,
                    max_chunk_chars=config.max_chunk_chars,
                    overlap_lines=config.chunk_overlap_lines,
                )
                if source.path.name.lower() == "knowledge.md":
                    field_count += _index_knowledge_fields(conn, source)
        schema_chunks, schema_fields = _index_schema_summaries(conn, task)
        chunk_count += schema_chunks
        field_count += schema_fields
        conn.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("indexer_version", INDEXER_VERSION),
                ("task_id", task.task_id),
                ("manifest", json.dumps(manifest, ensure_ascii=False, sort_keys=True)),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "enabled": True,
        "built": True,
        "cache_hit": False,
        "task_id": task.task_id,
        "indexer_version": INDEXER_VERSION,
        "index_path": str(index_path),
        "cache_key": _cache_key_for_manifest(manifest),
        "doc_count": doc_count,
        "chunk_count": chunk_count,
        "field_count": field_count,
        "max_results": config.max_results,
    }


def build_or_load_task_index(task: PublicTask, config: RetrievalIndexConfig) -> dict[str, Any]:
    """Build or reuse the local retrieval index for a task.

    The returned dict is safe to write into summary.json and tracing metadata.
    """
    sources = _source_files(task)
    manifest = _manifest_for_sources(task, sources)
    cache_key = _cache_key_for_manifest(manifest)
    index_dir = config.cache_dir / task.task_id / cache_key[:16]
    index_path = index_dir / _SQLITE_FILENAME
    build_meta_path = index_dir / _BUILD_META_FILENAME
    lock_path = config.cache_dir / task.task_id / ".lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock_path)):
        cached = _read_json(build_meta_path)
        if index_path.exists() and cached and cached.get("cache_key") == cache_key:
            payload = dict(cached)
            payload["cache_hit"] = True
            payload["built"] = True
            payload["max_results"] = config.max_results
            _write_enabled_metadata(task, payload)
            return payload

        payload = _build_index(
            index_path=index_path,
            task=task,
            sources=sources,
            manifest=manifest,
            config=config,
        )
        _write_json(build_meta_path, payload)
        _write_enabled_metadata(task, payload)
        return payload


def prepare_task_retrieval_index(
    task: PublicTask, config: RetrievalIndexConfig
) -> dict[str, Any] | None:
    """Prepare retrieval metadata for a task, swallowing failures into metadata.

    Disabled retrieval writes a disabled sidecar so stale indexes from earlier runs are not
    accidentally used by tools in a default-off run.
    """
    if not config.enabled:
        _write_disabled_metadata(task)
        return None
    try:
        return build_or_load_task_index(task, config)
    except Exception as exc:
        payload = {
            "enabled": True,
            "built": False,
            "cache_hit": False,
            "task_id": task.task_id,
            "indexer_version": INDEXER_VERSION,
            "error": str(exc),
            "max_results": config.max_results,
        }
        _write_enabled_metadata(task, payload)
        return payload


def _metadata_error(task: PublicTask) -> dict[str, Any] | None:
    metadata = load_task_index_metadata(task)
    if metadata is None:
        return {
            "enabled": False,
            "error": (
                "Local retrieval index metadata is missing. Run with "
                "run.retrieval_index.enabled=true and call inspect_files first."
            ),
        }
    if not metadata.get("enabled"):
        return {"enabled": False, "error": "Local retrieval index is disabled for this run."}
    if not metadata.get("built"):
        return {
            "enabled": True,
            "error": f"Local retrieval index is unavailable: {metadata.get('error', 'build failed')}",
        }
    index_path = metadata.get("index_path")
    if not isinstance(index_path, str) or not Path(index_path).exists():
        return {"enabled": True, "error": "Local retrieval index file is missing."}
    return None


def _query_to_fts(query: str) -> str:
    tokens = [token.lower() for token in _QUERY_TOKEN.findall(query)]
    unique_tokens = list(dict.fromkeys(tokens))
    if not unique_tokens:
        raise ValueError("query must contain at least one searchable word")
    return " OR ".join(f'"{token}"' for token in unique_tokens)


def _resolve_limit(metadata: dict[str, Any], requested: int | None) -> int:
    configured = metadata.get("max_results")
    max_results = configured if isinstance(configured, int) and configured >= 1 else 8
    if requested is None:
        return max_results
    return max(1, min(requested, max_results))


def search_context(
    task: PublicTask,
    *,
    query: str,
    path: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Search text/schema chunks in the task-local retrieval index."""
    error = _metadata_error(task)
    if error is not None:
        return error
    metadata = cast(dict[str, Any], load_task_index_metadata(task))
    index_path = Path(cast(str, metadata["index_path"]))
    fts_query = _query_to_fts(query)
    result_limit = _resolve_limit(metadata, limit)

    conn = sqlite3.connect(index_path)
    conn.row_factory = sqlite3.Row
    try:
        params: list[Any] = [fts_query]
        where = "chunks_fts MATCH ?"
        if path:
            where += " AND path = ?"
            params.append(path)
        params.append(result_limit)
        rows = conn.execute(
            f"""
            SELECT
                path,
                start_line,
                end_line,
                heading,
                chunk_kind,
                snippet(chunks_fts, 4, '[', ']', ' ... ', 24) AS snippet,
                text,
                bm25(chunks_fts) AS rank
            FROM chunks_fts
            WHERE {where}
            ORDER BY rank
            LIMIT ?
            """,
            params,
        ).fetchall()
    finally:
        conn.close()

    results = [
        {
            "path": row["path"],
            "start_line": row["start_line"],
            "end_line": row["end_line"],
            "heading": row["heading"],
            "chunk_kind": row["chunk_kind"],
            "score": -float(row["rank"]),
            "preview": _preview_text(row["snippet"] or row["text"]),
        }
        for row in rows
    ]
    return {
        "enabled": True,
        "query": query,
        "path_filter": path,
        "result_count": len(results),
        "results": results,
        "index": {
            "path": str(index_path),
            "cache_key": metadata.get("cache_key"),
        },
    }


def lookup_field(task: PublicTask, *, term: str, limit: int | None = None) -> dict[str, Any]:
    """Search schema and knowledge glossary entries in the task-local retrieval index."""
    error = _metadata_error(task)
    if error is not None:
        return error
    metadata = cast(dict[str, Any], load_task_index_metadata(task))
    index_path = Path(cast(str, metadata["index_path"]))
    fts_query = _query_to_fts(term)
    result_limit = _resolve_limit(metadata, limit)

    conn = sqlite3.connect(index_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                path,
                source_kind,
                canonical,
                alias,
                snippet(fields_fts, 4, '[', ']', ' ... ', 24) AS snippet,
                text,
                bm25(fields_fts) AS rank
            FROM fields_fts
            WHERE fields_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, result_limit),
        ).fetchall()
    finally:
        conn.close()

    results = [
        {
            "path": row["path"],
            "source_kind": row["source_kind"],
            "canonical": row["canonical"],
            "alias": row["alias"],
            "score": -float(row["rank"]),
            "preview": _preview_text(row["snippet"] or row["text"]),
        }
        for row in rows
    ]
    return {
        "enabled": True,
        "term": term,
        "result_count": len(results),
        "results": results,
        "index": {
            "path": str(index_path),
            "cache_key": metadata.get("cache_key"),
        },
    }
