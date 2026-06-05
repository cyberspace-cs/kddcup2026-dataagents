"""SQLiteTraceStore — read/write access to the traces database."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from data_agent_baseline.tracing.util import time_iso

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS traces (
    trace_id     TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL,
    task_id      TEXT NOT NULL,
    name         TEXT NOT NULL DEFAULT '',
    started_at   TEXT,
    ended_at     TEXT,
    status       TEXT NOT NULL DEFAULT 'running',
    total_tokens INTEGER DEFAULT 0,
    duration_ms  INTEGER,
    span_count   INTEGER DEFAULT 0,
    metadata_json TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_traces_run ON traces(run_id);
CREATE INDEX IF NOT EXISTS idx_traces_started ON traces(started_at DESC);

CREATE TABLE IF NOT EXISTS spans (
    span_id         TEXT PRIMARY KEY,
    trace_id        TEXT NOT NULL,
    parent_span_id  TEXT,
    kind            TEXT NOT NULL,
    name            TEXT NOT NULL DEFAULT '',
    started_at      TEXT,
    ended_at        TEXT,
    status          TEXT NOT NULL DEFAULT 'running',
    attributes_json TEXT DEFAULT '{}',
    task_id         TEXT NOT NULL DEFAULT '',
    run_id          TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_kind ON spans(kind);
CREATE INDEX IF NOT EXISTS idx_spans_started ON spans(started_at DESC);
"""


def _iso_to_epoch(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).timestamp()
    except (ValueError, TypeError):
        return None


def _compute_duration_ms(started: str | None, ended: str | None) -> int | None:
    s = _iso_to_epoch(started)
    e = _iso_to_epoch(ended)
    if s is not None and e is not None:
        return max(0, int((e - s) * 1000))
    return None


class SQLiteTraceStore:
    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def upsert_trace(self, data: dict[str, Any]) -> None:
        status = "completed" if data.get("ended_at") else "running"
        duration = _compute_duration_ms(data.get("started_at"), data.get("ended_at"))
        metadata_json = json.dumps(data.get("metadata", {}), ensure_ascii=False, default=str)
        with self._lock:
            self._conn.execute(
                """INSERT INTO traces (trace_id, run_id, task_id, name, started_at, ended_at,
                                       status, duration_ms, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(trace_id) DO UPDATE SET
                       ended_at = excluded.ended_at,
                       status = excluded.status,
                       duration_ms = excluded.duration_ms,
                       metadata_json = excluded.metadata_json
                """,
                (
                    data["id"],
                    data.get("run_id", ""),
                    data.get("task_id", ""),
                    data.get("name", ""),
                    data.get("started_at"),
                    data.get("ended_at"),
                    status,
                    duration,
                    metadata_json,
                ),
            )
            self._conn.commit()

    def upsert_span(self, data: dict[str, Any]) -> None:
        span_data = data.get("span_data", {})
        kind = span_data.get("type", "unknown")
        name = span_data.get("name", "") or kind
        status = (
            "error" if data.get("error") else ("completed" if data.get("ended_at") else "running")
        )
        attrs = {**span_data}
        if data.get("error"):
            attrs["error"] = data["error"]
        attrs_json = json.dumps(attrs, ensure_ascii=False, default=str)

        trace_id = data.get("trace_id", "")
        with self._lock:
            self._conn.execute(
                """INSERT INTO spans (span_id, trace_id, parent_span_id, kind, name,
                                      started_at, ended_at, status, attributes_json,
                                      task_id, run_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(span_id) DO UPDATE SET
                       name = excluded.name,
                       ended_at = excluded.ended_at,
                       status = excluded.status,
                       attributes_json = excluded.attributes_json
                """,
                (
                    data["id"],
                    trace_id,
                    data.get("parent_id"),
                    kind,
                    name,
                    data.get("started_at"),
                    data.get("ended_at"),
                    status,
                    attrs_json,
                    self._get_trace_task_id(trace_id),
                    self._get_trace_run_id(trace_id),
                ),
            )
            self._conn.commit()

    def _get_trace_task_id(self, trace_id: str) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT task_id FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        return row["task_id"] if row else ""

    def _get_trace_run_id(self, trace_id: str) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT run_id FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        return row["run_id"] if row else ""

    def finalize_trace(self, trace_id: str) -> None:
        with self._lock:
            row = self._conn.execute(
                """SELECT
                       COUNT(*) AS span_count,
                       COALESCE(SUM(
                           CASE WHEN kind = 'generation'
                                THEN CAST(json_extract(attributes_json, '$.usage.input_tokens') AS INTEGER)
                                     + CAST(json_extract(attributes_json, '$.usage.output_tokens') AS INTEGER)
                                ELSE 0
                           END
                       ), 0) AS total_tokens
                   FROM spans WHERE trace_id = ?""",
                (trace_id,),
            ).fetchone()
            if row:
                self._conn.execute(
                    "UPDATE traces SET span_count = ?, total_tokens = ? WHERE trace_id = ?",
                    (row["span_count"], row["total_tokens"], trace_id),
                )
                self._conn.commit()

    def mark_running_trace_error(self, *, run_id: str, task_id: str, failure_reason: str) -> int:
        """Mark in-flight traces for a failed task as errored.

        The parent process uses this when a traced child is terminated before
        its TraceCtxManager can run ``__exit__``.
        """
        ended_at = time_iso()
        with self._lock:
            rows = self._conn.execute(
                """SELECT trace_id, started_at, metadata_json FROM traces
                   WHERE run_id = ? AND task_id = ? AND status = 'running'""",
                (run_id, task_id),
            ).fetchall()
            trace_ids: list[str] = []
            for row in rows:
                trace_ids.append(row["trace_id"])
                try:
                    metadata = json.loads(row["metadata_json"] or "{}")
                except json.JSONDecodeError:
                    metadata = {}
                metadata["failure_reason"] = failure_reason
                self._conn.execute(
                    """UPDATE traces
                       SET ended_at = ?,
                           status = 'error',
                           duration_ms = ?,
                           metadata_json = ?
                       WHERE trace_id = ?""",
                    (
                        ended_at,
                        _compute_duration_ms(row["started_at"], ended_at),
                        json.dumps(metadata, ensure_ascii=False, default=str),
                        row["trace_id"],
                    ),
                )
            self._conn.commit()

        for trace_id in trace_ids:
            self.finalize_trace(trace_id)
        return len(trace_ids)

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT run_id,
                          COUNT(*) AS task_count,
                          SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS ok_count,
                          SUM(total_tokens) AS tokens,
                          AVG(duration_ms) AS avg_ms,
                          MIN(started_at) AS started
                   FROM traces
                   GROUP BY run_id
                   ORDER BY started DESC"""
            ).fetchall()
            return [dict(r) for r in rows]

    def list_traces(
        self,
        run_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "started_at",
        sort_order: str = "desc",
    ) -> dict[str, Any]:
        conditions: list[str] = []
        params: list[Any] = []
        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)
        if status:
            conditions.append("status = ?")
            params.append(status)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        allowed_sort = {"started_at", "ended_at", "duration_ms", "total_tokens", "task_id"}
        col = sort_by if sort_by in allowed_sort else "started_at"
        direction = "ASC" if sort_order.upper() == "ASC" else "DESC"

        with self._lock:
            total_row = self._conn.execute(
                f"SELECT COUNT(*) AS cnt FROM traces {where}", params
            ).fetchone()
            total = total_row["cnt"] if total_row else 0

            rows = self._conn.execute(
                f"SELECT * FROM traces {where} ORDER BY {col} {direction} LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            items: list[dict[str, Any]] = []
            for r in rows:
                item: dict[str, Any] = dict(r)
                item["metadata"] = json.loads(item.pop("metadata_json", "{}"))
                items.append(item)
            return {"items": items, "total": total, "has_more": offset + limit < total}

    def get_trace_with_spans(self, trace_id: str) -> dict[str, Any] | None:
        with self._lock:
            trace_row = self._conn.execute(
                "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
            if not trace_row:
                return None
            trace_dict = dict(trace_row)
            trace_dict["metadata"] = json.loads(trace_dict.pop("metadata_json", "{}"))

            span_rows = self._conn.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY started_at ASC",
                (trace_id,),
            ).fetchall()
            spans: list[dict[str, Any]] = []
            for r in span_rows:
                s: dict[str, Any] = dict(r)
                s["attributes"] = json.loads(s.pop("attributes_json", "{}"))
                spans.append(s)
            return {"trace": trace_dict, "spans": spans}

    def get_span(self, span_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM spans WHERE span_id = ?", (span_id,)).fetchone()
            if not row:
                return None
            result = dict(row)
            result["attributes"] = json.loads(result.pop("attributes_json", "{}"))
            return result

    def get_spans_after(
        self, iso_timestamp: str, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        with self._lock:
            if run_id:
                rows = self._conn.execute(
                    """SELECT * FROM spans
                       WHERE (started_at > ? OR ended_at > ?) AND run_id = ?
                       ORDER BY started_at ASC""",
                    (iso_timestamp, iso_timestamp, run_id),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT * FROM spans
                       WHERE started_at > ? OR ended_at > ?
                       ORDER BY started_at ASC""",
                    (iso_timestamp, iso_timestamp),
                ).fetchall()
            result: list[dict[str, Any]] = []
            for r in rows:
                s: dict[str, Any] = dict(r)
                s["attributes"] = json.loads(s.pop("attributes_json", "{}"))
                result.append(s)
            return result
