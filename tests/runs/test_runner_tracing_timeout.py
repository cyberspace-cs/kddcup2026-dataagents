"""Parent-process tracing cleanup for subprocess timeout paths."""

from __future__ import annotations

from pathlib import Path
from queue import Empty
from typing import Any

from agents.config import AgentConfig, AppConfig, RunConfig, TracingConfig
from agents.runs import subprocess as runs_subprocess
from agents.tracing.store import SQLiteTraceStore


class _EmptyQueue:
    def get(self, timeout: float) -> dict[str, Any]:
        raise Empty

    def get_nowait(self) -> dict[str, Any]:
        raise Empty


class _TimeoutProcess:
    exitcode: int | None = None

    def __init__(self, target: Any, args: tuple[Any, ...]) -> None:
        self._alive = False

    def start(self) -> None:
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self._alive = False
        self.exitcode = -15

    def kill(self) -> None:
        self._alive = False
        self.exitcode = -9

    def join(self, timeout: float | None = None) -> None:
        pass


def _make_running_trace(db_path: Path) -> None:
    store = SQLiteTraceStore(db_path)
    store.upsert_trace(
        {
            "object": "trace",
            "id": "trace-timeout",
            "run_id": "run-1",
            "task_id": "task_1",
            "name": "ReActAgent",
            "started_at": "2026-01-01T00:00:00+00:00",
            "ended_at": None,
            "metadata": {},
        }
    )
    store.close()


def test_timeout_marks_running_sqlite_trace_error(tmp_path: Path, monkeypatch: Any) -> None:
    db_path = tmp_path / "traces.db"
    _make_running_trace(db_path)

    monkeypatch.setattr(runs_subprocess._MP_SPAWN, "Queue", lambda: _EmptyQueue())
    monkeypatch.setattr(runs_subprocess._MP_SPAWN, "Process", _TimeoutProcess)
    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(runs_subprocess, "perf_counter", lambda: next(ticks))

    config = AppConfig(
        agent=AgentConfig(model="m", api_base="http://x", api_key="real-key"),
        run=RunConfig(run_id="run-1", task_timeout_seconds=1),
        tracing=TracingConfig(enabled=True, db_path=db_path),
    )

    payload = runs_subprocess.run_single_task_with_timeout(task_id="task_1", config=config)

    assert payload["failure_reason"] == "Task timed out after 1 seconds."
    store = SQLiteTraceStore(db_path)
    result = store.get_trace_with_spans("trace-timeout")
    store.close()
    assert result is not None
    trace = result["trace"]
    assert trace["status"] == "error"
    assert trace["ended_at"] is not None
    assert trace["metadata"]["failure_reason"] == "Task timed out after 1 seconds."
