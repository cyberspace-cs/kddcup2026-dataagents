"""Lightweight per-file ETL tracing — phase records + JSON sidecar."""

import contextlib
import logging
import time
from collections.abc import Generator
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from agents.tracing.create import function_span

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ETLPhaseRecord:
    name: str
    started_at: float = 0.0
    ended_at: float = 0.0
    skipped: bool = False
    metrics: dict[str, object] = field(default_factory=lambda: dict[str, object]())
    decisions: dict[str, object] = field(default_factory=lambda: dict[str, object]())
    error: str | None = None
    llm_calls: int = 0
    input_size: int = 0
    output_size: int = 0
    llm_exchanges: list[dict[str, str]] = field(default_factory=lambda: list[dict[str, str]]())


@dataclass
class ETLFileTrace:
    task_id: str
    stem: str
    phases: list[ETLPhaseRecord] = field(default_factory=lambda: list[ETLPhaseRecord]())
    start_time: float = field(default_factory=time.monotonic)
    start_wall: datetime = field(default_factory=lambda: datetime.now(UTC))


_etl_file_trace: ContextVar[ETLFileTrace | None] = ContextVar("_etl_file_trace", default=None)


@contextlib.contextmanager
def etl_phase(name: str, **initial_metrics: Any) -> Generator[ETLPhaseRecord, None, None]:
    """Context manager that records one ETL phase.

    - Opens a ``function_span(f"etl/{name}")`` so the phase nests in the
      span tree (traces.db).
    - Creates an ``ETLPhaseRecord``, yields it so the caller can set
      metrics / decisions after the wrapped call returns.
    - On exit: records timing, appends to the current ``ETLFileTrace``.
    - Gracefully no-ops when ``_etl_file_trace`` is None.
    """
    record = ETLPhaseRecord(name=name, metrics=dict(initial_metrics))
    if "input_size" in initial_metrics:
        record.input_size = initial_metrics["input_size"]
    record.started_at = time.monotonic()

    trace = _etl_file_trace.get(None)
    if trace is not None:
        trace.phases.append(record)

    with function_span(f"etl/{name}"):
        try:
            yield record
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.ended_at = time.monotonic()


def record_llm_exchange(
    phase_name: str,
    prompt: str,
    response: str,
    label: str = "",
) -> None:
    """Append an LLM prompt/response pair to the current phase or a named phase."""
    trace = _etl_file_trace.get(None)
    if trace is None:
        return
    for p in reversed(trace.phases):
        if p.name == phase_name and p.ended_at == 0.0:
            p.llm_exchanges.append({"label": label, "prompt": prompt, "response": response})
            return
    # No open phase with that name — attach to the last open phase
    for p in reversed(trace.phases):
        if p.ended_at == 0.0:
            p.llm_exchanges.append({"label": label, "prompt": prompt, "response": response})
            return


def get_etl_trace() -> ETLFileTrace | None:
    return _etl_file_trace.get(None)


def set_etl_trace(trace: ETLFileTrace) -> Any:
    return _etl_file_trace.set(trace)


def reset_etl_trace(token: Any) -> None:
    _etl_file_trace.reset(token)


def serialize_trace(trace: ETLFileTrace) -> dict[str, Any]:
    end = time.monotonic()
    total_llm = sum(p.llm_calls for p in trace.phases)
    phases: list[dict[str, Any]] = []
    for p in trace.phases:
        rec: dict[str, Any] = {
            "name": p.name,
            "duration_ms": round((p.ended_at - p.started_at) * 1000),
            "skipped": p.skipped,
            "error": p.error,
            "llm_calls": p.llm_calls,
            "input_size": p.input_size,
            "output_size": p.output_size,
            "metrics": p.metrics,
            "decisions": p.decisions,
        }
        if p.llm_exchanges:
            rec["llm_exchanges"] = p.llm_exchanges
        phases.append(rec)

    return {
        "version": 1,
        "task_id": trace.task_id,
        "stem": trace.stem,
        "started_at": trace.start_wall.isoformat(),
        "duration_ms": round((end - trace.start_time) * 1000),
        "total_llm_calls": total_llm,
        "phases": phases,
    }
