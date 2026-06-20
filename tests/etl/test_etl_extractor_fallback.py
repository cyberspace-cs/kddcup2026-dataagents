from __future__ import annotations

from pathlib import Path

import pytest

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.etl import extractor
from agents.llm import ModelMessage, ModelResponse


class NoFallbackAdapter:
    def complete(
        self, messages: list[ModelMessage], *, tools: object | None = None
    ) -> ModelResponse:
        raise AssertionError("ETL should not make any post-compression LLM CSV calls")


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_no_generated_code"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_no_generated_code", difficulty="easy", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_extract_prose_file_stops_when_deterministic_csv_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _make_task(tmp_path)
    monkeypatch.setattr(extractor, "ETL_SCRATCH_ROOT", tmp_path / "etl")
    monkeypatch.setattr(
        extractor,
        "compress_prose",
        lambda *_args, **_kwargs: ("id: 1 | value: source", None),
    )
    monkeypatch.setattr(extractor, "deterministic_kv_to_csv", lambda *_args, **_kwargs: None)

    prose_path = task.context_dir / "patient.md"
    prose_path.write_text("ID 1 has value source.", encoding="utf-8")

    result = extractor.extract_prose_file(
        NoFallbackAdapter(),
        prose_path,
        task,
        prose_path.read_text(encoding="utf-8"),
        schema_columns=["id", "value"],
        anchor_keys=["id"],
    )

    assert result is None
    assert not (tmp_path / "etl" / task.task_id / "_etl" / "patient.csv").exists()
