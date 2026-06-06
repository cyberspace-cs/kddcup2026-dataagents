"""PDF ETL edge-case tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.etl import extractor
from agents.llm import ModelMessage, ModelResponse


class ExplodingAdapter:
    def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        raise AssertionError("empty extracted text should not call the LLM")


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_pdf_etl"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_pdf_etl", difficulty="easy", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_extract_prose_file_returns_none_for_textless_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pymupdf

    task = _make_task(tmp_path)
    monkeypatch.setattr(extractor, "ETL_SCRATCH_ROOT", tmp_path / "etl")

    pdf_path = task.context_dir / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    result = extractor.extract_prose_file(ExplodingAdapter(), pdf_path, task, "")

    assert result is None
