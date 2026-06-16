from __future__ import annotations

from pathlib import Path

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.tools import context as context_mod


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "input" / "task_ctx"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_ctx", difficulty="hard", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_build_virtual_context_hides_prose_and_links_etl_csv(tmp_path: Path, monkeypatch) -> None:
    task = _make_task(tmp_path)
    doc_dir = task.context_dir / "doc"
    doc_dir.mkdir()
    (task.context_dir / "knowledge.md").write_text("# Knowledge\n", encoding="utf-8")
    (doc_dir / "source.md").write_text("raw markdown", encoding="utf-8")
    (doc_dir / "source.pdf").write_bytes(b"%PDF-1.4\n")
    (task.context_dir / "records.csv").write_text("id,value\n1,a\n", encoding="utf-8")

    scratch_root = tmp_path / "scratch"
    monkeypatch.setattr(context_mod, "ETL_SCRATCH_ROOT", scratch_root)
    etl_dir = scratch_root / task.task_id / "_etl"
    etl_dir.mkdir(parents=True)
    (etl_dir / "source.csv").write_text("id,value\n1,a\n", encoding="utf-8")

    virtual_context = context_mod.build_virtual_context(task)

    assert (virtual_context / "knowledge.md").exists()
    assert (virtual_context / "records.csv").exists()
    assert (virtual_context / "csv" / "source.csv").exists()
    assert not (virtual_context / "doc" / "source.md").exists()
    assert not (virtual_context / "doc" / "source.pdf").exists()
