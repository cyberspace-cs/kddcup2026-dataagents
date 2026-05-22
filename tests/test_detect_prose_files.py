"""Tests for _detect_prose_files expansion (Change 4)."""

from __future__ import annotations

from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.etl.extractor import _MIN_LINES, _detect_prose_files


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_etl"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_etl", difficulty="easy", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _write_lines(path: Path, n: int) -> None:
    path.write_text("\n".join(f"line {i}" for i in range(n)))


def test_txt_in_doc_dir_detected(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    doc_dir = task.context_dir / "doc"
    doc_dir.mkdir()
    _write_lines(doc_dir / "data.txt", _MIN_LINES + 10)
    result = _detect_prose_files(task)
    assert any(p.name == "data.txt" for p in result)


def test_non_utf8_txt_does_not_crash(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    bad = task.context_dir / "bad.txt"
    bad.write_bytes((b"line\n" * (_MIN_LINES + 1)) + b"\xff")

    result = _detect_prose_files(task)

    assert bad in result


def test_md_in_context_root_detected(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    _write_lines(task.context_dir / "heroes.md", _MIN_LINES + 10)
    result = _detect_prose_files(task)
    assert any(p.name == "heroes.md" for p in result)


def test_knowledge_md_excluded_everywhere(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    doc_dir = task.context_dir / "doc"
    doc_dir.mkdir()
    _write_lines(task.context_dir / "knowledge.md", _MIN_LINES + 10)
    _write_lines(doc_dir / "knowledge.md", _MIN_LINES + 10)
    result = _detect_prose_files(task)
    assert not any(p.name.lower() == "knowledge.md" for p in result)


def test_short_file_excluded(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    _write_lines(task.context_dir / "short.md", _MIN_LINES - 1)
    result = _detect_prose_files(task)
    assert result == []


def test_no_duplicates_when_file_in_both(tmp_path: Path) -> None:
    """doc/ takes priority; same file should not appear twice."""
    task = _make_task(tmp_path)
    doc_dir = task.context_dir / "doc"
    doc_dir.mkdir()
    _write_lines(doc_dir / "data.md", _MIN_LINES + 10)
    result = _detect_prose_files(task)
    names = [p.name for p in result]
    assert names.count("data.md") == 1
