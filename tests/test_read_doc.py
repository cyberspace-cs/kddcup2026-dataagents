from __future__ import annotations

from pathlib import Path

import pytest

from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.filesystem import read_doc_preview
from data_agent_baseline.tools.registry import create_default_tool_registry


@pytest.fixture
def task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(
            task_id="task_1",
            difficulty="easy",
            question="Read the document.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def write_doc(task: PublicTask, name: str, lines: list[str]) -> None:
    (task.context_dir / name).write_text("\n".join(lines))


def numbered_lines(count: int) -> list[str]:
    return [f"line-{line_number}" for line_number in range(1, count + 1)]


def test_read_doc_returns_requested_line_page(task: PublicTask) -> None:
    write_doc(task, "knowledge.md", numbered_lines(100))

    result = read_doc_preview(task, "knowledge.md", start_line=21, line_count=10)

    assert result == {
        "path": "knowledge.md",
        "total_lines": 100,
        "returned_range": {"start_line": 21, "end_line": 30},
        "has_more": True,
        "keyword_match": None,
        "preview": "\n".join(f"line-{line_number}" for line_number in range(21, 31)),
    }


def test_read_doc_returns_last_page_without_has_more(task: PublicTask) -> None:
    write_doc(task, "knowledge.md", numbered_lines(100))

    result = read_doc_preview(task, "knowledge.md", start_line=95, line_count=20)

    assert result["returned_range"] == {"start_line": 95, "end_line": 100}
    assert result["has_more"] is False
    assert result["preview"] == "\n".join(f"line-{line_number}" for line_number in range(95, 101))


def test_read_doc_normalizes_line_arguments(task: PublicTask) -> None:
    write_doc(task, "knowledge.md", numbered_lines(400))

    result = read_doc_preview(task, "knowledge.md", start_line=-10, line_count=500)

    assert result["returned_range"] == {"start_line": 1, "end_line": 300}
    assert result["has_more"] is True
    assert result["preview"].splitlines()[0] == "line-1"
    assert result["preview"].splitlines()[-1] == "line-300"


def test_read_doc_keyword_match_returns_context_window(task: PublicTask) -> None:
    lines = numbered_lines(100)
    lines[49] = "line-50 special Thrombosis finding"
    write_doc(task, "knowledge.md", lines)

    result = read_doc_preview(
        task,
        "knowledge.md",
        start_line=1,
        line_count=9,
        keyword="thrombosis",
    )

    assert result["returned_range"] == {"start_line": 47, "end_line": 55}
    assert result["has_more"] is True
    assert result["keyword_match"] == {
        "keyword": "thrombosis",
        "found": True,
        "line": 50,
    }
    assert "line-50 special Thrombosis finding" in result["preview"]


def test_read_doc_keyword_search_starts_at_start_line(task: PublicTask) -> None:
    lines = numbered_lines(100)
    lines[19] = "line-20 Thrombosis early"
    lines[59] = "line-60 Thrombosis later"
    write_doc(task, "knowledge.md", lines)

    result = read_doc_preview(
        task,
        "knowledge.md",
        start_line=21,
        line_count=12,
        keyword="thrombosis",
    )

    assert result["keyword_match"] == {
        "keyword": "thrombosis",
        "found": True,
        "line": 60,
    }
    assert "line-20 Thrombosis early" not in result["preview"]
    assert "line-60 Thrombosis later" in result["preview"]


def test_read_doc_keyword_miss_returns_regular_page(task: PublicTask) -> None:
    write_doc(task, "knowledge.md", numbered_lines(100))

    result = read_doc_preview(
        task,
        "knowledge.md",
        start_line=21,
        line_count=10,
        keyword="Thrombosis",
    )

    assert result["returned_range"] == {"start_line": 21, "end_line": 30}
    assert result["has_more"] is True
    assert result["keyword_match"] == {
        "keyword": "Thrombosis",
        "found": False,
        "line": None,
    }
    assert result["preview"] == "\n".join(f"line-{line_number}" for line_number in range(21, 31))


def test_read_doc_empty_file_returns_stable_empty_result(task: PublicTask) -> None:
    write_doc(task, "empty.md", [])

    result = read_doc_preview(task, "empty.md")

    assert result == {
        "path": "empty.md",
        "total_lines": 0,
        "returned_range": None,
        "has_more": False,
        "keyword_match": None,
        "preview": "",
    }


def test_read_doc_rejects_path_escape(task: PublicTask) -> None:
    with pytest.raises(ValueError, match="Path escapes context dir"):
        read_doc_preview(task, "../secret.md")


def test_read_doc_registry_schema_uses_line_pagination() -> None:
    registry = create_default_tool_registry()

    schema = registry.definitions["read_doc"].input_schema

    assert "max_chars" not in schema
    assert schema == {
        "path": "relative/path/to/file.md",
        "start_line": 1,
        "line_count": 80,
        "keyword": "optional search text",
    }
