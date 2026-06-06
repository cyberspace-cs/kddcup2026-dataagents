"""Tests for _sanity_check_answer."""

from __future__ import annotations

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.tools.handlers import (
    _sanity_check_answer,
    handle_answer,
)
from agents.tools.inputs import AnswerInput


def _make_task(tmp_path):
    task_dir = tmp_path / "task_v"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_v", difficulty="easy", question="What are the names?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


# --- Unit tests on _sanity_check_answer ---


def test_zero_row_warns() -> None:
    warnings = _sanity_check_answer(["col"], [], "What is the value?")
    assert any("0 rows" in w for w in warnings)


def test_id_column_warns() -> None:
    warnings = _sanity_check_answer(
        ["id", "name"],
        [[1, "Alice"], [2, "Bob"], [3, "Carol"]],
        "What are the names?",
    )
    assert any("id" in w.lower() and "sequential" in w.lower() for w in warnings)


def test_camelcase_id_column_warns() -> None:
    warnings = _sanity_check_answer(
        ["CustomerId", "name"],
        [[1, "Alice"], [2, "Bob"], [3, "Carol"]],
        "What are the names?",
    )
    assert any("CustomerId" in w and "sequential" in w.lower() for w in warnings)


def test_uppercase_camelcase_id_warns() -> None:
    warnings = _sanity_check_answer(
        ["PostID", "title"],
        [[1, "a"], [2, "b"], [3, "c"]],
        "What are the titles?",
    )
    assert any("PostID" in w and "sequential" in w.lower() for w in warnings)


def test_id_column_no_warn_when_question_asks_for_id() -> None:
    warnings = _sanity_check_answer(
        ["id", "name"],
        [[1, "Alice"], [2, "Bob"], [3, "Carol"]],
        "What are the IDs and names?",
    )
    assert not any("sequential" in w.lower() for w in warnings)


def test_non_consecutive_id_column_no_warns() -> None:
    warnings = _sanity_check_answer(
        ["id", "name"],
        [[1, "Alice"], [2, "Bob"], [4, "Carol"]],
        "What are the names?",
    )
    assert not any("sequential" in w.lower() for w in warnings)


def test_all_same_value_warns() -> None:
    warnings = _sanity_check_answer(
        ["status", "name"],
        [["active", "Alice"], ["active", "Bob"]],
        "Who are the users?",
    )
    assert any("same value" in w.lower() for w in warnings)


def test_clean_answer_no_warnings() -> None:
    warnings = _sanity_check_answer(
        ["name", "score"],
        [["Alice", 90], ["Bob", 85]],
        "What are the scores?",
    )
    assert warnings == []


# --- Regression: handle_answer no longer returns answer_warnings ---


def test_handle_answer_no_answer_warnings(tmp_path) -> None:
    task = _make_task(tmp_path)
    args = AnswerInput.model_validate(
        {"columns": ["id", "name"], "rows": [[1, "Alice"], [2, "Bob"], [3, "Carol"]]}
    )
    result = handle_answer(task, args)
    assert result.ok is True
    assert result.is_terminal is True
    assert "answer_warnings" not in result.content
