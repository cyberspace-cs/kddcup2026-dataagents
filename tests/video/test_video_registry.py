"""Video agent registry：仅 report 一个终止工具，带运行时校验与大小兜底。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.video.registry import VIDEO_REPORT_MAX_BYTES, create_video_tool_registry


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="q"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_registry_contains_only_terminal_report() -> None:
    registry = create_video_tool_registry()
    assert list(registry.definitions) == ["report"]
    assert registry.definitions["report"].is_terminal is True
    assert registry.definitions["report"].input_model is not None


def test_valid_report_roundtrips_findings(tmp_path: Path) -> None:
    registry = create_video_tool_registry()
    result = registry.execute(
        _make_task(tmp_path),
        "report",
        {
            "summary": "dashboard recording",
            "timeline": [{"time": "00:42", "event": "bar chart"}],
            "extracted_data": {
                "rules": {"filters": [{"field": "quarter", "op": "=", "value": "Q3"}]},
                "displayed_samples": {"q3_revenue": {"A": 10, "B": 20}},
            },
        },
    )
    assert result.ok is True and result.is_terminal is True
    findings = json.loads(result.content["findings"])
    assert findings["summary"] == "dashboard recording"
    assert findings["extracted_data"] == {
        "rules": {"filters": [{"field": "quarter", "op": "=", "value": "Q3"}]},
        "displayed_samples": {"q3_revenue": {"A": 10, "B": 20}},
    }
    assert findings["coverage"] == [] and findings["warnings"] == []


def test_missing_extracted_data_defaults_to_empty_contract(tmp_path: Path) -> None:
    registry = create_video_tool_registry()
    result = registry.execute(_make_task(tmp_path), "report", {"summary": "dashboard"})

    assert result.ok is True
    findings = json.loads(result.content["findings"])
    assert findings["extracted_data"] == {"rules": {}, "displayed_samples": {}}


def test_flat_extracted_data_rejected(tmp_path: Path) -> None:
    registry = create_video_tool_registry()
    with pytest.raises(
        ValueError,
        match=r"report: extra field 'extracted_data\.kpi' is not permitted",
    ):
        registry.execute(
            _make_task(tmp_path),
            "report",
            {"summary": "dashboard", "extracted_data": {"kpi": 42}},
        )


def test_missing_summary_raises_single_line_error(tmp_path: Path) -> None:
    registry = create_video_tool_registry()
    with pytest.raises(ValueError, match="report: field 'summary' is required"):
        registry.execute(_make_task(tmp_path), "report", {})


def test_oversized_report_rejected_not_terminal(tmp_path: Path) -> None:
    registry = create_video_tool_registry()
    big = "x" * (VIDEO_REPORT_MAX_BYTES + 1)
    result = registry.execute(
        _make_task(tmp_path),
        "report",
        {"summary": "s", "extracted_data": {"rules": {"blob": big}}},
    )
    assert result.ok is False
    assert "exceeds" in result.content["error"]
    assert result.is_terminal is False
