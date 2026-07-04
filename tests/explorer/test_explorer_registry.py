"""Explorer registry：video_tool 注入开关 + report 运行时校验 + 定向模式字段默认。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.explorer.registry import ReportInput, create_explorer_tool_registry
from agents.tools.registry import FunctionTool, ToolExecutionResult


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="q"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _stub_video_tool() -> FunctionTool:
    return FunctionTool(
        name="explore_video",
        description="stub",
        json_schema=None,
        handler=lambda _t, _a: ToolExecutionResult(ok=True, content={}),
    )


def test_video_tool_absent_by_default() -> None:
    registry = create_explorer_tool_registry()
    assert "explore_video" not in registry.definitions


def test_video_tool_injected_when_provided() -> None:
    registry = create_explorer_tool_registry(video_tool=_stub_video_tool())
    assert "explore_video" in registry.definitions
    assert registry.definitions["report"].is_terminal is True


def test_execute_context_sql_description_matches_local_index_contract() -> None:
    registry = create_explorer_tool_registry()
    description = registry.definitions["execute_context_sql"].description.lower()

    assert "virtual-context sqlite database copy" in description
    assert "create index if not exists" in description
    assert "read-only sql query" not in description


def test_preview_file_description_matches_sqlite_row_count_shape() -> None:
    registry = create_explorer_tool_registry()
    description = registry.definitions["preview_file"].description

    assert "{name, create_sql, row_count}" in description


def test_report_now_validated_at_registry_boundary(tmp_path: Path) -> None:
    registry = create_explorer_tool_registry()
    assert registry.definitions["report"].input_model is ReportInput
    with pytest.raises(ValueError, match="report: extra field"):
        registry.execute(_make_task(tmp_path), "report", {"bogus_field": 1})


def test_report_files_and_schema_map_default_empty(tmp_path: Path) -> None:
    # 定向模式允许精简报告：files / schema_map 不再必填
    registry = create_explorer_tool_registry()
    result = registry.execute(_make_task(tmp_path), "report", {"knowledge": {"answer_hint": "x"}})
    assert result.ok is True and result.is_terminal is True


@pytest.mark.parametrize("field", ["schema_map", "knowledge", "value_samples"])
def test_report_rejects_scalar_dict_fields(tmp_path: Path, field: str) -> None:
    registry = create_explorer_tool_registry()
    with pytest.raises(ValueError, match=rf"report: field '{field}'"):
        registry.execute(_make_task(tmp_path), "report", {field: "not a dict"})


def test_report_value_samples_preserve_scalar_types(tmp_path: Path) -> None:
    registry = create_explorer_tool_registry()
    result = registry.execute(
        _make_task(tmp_path),
        "report",
        {
            "value_samples": {
                "db/sub_db.sqlite.mf_fmretscaleanalysis.TotalAUM": [0.171053, 405.178231],
                "csv/foo.category": ["本科", "硕士"],
                "csv/foo.count": [1, 2],
            }
        },
    )

    assert result.ok is True and result.is_terminal is True
    findings = json.loads(result.content["findings"])
    assert findings["value_samples"] == {
        "db/sub_db.sqlite.mf_fmretscaleanalysis.TotalAUM": [0.171053, 405.178231],
        "csv/foo.category": ["本科", "硕士"],
        "csv/foo.count": [1, 2],
    }
