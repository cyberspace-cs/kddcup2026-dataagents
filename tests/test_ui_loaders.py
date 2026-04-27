"""UI 加载层（`data_agent_baseline.ui.loaders`）的纯函数单测。

验证点：
- `list_runs`：枚举 + 按时间倒序 + 缺 summary.json 兜底
- `list_tasks_in_run`：summary.json 路径 + 子目录兜底
- `load_gold` / `load_prediction` / `load_trace`：缺失文件返回 None
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_agent_baseline.ui.loaders import (
    list_runs,
    list_tasks_in_run,
    load_gold,
    load_prediction,
    load_trace,
)


def _make_run(
    artifacts_dir: Path,
    run_id: str,
    *,
    summary: dict | None = None,
    tasks: list[dict] | None = None,
) -> Path:
    """便捷：在 artifacts/runs/<id>/ 下铺好 summary.json + 子任务目录。"""
    run_dir = artifacts_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    if summary is not None:
        (run_dir / "summary.json").write_text(json.dumps(summary))
    for task in tasks or []:
        task_dir = run_dir / task["task_id"]
        task_dir.mkdir(parents=True)
        if "trace" in task:
            (task_dir / "trace.json").write_text(json.dumps(task["trace"]))
        if "prediction" in task:
            pd.DataFrame(task["prediction"]).to_csv(task_dir / "prediction.csv", index=False)
    return run_dir


def test_list_runs_returns_empty_when_dir_missing(tmp_path: Path) -> None:
    """artifacts/runs/ 不存在时优雅降级为空列表。"""
    assert list_runs(tmp_path / "nope") == []


def test_list_runs_sorts_by_id_desc_and_picks_summary_counts(tmp_path: Path) -> None:
    """有 summary.json 时使用其中的统计；按 run_id 倒序。"""
    artifacts = tmp_path / "artifacts"
    _make_run(
        artifacts,
        "20260101T000000Z",
        summary={"task_count": 3, "succeeded_task_count": 2},
    )
    _make_run(
        artifacts,
        "20260202T000000Z",
        summary={"task_count": 5, "succeeded_task_count": 5},
    )
    _make_run(artifacts, "20260303T000000Z", summary=None)  # 无 summary

    runs = list_runs(artifacts)
    assert [r.run_id for r in runs] == [
        "20260303T000000Z",
        "20260202T000000Z",
        "20260101T000000Z",
    ]

    by_id = {r.run_id: r for r in runs}
    assert by_id["20260202T000000Z"].task_count == 5
    assert by_id["20260202T000000Z"].succeeded_count == 5
    # 缺 summary.json 时计数为 None
    assert by_id["20260303T000000Z"].task_count is None
    assert by_id["20260303T000000Z"].succeeded_count is None


def test_list_runs_corrupt_summary_falls_back_to_none(tmp_path: Path) -> None:
    """summary.json 损坏不应阻止 run 列表渲染——计数置 None。"""
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "runs" / "broken"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{ not valid json")

    runs = list_runs(artifacts)
    assert len(runs) == 1
    assert runs[0].task_count is None
    assert runs[0].succeeded_count is None


def test_list_tasks_in_run_uses_summary_payload(tmp_path: Path) -> None:
    """summary.json 中的 tasks 数组应直接转换为 TaskInfo。"""
    artifacts = tmp_path / "artifacts"
    run_dir = _make_run(
        artifacts,
        "r1",
        summary={
            "tasks": [
                {
                    "task_id": "task_1",
                    "task_output_dir": str(artifacts / "runs" / "r1" / "task_1"),
                    "succeeded": True,
                    "failure_reason": None,
                },
                {
                    "task_id": "task_2",
                    "task_output_dir": str(artifacts / "runs" / "r1" / "task_2"),
                    "succeeded": False,
                    "failure_reason": "Task timed out.",
                },
            ],
        },
        tasks=[{"task_id": "task_1"}, {"task_id": "task_2"}],
    )

    tasks = list_tasks_in_run(run_dir)
    assert [t.task_id for t in tasks] == ["task_1", "task_2"]
    assert tasks[0].succeeded is True
    assert tasks[1].succeeded is False
    assert tasks[1].failure_reason == "Task timed out."


def test_list_tasks_in_run_falls_back_to_subdirs(tmp_path: Path) -> None:
    """summary.json 缺失时通过扫描子目录推断任务列表。"""
    artifacts = tmp_path / "artifacts"
    run_dir = _make_run(
        artifacts,
        "r2",
        summary=None,
        tasks=[
            {
                "task_id": "task_a",
                "trace": {"task_id": "task_a", "succeeded": True, "failure_reason": None},
            },
            {
                "task_id": "task_b",
                "trace": {"task_id": "task_b", "succeeded": False, "failure_reason": "boom"},
            },
        ],
    )

    tasks = list_tasks_in_run(run_dir)
    by_id = {t.task_id: t for t in tasks}
    assert by_id["task_a"].succeeded is True
    assert by_id["task_b"].succeeded is False
    assert by_id["task_b"].failure_reason == "boom"


def test_load_gold_missing_returns_none(tmp_path: Path) -> None:
    public_data = tmp_path / "public"
    public_data.mkdir()
    assert load_gold("task_x", public_data) is None


def test_load_gold_returns_dataframe(tmp_path: Path) -> None:
    public_data = tmp_path / "public"
    out_dir = public_data / "output" / "task_42"
    out_dir.mkdir(parents=True)
    pd.DataFrame({"v": [1, 2, 3]}).to_csv(out_dir / "gold.csv", index=False)

    df = load_gold("task_42", public_data)
    assert df is not None
    assert list(df.columns) == ["v"]
    assert df["v"].tolist() == [1, 2, 3]


def test_load_prediction_missing_returns_none(tmp_path: Path) -> None:
    assert load_prediction(tmp_path / "task_dir") is None


def test_load_prediction_returns_dataframe(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_dir"
    task_dir.mkdir()
    pd.DataFrame({"a": ["x", "y"]}).to_csv(task_dir / "prediction.csv", index=False)

    df = load_prediction(task_dir)
    assert df is not None
    assert df["a"].tolist() == ["x", "y"]


def test_load_trace_missing_or_corrupt(tmp_path: Path) -> None:
    """trace.json 缺失或损坏都应返回 None，避免 UI 崩溃。"""
    task_dir = tmp_path / "task_dir"
    task_dir.mkdir()
    assert load_trace(task_dir) is None

    (task_dir / "trace.json").write_text("{ not valid json")
    assert load_trace(task_dir) is None


def test_load_trace_returns_payload(tmp_path: Path) -> None:
    task_dir = tmp_path / "task_dir"
    task_dir.mkdir()
    payload = {"task_id": "task_1", "steps": [], "succeeded": True}
    (task_dir / "trace.json").write_text(json.dumps(payload))

    loaded = load_trace(task_dir)
    assert loaded == payload
