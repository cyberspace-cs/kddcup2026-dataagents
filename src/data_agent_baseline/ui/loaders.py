"""单任务调试前端的纯 IO 加载层（不依赖 streamlit）。

复用现有：
- `data_agent_baseline.benchmark.dataset.DABenchPublicDataset`：加载 question
- `data_agent_baseline.run.runner_artifacts.load_partial_trace`：加载 trace.json

新增：
- `list_runs(artifacts_dir)`：枚举 `<artifacts>/runs/*` 下的 run，按时间倒序
- `list_tasks_in_run(run_dir)`：读取 summary.json，返回任务索引
- `load_gold(task_id, public_data_dir)`：读取 gold.csv（可能不存在）
- `load_prediction(task_artifact_dir)`：读取 prediction.csv（可能不存在）

字段尽量保持纯数据，不引入 streamlit 依赖；UI 渲染交给 `streamlit_app.py`。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd


@dataclass(frozen=True, slots=True)
class RunInfo:
    """一个 run 目录的可读视图。

    - run_id:           目录名（也即用户可见 ID）
    - run_dir:          绝对路径
    - task_count:       summary.json 中的 task_count（不存在时为 None）
    - succeeded_count:  summary.json 中的 succeeded_task_count（不存在时为 None）
    """

    run_id: str
    run_dir: Path
    task_count: int | None
    succeeded_count: int | None


@dataclass(frozen=True, slots=True)
class TaskInfo:
    """一个 task 在某次 run 中的状态汇总。"""

    task_id: str
    task_dir: Path
    succeeded: bool
    failure_reason: str | None


def list_runs(artifacts_dir: Path) -> list[RunInfo]:
    """枚举 `<artifacts_dir>/runs/*` 下所有 run 子目录，按目录名倒序（默认时间格式 YYYYMMDD…）。

    `artifacts_dir` 通常是项目根的 `artifacts/`，传 `<artifacts_dir>/runs/*` 太啰嗦，
    本函数会自动把 `runs/` 子目录拼上。若目录不存在则返回空列表，避免初次使用时抛错。
    """
    runs_root = artifacts_dir / "runs" if artifacts_dir.name != "runs" else artifacts_dir
    if not runs_root.is_dir():
        return []

    infos: list[RunInfo] = []
    for entry in runs_root.iterdir():
        if not entry.is_dir():
            continue
        summary_path = entry / "summary.json"
        task_count: int | None = None
        succeeded_count: int | None = None
        if summary_path.exists():
            try:
                payload = json.loads(summary_path.read_text())
                if isinstance(payload, dict):
                    payload_dict = cast(dict[str, Any], payload)
                    raw_total = payload_dict.get("task_count")
                    raw_ok = payload_dict.get("succeeded_task_count")
                    task_count = int(raw_total) if isinstance(raw_total, int) else None
                    succeeded_count = int(raw_ok) if isinstance(raw_ok, int) else None
            except (OSError, json.JSONDecodeError):
                # summary.json 损坏不应阻止 run 列表渲染——保持 None
                pass
        infos.append(
            RunInfo(
                run_id=entry.name,
                run_dir=entry,
                task_count=task_count,
                succeeded_count=succeeded_count,
            )
        )
    # 时间戳格式天然单调，按 run_id 倒序即"最新优先"
    infos.sort(key=lambda info: info.run_id, reverse=True)
    return infos


def list_tasks_in_run(run_dir: Path) -> list[TaskInfo]:
    """读取 `<run_dir>/summary.json`，返回任务列表 + 成败 + 失败原因。

    summary.json 缺失时回退：扫描 run_dir 下的子目录，每个子目录视为一个任务，
    通过 trace.json 里的 `succeeded` 字段判断成败（缺失则视作失败）。
    """
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            return _list_tasks_from_subdirs(run_dir)
        if not isinstance(payload, dict):
            return _list_tasks_from_subdirs(run_dir)
        tasks_payload = cast(dict[str, Any], payload).get("tasks")
        if not isinstance(tasks_payload, list):
            return _list_tasks_from_subdirs(run_dir)

        result: list[TaskInfo] = []
        for raw in cast(list[Any], tasks_payload):
            if not isinstance(raw, dict):
                continue
            task_id = str(cast(dict[str, Any], raw).get("task_id", ""))
            if not task_id:
                continue
            task_dir_str = cast(dict[str, Any], raw).get("task_output_dir")
            task_dir = Path(task_dir_str) if isinstance(task_dir_str, str) else run_dir / task_id
            result.append(
                TaskInfo(
                    task_id=task_id,
                    task_dir=task_dir,
                    succeeded=bool(cast(dict[str, Any], raw).get("succeeded")),
                    failure_reason=cast(dict[str, Any], raw).get("failure_reason"),
                )
            )
        return result

    return _list_tasks_from_subdirs(run_dir)


def _list_tasks_from_subdirs(run_dir: Path) -> list[TaskInfo]:
    """summary.json 缺失时的兜底：从 run_dir 的子目录中推断任务列表。"""
    if not run_dir.is_dir():
        return []
    out: list[TaskInfo] = []
    for entry in sorted(run_dir.iterdir()):
        if not entry.is_dir():
            continue
        trace_path = entry / "trace.json"
        succeeded = False
        failure_reason: str | None = "trace.json missing"
        if trace_path.exists():
            try:
                payload = json.loads(trace_path.read_text())
                if isinstance(payload, dict):
                    payload_dict = cast(dict[str, Any], payload)
                    succeeded = bool(payload_dict.get("succeeded"))
                    raw_reason = payload_dict.get("failure_reason")
                    failure_reason = raw_reason if isinstance(raw_reason, str) else None
            except (OSError, json.JSONDecodeError):
                failure_reason = "trace.json corrupted"
        out.append(
            TaskInfo(
                task_id=entry.name,
                task_dir=entry,
                succeeded=succeeded,
                failure_reason=failure_reason,
            )
        )
    return out


def load_gold(task_id: str, public_data_dir: Path) -> pd.DataFrame | None:
    """读 `<public_data_dir>/output/<task_id>/gold.csv`；不存在或读取失败返回 None。

    gold 文件不一定每个 task 都有（公开赛题可能只发了输入），所以用 None 表达"不可用"
    而不是抛错。
    """
    gold_path = public_data_dir / "output" / task_id / "gold.csv"
    if not gold_path.exists():
        return None
    try:
        return pd.read_csv(gold_path)
    except (OSError, ValueError):
        return None


def load_prediction(task_artifact_dir: Path) -> pd.DataFrame | None:
    """读 `<task_artifact_dir>/prediction.csv`；不存在或读取失败返回 None。

    Agent 没调用 answer 工具的任务不会有 prediction.csv（参见 runner_artifacts），
    UI 上需要把这种情况显著提示出来。
    """
    pred_path = task_artifact_dir / "prediction.csv"
    if not pred_path.exists():
        return None
    try:
        return pd.read_csv(pred_path)
    except (OSError, ValueError):
        return None


def load_trace(task_artifact_dir: Path) -> dict[str, Any] | None:
    """读 `<task_artifact_dir>/trace.json` 并返回 dict；缺失/损坏时返回 None。

    薄包装，不依赖 runner_artifacts 内部细节，便于 UI 单独使用。
    """
    trace_path = task_artifact_dir / "trace.json"
    if not trace_path.exists():
        return None
    try:
        payload = json.loads(trace_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return cast(dict[str, Any], payload) if isinstance(payload, dict) else None
