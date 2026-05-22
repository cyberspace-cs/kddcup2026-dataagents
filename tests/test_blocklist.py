"""`run.blocklist` 跳过名单端到端测试。

覆盖三层：
- Pydantic 校验（int / str / "task_<n>" 归一化、负数 / 重复 / 类型错误）
- `run_single_task` 短路：命中名单时不走 timeout 子进程，落盘合成 trace.json
- `run_benchmark` 整合：blocklist + 限制选项叠加时统计与 summary.json 形态正确
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from data_agent_baseline.config import (
    AgentConfig,
    AppConfig,
    DatasetConfig,
    RunConfig,
    load_app_config,
)
from data_agent_baseline.run import runner as runner_mod
from data_agent_baseline.run.runner import TaskRunArtifacts, run_benchmark, run_single_task
from data_agent_baseline.run.runner_artifacts import (
    BLOCKLIST_FAILURE_REASON,
    blocklist_payload,
    write_task_outputs,
)

# ---------- Pydantic 校验层 ------------------------------------------------------


def _write_minimal_config(path: Path, run_body: str) -> Path:
    """写一个最小可解析 YAML，把 run_body 插到 run: 下作为额外字段。"""
    path.write_text(f"agent:\n  model: m\n  api_base: http://x\n  api_key: k\nrun:\n{run_body}")
    return path


def test_blocklist_default_is_empty(tmp_path: Path) -> None:
    cfg_path = _write_minimal_config(tmp_path / "cfg.yaml", "  output_dir: /tmp/run\n")
    cfg = load_app_config(cfg_path)
    assert cfg.run.blocklist == ()


def test_blocklist_accepts_mixed_int_and_string(tmp_path: Path) -> None:
    cfg_path = _write_minimal_config(
        tmp_path / "cfg.yaml",
        '  blocklist:\n    - 5\n    - task_19\n    - "7"\n',
    )
    cfg = load_app_config(cfg_path)
    # 都规整成 task_<n> 形式
    assert cfg.run.blocklist == ("task_5", "task_19", "task_7")


def test_blocklist_rejects_non_list(tmp_path: Path) -> None:
    cfg_path = _write_minimal_config(tmp_path / "cfg.yaml", "  blocklist: 5\n")
    with pytest.raises(ValueError, match="blocklist must be a list"):
        load_app_config(cfg_path)


def test_blocklist_rejects_zero_or_negative(tmp_path: Path) -> None:
    cfg_path = _write_minimal_config(tmp_path / "cfg.yaml", "  blocklist:\n    - 0\n")
    with pytest.raises(ValueError, match="task number must be >= 1"):
        load_app_config(cfg_path)


def test_blocklist_rejects_unknown_string(tmp_path: Path) -> None:
    cfg_path = _write_minimal_config(
        tmp_path / "cfg.yaml",
        "  blocklist:\n    - random_string\n",
    )
    with pytest.raises(ValueError, match="must be an int or 'task_<n>'"):
        load_app_config(cfg_path)


def test_blocklist_rejects_duplicates(tmp_path: Path) -> None:
    # int 5 和 "task_5" 经规整后等价，应被识别为重复
    cfg_path = _write_minimal_config(
        tmp_path / "cfg.yaml",
        "  blocklist:\n    - 5\n    - task_5\n",
    )
    with pytest.raises(ValueError, match="contains duplicate entries"):
        load_app_config(cfg_path)


def test_blocklist_rejects_bool(tmp_path: Path) -> None:
    # bool 是 int 子类；显式 false → 0 会绕过 number 校验，所以单独排除
    cfg_path = _write_minimal_config(
        tmp_path / "cfg.yaml",
        "  blocklist:\n    - true\n",
    )
    with pytest.raises(ValueError, match="must be an int or string"):
        load_app_config(cfg_path)


# ---------- runner 短路 ----------------------------------------------------------


def _make_config(*, tmp_path: Path, blocklist: tuple[str, ...] = ()) -> AppConfig:
    return AppConfig(
        dataset=DatasetConfig(root_path=tmp_path / "data"),
        agent=AgentConfig(model="m", api_base="http://x", api_key="primary-key"),
        run=RunConfig(
            output_dir=tmp_path / "runs",
            run_id=None,
            max_workers=2,
            task_timeout_seconds=60,
            blocklist=blocklist,
        ),
    )


def test_blocklist_payload_shape() -> None:
    payload = blocklist_payload("task_7")
    # 与 _failure_run_result_payload 同形，下游 write_task_outputs 不需要分支
    assert payload["task_id"] == "task_7"
    assert payload["answer"] is None
    assert payload["steps"] == []
    assert payload["failure_reason"] == BLOCKLIST_FAILURE_REASON
    assert payload["succeeded"] is False


def test_run_single_task_short_circuits_blocklisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """命中名单时不应触发子进程或同进程 core 路径。"""
    invoked = {"timeout": 0, "core": 0}

    def _fail_timeout(**_: Any) -> dict[str, Any]:
        invoked["timeout"] += 1
        raise AssertionError("timeout dispatcher must not be called for blocklisted task")

    def _fail_core(**_: Any) -> dict[str, Any]:
        invoked["core"] += 1
        raise AssertionError("core dispatcher must not be called for blocklisted task")

    monkeypatch.setattr(runner_mod, "run_single_task_with_timeout", _fail_timeout)
    monkeypatch.setattr(runner_mod, "run_single_task_core", _fail_core)

    cfg = _make_config(tmp_path=tmp_path, blocklist=("task_5",))
    run_output = tmp_path / "outdir"
    run_output.mkdir()

    artifact = run_single_task(task_id="task_5", config=cfg, run_output_dir=run_output)

    assert invoked == {"timeout": 0, "core": 0}
    assert artifact.succeeded is False
    assert artifact.failure_reason == BLOCKLIST_FAILURE_REASON
    assert artifact.prediction_csv_path is None
    # trace.json 必须落盘，UI / 评测脚本依赖它存在
    assert artifact.trace_path.exists()
    payload = json.loads(artifact.trace_path.read_text())
    assert payload["task_id"] == "task_5"
    assert payload["succeeded"] is False
    assert payload["failure_reason"] == BLOCKLIST_FAILURE_REASON
    assert payload["steps"] == []


def test_run_single_task_runs_normally_when_not_blocklisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """blocklist 不命中时仍走原 timeout dispatcher——证明短路只在命中时触发。"""
    seen: list[str] = []

    def _ok_timeout(*, task_id: str, config: AppConfig, partial_trace_path: Any) -> dict[str, Any]:
        del config, partial_trace_path
        seen.append(task_id)
        return {
            "task_id": task_id,
            "answer": None,
            "steps": [],
            "failure_reason": None,
            "succeeded": True,
        }

    monkeypatch.setattr(runner_mod, "run_single_task_with_timeout", _ok_timeout)

    cfg = _make_config(tmp_path=tmp_path, blocklist=("task_5",))
    run_output = tmp_path / "outdir"
    run_output.mkdir()

    artifact = run_single_task(task_id="task_7", config=cfg, run_output_dir=run_output)

    assert seen == ["task_7"]
    assert artifact.succeeded is True


# ---------- run_benchmark 整合 ---------------------------------------------------


class _CapturingRunSingleTask:
    """记录被实际调度的 task_id；返回成功占位 artifacts。"""

    def __init__(self) -> None:
        self.captured: list[str] = []

    def __call__(
        self,
        *,
        task_id: str,
        config: AppConfig,
        run_output_dir: Path,
        model: Any = None,
        tools: Any = None,
        answer_verifier: Any = None,
    ) -> TaskRunArtifacts:
        del model, tools, answer_verifier
        # 模拟 run_single_task 内部的 blocklist 短路：保留行为以便 summary 统计正确
        if task_id in config.run.blocklist:
            return write_task_outputs(task_id, run_output_dir, blocklist_payload(task_id))
        self.captured.append(task_id)
        task_output_dir = run_output_dir / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = task_output_dir / "trace.json"
        trace_path.write_text("{}")
        return TaskRunArtifacts(
            task_id=task_id,
            task_output_dir=task_output_dir,
            prediction_csv_path=None,
            trace_path=trace_path,
            succeeded=True,
            failure_reason=None,
        )


def _patch_dataset(monkeypatch: pytest.MonkeyPatch, task_ids: list[str]) -> None:
    class _FakeTask:
        def __init__(self, tid: str) -> None:
            self.task_id = tid

    class _FakeDataset:
        def __init__(self, root: Path) -> None:
            del root

        def iter_tasks(self) -> list[_FakeTask]:
            return [_FakeTask(tid) for tid in task_ids]

    monkeypatch.setattr(runner_mod, "DABenchPublicDataset", _FakeDataset)


def test_run_benchmark_records_blocked_tasks_as_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """blocklist 内的任务必须出现在 summary 的 tasks 列表里且 succeeded=False。"""
    fake = _CapturingRunSingleTask()
    monkeypatch.setattr(runner_mod, "run_single_task", fake)
    _patch_dataset(monkeypatch, ["task_1", "task_2", "task_3"])

    cfg = _make_config(tmp_path=tmp_path, blocklist=("task_2",))
    cfg = replace(cfg, run=replace(cfg.run, max_workers=1))

    run_output_dir, artifacts = run_benchmark(config=cfg)

    # 只有非 blocked 的任务真正调用过 dispatcher
    assert sorted(fake.captured) == ["task_1", "task_3"]
    # 但 summary 应包含全部 3 个任务
    assert len(artifacts) == 3
    by_id = {a.task_id: a for a in artifacts}
    assert by_id["task_2"].succeeded is False
    assert by_id["task_2"].failure_reason == BLOCKLIST_FAILURE_REASON
    assert by_id["task_1"].succeeded is True
    assert by_id["task_3"].succeeded is True

    # summary.json 同形：succeeded 计数排除 blocked
    summary_path = run_output_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["task_count"] == 3
    assert summary["succeeded_task_count"] == 2
    blocked_entry = next(t for t in summary["tasks"] if t["task_id"] == "task_2")
    assert blocked_entry["failure_reason"] == BLOCKLIST_FAILURE_REASON
    assert blocked_entry["prediction_csv_path"] is None
