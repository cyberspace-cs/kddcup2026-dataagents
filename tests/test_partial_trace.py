"""P0 "partial trace on timeout" 回归测试。

覆盖点：
- `ReActAgent.step_callback` 在每步 append 后被调用，可以观察当时的 `state.steps`
- callback 抛异常不会污染循环
- `_load_partial_trace` 容错：缺文件 / JSON 坏 / task_id 不匹配 都返回 None
- `_merge_partial_with_failure` 的合并行为：保留 partial 的 steps/answer，失败字段覆写
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_agent_baseline.agents.model import (
    ModelResponse,
    ModelToolCall,
    ScriptedNativeModelAdapter,
)
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.agents.runtime import AgentRuntimeState
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.run.runner import (
    _load_partial_trace,
    _make_partial_trace_writer,
    _merge_partial_with_failure,
)
from data_agent_baseline.tools.registry import create_default_tool_registry


@pytest.fixture
def task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="q?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _tool_call(call_id: str, name: str, arguments: dict) -> tuple[ModelToolCall, dict]:
    raw = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    return ModelToolCall(id=call_id, name=name, arguments=arguments), raw


def test_step_callback_fires_after_each_append(task: PublicTask) -> None:
    list_call, list_raw = _tool_call("c1", "list_context", {"max_depth": 1})
    answer_call, answer_raw = _tool_call("c2", "answer", {"columns": ["v"], "rows": [["ok"]]})
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="look",
                tool_calls=[list_call],
                raw_response="look",
                raw_tool_calls=[list_raw],
            ),
            ModelResponse(
                content="done",
                tool_calls=[answer_call],
                raw_response="done",
                raw_tool_calls=[answer_raw],
            ),
        ]
    )

    snapshots: list[int] = []

    def cb(state: AgentRuntimeState) -> None:
        # 捕获每次 callback 时的步数
        snapshots.append(len(state.steps))

    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=4),
        protocol="native",
        step_callback=cb,
    )
    result = agent.run(task)

    assert result.succeeded is True
    # 2 次 append（list + answer）→ 2 次 callback，且步数单调递增
    assert snapshots == [1, 2]


def test_step_callback_exception_is_swallowed(task: PublicTask) -> None:
    answer_call, answer_raw = _tool_call("c1", "answer", {"columns": ["v"], "rows": [["ok"]]})
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="",
                tool_calls=[answer_call],
                raw_response="",
                raw_tool_calls=[answer_raw],
            ),
        ]
    )

    call_count = {"n": 0}

    def broken_cb(_state: AgentRuntimeState) -> None:
        call_count["n"] += 1
        raise RuntimeError("observer boom")

    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=4),
        protocol="native",
        step_callback=broken_cb,
    )
    # callback 抛错不能影响循环；任务照样成功
    result = agent.run(task)
    assert result.succeeded is True
    assert call_count["n"] == 1


def test_partial_trace_writer_round_trip(tmp_path: Path, task: PublicTask) -> None:
    """端到端：writer 写盘 → load 回来应能命中 task_id + steps。"""
    trace_path = tmp_path / "task_1" / "trace.json"
    list_call, list_raw = _tool_call("c1", "list_context", {"max_depth": 1})
    # 刻意不给 answer：模拟"任务中途"快照
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="mid-run",
                tool_calls=[list_call],
                raw_response="mid-run",
                raw_tool_calls=[list_raw],
            ),
        ]
    )
    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=1),  # 刻意 1 步就退出，模拟 kill 前最后状态
        protocol="native",
        step_callback=_make_partial_trace_writer("task_1", trace_path),
    )
    agent.run(task)

    loaded = _load_partial_trace("task_1", trace_path)
    assert loaded is not None
    assert loaded["task_id"] == "task_1"
    assert loaded["succeeded"] is False  # 增量快照永远标 False
    assert loaded["answer"] is None
    assert len(loaded["steps"]) == 1
    assert loaded["steps"][0]["action"] == "list_context"


def test_load_partial_trace_handles_missing_and_corrupt(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    assert _load_partial_trace("task_1", missing) is None

    corrupt = tmp_path / "bad.json"
    corrupt.write_text("{ not valid json")
    assert _load_partial_trace("task_1", corrupt) is None

    # task_id 不匹配也应返回 None —— 避免残留的旧 trace 串到别的任务上
    wrong_task = tmp_path / "other.json"
    wrong_task.write_text(json.dumps({"task_id": "task_999", "steps": []}))
    assert _load_partial_trace("task_1", wrong_task) is None


def test_merge_partial_with_failure_preserves_steps(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "steps": [{"step_index": 1, "action": "list_context"}],
                "answer": None,
                "failure_reason": None,
                "succeeded": False,
            }
        )
    )
    merged = _merge_partial_with_failure("task_1", trace_path, "Task timed out.")
    assert merged["failure_reason"] == "Task timed out."
    assert merged["succeeded"] is False
    # 关键断言：steps 没有被清空
    assert len(merged["steps"]) == 1


def test_merge_partial_with_failure_falls_back_when_trace_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    merged = _merge_partial_with_failure("task_1", missing, "Task timed out.")
    assert merged["task_id"] == "task_1"
    assert merged["failure_reason"] == "Task timed out."
    assert merged["steps"] == []
    assert merged["succeeded"] is False
