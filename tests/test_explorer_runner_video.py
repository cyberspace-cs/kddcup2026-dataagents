"""run_explorer：video_findings 提取、focused 模式首条消息、video_tool 穿透。"""

from __future__ import annotations

import json
from pathlib import Path

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.explorer.runner import run_explorer
from agents.llm.types import ModelMessage, ModelResponse, ModelToolCall
from agents.tools.handlers import ToolExecutionResult
from agents.tools.registry import ToolDefinition
from tests.helpers.scripted_adapters import ScriptedNativeModelAdapter


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="What is X?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _call(call_id: str, name: str, arguments: dict) -> ModelResponse:
    raw = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    return ModelResponse(
        content="",
        tool_calls=[ModelToolCall(id=call_id, name=name, arguments=arguments)],
        raw_response="",
        raw_tool_calls=[raw],
    )


def _stub_video_tool() -> ToolDefinition:
    return ToolDefinition(
        name="explore_video",
        description="stub",
        json_schema=None,
        handler=lambda _t, _a: ToolExecutionResult(
            ok=True,
            content={"video_status": "ok", "findings": {"summary": "chart video"}},
        ),
    )


class _RecordingAdapter:
    def __init__(self, inner: ScriptedNativeModelAdapter) -> None:
        self.inner = inner
        self.seen: list[list[ModelMessage]] = []

    def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        self.seen.append(messages)
        return self.inner.complete(messages)


def test_video_findings_extracted_from_steps(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    adapter = ScriptedNativeModelAdapter(
        [
            _call("v1", "explore_video", {"path": "demo.mp4", "instructions": "look"}),
            _call("r1", "report", {"knowledge": {"from_video": "chart video"}}),
        ]
    )
    result = run_explorer(task, task.question, adapter, video_tool=_stub_video_tool())
    assert result.success is True
    assert result.video_findings is not None
    assert any("chart video" in json.dumps(v) for v in result.video_findings)


def test_no_video_tool_no_video_findings(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    adapter = ScriptedNativeModelAdapter([_call("r1", "report", {"knowledge": {}})])
    result = run_explorer(task, task.question, adapter)
    assert result.success is True
    assert result.video_findings is None


def test_focus_question_appends_directive_to_first_message(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    adapter = _RecordingAdapter(
        ScriptedNativeModelAdapter([_call("r1", "report", {"knowledge": {"x": 1}})])
    )
    result = run_explorer(task, "check the orders.status values", adapter)
    assert result.success is True
    first_user = adapter.seen[0][1]
    assert first_user.role == "user"
    assert "## Focused re-exploration directive" in str(first_user.content)
    assert "check the orders.status values" in str(first_user.content)


def test_plain_question_keeps_default_first_message(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    adapter = _RecordingAdapter(
        ScriptedNativeModelAdapter([_call("r1", "report", {"knowledge": {}})])
    )
    run_explorer(task, task.question, adapter)
    assert "Focused re-exploration" not in str(adapter.seen[0][1].content)
