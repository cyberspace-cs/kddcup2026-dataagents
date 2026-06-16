"""run_explorer：video_findings 提取、video_tool 穿透。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.explorer.runner import run_explorer
from agents.llm.types import ModelMessage, ModelResponse, ModelToolCall
from agents.tools.registry import FunctionTool, ToolExecutionResult
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


def _stub_video_tool() -> FunctionTool:
    # json_schema 非 None：to_openai_tools 会跳过无 schema 的工具，
    # 同源断言需要 explore_video 出现在每轮请求的广告里
    return FunctionTool(
        name="explore_video",
        description="stub",
        json_schema={"type": "object", "properties": {}},
        handler=lambda _t, _a: ToolExecutionResult(
            ok=True,
            content={"video_status": "ok", "findings": {"summary": "chart video"}},
        ),
    )


class _RecordingAdapter:
    def __init__(self, inner: ScriptedNativeModelAdapter) -> None:
        self.inner = inner
        self.seen: list[list[ModelMessage]] = []
        self.tools_seen: list[Any] = []

    def complete(
        self, messages: list[ModelMessage], *, tools: object | None = None
    ) -> ModelResponse:
        self.seen.append(messages)
        self.tools_seen.append(tools)
        return self.inner.complete(messages, tools=tools)


def _advertised_names_per_turn(adapter: _RecordingAdapter) -> list[set[str]]:
    """渲染每轮下推 registry 的 OpenAI tools 广告，取工具名集合。"""
    per_turn: list[set[str]] = []
    for tools in adapter.tools_seen:
        assert tools is not None, "agent must push its registry on every turn"
        per_turn.append({entry["function"]["name"] for entry in tools.to_openai_tools()})
    return per_turn


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


def test_each_turn_advertises_video_tool_when_injected(tmp_path: Path) -> None:
    """同源行为：注入 video_tool 的 run，每轮请求的工具广告都含 explore_video。"""
    task = _make_task(tmp_path)
    adapter = _RecordingAdapter(
        ScriptedNativeModelAdapter(
            [
                _call("v1", "explore_video", {"path": "demo.mp4", "instructions": "look"}),
                _call("r1", "report", {"knowledge": {"from_video": "chart video"}}),
            ]
        )
    )
    run_explorer(task, task.question, adapter, video_tool=_stub_video_tool())
    per_turn = _advertised_names_per_turn(adapter)
    assert len(per_turn) == 2
    assert all("explore_video" in names for names in per_turn)


def test_each_turn_omits_video_tool_when_not_injected(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    adapter = _RecordingAdapter(
        ScriptedNativeModelAdapter([_call("r1", "report", {"knowledge": {}})])
    )
    run_explorer(task, task.question, adapter)
    per_turn = _advertised_names_per_turn(adapter)
    assert per_turn
    assert all("explore_video" not in names for names in per_turn)
