"""explore 工具：按任务选 (adapter, video_tool) 配对、focus 穿透、video_findings 注入。"""

from __future__ import annotations

from pathlib import Path

import pytest

import agents.explorer.runner as runner_mod
from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.explorer.runner import ExplorerResult
from agents.tools.explore import create_explore_tool_definition
from agents.tools.handlers import ToolExecutionResult
from agents.tools.inputs import ExploreInput
from agents.tools.registry import ToolDefinition


class _Adapter:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def complete(self, messages):  # pragma: no cover
        raise AssertionError("not called")


def _video_tool() -> ToolDefinition:
    return ToolDefinition(
        name="explore_video",
        description="stub",
        json_schema=None,
        handler=lambda _t, _a: ToolExecutionResult(ok=True, content={}),
    )


def _make_task(tmp_path: Path, *, with_video: bool) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    if with_video:
        (context_dir / "demo.mp4").write_bytes(b"\x00")
    return PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="What is X?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> dict:
    seen: dict = {}

    def fake_run_explorer(task, question, model, *, video_tool=None, **kwargs):
        seen.update(question=question, model=model, video_tool=video_tool)
        return ExplorerResult(
            findings='{"files": [], "schema_map": {}}',
            steps_used=2,
            success=True,
            video_findings=({"video_status": "ok", "findings": {"summary": "v"}},),
        )

    monkeypatch.setattr(runner_mod, "run_explorer", fake_run_explorer)
    return seen


def _make_tool(video_tool: ToolDefinition | None):
    plain = _Adapter("plain")
    with_video = _Adapter("with_video")
    return (
        create_explore_tool_definition(
            plain, explorer_model_with_video=with_video, video_tool=video_tool
        ),
        plain,
        with_video,
    )


def test_video_task_picks_video_pair(tmp_path: Path, recorded: dict) -> None:
    tool, _plain, with_video = _make_tool(_video_tool())
    task = _make_task(tmp_path, with_video=True)
    result = tool.handler(task, ExploreInput())
    assert result.ok is True
    assert recorded["model"] is with_video
    assert recorded["video_tool"] is not None
    assert result.content["video_findings"][0]["findings"]["summary"] == "v"


def test_no_video_task_picks_plain_pair(tmp_path: Path, recorded: dict) -> None:
    tool, plain, _with_video = _make_tool(_video_tool())
    task = _make_task(tmp_path, with_video=False)
    result = tool.handler(task, ExploreInput())
    assert recorded["model"] is plain
    assert recorded["video_tool"] is None
    assert "video_findings" not in result.content


def test_focus_overrides_question(tmp_path: Path, recorded: dict) -> None:
    tool, _p, _w = _make_tool(_video_tool())
    task = _make_task(tmp_path, with_video=False)
    tool.handler(task, ExploreInput(focus="re-check orders.status"))
    assert recorded["question"] == "re-check orders.status"


def test_without_focus_uses_task_question(tmp_path: Path, recorded: dict) -> None:
    tool, _p, _w = _make_tool(_video_tool())
    task = _make_task(tmp_path, with_video=False)
    tool.handler(task, ExploreInput())
    assert recorded["question"] == "What is X?"


def test_backward_compat_single_adapter(tmp_path: Path, recorded: dict) -> None:
    plain = _Adapter("plain")
    tool = create_explore_tool_definition(plain)
    task = _make_task(tmp_path, with_video=True)
    result = tool.handler(task, ExploreInput())
    assert result.ok is True
    assert recorded["model"] is plain and recorded["video_tool"] is None
