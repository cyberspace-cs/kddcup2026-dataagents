"""explore 工具：explore_video 按任务动态注入、video_findings 注入。"""

from __future__ import annotations

from pathlib import Path

import pytest

import agents.explorer.runner as runner_mod
from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.explorer.runner import ExplorerResult
from agents.tools.explore import ExploreInput, create_explore_tool_definition
from agents.tools.registry import FunctionTool, ToolExecutionResult


class _Adapter:
    def complete(self, messages, *, tools=None, **kwargs):  # pragma: no cover
        raise AssertionError("not called")


def _video_tool() -> FunctionTool:
    return FunctionTool(
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


def test_explore_tool_description_matches_discovery_contract() -> None:
    tool = create_explore_tool_definition(_Adapter())
    description = tool.description.lower()

    assert "discovery-only" in description
    assert "read-only sub-agent" not in description


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


@pytest.mark.parametrize("task_has_video", [True, False], ids=["video-task", "plain-task"])
@pytest.mark.parametrize("pass_video_tool", [True, False], ids=["with-tool", "no-tool"])
def test_video_tool_injected_only_when_task_has_video(
    tmp_path: Path, recorded: dict, task_has_video: bool, pass_video_tool: bool
) -> None:
    adapter = _Adapter()
    tool = create_explore_tool_definition(
        adapter, video_tool=_video_tool() if pass_video_tool else None
    )
    task = _make_task(tmp_path, with_video=task_has_video)

    result = tool.handler(task, ExploreInput())

    assert result.ok is True
    assert recorded["model"] is adapter
    if task_has_video and pass_video_tool:
        assert recorded["video_tool"] is not None
        assert result.content["video_findings"][0]["findings"]["summary"] == "v"
    else:
        assert recorded["video_tool"] is None
        assert "video_findings" not in result.content
