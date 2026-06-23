"""Integration tests for the explore_video tool's filter-recipe wiring.

Pin that task.question flows from the PublicTask into rule_templater so that
abbreviation-aware prompting reaches the agent. We stub video execution: a fake
findings dict comes back from run_video_agent and we only assert on the recipe
text the handler attaches to its observation.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from agents.tools.explore_video import create_explore_video_tool_definition
from agents.tools.registry import ToolExecutionResult


class _StubVideoResult:
    def __init__(self, findings_dict: dict[str, Any]) -> None:
        self.findings = json.dumps(findings_dict, ensure_ascii=False)
        self.steps_used = 1
        self.success = True
        self.failure_reason = None

    def to_tool_content(self) -> dict[str, Any]:
        return {"video_status": "failed", "failure_reason": "stub", "steps_used": self.steps_used}


def _make_task_with_question(question: str, video_path: Any) -> Any:
    """Minimal PublicTask stub; only `question` and `assets` are touched by the handler."""
    assets = SimpleNamespace(
        task_dir=video_path.parent,
        context_dir=video_path.parent,
        input_dir=video_path.parent,
    )
    return SimpleNamespace(task_id="task_x", question=question, assets=assets)


def _findings_dict() -> dict[str, Any]:
    return {
        "summary": "video",
        "timeline": [],
        "extracted_data": {
            "rules": {"field": "dailybenchgr", "operator": ">", "threshold": "0.00%"},
            "displayed_samples": {},
        },
        "coverage": [],
        "warnings": [],
    }


def _invoke_handler(
    *,
    question: str,
    tmp_path: Any,
) -> ToolExecutionResult:
    video_file = tmp_path / "demo.mp4"
    video_file.write_bytes(b"x" * 1024)

    tool_def = create_explore_video_tool_definition(video_model=None)  # type: ignore[arg-type]
    task = _make_task_with_question(question, video_file)

    with patch("agents.runtime.media.VIDEO_EXTENSIONS", (".mp4",)), \
         patch("agents.runtime.media.VIDEO_MAX_BYTES", 10 * 1024 * 1024), \
         patch("agents.tools.context.resolve_context_path", return_value=video_file), \
         patch(
             "agents.video.runner.run_video_agent",
             return_value=_StubVideoResult(_findings_dict()),
         ):
        return tool_def.handler(
            task,
            tool_def.input_model.model_validate({"path": "demo.mp4", "instructions": "x"}),
        )


def test_explore_video_threads_question_into_filter_recipe(tmp_path: Any) -> None:
    """When the task asks for 简称, the resulting filter_recipe text mentions abbreviation."""
    result = _invoke_handler(question="请列出基金简称", tmp_path=tmp_path)
    assert result.ok is True
    assert "filter_recipe" in result.content
    assert "prefer_column_kind: abbreviation" in result.content["filter_recipe"]


def test_explore_video_omits_abbr_hint_when_question_full_name(tmp_path: Any) -> None:
    """When the task asks for 全称, the recipe omits the abbreviation hint."""
    result = _invoke_handler(question="请列出基金全称", tmp_path=tmp_path)
    assert result.ok is True
    assert "prefer_column_kind" not in result.content.get("filter_recipe", "")
