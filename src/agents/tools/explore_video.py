"""explore_video tool: wraps the video sub-agent as an explorer-only tool.

与 `tools/explore.py` 同款 agent-as-tool 模式；重 import（agents.video.runner）
推迟到调用期以断开 registry → video → agent → runtime → tools 的环。
该工具的全部使用指引都在 description 里：registry 不含本工具时模型完全无感知
（`build_system_message` 不渲染工具文档，说明只经请求体 tools=[...] 下推）。
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from agents.benchmark.schema import PublicTask
from agents.tools.handlers import ToolExecutionResult
from agents.tools.inputs import EXPLORE_VIDEO_SCHEMA, ExploreVideoInput

if TYPE_CHECKING:
    from agents.llm import ModelAdapter
    from agents.tools.registry import ToolDefinition

logger = logging.getLogger(__name__)


def create_explore_video_tool_definition(video_model: ModelAdapter) -> ToolDefinition:
    """Build the ToolDefinition that runs the video sub-agent on one context video."""
    from agents.tools.registry import ToolDefinition as _ToolDefinition

    def handler(task: PublicTask, args: ExploreVideoInput) -> ToolExecutionResult:
        import agents.runtime.media as media
        import agents.video.runner as video_runner
        from agents.tools.filesystem.path import resolve_context_path

        try:
            video_path = resolve_context_path(task, args.path)
        except (ValueError, FileNotFoundError) as exc:
            return ToolExecutionResult(ok=False, content={"error": str(exc)})
        if video_path.suffix.lower() not in media.VIDEO_EXTENSIONS:
            return ToolExecutionResult(
                ok=False, content={"error": f"{args.path} is not a video file"}
            )
        size = video_path.stat().st_size
        if size > media.VIDEO_MAX_BYTES:
            return ToolExecutionResult(
                ok=False,
                content={
                    "error": (
                        f"{args.path} is {size} bytes, which exceeds the "
                        f"{media.VIDEO_MAX_BYTES} byte limit"
                    )
                },
            )

        result = video_runner.run_video_agent(task, video_path, args.instructions, video_model)
        if not result.success:
            return ToolExecutionResult(ok=False, content=result.to_tool_content())
        try:
            findings: Any = json.loads(result.findings)
        except (json.JSONDecodeError, TypeError):
            findings = result.findings
        return ToolExecutionResult(
            ok=True,
            content={
                "video_status": "ok",
                "path": args.path,
                "findings": findings,
                "video_steps_used": result.steps_used,
            },
        )

    return _ToolDefinition(
        name="explore_video",
        description=(
            "Launch the video analysis sub-agent on one video file from context. Use "
            "ONLY when the task context contains a video file (the task message lists "
            "them). Write `instructions` as an analysis brief tied to the task "
            "question: which charts, tables, values, or text to transcribe. Call once "
            "per video. The findings return to you AND pass through to the main agent "
            "verbatim — reference key facts in your report's knowledge/value_samples "
            "but do not re-transcribe numbers."
        ),
        json_schema=EXPLORE_VIDEO_SCHEMA,
        handler=handler,
        input_model=ExploreVideoInput,
    )
