"""Explore tool: wraps the Explorer sub-agent as a main-agent tool.

The `explore` tool replaces `inspect_files` in production: a single call
runs the Explorer sub-agent (3–10 read-only steps) and returns a complete
data map — file inventory, schemas, knowledge extraction, join paths, and
warnings — so the main agent can skip manual multi-step discovery.

Heavy imports (``agents.explorer.runner``) are deferred to call-time to
break the circular import chain: ``registry`` → ``explore`` → ``explorer``
→ ``agents.agent`` → ``runtime`` → ``tools`` → ``registry``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict

from agents.benchmark.schema import PublicTask
from agents.tools.registry import ToolExecutionResult
from agents.tools.schema_normalize import normalize_for_vllm


class ExploreInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


if TYPE_CHECKING:
    from agents.llm import ModelAdapter
    from agents.tools.registry import FunctionTool

logger = logging.getLogger(__name__)

EXPLORE_SCHEMA = normalize_for_vllm(ExploreInput.model_json_schema())  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]


def _ensure_column_conversions(task: PublicTask, content: dict[str, Any]) -> None:
    """Deterministically inject column_conversions from ``_units.json``.

    1. Fills missing ``column_conversions`` in ``schema_map`` entries.
    2. Appends a top-level ``unit_conversion`` array as a prominent reminder.
    """
    from agents.config import ETL_SCRATCH_ROOT

    cache_dir = ETL_SCRATCH_ROOT / task.task_id / "_cache"
    if not cache_dir.is_dir():
        return

    schema_map = cast(dict[str, Any], content.get("schema_map", {}) or {})

    top_level: list[dict[str, Any]] = []

    for units_path in sorted(cache_dir.glob("*_units.json")):
        try:
            units = json.loads(units_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        by_group: dict[tuple[str, str, float], list[str]] = {}
        col_conversions: dict[str, str] = {}
        for field, source_unit in units.items():
            if field.startswith("_"):
                continue
            target = units.get(f"_target_{field}")
            factor_str = units.get(f"_factor_{field}")
            if not target or str(source_unit) == str(target) or not factor_str:
                continue
            try:
                factor = float(factor_str)
            except (ValueError, TypeError):
                continue
            if factor == 0 or factor == 1:
                continue
            col_conversions[field] = f"CONVERT: ×{factor_str} ({source_unit} → {target})"
            by_group.setdefault((source_unit, target, factor), []).append(field)

        if not col_conversions:
            continue

        stem = units_path.stem.removesuffix("_units")

        # 1. Inject into schema_map (match by exact key or Path stem)
        for key, raw_entry in schema_map.items():
            if not isinstance(raw_entry, dict):
                continue
            entry = cast(dict[str, Any], raw_entry)
            key_stem = Path(key).stem if "/" in key else key
            if key_stem != stem:
                continue
            existing = cast(dict[str, str], entry.get("column_conversions", {}) or {})
            for col, note in col_conversions.items():
                if col not in existing:
                    existing[col] = note
            if existing:
                entry["column_conversions"] = existing

        # 2. Build top-level unit_conversion entries
        for (from_unit, to_unit, factor), columns in by_group.items():
            top_level.append(
                {
                    "file": stem,
                    "columns": columns,
                    "from": from_unit,
                    "to": to_unit,
                    "factor": factor,
                    "instruction": (
                        f"Multiply {', '.join(columns)} by {factor} "
                        f"to convert from {from_unit} to {to_unit} before submitting."
                    ),
                }
            )

    if top_level:
        content["unit_conversion"] = top_level


def create_explore_tool_definition(
    explorer_model: ModelAdapter,
    *,
    video_tool: FunctionTool | None = None,
) -> FunctionTool:
    """Build a FunctionTool that runs the Explorer sub-agent.

    `explore_video` 按任务动态注入：含视频任务把 ``video_tool`` 传给
    `run_explorer`，registry 在其内部组装一次——既是执行分发表，也是该 run
    每轮请求下推的工具广告源，广告与执行结构性同源。
    """
    from agents.tools.registry import FunctionTool as _FunctionTool

    def handler(task: PublicTask, _args: Any) -> ToolExecutionResult:
        import agents.explorer.runner as explorer_runner
        from agents.runtime.media import find_videos

        has_video = video_tool is not None and bool(find_videos(task.context_dir))
        result = explorer_runner.run_explorer(
            task, task.question, explorer_model, video_tool=video_tool if has_video else None
        )
        content: dict[str, Any]
        if result.success:
            try:
                parsed: Any = json.loads(result.findings)
                content = (
                    cast(dict[str, Any], parsed)
                    if isinstance(parsed, dict)
                    else {"findings": result.findings}
                )
            except (json.JSONDecodeError, TypeError):
                content = {"findings": result.findings}
            content["explorer_steps_used"] = result.steps_used
            if has_video and result.video_findings:
                # 确定性透传：视频精确数值不经 explorer 二次转写直达主 agent
                content["video_findings"] = list(result.video_findings)
            _ensure_column_conversions(task, content)
        else:
            content = result.to_tool_content()
        return ToolExecutionResult(ok=result.success, content=content)

    return _FunctionTool(
        name="explore",
        description=(
            "Use first to bootstrap context exploration: launches a sub-agent that "
            "inspects every file under context (columns, dtypes, row counts, table "
            "schemas, unit_hint), reads knowledge.md, samples key values, and "
            "identifies join paths — all in one call. Returns a complete data map "
            "so you can skip manual multi-step discovery."
        ),
        json_schema=EXPLORE_SCHEMA,
        handler=handler,
    )
