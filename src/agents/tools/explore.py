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
from typing import TYPE_CHECKING, Any

from agents.benchmark.schema import PublicTask
from agents.tools.handlers import ToolExecutionResult
from agents.tools.inputs import ExploreInput
from agents.tools.schema_normalize import normalize_for_vllm

if TYPE_CHECKING:
    from agents.llm import ModelAdapter, ToolProtocol
    from agents.tools.registry import ToolDefinition

logger = logging.getLogger(__name__)

EXPLORE_SCHEMA = normalize_for_vllm(ExploreInput.model_json_schema())  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

_UNIT_FACTORS: dict[tuple[str, str], float] = {}
_SCALE = {"元": 1, "万元": 1e4, "百万元": 1e6, "亿元": 1e8}
for _src, _sv in _SCALE.items():
    for _tgt, _tv in _SCALE.items():
        if _src != _tgt:
            _UNIT_FACTORS[(_src, _tgt)] = _sv / _tv


def _detect_unit_conversions(task: PublicTask) -> list[dict[str, Any]]:
    """Compare each field's source unit against its ``_target_<field>`` unit.

    The ETL stage writes per-field target units (via majority-vote LLM
    extraction from knowledge.md) as ``_target_<field>=<unit>`` entries
    in ``_cache/*_units.json``.

    Returns a list of conversion instructions, one per (file, source_unit) group.
    """
    from agents.config import ETL_SCRATCH_ROOT

    cache_dir = ETL_SCRATCH_ROOT / task.task_id / "_cache"
    if not cache_dir.is_dir():
        return []

    conversions: list[dict[str, Any]] = []
    for units_path in sorted(cache_dir.glob("*_units.json")):
        try:
            units = json.loads(units_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # Group fields by (source_unit, target_unit) pair
        by_pair: dict[tuple[str, str], list[str]] = {}
        for field, source_unit in units.items():
            if field.startswith("_"):
                continue
            target_unit = units.get(f"_target_{field}")
            if not target_unit or source_unit == target_unit:
                continue
            if (source_unit, target_unit) in _UNIT_FACTORS:
                by_pair.setdefault((source_unit, target_unit), []).append(field)
        stem = units_path.stem.removesuffix("_units")
        for (from_unit, to_unit), columns in by_pair.items():
            factor = _UNIT_FACTORS[(from_unit, to_unit)]
            conversions.append(
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
    return conversions


def create_explore_tool_definition(
    explorer_model: ModelAdapter,
    protocol: ToolProtocol,
) -> ToolDefinition:
    """Build a ToolDefinition that runs the Explorer sub-agent.

    The returned handler closes over ``explorer_model`` and ``protocol``
    so the main agent's ``ToolRegistry.execute`` can dispatch to it
    without knowing about the Explorer internals.
    """
    from agents.tools.registry import ToolDefinition as _ToolDefinition

    def handler(task: PublicTask, _args: Any) -> ToolExecutionResult:
        from agents.explorer.runner import run_explorer

        result = run_explorer(task, task.question, explorer_model, protocol=protocol)
        content: dict[str, Any]
        if result.success:
            try:
                content = json.loads(result.findings)
            except (json.JSONDecodeError, TypeError):
                content = {"findings": result.findings}
            content["explorer_steps_used"] = result.steps_used
            unit_conversions = _detect_unit_conversions(task)
            if unit_conversions:
                content["unit_conversion"] = unit_conversions
        else:
            content = result.to_tool_content()
        return ToolExecutionResult(ok=result.success, content=content)

    return _ToolDefinition(
        name="explore",
        description=(
            "Use first to bootstrap context exploration: launches a sub-agent that "
            "inspects every file under context (columns, dtypes, row counts, table "
            "schemas, unit_hint), reads knowledge.md, samples key values, and "
            "identifies join paths — all in one call. Returns a complete data map "
            "so you can skip manual multi-step discovery."
        ),
        input_schema=ExploreInput.example_value(),
        json_schema=EXPLORE_SCHEMA,
        handler=handler,
    )
