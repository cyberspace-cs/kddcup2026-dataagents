"""Helpers for extracting video findings from recorded agent state."""

from __future__ import annotations

from typing import Any, cast

from agents.runtime.state import AgentRuntimeState


def _as_dict(value: Any) -> dict[str, Any] | None:
    """Narrow an unknown payload to ``dict[str, Any]`` for pyright; None otherwise."""
    return cast(dict[str, Any], value) if isinstance(value, dict) else None


def _findings_from_tool_content(content: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the findings dict from an explore_video tool content payload.

    Production ``StepRecord.observation`` wraps tool output under
    ``observation.content``. Some older tests/fixtures used flatter shapes, so
    accept both ``{findings: ...}`` and a direct findings dict with
    ``extracted_data``.
    """
    findings = _as_dict(content.get("findings"))
    if findings is not None:
        return findings
    if "extracted_data" in content:
        return content
    return None


def _findings_from_video_findings(raw: Any) -> dict[str, Any] | None:
    """Extract latest findings from explore tool's nested ``video_findings``."""
    if isinstance(raw, list):
        for item in reversed(cast(list[Any], raw)):
            item_dict = _as_dict(item)
            if item_dict is None:
                continue
            findings = _findings_from_tool_content(item_dict)
            if findings is not None:
                return findings
        return None
    item_dict = _as_dict(raw)
    if item_dict is None:
        return None
    return _findings_from_tool_content(item_dict)


def latest_video_findings(state: AgentRuntimeState | None) -> dict[str, Any] | None:
    """Return the most recent video findings dict from direct or nested tool output."""
    if state is None:
        return None
    for step in reversed(state.steps):
        action = getattr(step, "action", None)
        observation = _as_dict(getattr(step, "observation", None))
        if observation is None:
            continue

        if action == "explore_video":
            content = _as_dict(observation.get("content"))
            if content is not None:
                findings = _findings_from_tool_content(content)
                if findings is not None:
                    return findings
            findings = _findings_from_tool_content(observation)
            if findings is not None:
                return findings

        if action == "explore":
            content = _as_dict(observation.get("content"))
            if content is None:
                continue
            findings = _findings_from_video_findings(content.get("video_findings"))
            if findings is not None:
                return findings
    return None


def extract_video_rules_and_samples(
    state: AgentRuntimeState | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | list[Any] | None]:
    """Pull ``rules`` and ``displayed_samples`` from the latest video findings."""
    findings = latest_video_findings(state)
    if findings is None:
        return None, None
    extracted = _as_dict(findings.get("extracted_data"))
    if extracted is None:
        return None, None

    rules = _as_dict(extracted.get("rules"))
    samples_raw = extracted.get("displayed_samples")
    samples: dict[str, Any] | list[Any] | None
    if isinstance(samples_raw, dict):
        samples = cast(dict[str, Any], samples_raw)
    elif isinstance(samples_raw, list):
        samples = cast(list[Any], samples_raw)
    else:
        samples = None
    return rules, samples
