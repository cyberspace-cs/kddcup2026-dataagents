"""Explorer sub-agent runner.

Spawns a short ReAct loop with read-only tools to explore the data
landscape for a given task+question, then returns the findings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

from agents.agent import ReActAgent, ReActAgentConfig
from agents.benchmark.schema import PublicTask
from agents.explorer.prompt import EXPLORER_SYSTEM_PROMPT
from agents.explorer.registry import create_explorer_tool_registry
from agents.llm import ModelAdapter, ToolProtocol
from agents.runtime import StepCallback

logger = logging.getLogger(__name__)

EXPLORER_MAX_STEPS = 16


@dataclass(frozen=True, slots=True)
class ExplorerResult:
    """Structured output from an Explorer run."""

    findings: str
    steps_used: int
    success: bool
    failure_reason: str | None = None

    def to_tool_content(self) -> dict[str, Any]:
        """Format as a tool observation for the main agent."""
        if not self.success:
            return {
                "explorer_status": "failed",
                "failure_reason": self.failure_reason or "unknown",
                "steps_used": self.steps_used,
            }
        return {
            "explorer_status": "ok",
            "findings": self.findings,
            "steps_used": self.steps_used,
        }


def run_explorer(
    task: PublicTask,
    question: str,  # reserved for future prompt override
    model: ModelAdapter,
    *,
    protocol: ToolProtocol = "native",
    max_steps: int = EXPLORER_MAX_STEPS,
    step_callback: StepCallback | None = None,
) -> ExplorerResult:
    """Run the Explorer sub-agent on a task and return findings.

    The Explorer uses a read-only tool subset (no execute_python, no answer)
    and terminates via a `report` tool. It aims to complete in 3-6 steps.
    """
    registry = create_explorer_tool_registry()

    agent = ReActAgent(
        model=model,
        tools=registry,
        config=ReActAgentConfig(
            max_steps=max_steps,
            max_empty_tool_call_retries=3,
        ),
        system_prompt=EXPLORER_SYSTEM_PROMPT,
        protocol=protocol,
        step_callback=step_callback,
    )

    result = agent.run(task)

    if result.answer is not None:
        # report tool returns findings in the answer table's first row
        # but we stored it in ToolExecutionResult.content, not answer table.
        # Extract from the last successful step instead.
        pass

    # Extract findings from steps: find the answer tool call
    findings = _extract_findings(result.steps)

    if findings is not None:
        return ExplorerResult(
            findings=findings,
            steps_used=len(result.steps),
            success=True,
        )

    # Fallback: model didn't call answer — synthesize findings from observations
    synthesized = _synthesize_findings(result.steps)
    if synthesized:
        logger.info("Explorer fallback: synthesized findings from %d steps", len(result.steps))
        return ExplorerResult(
            findings=synthesized,
            steps_used=len(result.steps),
            success=True,
        )

    return ExplorerResult(
        findings="",
        steps_used=len(result.steps),
        success=False,
        failure_reason=result.failure_reason or "Explorer did not submit a report",
    )


def _synthesize_findings(steps: list[Any]) -> str:
    """Build a findings string from successful tool observations when model
    didn't call answer. Extracts the most useful data from each step."""
    import json

    sections: dict[str, Any] = {}
    for step in steps:
        action: str = getattr(step, "action", "")
        ok: bool = getattr(step, "ok", False)
        if not ok or action == "__error__":
            continue
        obs_raw: Any = getattr(step, "observation", {})
        if not isinstance(obs_raw, dict):
            continue
        obs = cast(dict[str, Any], obs_raw)
        raw: Any = obs.get("content", obs)
        content = cast(dict[str, Any], raw) if isinstance(raw, dict) else obs

        if action == "inspect_files":
            files_summary: list[dict[str, Any]] = []
            file_list: list[Any] = cast(list[Any], content.get("files", []))
            for f_item in file_list:
                f = cast(dict[str, Any], f_item)
                entry: dict[str, Any] = {
                    "path": f.get("path", ""),
                    "format": f.get("format", ""),
                }
                raw_schema: Any = f.get("schema", {})
                if isinstance(raw_schema, dict):
                    schema = cast(dict[str, Any], raw_schema)
                    if "columns" in schema:
                        entry["columns"] = schema["columns"]
                    if "row_count" in schema:
                        entry["row_count"] = schema["row_count"]
                    if "unit_hint" in schema:
                        entry["unit_hint"] = schema["unit_hint"]
                    if "tables" in schema:
                        tables = cast(dict[str, Any], schema["tables"])
                        entry["tables"] = {
                            t: {
                                "columns": cast(dict[str, Any], info).get("columns", []),
                                "row_count": cast(dict[str, Any], info).get("row_count"),
                            }
                            for t, info in tables.items()
                            if isinstance(info, dict)
                        }
                files_summary.append(entry)
            sections["files"] = files_summary

        elif action == "read_csv":
            key: str = str(content.get("path", "csv"))
            sections.setdefault("csv_previews", {})[key] = {
                "columns": content.get("columns", []),
                "dtypes": content.get("dtypes", []),
                "row_count": content.get("row_count"),
            }

        elif action == "read_doc":
            raw_matches: Any = content.get("keyword_match", {})
            if isinstance(raw_matches, dict):
                matches = cast(dict[str, Any], raw_matches)
                if matches.get("found"):
                    match_list: list[Any] = cast(list[Any], matches.get("matches", []))
                    sections.setdefault("doc_matches", []).append(
                        {
                            "keyword": matches.get("keyword"),
                            "match_count": matches.get("match_count"),
                            "snippets": [
                                cast(dict[str, Any], m).get("text", "")[:200]
                                for m in match_list[:5]
                            ],
                        }
                    )

        elif action == "grep_context":
            sections.setdefault("grep_results", []).append(
                {
                    "pattern": content.get("pattern"),
                    "match_count": content.get("match_count"),
                    "results": cast(list[Any], content.get("results", []))[:10],
                }
            )

        elif action == "execute_context_sql":
            sections.setdefault("sql_results", []).append(
                {
                    "columns": content.get("columns", []),
                    "row_count": content.get("row_count"),
                    "rows": cast(list[Any], content.get("rows", []))[:5],
                }
            )

    return json.dumps(sections, ensure_ascii=False) if sections else ""


def _extract_findings(steps: list[Any]) -> str | None:
    """Walk steps backward to find the report tool's findings."""
    for step in reversed(steps):
        if getattr(step, "action", None) == "report":
            raw_args: Any = getattr(step, "action_input", None)
            if isinstance(raw_args, dict) and "findings" in raw_args:
                return str(cast(dict[str, Any], raw_args)["findings"])
    return None
