from __future__ import annotations

from pathlib import Path

from data_agent_baseline.agents.prompt import (
    build_native_system_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord


def _task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(
            task_id="task_1",
            difficulty="easy",
            question="Find the requested value.",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def test_native_system_prompt_requires_tool_calls() -> None:
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "must use tool_calls" in lower
    assert "do not reply with plain text" in lower
    assert "plain text will not deliver the answer" in lower
    assert "each row length must match the number of columns" in lower


def test_native_system_prompt_keeps_tool_calls_out_of_reasoning() -> None:
    prompt = build_native_system_prompt()
    lower = prompt.lower()

    assert "the only way to invoke a tool is to emit a structured tool call" in lower
    assert "reasoning/thinking text is a private scratchpad" in lower
    assert "never reaches tools and is never executed" in lower
    assert "textual tool-call drafts are ignored" in lower
    assert "do not rehearse or preview the call in reasoning" in lower
    assert "structured `answer` call" in lower


def test_native_system_prompt_includes_tool_decision_tree() -> None:
    prompt = build_native_system_prompt()

    for keyword in [
        "list_context",
        "read_csv",
        "read_json",
        "read_doc",
        "inspect_sqlite_schema",
        "execute_context_sql",
        "execute_python",
        "answer",
    ]:
        assert keyword in prompt


def test_native_system_prompt_custom_override_is_preserved() -> None:
    assert build_native_system_prompt("custom native prompt") == "custom native prompt"


def test_task_prompt_adds_near_tool_call_instruction(tmp_path: Path) -> None:
    prompt = build_task_prompt(_task(tmp_path))
    lower = prompt.lower()

    assert "paths inside the context directory" in lower
    assert "do not prepend 'context/'" in lower
    assert "use the provided tools to answer this task" in lower
    assert "do not reply with plain text; make a tool call" in lower
    assert "call list_context" in lower
    assert "when final, call answer" in lower
    assert "emit tool calls as structured calls, not in reasoning" in lower


def test_text_system_prompt_keeps_text_protocol_contract() -> None:
    prompt = build_system_prompt("- list_context: Use first to inspect context.")

    assert "single ```json fenced block" in prompt
    assert "tool_calls" not in prompt
