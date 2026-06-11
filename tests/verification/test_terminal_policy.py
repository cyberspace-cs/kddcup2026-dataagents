"""Behavior tests for terminal-policy producing-code extraction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from agents.runtime.state import AgentRuntimeState
from agents.verification.terminal_policy import recent_execute_code


def _state(*steps: tuple[str, dict[str, Any]]) -> AgentRuntimeState:
    state = AgentRuntimeState()
    state.steps.extend(
        SimpleNamespace(action=action, action_input=action_input)  # type: ignore[arg-type]
        for action, action_input in steps
    )
    return state


def test_recent_execute_code_returns_last_two_in_order() -> None:
    state = _state(
        ("execute_python", {"code": "first = 1"}),
        ("execute_context_sql", {"sql": "SELECT 1"}),
        ("read_csv", {"path": "a.csv"}),
        ("execute_python", {"code": "final = build_answer()"}),
    )

    code = recent_execute_code(state)

    assert code is not None
    assert "SELECT 1" in code and "final = build_answer()" in code
    assert "first = 1" not in code  # only the most recent two execute steps
    assert code.index("SELECT 1") < code.index("final = build_answer()")


def test_recent_execute_code_skips_empty_inputs_and_handles_absence() -> None:
    assert recent_execute_code(None) is None
    assert recent_execute_code(_state(("read_csv", {"path": "a.csv"}))) is None

    state = _state(
        ("execute_python", {"code": "real = 1"}),
        ("execute_python", {"code": "   "}),
    )
    code = recent_execute_code(state)
    assert code is not None
    assert "real = 1" in code
