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


def test_recent_execute_code_returns_last_execute_step_only() -> None:
    state = _state(
        ("execute_python", {"code": "first = 1"}),
        ("execute_context_sql", {"sql": "SELECT 1"}),
        ("read_csv", {"path": "a.csv"}),
        ("execute_python", {"code": "final = build_answer()"}),
    )

    code = recent_execute_code(state)

    assert code is not None
    assert "final = build_answer()" in code
    assert "SELECT 1" not in code
    assert "first = 1" not in code


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


# ---------- filter_audit_enabled field ----------------------------------------


from agents.verification.terminal_policy import TerminalAnswerPolicy


def test_terminal_policy_filter_audit_enabled_defaults_false() -> None:
    # Backward-compatible default: existing call sites don't pass it.
    policy = TerminalAnswerPolicy(verifier=None)
    assert policy.filter_audit_enabled is False


def test_terminal_policy_filter_audit_enabled_can_be_true() -> None:
    policy = TerminalAnswerPolicy(verifier=None, filter_audit_enabled=True)
    assert policy.filter_audit_enabled is True


# ---------- maybe_reject + auditor wiring -------------------------------------


from agents.benchmark.schema import AnswerTable
from agents.verification.video_filter_audit import AuditFinding


def _stub_task() -> Any:
    """Minimal SimpleNamespace task that exercises build_answer_table_without_side_effects."""
    assets = SimpleNamespace(task_dir="/tmp", context_dir="/tmp")
    return SimpleNamespace(task_id="task_x", question="qq", assets=assets)


def test_maybe_reject_runs_auditor_when_flag_enabled(monkeypatch: Any) -> None:
    """Auditor reject yields a TerminalAnswerRejection and increments rejections_used."""
    from agents.verification import terminal_policy as tp

    monkeypatch.setattr(
        tp,
        "build_answer_table_without_side_effects",
        lambda task, args: AnswerTable(columns=["x"], rows=[["a"]]),
    )
    monkeypatch.setattr(
        tp,
        "evaluate_video_constraints",
        lambda state, q, n: None,  # let auditor be the rejecter
    )
    monkeypatch.setattr(
        tp,
        "audit_answer_table",
        lambda table, rules, samples: [
            AuditFinding(
                code="duplicate_export_rows",
                severity="reject",
                message="dup",
                evidence={},
            )
        ],
    )
    monkeypatch.setattr(
        tp,
        "_extract_video_rules_and_samples",
        lambda state: ({"export_fields": ["x"]}, None),
    )

    policy = tp.TerminalAnswerPolicy(verifier=None, filter_audit_enabled=True)
    result = policy.maybe_reject(
        task=_stub_task(),
        arguments={},
        is_terminal=True,
        remaining_steps=5,
        state=None,
    )
    assert result is not None
    assert "answer_rejected" in result.content
    assert policy.rejections_used == 1


def test_maybe_reject_skips_auditor_when_flag_disabled(monkeypatch: Any) -> None:
    from agents.verification import terminal_policy as tp

    monkeypatch.setattr(
        tp,
        "build_answer_table_without_side_effects",
        lambda task, args: AnswerTable(columns=["x"], rows=[["a"]]),
    )
    monkeypatch.setattr(
        tp,
        "evaluate_video_constraints",
        lambda state, q, n: None,
    )
    monkeypatch.setattr(
        tp,
        "audit_answer_table",
        lambda table, rules, samples: [
            AuditFinding(code="duplicate_export_rows", severity="reject", message="dup", evidence={})
        ],
    )

    policy = tp.TerminalAnswerPolicy(verifier=None, filter_audit_enabled=False)
    result = policy.maybe_reject(
        task=_stub_task(),
        arguments={},
        is_terminal=True,
        remaining_steps=5,
        state=None,
    )
    # Verifier is None and auditor is disabled — no rejection
    assert result is None
    assert policy.rejections_used == 0


def test_maybe_reject_auditor_warns_do_not_block(monkeypatch: Any) -> None:
    from agents.verification import terminal_policy as tp

    monkeypatch.setattr(
        tp,
        "build_answer_table_without_side_effects",
        lambda task, args: AnswerTable(columns=["x"], rows=[["a"]]),
    )
    monkeypatch.setattr(
        tp,
        "evaluate_video_constraints",
        lambda state, q, n: None,
    )
    monkeypatch.setattr(
        tp,
        "audit_answer_table",
        lambda table, rules, samples: [
            AuditFinding(code="boundary_ambiguous", severity="warn", message="warn", evidence={})
        ],
    )
    monkeypatch.setattr(
        tp,
        "_extract_video_rules_and_samples",
        lambda state: ({}, None),
    )

    policy = tp.TerminalAnswerPolicy(verifier=None, filter_audit_enabled=True)
    result = policy.maybe_reject(
        task=_stub_task(),
        arguments={},
        is_terminal=True,
        remaining_steps=5,
        state=None,
    )
    assert result is None
    assert policy.rejections_used == 0


def test_extract_video_rules_and_samples_pulls_from_latest_explore_video() -> None:
    from agents.verification.terminal_policy import _extract_video_rules_and_samples

    state = AgentRuntimeState()
    state.steps.extend(
        SimpleNamespace(  # type: ignore[arg-type]
            action="explore_video",
            ok=True,
            observation={
                "findings": {
                    "extracted_data": {
                        "rules": {"field": "value", "operator": ">"},
                        "displayed_samples": {"total_count": 7},
                    }
                }
            },
        )
        for _ in range(1)
    )
    rules, samples = _extract_video_rules_and_samples(state)
    assert rules == {"field": "value", "operator": ">"}
    assert samples == {"total_count": 7}


def test_extract_video_rules_and_samples_returns_none_on_missing() -> None:
    from agents.verification.terminal_policy import _extract_video_rules_and_samples

    assert _extract_video_rules_and_samples(None) == (None, None)
    assert _extract_video_rules_and_samples(_state(("read_csv", {"path": "a.csv"}))) == (None, None)
