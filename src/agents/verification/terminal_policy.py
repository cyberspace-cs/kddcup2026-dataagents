"""Terminal-answer verification and fallback policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.benchmark.schema import AnswerTable, PublicTask
from agents.runtime.state import AgentRuntimeState
from agents.tools import (
    build_answer_table_without_side_effects,
    cleanup_answer_artifacts,
)
from agents.verification.answer import AnswerVerifier

_EXECUTE_ACTIONS = ("execute_python", "execute_context_sql")
_PRODUCING_CODE_STEPS = 2
_PRODUCING_CODE_CHARS = 2000


def recent_execute_code(state: AgentRuntimeState | None) -> str | None:
    """Inputs of the most recent execute steps, for the answer verifier.

    The verifier's NULL-filter check needs to see the SQL/python that built
    the submitted table; the last couple of execute steps before the answer
    call are that code in practice. Returns None when no execute step exists.
    """
    if state is None:
        return None
    snippets: list[str] = []
    for step in reversed(state.steps):
        if step.action not in _EXECUTE_ACTIONS:
            continue
        payload = step.action_input or {}
        code = payload.get("code") or payload.get("sql") or ""
        if not str(code).strip():
            continue
        snippets.append(f"[{step.action}]\n{str(code)[:_PRODUCING_CODE_CHARS]}")
        if len(snippets) >= _PRODUCING_CODE_STEPS:
            break
    if not snippets:
        return None
    return "\n\n".join(reversed(snippets))


@dataclass(frozen=True, slots=True)
class TerminalAnswerRejection:
    content: dict[str, str]


@dataclass(slots=True)
class TerminalAnswerPolicy:
    """Coordinate verifier rejection limits and last-valid-answer fallback."""

    verifier: AnswerVerifier | None
    rejections_used: int = 0
    fallback_answer: AnswerTable | None = None
    fallback_answer_arguments: dict[str, Any] | None = field(default=None)

    def maybe_reject(
        self,
        *,
        task: PublicTask,
        arguments: dict[str, Any],
        is_terminal: bool,
        remaining_steps: int,
        state: AgentRuntimeState | None = None,
    ) -> TerminalAnswerRejection | None:
        """Return a rejection observation when the terminal answer verifier rejects."""
        if (
            self.verifier is None
            or not is_terminal
            or self.rejections_used >= self.verifier.max_rejections
            or remaining_steps <= 1
        ):
            return None
        rejection = self.verifier.verify(
            task.question,
            arguments,
            producing_code=recent_execute_code(state),
        )
        if rejection is None:
            return None

        self.rejections_used += 1
        candidate = build_answer_table_without_side_effects(task, arguments)
        if candidate is not None and candidate.columns and candidate.rows:
            self.fallback_answer = candidate
            self.fallback_answer_arguments = dict(arguments)
        return TerminalAnswerRejection(content={"answer_rejected": rejection})

    def promote_fallback_answer(self, *, task: PublicTask, state: AgentRuntimeState) -> None:
        """Promote the latest valid verifier-rejected answer if the loop otherwise failed."""
        if state.answer is not None or self.fallback_answer is None:
            return
        if self.fallback_answer_arguments is not None:
            cleanup_answer_artifacts(task, self.fallback_answer_arguments)
        state.answer = self.fallback_answer
        state.failure_reason = None
