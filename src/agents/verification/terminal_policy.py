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
    ) -> TerminalAnswerRejection | None:
        """Return a rejection observation when the terminal answer verifier rejects."""
        if (
            self.verifier is None
            or not is_terminal
            or self.rejections_used >= self.verifier.max_rejections
            or remaining_steps <= 1
        ):
            return None
        rejection = self.verifier.verify(task.question, arguments)
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
