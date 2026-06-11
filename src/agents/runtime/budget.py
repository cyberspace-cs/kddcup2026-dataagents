"""Step-budget prompts and final-turn guardrails."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agents.llm.types import ModelMessage


@dataclass(frozen=True, slots=True)
class BudgetStatus:
    remaining: int
    ratio: float


def compute_budget_status(*, effective_turns: int, max_steps: int) -> BudgetStatus:
    """Return the current budget ratio and remaining effective model turns."""
    ratio = effective_turns / max_steps if max_steps > 0 else 0
    return BudgetStatus(remaining=max_steps - effective_turns, ratio=ratio)


def build_budget_prompt(
    *,
    status: BudgetStatus,
    effective_turns: int,
    max_steps: int,
    warning_fired: bool,
) -> tuple[ModelMessage | None, bool]:
    """Return an optional budget pressure message and the updated warning state."""
    if status.ratio >= 0.90:
        return (
            ModelMessage(
                role="user",
                content=(
                    f"CRITICAL: {status.remaining} step(s) left. "
                    "You MUST call `answer` NOW with whatever result you have. "
                    "Failure to answer = score 0."
                ),
            ),
            warning_fired,
        )
    if not warning_fired and status.ratio >= 0.70:
        return (
            ModelMessage(
                role="user",
                content=(
                    f"BUDGET WARNING: {effective_turns}/{max_steps} steps used, "
                    f"{status.remaining} remaining. "
                    "If you have computed ANY answer, call `answer` NOW — "
                    "an imperfect answer scores higher than no answer. "
                    "If not, do ONE final computation and submit."
                ),
            ),
            True,
        )
    return None, warning_fired


def is_final_step_blocking_active(*, status: BudgetStatus, last_step_retries: int) -> bool:
    """末步守卫是否处于激活区间（预算 ≥90%、剩余 ≤1、重试次数未耗尽）。

    不检查 calls 内容——仅判断"是否处于需要守卫干预的时间窗口"。
    供 agent.py 在末步混合轮场景下复用，无需重复阈值数字。
    """
    return status.remaining <= 1 and status.ratio >= 0.90 and last_step_retries < 2


def should_block_non_answer_final_turn(
    *,
    status: BudgetStatus,
    last_step_retries: int,
    calls: Sequence[Any],
) -> bool:
    """Return whether the final-step guard should force an immediate answer call."""
    return is_final_step_blocking_active(
        status=status, last_step_retries=last_step_retries
    ) and not any(
        getattr(call, "name", None) == "answer" or getattr(call, "name", None) == "report"
        for call in calls
    )


def final_step_block_error() -> str:
    """Return the observation error used when the final-step guard blocks a call."""
    return (
        "BLOCKED: final step must be a terminal tool call (`answer` or `report`). "
        "Call it with your best result immediately."
    )
