"""Behavior of the container path's budget-derived per-task timeout."""

from __future__ import annotations

import pytest

from agents.submission import derive_task_timeout


@pytest.mark.parametrize(
    ("configured", "global_budget", "workers", "task_count", "expected"),
    [
        # Fair share above the configured floor → fair share wins.
        (1200, 42_000.0, 4, 50, 3360),
        # Single task → fair share exceeds the budget → capped at the budget.
        (1200, 42_000.0, 4, 1, 42_000),
        # Many tasks → fair share below the floor → floor wins.
        (1200, 42_000.0, 4, 500, 1200),
        # Budget below the configured floor → the budget cap wins over the floor.
        (1200, 600.0, 4, 50, 600),
        # configured <= 0 means "timeout disabled"; derivation must not enable it.
        (0, 42_000.0, 4, 50, 0),
        (-1, 42_000.0, 4, 50, -1),
        # No tasks → nothing to scale, configured value passes through.
        (1200, 42_000.0, 4, 0, 1200),
    ],
)
def test_derive_task_timeout(
    configured: int, global_budget: float, workers: int, task_count: int, expected: int
) -> None:
    assert (
        derive_task_timeout(
            configured,
            global_budget=global_budget,
            workers=workers,
            task_count=task_count,
        )
        == expected
    )
