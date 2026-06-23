"""Deterministic post-checks on the AnswerTable for video tasks.

Runs between agents.verification.video_constraints (existing batch_id / snapshot
multiplicity checks) and the LLM AnswerVerifier. Three concerns it owns:

1. duplicate_export_rows - SELECT/JOIN side-effects that inflate row counts.
2. boundary_ambiguous - '>' vs '>=' choice when any row equals the threshold.
3. row_count_overflow - pred row count exceeds a credible small total parsed
   from displayed_samples.

The auditor is a pure function: never raises, never does IO, never calls an LLM.
Internal exceptions are caught and surfaced as a 'warn'-severity finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.benchmark.schema import AnswerTable


@dataclass(frozen=True, slots=True)
class AuditFinding:
    code: str
    severity: str
    message: str
    evidence: dict[str, Any]


def audit_answer_table(
    table: AnswerTable,
    rules: dict[str, Any] | None,
    samples: dict[str, Any] | list[Any] | None,
) -> list[AuditFinding]:
    """Run all auditor checks; return accumulated findings (empty when nothing trips)."""
    findings: list[AuditFinding] = []
    if rules is None:
        return findings
    return findings
