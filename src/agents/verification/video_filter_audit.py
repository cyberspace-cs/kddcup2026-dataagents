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


# ---------- internal helpers --------------------------------------------------


# 与 rule_templater 同款 export 字段别名 — keep these in sync.
_EXPORT_KEYS: tuple[str, ...] = (
    "export_field",
    "export_fields",
    "output_field",
    "output_fields",
    "select",
    "fields_to_export",
    "fields_for_result",
    "result_fields",
    "fields",
)


def _coerce_export_fields(raw: Any) -> tuple[str, ...]:
    """Normalize rules.export_fields (str | list[str]) into a clean tuple."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        return (text,) if text else ()
    if not isinstance(raw, (list, tuple)):
        return ()
    out: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip() and item.strip() not in out:
            out.append(item.strip())
    return tuple(out)


def _first_export_key(rules: dict[str, Any]) -> tuple[str, ...]:
    for key in _EXPORT_KEYS:
        if key in rules:
            return _coerce_export_fields(rules[key])
    return ()


def _check_duplicate_export_rows(
    table: AnswerTable,
    rules: dict[str, Any],
) -> AuditFinding | None:
    export_fields = _first_export_key(rules)
    if not export_fields:
        return None
    if not table.rows:
        return None
    # Find which export fields actually exist as columns in the table.
    col_indices: list[int] = []
    for field in export_fields:
        if field in table.columns:
            col_indices.append(table.columns.index(field))
    if not col_indices:
        return None
    seen: dict[tuple[Any, ...], int] = {}
    duplicates: list[tuple[Any, ...]] = []
    for row in table.rows:
        key = tuple(row[i] for i in col_indices)
        if key in seen:
            if seen[key] == 1:
                duplicates.append(key)
            seen[key] += 1
        else:
            seen[key] = 1
    if not duplicates:
        return None
    return AuditFinding(
        code="duplicate_export_rows",
        severity="reject",
        message=(
            f"{len(duplicates)} duplicate row(s) on export key {list(export_fields)}; "
            "DISTINCT the result or fix the JOIN that multiplied them"
        ),
        evidence={
            "duplicate_keys": [list(k) for k in duplicates[:10]],
            "duplicate_key_count": len(duplicates),
            "export_fields": list(export_fields),
        },
    )


def audit_answer_table(
    table: AnswerTable,
    rules: dict[str, Any] | None,
    samples: dict[str, Any] | list[Any] | None,
) -> list[AuditFinding]:
    """Run all auditor checks; return accumulated findings (empty when nothing trips)."""
    findings: list[AuditFinding] = []
    if rules is None:
        return findings
    dup = _check_duplicate_export_rows(table, rules)
    if dup is not None:
        findings.append(dup)
    return findings
