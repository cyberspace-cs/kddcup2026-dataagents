"""Unit tests for the video_filter_audit module.

Each check is asserted in isolation: trigger case, no-trigger case, safe-edge case.
"""

from __future__ import annotations

from agents.benchmark.schema import AnswerTable
from agents.verification.video_filter_audit import (
    AuditFinding,
    audit_answer_table,
)


def test_audit_finding_dataclass_shape() -> None:
    f = AuditFinding(
        code="duplicate_export_rows",
        severity="reject",
        message="3 rows duplicated on export key",
        evidence={"duplicate_keys": ["A", "B", "C"]},
    )
    assert f.code == "duplicate_export_rows"
    assert f.severity == "reject"
    assert "duplicated" in f.message
    assert f.evidence["duplicate_keys"] == ["A", "B", "C"]


def test_audit_answer_table_returns_empty_for_no_rules() -> None:
    table = AnswerTable(columns=["x"], rows=[["a"], ["b"]])
    assert audit_answer_table(table, rules=None, samples=None) == []


def test_audit_answer_table_returns_empty_for_no_signal() -> None:
    table = AnswerTable(columns=["x"], rows=[["a"], ["b"]])
    rules = {"unrelated_key": "value"}
    assert audit_answer_table(table, rules=rules, samples=None) == []
