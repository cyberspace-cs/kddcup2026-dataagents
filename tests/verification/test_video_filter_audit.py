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


# ---------- duplicate_export_rows ----------------------------------------------


def test_duplicate_export_rows_triggers_on_repeat() -> None:
    table = AnswerTable(
        columns=["shareholder"],
        rows=[["国家股"], ["公司A"], ["国家股"], ["公司B"]],
    )
    rules = {"export_fields": ["shareholder"]}
    findings = audit_answer_table(table, rules=rules, samples=None)
    rejects = [f for f in findings if f.code == "duplicate_export_rows"]
    assert len(rejects) == 1
    assert rejects[0].severity == "reject"
    assert "国家股" in str(rejects[0].evidence)


def test_duplicate_export_rows_silent_on_unique() -> None:
    table = AnswerTable(columns=["x"], rows=[["a"], ["b"], ["c"]])
    rules = {"export_fields": ["x"]}
    findings = audit_answer_table(table, rules=rules, samples=None)
    assert [f for f in findings if f.code == "duplicate_export_rows"] == []


def test_duplicate_export_rows_silent_on_missing_export_fields() -> None:
    table = AnswerTable(columns=["x"], rows=[["a"], ["a"]])
    rules = {"unrelated_key": "value"}
    findings = audit_answer_table(table, rules=rules, samples=None)
    assert findings == []


def test_duplicate_export_rows_silent_on_empty_table() -> None:
    table = AnswerTable(columns=["x"], rows=[])
    rules = {"export_fields": ["x"]}
    findings = audit_answer_table(table, rules=rules, samples=None)
    assert findings == []


def test_duplicate_export_rows_silent_when_export_col_absent_in_table() -> None:
    table = AnswerTable(columns=["other"], rows=[["a"], ["b"]])
    rules = {"export_fields": ["shareholder"]}
    findings = audit_answer_table(table, rules=rules, samples=None)
    # export field not in table columns - auditor cannot judge, stays silent
    assert findings == []
