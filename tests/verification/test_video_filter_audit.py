"""Unit tests for the video_filter_audit module.

Each check is asserted in isolation: trigger case, no-trigger case, safe-edge case.
"""

from __future__ import annotations

from typing import Any as _Any

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


# ---------- boundary_ambiguous -------------------------------------------------


def test_boundary_ambiguous_triggers_when_row_equals_threshold() -> None:
    table = AnswerTable(
        columns=["company", "value"],
        rows=[["A", 100.0], ["B", 200.0], ["C", 100.0]],
    )
    rules = {
        "field": "value",
        "operator": ">",
        "threshold": "100.0",
    }
    findings = audit_answer_table(table, rules=rules, samples=None)
    warns = [f for f in findings if f.code == "boundary_ambiguous"]
    assert len(warns) == 1
    assert warns[0].severity == "warn"
    ev = warns[0].evidence
    assert ev["operator_chosen"] == ">"
    assert ev["rows_at_threshold"] == 2
    assert ev["count_if_strict"] == 1
    assert ev["count_if_inclusive"] == 3


def test_boundary_ambiguous_silent_when_no_equal_row() -> None:
    table = AnswerTable(columns=["v"], rows=[[150.0], [200.0]])
    rules = {"field": "v", "operator": ">", "threshold": "100.0"}
    findings = audit_answer_table(table, rules=rules, samples=None)
    assert [f for f in findings if f.code == "boundary_ambiguous"] == []


def test_boundary_ambiguous_silent_when_operator_not_inequality() -> None:
    table = AnswerTable(columns=["v"], rows=[[100.0]])
    rules = {"field": "v", "operator": "==", "threshold": "100.0"}
    findings = audit_answer_table(table, rules=rules, samples=None)
    assert [f for f in findings if f.code == "boundary_ambiguous"] == []


def test_boundary_ambiguous_silent_when_threshold_missing() -> None:
    table = AnswerTable(columns=["v"], rows=[[100.0]])
    rules = {"field": "v", "operator": ">"}
    findings = audit_answer_table(table, rules=rules, samples=None)
    assert findings == []


def test_boundary_ambiguous_silent_when_threshold_unparseable() -> None:
    table = AnswerTable(columns=["v"], rows=[[100.0]])
    rules = {"field": "v", "operator": ">", "threshold": "high"}
    findings = audit_answer_table(table, rules=rules, samples=None)
    assert findings == []


def test_boundary_ambiguous_silent_when_field_absent_in_table() -> None:
    table = AnswerTable(columns=["other"], rows=[[100.0]])
    rules = {"field": "v", "operator": ">", "threshold": "100.0"}
    findings = audit_answer_table(table, rules=rules, samples=None)
    assert findings == []


# ---------- row_count_overflow -------------------------------------------------


def test_row_count_overflow_triggers_when_pred_much_larger_than_samples_total() -> None:
    table = AnswerTable(columns=["x"], rows=[[i] for i in range(50)])
    rules = {"field": "x"}
    samples = {"total_count": 10}  # video says ~10 items, pred has 50
    findings = audit_answer_table(table, rules=rules, samples=samples)
    warns = [f for f in findings if f.code == "row_count_overflow"]
    assert len(warns) == 1
    assert warns[0].severity == "warn"
    assert warns[0].evidence["samples_total"] == 10
    assert warns[0].evidence["df_rows"] == 50


def test_row_count_overflow_triggers_when_pred_at_least_double_list_length() -> None:
    table = AnswerTable(columns=["x"], rows=[[i] for i in range(20)])
    rules = {"field": "x"}
    samples = ["a", "b", "c", "d", "e"]  # 5 items - pred 20 (>= 2x)
    findings = audit_answer_table(table, rules=rules, samples=samples)
    warns = [f for f in findings if f.code == "row_count_overflow"]
    assert len(warns) == 1


def test_row_count_overflow_silent_when_within_2x() -> None:
    table = AnswerTable(columns=["x"], rows=[[i] for i in range(8)])
    rules = {"field": "x"}
    samples = {"total_count": 5}  # 8 < 2*5
    findings = audit_answer_table(table, rules=rules, samples=samples)
    assert [f for f in findings if f.code == "row_count_overflow"] == []


def test_row_count_overflow_silent_when_samples_unparseable() -> None:
    table = AnswerTable(columns=["x"], rows=[[i] for i in range(50)])
    rules = {"field": "x"}
    samples = {"description": "many rows of data"}  # no number
    findings = audit_answer_table(table, rules=rules, samples=samples)
    assert [f for f in findings if f.code == "row_count_overflow"] == []


def test_row_count_overflow_silent_when_samples_none() -> None:
    table = AnswerTable(columns=["x"], rows=[[i] for i in range(50)])
    rules = {"field": "x"}
    findings = audit_answer_table(table, rules=rules, samples=None)
    assert findings == []


# ---------- aggregator: never raises -------------------------------------------


def test_audit_answer_table_swallows_internal_exception(monkeypatch: _Any) -> None:
    """If a check raises, the auditor must catch it and emit a warn finding."""
    from agents.verification import video_filter_audit as vfa

    def boom(*_args: _Any, **_kwargs: _Any) -> _Any:
        raise RuntimeError("forced check failure")

    monkeypatch.setattr(vfa, "_check_duplicate_export_rows", boom)
    table = AnswerTable(columns=["x"], rows=[["a"]])
    findings = audit_answer_table(table, rules={"export_fields": ["x"]}, samples=None)
    assert any(f.code == "audit_internal_error" and f.severity == "warn" for f in findings)


def test_audit_answer_table_rejects_non_dict_rules_safely() -> None:
    """Rules of an unexpected shape (e.g. accidentally a string) yield no findings."""
    table = AnswerTable(columns=["x"], rows=[["a"]])
    bogus: _Any = "not a dict"
    findings = audit_answer_table(table, rules=bogus, samples=None)
    assert findings == []


def test_audit_answer_table_returns_multiple_findings() -> None:
    """Duplicate export rows + boundary ambiguity accumulate into one finding list."""
    table = AnswerTable(
        columns=["company", "value"],
        rows=[["A", 100.0], ["A", 100.0], ["B", 150.0]],
    )
    rules = {
        "export_fields": ["company"],
        "field": "value",
        "operator": ">",
        "threshold": "100.0",
    }
    findings = audit_answer_table(table, rules=rules, samples=None)
    codes = {f.code for f in findings}
    assert "duplicate_export_rows" in codes
    assert "boundary_ambiguous" in codes
