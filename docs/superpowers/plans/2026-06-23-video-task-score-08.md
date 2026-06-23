# Video Task Score 0.745 → 0.80 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the 30-task video subset mean from ~0.745 to ≥ 0.80 by adding deterministic post-checks (filter audit) and an abbreviation-column preference hint, validated by per-Phase A/B runs.

**Architecture:** Two interventions hook into the existing pipeline without touching the ReAct main loop or sub-agent prompt. (1) A new `video_filter_audit.py` module inspects the `AnswerTable` between the existing `evaluate_video_constraints` and the LLM `AnswerVerifier`, sharing the same `rejections_used` budget. (2) `rule_templater.py` gains a `question=` parameter; when the question text contains abbreviation keywords, the rendered filter recipe surfaces a `prefer_column_kind: abbreviation` hint. Both ship behind toggles for clean rollback.

**Tech Stack:** Python 3.14, `uv run` for all commands (per CLAUDE.md), pydantic dataclasses for config, pytest for tests, the existing `AnswerTable(columns, rows)` shape (no pandas).

**Spec:** `docs/superpowers/specs/2026-06-23-video-task-score-08-design.md`

---

## File map

**Create:**
- `src/agents/verification/video_filter_audit.py` — auditor module
- `tests/verification/test_video_filter_audit.py` — auditor unit tests
- `scripts/triage_failures.py` — diagnostic-only failure-mode classifier

**Modify:**
- `src/agents/config.py` — add `video_filter_audit` literal field on `AgentConfig`
- `src/agents/verification/terminal_policy.py` — add `filter_audit_enabled` flag + auditor call in `maybe_reject`
- `src/agents/agent.py` — thread the flag through `ReActAgent` constructor into `TerminalAnswerPolicy(...)` call site
- `src/agents/application.py` — read `config.agent.video_filter_audit == "enabled"` and pass through
- `tests/verification/test_terminal_policy.py` — add wiring tests
- `src/agents/video/rule_templater.py` — add `question=` param + `prefer_abbr_column` field + render hint line
- `src/agents/tools/explore_video.py` — pass `task.question` into `synthesize_filter_recipe`
- `tests/video/test_rule_templater.py` — extend with question-aware cases
- `configs/react.example.yaml` — document `video_filter_audit` toggle

---

## Conventions

- **Run Python via `uv run`** — never `python`, `pip`, or `.venv/bin/python` (CLAUDE.md).
- **Commit after each task** (or task group when explicitly noted).
- **One assertion per logical concern in tests.** Use parametrize where shapes repeat.
- **Never edit `configs/react.local.yaml` in a commit** — it carries real API keys. Local overrides in that file are operator-only.
- **Use chained `cmd1 && cmd2` in bash** when running multiple commands in one step.

---

## Task 1: Phase 1 baseline — measurement procedure

**Goal:** Anchor the real `react.local.yaml + thinking=true + max_tokens=16384` baseline on `main@746b343` before writing any code. Refresh the failure-mode classification. **This is a go/no-go gate** for the rest of the plan.

**Files:** none (procedure only; no commits).

- [ ] **Step 1: Apply local-only config overrides (do not commit)**

Edit `configs/react.local.yaml` in this worktree to set:

```yaml
agent:
  enable_thinking: true       # was false
  max_tokens: 16384           # was 8192
```

These are operator overrides for the baseline; they are not committed (the file holds real API keys).

- [ ] **Step 2: Verify the worktree still tests clean before running**

```bash
uv run pytest tests/verification tests/video -q
```

Expected: all tests pass. If anything fails, stop and investigate before running the baseline.

- [ ] **Step 3: Run baseline #1 (fresh)**

```bash
uv run dabench run-benchmark --config configs/react.local.yaml
```

Expected: produces `artifacts/runs/<run_id_1>/` with 30 video tasks (the 30 non-video tasks are blocklisted in `react.local.yaml`).

- [ ] **Step 4: Score baseline #1**

```bash
uv run python scripts/score_predictions.py \
    --pred artifacts/runs/<run_id_1> \
    --gold data/public/output \
    --lambda 0.1 \
    --output artifacts/runs/<run_id_1>/scores.json \
    --verbose
```

Record: `aggregate.total_score`, `aggregate.mean_recall`, `aggregate.tasks_scored_nonzero`, walltime/task = total walltime / 30.

- [ ] **Step 5: Run baseline #2 (warm — same config, different `run_id`)**

```bash
uv run dabench run-benchmark --config configs/react.local.yaml
```

The seed in `react.local.yaml` is fixed at 1024, so this is a noise-floor measurement of the same config. Score it the same way.

- [ ] **Step 6: Per-task triage**

```bash
uv run python -c "
import json
from pathlib import Path
run = Path('artifacts/runs/<run_id_1>')
scores = json.loads((run / 'scores.json').read_text(encoding='utf-8'))
per = scores['per_task']
imperfect = sorted(
    [(t, info) for t, info in per.items()
     if info.get('status') in ('no_match', 'scored') and info.get('score', 0) < 0.99],
    key=lambda x: x[1].get('score', 0)
)
for t, info in imperfect:
    print(f\"{t}: score={info.get('score',0):.3f} status={info.get('status')} \"
          f\"gold={info.get('gold_columns',0)} pred={info.get('predicted_columns',0)} \"
          f\"matched={info.get('matched_columns',0)} extra={info.get('extra_columns',0)}\")
"
```

For each imperfect task, diff `data/public/output/<task>/gold.csv` against `artifacts/runs/<run_id_1>/<task>/prediction.csv` and assign one of: `threshold/filter`, `boundary/duplicate`, `abbr_vs_full`, `snapshot_multiplicity`, `zero_count`, `other`.

- [ ] **Step 7: Cross-check `task_41` zero-count fix**

`task_41` was the Category-A failure addressed by `746b343 fix(prompts): video qualifier matching, zero-count validity`. Check whether it scores ≥ 0.99 in either baseline.

- [ ] **Step 8: Cross-check `finish_reason='length'`**

Inspect `artifacts/traces.db` for any LLM generation span with `finish_reason='length'`:

```bash
uv run python -c "
import sqlite3
conn = sqlite3.connect('artifacts/traces.db')
cur = conn.execute(\"SELECT span_id, attributes FROM spans WHERE attributes LIKE '%length%' LIMIT 20\")
for row in cur.fetchall(): print(row)
"
```

Count distinct tasks that hit `length`. If > 5 tasks hit it at `max_tokens=16384`, **stop and revise the design** (this indicates a deeper budget issue beyond what raising max_tokens fixes).

- [ ] **Step 9: Go / no-go decision**

Append the following to `docs/superpowers/specs/2026-06-23-video-task-score-08-design.md` §11 (Attempt log) as the first row:

```markdown
| 2026-06-23 | Phase 1 | baseline (react.local.yaml + thinking=true + max_tokens=16384) | n/a | <mean from run #1> | n/a | <walltime/task> | reference |
| 2026-06-23 | Phase 1 | warm rerun, same config | n/a | <mean from run #2> | n/a | <walltime/task> | reference |
```

Commit the doc update:

```bash
git add docs/superpowers/specs/2026-06-23-video-task-score-08-design.md
git commit -m "docs(video): Phase 1 baseline measurement (react.local.yaml + thinking=true)"
```

**Decision rule:**

- If both baselines have mean ≥ 0.80: STOP. The project is already done; document and exit.
- If `task_41` is now ≥ 0.99 AND the dominant failure modes are still threshold/boundary + abbr_vs_full as in spec §2: proceed to Task 2.
- If `task_41` failed differently OR the dominant failure modes shifted: STOP and revise the spec (`docs/superpowers/specs/2026-06-23-video-task-score-08-design.md` §2, §5 Phase 2/3 priorities) before continuing.

---

## Task 2: Auditor module skeleton + AuditFinding dataclass

**Goal:** Land the empty auditor module so subsequent checks have a place to live and tests are wired.

**Files:**
- Create: `src/agents/verification/video_filter_audit.py`
- Create: `tests/verification/test_video_filter_audit.py`

- [ ] **Step 1: Write a failing test for `AuditFinding` and empty `audit_answer_table`**

Create `tests/verification/test_video_filter_audit.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/verification/test_video_filter_audit.py -v
```

Expected: ImportError or ModuleNotFoundError on `agents.verification.video_filter_audit`.

- [ ] **Step 3: Create the module with the dataclass and empty function**

Create `src/agents/verification/video_filter_audit.py`:

```python
"""Deterministic post-checks on the AnswerTable for video tasks.

Runs between agents.verification.video_constraints (existing batch_id / snapshot
multiplicity checks) and the LLM AnswerVerifier. Three concerns it owns:

1. duplicate_export_rows — SELECT/JOIN side-effects that inflate row counts.
2. boundary_ambiguous — '>' vs '>=' choice when any row equals the threshold.
3. row_count_overflow — pred row count exceeds a credible small total parsed
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
    code: str        # 'duplicate_export_rows' | 'boundary_ambiguous' | 'row_count_overflow' | 'audit_internal_error'
    severity: str    # 'reject' | 'warn'
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/verification/test_video_filter_audit.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agents/verification/video_filter_audit.py tests/verification/test_video_filter_audit.py
git commit -m "feat(verify): scaffold video_filter_audit module + AuditFinding dataclass"
```

---

## Task 3: `duplicate_export_rows` check

**Goal:** Reject answer tables whose `export_fields`-keyed rows contain duplicates. This is the leading failure mode on task_4 (138 vs 125 rows with duplicate entries).

**Files:**
- Modify: `src/agents/verification/video_filter_audit.py`
- Modify: `tests/verification/test_video_filter_audit.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/verification/test_video_filter_audit.py`:

```python
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
    # export field not in table columns — auditor cannot judge, stays silent
    assert findings == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/verification/test_video_filter_audit.py::test_duplicate_export_rows_triggers_on_repeat -v
```

Expected: FAIL — auditor currently returns `[]`.

- [ ] **Step 3: Implement the check**

In `src/agents/verification/video_filter_audit.py`, add `_check_duplicate_export_rows` and wire it into `audit_answer_table`:

```python
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
        return None  # Auditor cannot decide without alignment
    seen: dict[tuple[Any, ...], int] = {}
    duplicates: list[tuple[Any, ...]] = []
    for row in table.rows:
        key = tuple(row[i] for i in col_indices)
        if key in seen:
            if seen[key] == 1:  # only record on the first duplicate, not every repeat
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
```

Update `audit_answer_table` to call it:

```python
def audit_answer_table(
    table: AnswerTable,
    rules: dict[str, Any] | None,
    samples: dict[str, Any] | list[Any] | None,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    if rules is None:
        return findings
    dup = _check_duplicate_export_rows(table, rules)
    if dup is not None:
        findings.append(dup)
    return findings
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/verification/test_video_filter_audit.py -v
```

Expected: all duplicate_export_rows tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agents/verification/video_filter_audit.py tests/verification/test_video_filter_audit.py
git commit -m "feat(verify): video_filter_audit duplicate_export_rows check"
```

---

## Task 4: `boundary_ambiguous` check

**Goal:** Warn when the operator is `>` or `>=` and any row's value equals the threshold — that ambiguity drove task_4 to land 13 extra rows.

**Files:**
- Modify: `src/agents/verification/video_filter_audit.py`
- Modify: `tests/verification/test_video_filter_audit.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/verification/test_video_filter_audit.py`:

```python
# ---------- boundary_ambiguous -------------------------------------------------


def test_boundary_ambiguous_triggers_when_row_equals_threshold() -> None:
    table = AnswerTable(
        columns=["company", "value"],
        rows=[["A", 100.0], ["B", 200.0], ["C", 100.0]],  # two rows == threshold
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/verification/test_video_filter_audit.py -v -k boundary
```

Expected: 6 boundary tests fail.

- [ ] **Step 3: Implement the check**

Add to `src/agents/verification/video_filter_audit.py`:

```python
_FIELD_KEYS: tuple[str, ...] = (
    "field",
    "filter_field",
    "metric",
    "column",
    "attribute",
    "aggregation_field",
)
_THRESHOLD_KEYS: tuple[str, ...] = (
    "threshold",
    "threshold_value",
    "value",
    "condition_value",
)


def _first_present(rules: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in rules and rules[key] not in (None, ""):
            return rules[key]
    return None


def _coerce_float(raw: Any) -> float | None:
    if isinstance(raw, bool):  # bool is a subclass of int — exclude
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if not isinstance(raw, str):
        return None
    text = raw.strip().replace(",", "").replace("，", "").rstrip("%")
    try:
        return float(text)
    except ValueError:
        return None


def _row_value_at(table: AnswerTable, col_index: int, row_index: int) -> Any:
    row = table.rows[row_index]
    if col_index >= len(row):
        return None
    return row[col_index]


def _check_boundary_ambiguous(
    table: AnswerTable,
    rules: dict[str, Any],
) -> AuditFinding | None:
    operator_raw = _first_present(rules, ("operator", "comparator", "op"))
    if operator_raw not in (">", ">=", "<", "<="):
        return None
    raw_field = _first_present(rules, _FIELD_KEYS)
    if not isinstance(raw_field, str):
        return None
    if raw_field not in table.columns:
        return None
    threshold = _coerce_float(_first_present(rules, _THRESHOLD_KEYS))
    if threshold is None:
        return None
    col_index = table.columns.index(raw_field)
    equal_count = 0
    strict_count = 0
    inclusive_count = 0
    for row_index in range(len(table.rows)):
        cell = _row_value_at(table, col_index, row_index)
        value = _coerce_float(cell)
        if value is None:
            continue
        inclusive_count += 1
        if value > threshold or value < threshold:
            strict_count += 1
        if value == threshold:
            equal_count += 1
    if equal_count == 0:
        return None
    return AuditFinding(
        code="boundary_ambiguous",
        severity="warn",
        message=(
            f"{equal_count} row(s) equal the threshold {threshold}; "
            f"operator '{operator_raw}' is ambiguous — verify >= vs > against the video"
        ),
        evidence={
            "operator_chosen": operator_raw,
            "threshold": threshold,
            "rows_at_threshold": equal_count,
            "count_if_strict": strict_count,
            "count_if_inclusive": inclusive_count,
        },
    )
```

Wire it into `audit_answer_table`:

```python
def audit_answer_table(
    table: AnswerTable,
    rules: dict[str, Any] | None,
    samples: dict[str, Any] | list[Any] | None,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    if rules is None:
        return findings
    dup = _check_duplicate_export_rows(table, rules)
    if dup is not None:
        findings.append(dup)
    boundary = _check_boundary_ambiguous(table, rules)
    if boundary is not None:
        findings.append(boundary)
    return findings
```

- [ ] **Step 4: Run tests to verify all pass**

```bash
uv run pytest tests/verification/test_video_filter_audit.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/agents/verification/video_filter_audit.py tests/verification/test_video_filter_audit.py
git commit -m "feat(verify): video_filter_audit boundary_ambiguous check"
```

---

## Task 5: `row_count_overflow` check

**Goal:** Soft-warn when the prediction has many more rows than a credible total parsed from `displayed_samples`. Catches the over-permissive filter pattern as a safety net.

**Files:**
- Modify: `src/agents/verification/video_filter_audit.py`
- Modify: `tests/verification/test_video_filter_audit.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/verification/test_video_filter_audit.py`:

```python
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
    samples = ["a", "b", "c", "d", "e"]  # 5 items — pred 20 (>= 2x)
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/verification/test_video_filter_audit.py -v -k row_count_overflow
```

Expected: 2 trigger tests fail.

- [ ] **Step 3: Implement the check**

Add to `src/agents/verification/video_filter_audit.py`:

```python
_SAMPLES_TOTAL_KEYS: tuple[str, ...] = (
    "total_count",
    "row_count",
    "total",
    "count",
    "n",
)


def _extract_samples_total(samples: dict[str, Any] | list[Any] | None) -> int | None:
    """Best-effort: pull a credible small total from displayed_samples.

    Two shapes accepted:
    - dict: look for known total keys with int value
    - list: use list length as the proxy total
    Anything else returns None — the check is then silent.
    """
    if samples is None:
        return None
    if isinstance(samples, list):
        return len(samples) if samples else None
    if not isinstance(samples, dict):
        return None
    for key in _SAMPLES_TOTAL_KEYS:
        value = samples.get(key)
        if isinstance(value, bool):  # bool is int subclass — exclude
            continue
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str):
            text = value.strip().replace(",", "").replace("，", "")
            if text.isdigit():
                n = int(text)
                if n > 0:
                    return n
    return None


def _check_row_count_overflow(
    table: AnswerTable,
    samples: dict[str, Any] | list[Any] | None,
) -> AuditFinding | None:
    samples_total = _extract_samples_total(samples)
    if samples_total is None:
        return None
    df_rows = len(table.rows)
    if df_rows <= 2 * samples_total:
        return None
    return AuditFinding(
        code="row_count_overflow",
        severity="warn",
        message=(
            f"prediction has {df_rows} rows but displayed_samples implies ~{samples_total}; "
            "double-check the filter is not over-permissive"
        ),
        evidence={
            "samples_total": samples_total,
            "df_rows": df_rows,
        },
    )
```

Update `audit_answer_table`:

```python
def audit_answer_table(
    table: AnswerTable,
    rules: dict[str, Any] | None,
    samples: dict[str, Any] | list[Any] | None,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    if rules is None:
        return findings
    dup = _check_duplicate_export_rows(table, rules)
    if dup is not None:
        findings.append(dup)
    boundary = _check_boundary_ambiguous(table, rules)
    if boundary is not None:
        findings.append(boundary)
    overflow = _check_row_count_overflow(table, samples)
    if overflow is not None:
        findings.append(overflow)
    return findings
```

- [ ] **Step 4: Run tests to verify all pass**

```bash
uv run pytest tests/verification/test_video_filter_audit.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/agents/verification/video_filter_audit.py tests/verification/test_video_filter_audit.py
git commit -m "feat(verify): video_filter_audit row_count_overflow check"
```

---

## Task 6: Auditor exception swallow + aggregator robustness

**Goal:** Guarantee `audit_answer_table` never raises. Any internal exception becomes an `audit_internal_error` warn finding — the answer is then never blocked by an auditor bug.

**Files:**
- Modify: `src/agents/verification/video_filter_audit.py`
- Modify: `tests/verification/test_video_filter_audit.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/verification/test_video_filter_audit.py`:

```python
# ---------- aggregator: never raises -------------------------------------------

from typing import Any as _Any


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
```

- [ ] **Step 2: Run the new tests to verify the expected fail/pass split**

```bash
uv run pytest tests/verification/test_video_filter_audit.py -v -k "swallows or multiple or rejects_non_dict"
```

Expected:
- `test_audit_answer_table_swallows_internal_exception` FAILS — the helper raises RuntimeError and the aggregator currently does not catch it.
- `test_audit_answer_table_rejects_non_dict_rules_safely` PASSES — the current early-return for `rules is None` does not handle strings; the aggregator must be tightened in Step 3 to also reject non-dict shapes.
- `test_audit_answer_table_returns_multiple_findings` PASSES — Tasks 3 and 4 already accumulate findings.

(If the second test passes for the wrong reason — i.e. the implementation crashes — note it and move on; Step 3 fixes both gaps.)

- [ ] **Step 3: Wrap `audit_answer_table` in try/except**

Replace `audit_answer_table` in `src/agents/verification/video_filter_audit.py` with the swallow-and-warn version:

```python
def audit_answer_table(
    table: AnswerTable,
    rules: dict[str, Any] | None,
    samples: dict[str, Any] | list[Any] | None,
) -> list[AuditFinding]:
    """Run all auditor checks; never raises.

    Internal exceptions become a single warn-severity finding so that an auditor
    bug never blocks an otherwise-valid answer.
    """
    findings: list[AuditFinding] = []
    if rules is None or not isinstance(rules, dict):
        return findings
    try:
        dup = _check_duplicate_export_rows(table, rules)
        if dup is not None:
            findings.append(dup)
        boundary = _check_boundary_ambiguous(table, rules)
        if boundary is not None:
            findings.append(boundary)
        overflow = _check_row_count_overflow(table, samples)
        if overflow is not None:
            findings.append(overflow)
    except Exception as exc:  # noqa: BLE001 — fall-through is intentional
        findings.append(
            AuditFinding(
                code="audit_internal_error",
                severity="warn",
                message=f"auditor internal error: {type(exc).__name__}: {exc}",
                evidence={"exception_type": type(exc).__name__},
            )
        )
    return findings
```

- [ ] **Step 4: Run all auditor tests**

```bash
uv run pytest tests/verification/test_video_filter_audit.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/agents/verification/video_filter_audit.py tests/verification/test_video_filter_audit.py
git commit -m "feat(verify): video_filter_audit swallows internal exceptions as warn"
```

---

## Task 7: Add `video_filter_audit` config field

**Goal:** Make the auditor toggle-able via `agent.video_filter_audit: enabled|disabled` in YAML config.

**Files:**
- Modify: `src/agents/config.py`
- Modify: `tests/config/test_config_new_fields.py` (existing file per `git status`)

- [ ] **Step 1: Add a failing test for the new config field**

Append to `tests/config/test_config_new_fields.py` (read the file first to match its style):

```python
# ---------- video_filter_audit toggle ----------------------------------------


def test_agent_config_video_filter_audit_default_enabled() -> None:
    from agents.config import AgentConfig

    cfg = AgentConfig(
        model="m",
        api_base="https://example.com",
        api_key="sk-test",
    )
    assert cfg.video_filter_audit == "enabled"


def test_agent_config_video_filter_audit_accepts_disabled() -> None:
    from agents.config import AgentConfig

    cfg = AgentConfig(
        model="m",
        api_base="https://example.com",
        api_key="sk-test",
        video_filter_audit="disabled",
    )
    assert cfg.video_filter_audit == "disabled"


def test_agent_config_video_filter_audit_rejects_unknown() -> None:
    from agents.config import AgentConfig
    import pytest

    with pytest.raises(Exception):  # pydantic ValidationError
        AgentConfig(
            model="m",
            api_base="https://example.com",
            api_key="sk-test",
            video_filter_audit="maybe",  # type: ignore[arg-type]
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/config/test_config_new_fields.py -v -k video_filter_audit
```

Expected: FAIL — `AgentConfig` does not have `video_filter_audit`.

- [ ] **Step 3: Add field to `AgentConfig`**

In `src/agents/config.py`, add a new field next to the other video-related ones (after `video_max_tokens` around line 305):

```python
    # When "enabled", terminal_policy.maybe_reject runs video_filter_audit on the
    # candidate AnswerTable between evaluate_video_constraints and the LLM verifier.
    # Shares the AnswerVerifier.max_rejections budget — no new retry quota added.
    video_filter_audit: Literal["enabled", "disabled"] = "enabled"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/config/test_config_new_fields.py -v -k video_filter_audit
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agents/config.py tests/config/test_config_new_fields.py
git commit -m "feat(config): add video_filter_audit toggle (default enabled)"
```

---

## Task 8: Add `filter_audit_enabled` field to `TerminalAnswerPolicy`

**Goal:** Land the policy-side switch so a False value preserves exact current behavior, but a True value gates the auditor in Task 9.

**Files:**
- Modify: `src/agents/verification/terminal_policy.py`
- Modify: `tests/verification/test_terminal_policy.py`

- [ ] **Step 1: Add a failing test**

Append to `tests/verification/test_terminal_policy.py`:

```python
from agents.verification.terminal_policy import TerminalAnswerPolicy


def test_terminal_policy_filter_audit_enabled_defaults_false() -> None:
    # Backward-compatible default: existing call sites don't pass it.
    policy = TerminalAnswerPolicy(verifier=None)
    assert policy.filter_audit_enabled is False


def test_terminal_policy_filter_audit_enabled_can_be_true() -> None:
    policy = TerminalAnswerPolicy(verifier=None, filter_audit_enabled=True)
    assert policy.filter_audit_enabled is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/verification/test_terminal_policy.py -v -k filter_audit_enabled
```

Expected: FAIL — `TerminalAnswerPolicy` does not have the field.

- [ ] **Step 3: Add field to the dataclass**

In `src/agents/verification/terminal_policy.py`, add a field to `TerminalAnswerPolicy` (after `fallback_answer_arguments`):

```python
@dataclass(slots=True)
class TerminalAnswerPolicy:
    """Coordinate verifier rejection limits and last-valid-answer fallback."""

    verifier: AnswerVerifier | None
    rejections_used: int = 0
    fallback_answer: AnswerTable | None = None
    fallback_answer_arguments: dict[str, Any] | None = field(default=None)
    # When True, run audit_answer_table between evaluate_video_constraints and
    # the LLM verifier. Default False keeps existing call sites unchanged.
    filter_audit_enabled: bool = False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/verification/test_terminal_policy.py -v
```

Expected: all green, including pre-existing tests.

- [ ] **Step 5: Commit**

```bash
git add src/agents/verification/terminal_policy.py tests/verification/test_terminal_policy.py
git commit -m "feat(verify): TerminalAnswerPolicy.filter_audit_enabled flag (default False)"
```

---

## Task 9: Wire auditor into `TerminalAnswerPolicy.maybe_reject`

**Goal:** Run `audit_answer_table` between `evaluate_video_constraints` and the LLM verifier when the flag is on. Reject when any finding has severity `reject`; consume one slot of `rejections_used`. Behavior must match the existing video_constraints reject flow.

**Files:**
- Modify: `src/agents/verification/terminal_policy.py`
- Modify: `tests/verification/test_terminal_policy.py`

- [ ] **Step 1: Add failing tests for the wiring**

Append to `tests/verification/test_terminal_policy.py`:

```python
from typing import Any

from agents.benchmark.schema import AnswerTable, PublicTask
from agents.verification.video_filter_audit import AuditFinding


def _make_task() -> PublicTask:
    # Minimal stub — actual fields exercised only by build_answer_table_without_side_effects
    from types import SimpleNamespace
    assets = SimpleNamespace(task_dir="/tmp", context_dir="/tmp")
    return SimpleNamespace(task_id="task_x", question="qq", assets=assets)  # type: ignore[return-value]


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
        task=_make_task(),
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
        task=_make_task(),
        arguments={},
        is_terminal=True,
        remaining_steps=5,
        state=None,
    )
    # Verifier is None and auditor is disabled → no rejection
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
        task=_make_task(),
        arguments={},
        is_terminal=True,
        remaining_steps=5,
        state=None,
    )
    assert result is None
    assert policy.rejections_used == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/verification/test_terminal_policy.py -v -k "maybe_reject"
```

Expected: tests fail — `_extract_video_rules_and_samples` and the auditor wiring don't exist yet.

- [ ] **Step 3: Add helper + auditor call to `terminal_policy.py`**

In `src/agents/verification/terminal_policy.py`:

(a) Add imports at the top:

```python
from agents.verification.video_filter_audit import (
    AuditFinding,
    audit_answer_table,
)
```

(b) Add the rules/samples extraction helper after the `recent_execute_code` function:

```python
def _extract_video_rules_and_samples(
    state: AgentRuntimeState | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | list[Any] | None]:
    """Pull (rules, displayed_samples) from the latest explore_video step.

    Walks ``state.steps`` backward for an ``explore_video`` action whose
    observation carries ``findings.extracted_data.{rules, displayed_samples}``.
    Returns (None, None) when no such step is found or the shape is missing.
    """
    if state is None:
        return None, None
    for step in reversed(state.steps):
        if getattr(step, "action", None) != "explore_video":
            continue
        if not getattr(step, "ok", False):
            continue
        observation = getattr(step, "observation", None)
        if not isinstance(observation, dict):
            continue
        findings = observation.get("findings")
        if not isinstance(findings, dict):
            continue
        extracted = findings.get("extracted_data")
        if not isinstance(extracted, dict):
            continue
        rules = extracted.get("rules")
        samples = extracted.get("displayed_samples")
        return (
            rules if isinstance(rules, dict) else None,
            samples if isinstance(samples, (dict, list)) else None,
        )
    return None, None


def _format_audit_rejections(rejects: list[AuditFinding]) -> str:
    lines = ["Answer rejected by video_filter_audit:"]
    for f in rejects:
        lines.append(f"- [{f.code}] {f.message}")
        if f.evidence:
            lines.append(f"  evidence: {f.evidence}")
    return "\n".join(lines)
```

(c) Modify `maybe_reject` to insert the auditor between the existing deterministic check and the LLM verifier:

Find the section:

```python
        if deterministic_rejection is not None:
            self.rejections_used += 1
            if candidate is not None and candidate.columns and candidate.rows:
                self.fallback_answer = candidate
                self.fallback_answer_arguments = dict(arguments)
            return TerminalAnswerRejection(content={"answer_rejected": deterministic_rejection})

        if self.verifier is None:
            return None
```

Insert between them:

```python
        if deterministic_rejection is not None:
            self.rejections_used += 1
            if candidate is not None and candidate.columns and candidate.rows:
                self.fallback_answer = candidate
                self.fallback_answer_arguments = dict(arguments)
            return TerminalAnswerRejection(content={"answer_rejected": deterministic_rejection})

        if self.filter_audit_enabled and candidate is not None:
            rules, samples = _extract_video_rules_and_samples(state)
            findings = audit_answer_table(candidate, rules, samples)
            rejects = [f for f in findings if f.severity == "reject"]
            if rejects:
                self.rejections_used += 1
                if candidate.columns and candidate.rows:
                    self.fallback_answer = candidate
                    self.fallback_answer_arguments = dict(arguments)
                return TerminalAnswerRejection(
                    content={"answer_rejected": _format_audit_rejections(rejects)}
                )

        if self.verifier is None:
            return None
```

- [ ] **Step 4: Run all terminal_policy tests**

```bash
uv run pytest tests/verification/test_terminal_policy.py -v
```

Expected: all green. If a pre-existing test relies on `verifier=None → result is None`, it should still pass because the auditor is gated by `filter_audit_enabled`.

- [ ] **Step 5: Commit**

```bash
git add src/agents/verification/terminal_policy.py tests/verification/test_terminal_policy.py
git commit -m "feat(verify): wire video_filter_audit into TerminalAnswerPolicy"
```

---

## Task 10: Thread config flag from `AppConfig` to `TerminalAnswerPolicy`

**Goal:** Set `filter_audit_enabled` based on `config.agent.video_filter_audit == "enabled"`. End-to-end from YAML to dataclass.

**Files:**
- Modify: `src/agents/agent.py`
- Modify: `src/agents/application.py`
- Modify: `tests/application/test_app_factory.py` (existing file per `git status`)

- [ ] **Step 1: Open `tests/application/test_app_factory.py` and identify the existing `build_react_agent` fixture pattern**

```bash
uv run python -c "
from pathlib import Path
print(Path('tests/application/test_app_factory.py').read_text(encoding='utf-8'))" | head -120
```

Note how AppConfig / build_react_agent are constructed in the existing tests (likely via a `tmp_path` fixture or a helper that builds a minimal AppConfig). Reuse that exact pattern.

- [ ] **Step 2: Append two tests reusing the established factory pattern**

Pseudo-template (replace `_make_app_config(...)` with the helper or fixture call the existing tests use; AppConfig requires `dataset`, `agent`, `run` — copy the minimal shape from the file's earliest existing test):

```python
def test_build_react_agent_passes_video_filter_audit_enabled(tmp_path) -> None:
    cfg = _make_app_config(tmp_path, video_filter_audit="enabled")
    agent = build_react_agent(cfg)
    assert agent.filter_audit_enabled is True


def test_build_react_agent_passes_video_filter_audit_disabled(tmp_path) -> None:
    cfg = _make_app_config(tmp_path, video_filter_audit="disabled")
    agent = build_react_agent(cfg)
    assert agent.filter_audit_enabled is False
```

If the file does not yet have a `_make_app_config` helper, add it next to the new tests. The helper should construct a minimum-viable AppConfig from `tmp_path` with these overrides allowed: `video_filter_audit: Literal["enabled","disabled"] = "enabled"`. Look at how existing tests pass `agent=AgentConfig(...)` to AppConfig and replicate.

- [ ] **Step 2: Run the test and confirm it fails**

```bash
uv run pytest tests/application/test_app_factory.py -v -k video_filter_audit
```

Expected: FAIL — `ReActAgent` does not have `filter_audit_enabled`.

- [ ] **Step 3: Thread the flag through `ReActAgent`**

In `src/agents/agent.py`, find the constructor (around line 114) and add a parameter + attribute:

```python
    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig,
        system_prompt: str = "...",
        answer_verifier: AnswerVerifier | None = None,
        filter_audit_enabled: bool = False,   # NEW
        step_callback: StepCallback | None = None,
    ) -> None:
        # ...existing assignments...
        self.answer_verifier = answer_verifier
        self.filter_audit_enabled = filter_audit_enabled   # NEW
```

Then at the construction site of `TerminalAnswerPolicy` (around line 198) change:

```python
        terminal_policy = TerminalAnswerPolicy(self.answer_verifier)
```

to:

```python
        terminal_policy = TerminalAnswerPolicy(
            verifier=self.answer_verifier,
            filter_audit_enabled=self.filter_audit_enabled,
        )
```

- [ ] **Step 4: Pass the config flag from `application.py`**

In `src/agents/application.py`, find the `build_react_agent` function. It already wires `answer_verifier=verifier`. Add the flag:

```python
    return ReActAgent(
        # ...existing args...
        answer_verifier=verifier,
        filter_audit_enabled=(config.agent.video_filter_audit == "enabled"),
    )
```

- [ ] **Step 5: Run the failing test to verify it now passes**

```bash
uv run pytest tests/application/test_app_factory.py -v -k video_filter_audit
```

Expected: PASS.

- [ ] **Step 6: Run the full suite to catch regressions**

```bash
uv run pytest tests/ -q
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/agents/agent.py src/agents/application.py tests/application/test_app_factory.py
git commit -m "feat(verify): pipe video_filter_audit toggle from config through ReActAgent"
```

---

## Task 11: Document the toggle in `react.example.yaml`

**Goal:** Surface the new toggle to operators who copy `react.example.yaml` as their starting config.

**Files:**
- Modify: `configs/react.example.yaml`

- [ ] **Step 1: Add the toggle near the other video-related options**

Find the block around `video_tool_choice` / `video_max_tokens` in `configs/react.example.yaml` and add:

```yaml
  video_filter_audit: enabled              # "enabled" | "disabled"; runs deterministic post-checks on the AnswerTable (duplicate rows, threshold boundary, displayed_samples overflow) between video_constraints and the LLM verifier — shares the AnswerVerifier rejection budget
```

- [ ] **Step 2: Verify config loads with the new field**

```bash
uv run python -c "
from agents.config import load_app_config
cfg = load_app_config('configs/react.example.yaml')
print('video_filter_audit:', cfg.agent.video_filter_audit)
"
```

Expected: prints `video_filter_audit: enabled`.

- [ ] **Step 3: Commit**

```bash
git add configs/react.example.yaml
git commit -m "docs(config): document video_filter_audit toggle in react.example.yaml"
```

---

## Task 12: Add `question=` param + `prefer_abbr_column` to `extract_filter_recipe`

**Goal:** When the task question text contains abbreviation keywords (`简称`, `abbreviation`, `abbr`, `代号`), set a flag on the recipe so the renderer can surface the hint.

**Files:**
- Modify: `src/agents/video/rule_templater.py`
- Modify: `tests/video/test_rule_templater.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/video/test_rule_templater.py`:

```python
# ---------- prefer_abbr_column hint -------------------------------------------


def test_extract_prefer_abbr_column_default_false_when_question_none() -> None:
    rules = {"field": "f", "operator": ">", "threshold": "0"}
    r = extract_filter_recipe(rules)
    assert r is not None
    assert r.prefer_abbr_column is False


def test_extract_prefer_abbr_column_true_for_chinese_jian_cheng() -> None:
    rules = {"field": "f", "operator": ">", "threshold": "0"}
    r = extract_filter_recipe(rules, question="请列出符合条件的基金简称")
    assert r is not None
    assert r.prefer_abbr_column is True


def test_extract_prefer_abbr_column_true_for_english_abbreviation() -> None:
    rules = {"field": "f", "operator": ">", "threshold": "0"}
    r = extract_filter_recipe(rules, question="Which funds match? List the abbreviation.")
    assert r is not None
    assert r.prefer_abbr_column is True


def test_extract_prefer_abbr_column_true_for_english_abbr_short() -> None:
    rules = {"field": "f", "operator": ">", "threshold": "0"}
    r = extract_filter_recipe(rules, question="Return the fund abbr.")
    assert r is not None
    assert r.prefer_abbr_column is True


def test_extract_prefer_abbr_column_true_for_chinese_dai_hao() -> None:
    rules = {"field": "f", "operator": ">", "threshold": "0"}
    r = extract_filter_recipe(rules, question="请列出符合条件的公司代号")
    assert r is not None
    assert r.prefer_abbr_column is True


def test_extract_prefer_abbr_column_false_for_full_name_question() -> None:
    rules = {"field": "f", "operator": ">", "threshold": "0"}
    r = extract_filter_recipe(rules, question="请列出公司全称")
    assert r is not None
    assert r.prefer_abbr_column is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/video/test_rule_templater.py -v -k prefer_abbr_column
```

Expected: FAIL — `extract_filter_recipe` does not accept `question=` or `FilterRecipe` does not have `prefer_abbr_column`.

- [ ] **Step 3: Add the field and parameter**

In `src/agents/video/rule_templater.py`:

(a) Add the keyword tuple near other module constants (after `_PAREN_ANY_RE`):

```python
# Abbreviation column keywords (case-insensitive substring). Mark recipes whose
# task question requests an abbreviation so the rendered hint can guide column
# choice when both *Abbr and full-name columns exist.
_ABBR_QUESTION_KEYWORDS: tuple[str, ...] = (
    "简称",
    "代号",
    "abbreviation",
    "abbr",
)


def _question_prefers_abbr(question: str | None) -> bool:
    if not isinstance(question, str) or not question:
        return False
    text = question.lower()
    return any(k.lower() in text for k in _ABBR_QUESTION_KEYWORDS)
```

(b) Add the field to `FilterRecipe`. Find the existing dataclass (it currently ends with `snapshot_dates: tuple[str, ...] = ()` before `def is_sufficient`). Add ONE new field directly before `def is_sufficient`:

```python
    prefer_abbr_column: bool = False
```

Do NOT modify any other field, ordering, or the `is_sufficient` method.

(c) Modify the `extract_filter_recipe` signature to accept the new keyword argument. Find the current signature:

```python
def extract_filter_recipe(rules: dict[str, Any] | None) -> FilterRecipe | None:
```

Change to:

```python
def extract_filter_recipe(
    rules: dict[str, Any] | None,
    *,
    question: str | None = None,
) -> FilterRecipe | None:
```

Then find the `recipe = FilterRecipe(...)` call inside the function. Add one new keyword argument at the bottom of the constructor call (preserving every existing argument verbatim):

```python
        prefer_abbr_column=_question_prefers_abbr(question),
```

So the call becomes (for reference; only the last line is new):

```python
    recipe = FilterRecipe(
        field=field_value,
        field_label=field_label,
        operator=operator,
        threshold=threshold,
        threshold_unit=threshold_unit,
        sort_field=sort_field,
        sort_descending=sort_descending,
        top_n=top_n,
        export_fields=export_fields,
        group_by=group_by,
        table_hint=table_hint,
        table_hints=table_hints,
        batch_id=batch_id,
        snapshot_dates=snapshot_dates,
        prefer_abbr_column=_question_prefers_abbr(question),
    )
```

Do NOT change the `return (recipe if any(...) else None)` block — `prefer_abbr_column` must not affect `is_sufficient`'s gating.

(d) Add `question=` to `synthesize_filter_recipe` and pass through:

```python
def synthesize_filter_recipe(
    rules: dict[str, Any] | None,
    *,
    question: str | None = None,
) -> str | None:
    """One-shot convenience: rules dict → rendered hint or None."""
    recipe = extract_filter_recipe(rules, question=question)
    if recipe is None or not recipe.is_sufficient():
        return None
    return render_filter_recipe(recipe)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/video/test_rule_templater.py -v -k prefer_abbr_column
```

Expected: all 6 tests pass.

- [ ] **Step 5: Run all templater tests to catch regression**

```bash
uv run pytest tests/video/test_rule_templater.py -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/agents/video/rule_templater.py tests/video/test_rule_templater.py
git commit -m "feat(video): rule_templater question-aware prefer_abbr_column hint"
```

---

## Task 13: Render abbreviation hint in `render_filter_recipe`

**Goal:** Surface the new flag as a one-line hint at the top of the rendered recipe.

**Files:**
- Modify: `src/agents/video/rule_templater.py`
- Modify: `tests/video/test_rule_templater.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/video/test_rule_templater.py`:

```python
# ---------- render_filter_recipe: abbreviation hint ----------------------------


def test_render_filter_recipe_includes_abbr_hint_when_flag_set() -> None:
    from agents.video.rule_templater import render_filter_recipe, FilterRecipe

    recipe = FilterRecipe(
        field="dailybenchgr",
        operator=">",
        threshold="0.00%",
        sort_field="dailybenchgr",
        sort_descending=True,
        top_n=10,
        prefer_abbr_column=True,
    )
    text = render_filter_recipe(recipe)
    assert "prefer_column_kind: abbreviation" in text


def test_render_filter_recipe_omits_abbr_hint_when_flag_unset() -> None:
    from agents.video.rule_templater import render_filter_recipe, FilterRecipe

    recipe = FilterRecipe(
        field="dailybenchgr",
        operator=">",
        threshold="0.00%",
        sort_field="dailybenchgr",
        sort_descending=True,
        top_n=10,
    )
    text = render_filter_recipe(recipe)
    assert "prefer_column_kind" not in text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/video/test_rule_templater.py -v -k abbr_hint
```

Expected: `test_render_filter_recipe_includes_abbr_hint_when_flag_set` fails.

- [ ] **Step 3: Modify `render_filter_recipe`**

In `src/agents/video/rule_templater.py`, find the start of `render_filter_recipe`:

```python
def render_filter_recipe(recipe: FilterRecipe) -> str:
    """Format a FilterRecipe into the plaintext hint surfaced to the agent."""
    lines: list[str] = [
        "[video-rule recipe — VERIFY each name against actual schema before use]",
    ]
```

Insert the abbreviation hint immediately after the header line:

```python
def render_filter_recipe(recipe: FilterRecipe) -> str:
    """Format a FilterRecipe into the plaintext hint surfaced to the agent."""
    lines: list[str] = [
        "[video-rule recipe — VERIFY each name against actual schema before use]",
    ]
    if recipe.prefer_abbr_column:
        lines.append(
            "  prefer_column_kind: abbreviation — when both 'XxxAbbr' and "
            "full-name columns exist, use the *Abbr column"
        )
    # ... rest unchanged ...
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/video/test_rule_templater.py -v
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/agents/video/rule_templater.py tests/video/test_rule_templater.py
git commit -m "feat(video): render prefer_column_kind hint in filter recipe"
```

---

## Task 14: Thread `task.question` through `explore_video.py`

**Goal:** The templater can only judge from `question` if the caller passes it. `explore_video`'s handler has the `PublicTask`; pass `task.question` into `synthesize_filter_recipe`.

**Files:**
- Modify: `src/agents/tools/explore_video.py`

- [ ] **Step 1: Add an integration-style failing test**

Create or extend `tests/video/test_explore_video_integration.py`. If the file does not exist, create it:

```python
"""Integration tests for the explore_video tool's filter-recipe wiring."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from agents.benchmark.schema import PublicTask
from agents.tools.explore_video import create_explore_video_tool_definition
from agents.tools.registry import ToolExecutionResult


class _StubVideoResult:
    def __init__(self, findings_dict: dict[str, Any]) -> None:
        self.findings = json.dumps(findings_dict, ensure_ascii=False)
        self.steps_used = 1
        self.success = True


def _make_task_with_question(question: str) -> PublicTask:
    assets = SimpleNamespace(task_dir="/tmp", context_dir="/tmp", input_dir="/tmp")
    return SimpleNamespace(task_id="task_x", question=question, assets=assets)  # type: ignore[return-value]


def test_explore_video_threads_question_into_filter_recipe(tmp_path: Any) -> None:
    """When the task asks for 简称, the resulting filter_recipe text mentions abbreviation."""
    video_file = tmp_path / "demo.mp4"
    video_file.write_bytes(b"x" * 1024)

    findings_dict = {
        "summary": "video",
        "timeline": [],
        "extracted_data": {
            "rules": {"field": "dailybenchgr", "operator": ">", "threshold": "0.00%"},
            "displayed_samples": {},
        },
        "coverage": [],
        "warnings": [],
    }

    tool_def = create_explore_video_tool_definition(video_model=None)  # type: ignore[arg-type]
    task = _make_task_with_question("请列出基金简称")

    with patch("agents.runtime.media.VIDEO_EXTENSIONS", (".mp4",)), \
         patch("agents.runtime.media.VIDEO_MAX_BYTES", 10 * 1024 * 1024), \
         patch("agents.tools.context.resolve_context_path", return_value=video_file), \
         patch("agents.video.runner.run_video_agent", return_value=_StubVideoResult(findings_dict)):
        result: ToolExecutionResult = tool_def.handler(
            task,
            tool_def.input_model.model_validate({"path": "demo.mp4", "instructions": "x"}),
        )

    assert result.ok is True
    assert "filter_recipe" in result.content
    assert "prefer_column_kind: abbreviation" in result.content["filter_recipe"]


def test_explore_video_omits_abbr_hint_when_question_full_name(tmp_path: Any) -> None:
    """When the task asks for 全称, the recipe omits the abbreviation hint."""
    video_file = tmp_path / "demo.mp4"
    video_file.write_bytes(b"x" * 1024)

    findings_dict = {
        "summary": "video",
        "timeline": [],
        "extracted_data": {
            "rules": {"field": "dailybenchgr", "operator": ">", "threshold": "0.00%"},
            "displayed_samples": {},
        },
        "coverage": [],
        "warnings": [],
    }

    tool_def = create_explore_video_tool_definition(video_model=None)  # type: ignore[arg-type]
    task = _make_task_with_question("请列出基金全称")

    with patch("agents.runtime.media.VIDEO_EXTENSIONS", (".mp4",)), \
         patch("agents.runtime.media.VIDEO_MAX_BYTES", 10 * 1024 * 1024), \
         patch("agents.tools.context.resolve_context_path", return_value=video_file), \
         patch("agents.video.runner.run_video_agent", return_value=_StubVideoResult(findings_dict)):
        result: ToolExecutionResult = tool_def.handler(
            task,
            tool_def.input_model.model_validate({"path": "demo.mp4", "instructions": "x"}),
        )

    assert result.ok is True
    assert "prefer_column_kind" not in result.content.get("filter_recipe", "")
```

Read the existing `tests/video/` style first if you intend to refactor the boilerplate; minimal stubbing is preferred.

- [ ] **Step 2: Run the test to verify it fails**

```bash
uv run pytest tests/video/test_explore_video_integration.py -v
```

Expected: `prefer_column_kind: abbreviation` not in the rendered recipe — the handler does not pass `task.question` through yet.

- [ ] **Step 3: Modify the handler**

In `src/agents/tools/explore_video.py`, find the call inside `handler`:

```python
        recipe_text = synthesize_filter_recipe(rules)
```

Change to:

```python
        recipe_text = synthesize_filter_recipe(rules, question=task.question)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/video/test_explore_video_integration.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/agents/tools/explore_video.py tests/video/test_explore_video_integration.py
git commit -m "feat(video): thread task.question into rule_templater via explore_video"
```

---

## Task 15: Add `scripts/triage_failures.py` diagnostic tool

**Goal:** A single command to convert a run's `scores.json` + per-task gold/pred diff into a markdown failure-mode classification table. Diagnostic-only; not on the runtime path.

**Files:**
- Create: `scripts/triage_failures.py`

- [ ] **Step 1: Write the script**

Create `scripts/triage_failures.py`:

```python
#!/usr/bin/env python
"""Triage failing video tasks in a run by comparing prediction.csv vs gold.csv.

Usage:
    uv run python scripts/triage_failures.py --run artifacts/runs/<run_id> [--gold data/public/output]

Emits a markdown table with one row per imperfect task and a coarse failure-mode
guess (column_count_mismatch, value_mismatch_short, value_mismatch_large) so
the engineer can quickly bucket failures before deeper diff work.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _classify(gold_rows: list[list[str]], pred_rows: list[list[str]]) -> str:
    g, p = len(gold_rows), len(pred_rows)
    if g == 0 and p == 0:
        return "both_empty"
    if g == 0 or p == 0:
        return "one_side_empty"
    if g == p:
        return "value_mismatch_same_count"
    if p > g * 2:
        return "value_mismatch_large_excess"
    if g > p * 2:
        return "value_mismatch_large_shortfall"
    return "value_mismatch_small_diff"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="artifacts/runs/<run_id>")
    parser.add_argument("--gold", type=Path, default=Path("data/public/output"))
    args = parser.parse_args()

    scores_path = args.run / "scores.json"
    if not scores_path.exists():
        print(f"ERROR: {scores_path} not found", flush=True)
        return 2
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    per_task = scores.get("per_task", {})

    print("| task | score | status | gold_rows | pred_rows | classification |")
    print("|---|---|---|---|---|---|")
    for tid, info in sorted(per_task.items(), key=lambda kv: kv[1].get("score", 0.0)):
        score = info.get("score", 0.0)
        status = info.get("status", "")
        if score >= 0.99 or status == "missing_prediction":
            continue
        gold_path = args.gold / tid / "gold.csv"
        pred_path = args.run / tid / "prediction.csv"
        _, gold_rows = _read_csv(gold_path)
        _, pred_rows = _read_csv(pred_path)
        classification = _classify(gold_rows, pred_rows)
        print(
            f"| {tid} | {score:.3f} | {status} | {len(gold_rows)} | {len(pred_rows)} | {classification} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test the script against the worktree's existing run output**

If the worktree has any older run under `artifacts/runs/`, smoke it:

```bash
uv run python scripts/triage_failures.py --run artifacts/runs/<some-run-id>
```

Expected: prints a markdown table with one row per imperfect task. Does not raise.

If no runs exist yet, skip this step — Task 1's Phase 1 baseline will be the first input.

- [ ] **Step 3: Commit**

```bash
git add scripts/triage_failures.py
git commit -m "feat(scripts): triage_failures.py — quick markdown bucket of imperfect tasks"
```

---

## Task 16: Phase 2 A/B validation

**Goal:** Verify Phase 2 (filter audit) earns at least +0.03 mean over the Phase 1 baseline and stays under 800 s/task. Decide keep or revert.

**Files:** none (procedural). Only updates `docs/superpowers/specs/2026-06-23-video-task-score-08-design.md` §11 attempt log.

- [ ] **Step 1: Confirm Phase 2 commits are all on the branch**

```bash
git log --oneline ea97862^..HEAD
```

Expected: includes commits from Tasks 2-11 (auditor module, three checks, exception swallow, config field, terminal_policy wiring, agent threading, application threading, react.example.yaml).

- [ ] **Step 2: Run Phase 2 A/B #1 (fresh)**

```bash
uv run dabench run-benchmark --config configs/react.local.yaml
```

- [ ] **Step 3: Run Phase 2 A/B #2 (warm)**

```bash
uv run dabench run-benchmark --config configs/react.local.yaml
```

- [ ] **Step 4: Score both**

```bash
uv run python scripts/score_predictions.py \
    --pred artifacts/runs/<run_id_phase2_1> --gold data/public/output \
    --lambda 0.1 --output artifacts/runs/<run_id_phase2_1>/scores.json --verbose
uv run python scripts/score_predictions.py \
    --pred artifacts/runs/<run_id_phase2_2> --gold data/public/output \
    --lambda 0.1 --output artifacts/runs/<run_id_phase2_2>/scores.json --verbose
```

Compute mean over the two runs and per-task wall-clock = total walltime / 30.

- [ ] **Step 5: Apply decision rule**

- If mean delta vs Phase 1 baseline ≥ +0.03 AND per-task walltime ≤ 800s → keep. Append rows to §11 attempt log.
- Else → revert. Either:
  - Flip `video_filter_audit: disabled` in `react.example.yaml` (single commit) and proceed to Phase 3 with the auditor inactive
  - OR `git revert` the Phase 2 commits if a check turned out structurally wrong

In either case, append the result and the decision to §11 attempt log:

```markdown
| 2026-06-23 | Phase 2 | filter_audit (3 checks) | <P1 mean> | <P2 mean> | <delta> | <walltime/task> | <keep|revert> |
```

- [ ] **Step 6: Commit the doc update**

```bash
git add docs/superpowers/specs/2026-06-23-video-task-score-08-design.md
git commit -m "docs(video): Phase 2 A/B result — <keep|revert> with delta <±X.XX>"
```

---

## Task 17: Phase 3 A/B validation

**Goal:** Verify Phase 3 (abbreviation hint) earns at least +0.03 mean over the Phase 2 cumulative state and stays under 800 s/task.

**Files:** none (procedural). Only updates §11 attempt log.

- [ ] **Step 1: Confirm Phase 3 commits**

```bash
git log --oneline
```

Expected: includes Tasks 12-14 commits (rule_templater prefer_abbr_column, render hint, explore_video thread question).

- [ ] **Step 2: Run Phase 3 A/B × 2**

Same procedure as Task 16, Steps 2-4.

- [ ] **Step 3: Apply decision rule**

- If mean delta vs Phase 2 cumulative ≥ +0.03 AND per-task walltime ≤ 800s → keep. Append rows to §11 attempt log.
- Else → revert the Phase 3 commits (no toggle to flip — it's a soft text hint):

```bash
git revert <task_14_commit> <task_13_commit> <task_12_commit>
```

Append result to §11 attempt log.

- [ ] **Step 4: Commit the doc update**

```bash
git add docs/superpowers/specs/2026-06-23-video-task-score-08-design.md
git commit -m "docs(video): Phase 3 A/B result — <keep|revert> with delta <±X.XX>"
```

---

## Task 18: Final cumulative check + readiness

**Goal:** Confirm cumulative mean ≥ 0.80 and the worktree is ready to be merged or pushed for review.

**Files:** none (procedural).

- [ ] **Step 1: Compute cumulative state**

Look at the §11 attempt log and confirm the last "keep" row's mean ≥ 0.80.

- [ ] **Step 2: If ≥ 0.80 — celebrate, prepare for review**

```bash
git log --oneline main..HEAD
git diff --stat main..HEAD
```

Append a one-line "DONE" entry to §11 attempt log. Commit.

- [ ] **Step 3: If < 0.80 — escalate**

Document the gap and the most likely Plan B (port the deferred Phase 4 GCD work) in a new design doc under `docs/superpowers/specs/`. Do not start coding without a fresh design.

---

## Notes for executor

- **Where the spec disagrees with reality, the spec is wrong.** If a code file's actual shape differs from what this plan describes (function signature, return type, field name), pause, update the spec inline, and re-derive the plan task. Do not paper over the gap.
- **Skim spec §7 (error handling) before any auditor work.** It defines the contract: `audit_answer_table` never raises; toggle off = current behavior.
- **Never commit `configs/react.local.yaml`.** It carries real API keys.
- **Run the full suite (`uv run pytest tests/ -q`) after Task 10 and after Task 14** — those are the two cross-cutting threads.
- **Tracing on.** `react.local.yaml` already enables tracing. Use `artifacts/traces.db` for any A/B post-mortem.
