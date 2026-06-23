# Video Task Score 0.745 → 0.80 — Design

**Status**: drafted 2026-06-23, awaiting user review
**Branch**: `worktree-feat-video-score-08` (off `main@746b343`)
**Config**: `configs/react.local.yaml` with `enable_thinking: true`

---

## 1. Goal & success criteria

Push the local 30-task video subset mean from ~0.745 to **≥ 0.80** on top of `main@746b343` without breaking the online 2h / 60-task budget.

A change is kept only if **all** of:

1. Two consecutive A/B runs (one fresh seed, one warm seed) show video-subset mean ≥ 0.80 at the cumulative point
2. The change's own A/B delta is **≥ +0.03 mean** vs the prior cumulative state (noise band ≈ ±0.017 mean → ~2σ guard)
3. Per-task wall-clock (run total / 30) stays ≤ **800 s** (online safety margin against ≤ 960 s/task allowance at 8 workers)
4. The change is toggle-able off (single flag flip reverts to prior behavior)

**Non-goals**:

- Not changing the model (`qwen3.5-35b-a3b` is locked)
- Not increasing per-task LLM-call count beyond +1 (no self-consistency / multi-judge)
- Not editing the grader / scorer
- Not retuning temperature / top_p / top_k (current `T=0.6 top_p=0.9 top_k=15` is the benchmarked optimum)
- Not adding hardcoded task-ID rules (CLAUDE.md generalize policy)

---

## 2. Diagnostic baseline (evidence)

Failure breakdown of the 7 zero-score tasks in run `20260621-018` (latest local run on `feat/video-rule-templater`, thinking=false config):

| task | mode | gold vs pred |
|---|---|---|
| task_1 | threshold / filter wrong | gold 1 row vs pred 8 rows; wrong industry filter |
| task_4 | boundary + duplicate rows | gold 125 vs pred 138; extra "国家股" duplicates, "中国太平洋..." duplicated |
| task_12 | filter wrong | count 0 vs 1 |
| task_25 | abbr vs full-name column | pred "易方达上证50ETF"/"泰达宏利中证500指数增强(LOF)"; gold "易方达50ETF"/"泰达中证500" |
| task_31 | abbr vs full-name column (textbook) | pred "富国天丰强化收益债券(LOF)"; gold "富国天丰" |
| task_41 | zero-count broadening | all groups returned 0 (`746b343` may have fixed this) |
| task_52 | filter wrong | count 43 vs 50 |

Aggregated:

| failure mode | tasks | share |
|---|---|---|
| threshold / filter / boundary | 1, 4, 12, 52 | **4/7** |
| abbreviation vs full-name column | 25, 31 | **2/7** |
| zero-count broadening | 41 | 1/7 (possibly fixed on main) |

This drives Phase ordering: threshold/boundary first (largest failure cluster), abbreviation preference second.

**Caveat**: this data is from a `thinking=false` run. The chosen config in §1 is `thinking=true`, so Phase 1 must rebaseline because failure structure can shift ±1-2 tasks across thinking states (doc §1.1 row 015 vs 016 evidence).

---

## 3. Architecture

Current video task data flow on `main@746b343`:

```
task.json (question)
   │
   ▼
[main ReAct agent] ── inspect_files ── explore_video(path, instructions)
   │                                            │
   │                                            ▼
   │                                       [video sub-agent, max_steps=3]
   │                                            │  report-only registry
   │                                            ▼
   │                                       VideoReportInput {
   │                                         summary, timeline,
   │                                         extracted_data{rules, samples},
   │                                         coverage, warnings }
   │                                            │
   │                                            ▼
   │                                  rule_templater.synthesize_filter_recipe(rules)
   │                                            │
   │           ◄────────────────────────────────┘  observation = findings + filter_recipe
   ▼
execute_python / execute_context_sql ──► AnswerTable(columns, rows)
   │
   ▼
answer tool ──► [TerminalAnswerPolicy.maybe_reject]
                    │
                    ├── evaluate_video_constraints (existing — batch_id, snapshot multiplicity)
                    ├── audit_prediction  ←── NEW (this design)
                    └── AnswerVerifier (LLM)
   │
   ▼
prediction.csv
```

The four targeted modification points (numbered for cross-reference with §4):

| # | Layer | Change | Failure mode addressed |
|---|---|---|---|
| ① | new `scripts/triage_failures.py` | diagnostic-only script, reads traces.db + scores.json | Phase 1 measurement |
| ② | new `verification/video_filter_audit.py` + hook in `verification/terminal_policy.py` | deterministic checks on `AnswerTable` between existing video_constraints check and LLM verifier | threshold/boundary (4/7) |
| ③ | `video/rule_templater.py` extend with `question` param; `tools/explore_video.py` threads `task.question` through | abbreviation column preference hint | abbr (2/7) |
| ④ | port the uncommitted `video_structured_outputs` work from the user's working tree onto this branch | GCD on video.report boundary (CRANE pattern) | online-only, ~0 local on DashScope |

**Out of architecture scope**:

- ❌ Not adding a new agent layer (main + video stays two-tier)
- ❌ Not touching ReAct main loop
- ❌ Not changing `explore_video` tool signature (main agent stays unaware of auditor)
- ❌ Not editing `video/runner.py` or sub-agent prompt (rules extraction already correct)

---

## 4. Component contracts

Minimal interfaces only — each unit is independently testable and toggle-able.

### 4.1 `src/agents/verification/video_filter_audit.py` (new)

```python
@dataclass(frozen=True, slots=True)
class AuditFinding:
    code: str        # 'duplicate_export_rows' | 'boundary_ambiguous' | 'row_count_overflow' | 'audit_internal_error'
    severity: str    # 'reject' | 'warn'
    message: str     # human-readable for the ReAct agent
    evidence: dict   # finding-specific structured data

def audit_answer_table(
    table: AnswerTable,           # columns: list[str], rows: list[list[Any]]
    rules: dict[str, Any] | None,
    samples: dict | list | None,
) -> list[AuditFinding]: ...
```

- **Pure function**: no IO, no LLM, no side effects
- **Never raises**: internal exceptions caught and wrapped as `AuditFinding(code='audit_internal_error', severity='warn')`
- Inputs: `AnswerTable` produced by `build_answer_table_without_side_effects`; `rules` from `findings.extracted_data.rules`; `samples` from `displayed_samples`
- Operates on `table.columns` + `table.rows` directly — no pandas dependency (matches the codebase's existing post-check style)

### 4.2 `src/agents/verification/terminal_policy.py` (modify)

Insert the auditor between the existing `evaluate_video_constraints` deterministic check and the LLM `AnswerVerifier.verify`, sharing the same `rejections_used` budget:

```python
# Existing
deterministic_rejection = evaluate_video_constraints(state, task.question, row_count)
if deterministic_rejection is not None:
    self.rejections_used += 1
    ...
    return TerminalAnswerRejection(content={"answer_rejected": deterministic_rejection})

# New — between existing deterministic check and LLM verifier
if self.filter_audit_enabled and candidate is not None:
    rules, samples = _extract_video_rules_and_samples(state)
    findings = audit_answer_table(candidate, rules, samples)
    rejects = [f for f in findings if f.severity == "reject"]
    if rejects:
        self.rejections_used += 1
        if candidate.columns and candidate.rows:
            self.fallback_answer = candidate
            self.fallback_answer_arguments = dict(arguments)
        return TerminalAnswerRejection(content={"answer_rejected": _format_audit_rejections(rejects)})

if self.verifier is None:
    return None
rejection = self.verifier.verify(...)
```

`filter_audit_enabled` is a boolean dataclass field on `TerminalAnswerPolicy`, wired from `app_config.agent.video_filter_audit == "enabled"` at construction time.

No new retry quota — `rejections_used` is the existing shared budget (`AnswerVerifier.max_rejections` default 2).

`_extract_video_rules_and_samples(state)` reads the latest `explore_video` step from `state.steps` and returns `findings.extracted_data.rules` + `displayed_samples`. Helper lives in `video_filter_audit.py` to keep `terminal_policy.py` thin.

### 4.3 `src/agents/video/rule_templater.py` (modify)

```python
def extract_filter_recipe(
    rules: dict[str, Any] | None,
    *,
    question: str | None = None,
) -> FilterRecipe | None: ...
```

Add field on `FilterRecipe`: `prefer_abbr_column: bool = False`.

Set True when `question` contains any of `{简称, abbreviation, abbr, 代号}` (case-insensitive substring match).

`render_filter_recipe` adds one line at the top of the recipe when True:

```
prefer_column_kind: abbreviation — when both 'XxxAbbr' and full-name columns exist, use the *Abbr column
```

This is a **soft hint**, not a structural change to the recipe. No toggle needed (the worst outcome is silent no-op).

### 4.4 `src/agents/tools/explore_video.py` (1-line modify)

Pass `task.question` into `extract_filter_recipe` so the templater can branch on it. This is the only signature change in the call chain.

### 4.5 `configs/react.example.yaml` + `src/agents/config.py` (extend)

```yaml
agent:
  video_filter_audit: enabled        # "enabled" | "disabled"; new in Phase 2
  video_structured_outputs: enabled  # added in Phase 4 alongside ported GCD wiring
```

`react.local.yaml` is **not** edited via PR (contains real API keys); these toggles default to the new values for online deploy.

**Important on `video_structured_outputs`**: this field and the supporting `_video_structured_outputs(config)` helper + `agents/llm/openai.py:OpenAIModelAdapter.structured_outputs` parameter are **NOT on `main@746b343`**. They live as uncommitted modifications in the user's original working tree on branch `feat/video-rule-templater`. Phase 4 will port that work onto `worktree-feat-video-score-08` as a clean, isolated commit (no API-key bleed) before the toggle is meaningful. If the port is deferred, Phase 4 is a no-op and the toggle is not added.

### 4.6 `scripts/triage_failures.py` (new, diagnostic-only)

Read `artifacts/runs/<run_id>/scores.json` + `artifacts/traces.db`; emit per-task failure classification in a single markdown table. No runtime code dependency — pure analysis tool.

---

## 5. Phases

### Phase 1 — Measurement (1-2 days)

**Goal**: anchor the actual baseline on `react.local.yaml + thinking=true` and refresh the failure classification.

Steps:

1. Set `enable_thinking: true` in local `react.local.yaml` (local-only, not committed)
2. Run `uv run dabench run-benchmark --config configs/react.local.yaml` twice (one fresh seed via run_id sequence, one warm using cached responses where possible)
3. Score each with `scripts/score_predictions.py --pred artifacts/runs/<id> --gold data/public/output --lambda 0.1 --output <path>`
4. Use `scripts/triage_failures.py` to bucket failing tasks by mode

**Exit criteria**:

- Two `scores.json` files, both with mean computable
- A 1-3 word root-cause label per failing task (≤0.5 score)
- A refreshed mode-distribution table to compare with the §2 evidence
- Confirmation of `task_41` status (was main `746b343` fix enough?)
- Confirmation that `max_tokens=8192` is not being hit hard (count tasks with `finish_reason=length`)

**Go / no-go**: if the new baseline mean already ≥ 0.80, the project is done (unlikely but possible). If the failure mode distribution differs materially from §2, revise Phase 2 / 3 priorities before coding.

### Phase 2 — `video_filter_audit` (2-3 days, primary leverage)

**Implements**: §4.1 + §4.2.

Sub-checks (each independently testable):

1. **`duplicate_export_rows`** (severity=reject)
   - Trigger: `rules.export_fields` non-empty AND `df.duplicated(subset=export_fields).any()`
   - Evidence payload: list of first-N duplicate key values
2. **`boundary_ambiguous`** (severity=warn)
   - Trigger: rules.operator ∈ `{>, >=}` AND any row has the export value exactly equal to the threshold
   - Evidence payload: `{operator_chosen, threshold, rows_at_threshold, count_if_strict, count_if_inclusive}`
3. **`row_count_overflow`** (severity=warn)
   - Trigger: `displayed_samples` parses out a credible small total N (single-cell number, or list len < df_len/2), and `len(df) > 2 * N`
   - Evidence payload: `{samples_total, df_rows}`

Implementation order: 1 → 3 → 2 (simplest first to validate the wiring).

**A/B gate**: must show delta ≥ +0.03 mean vs Phase 1 cumulative baseline; otherwise revert and disable toggle.

### Phase 3 — Abbreviation preference (1-2 days)

**Implements**: §4.3 + §4.4.

No toggle — output text-only change.

**A/B gate**: must show delta ≥ +0.03 mean vs Phase 2 cumulative; otherwise revert.

### Phase 4 — Port GCD wiring (CRANE boundary) (1-2 days, conditional)

**Implements**: §4.5 final state.

The `video_structured_outputs` switch + `OpenAIModelAdapter.structured_outputs` plumbing is not on `main@746b343`; it sits as uncommitted edits in the user's original working tree. This phase is gated on the user confirming the port is wanted (the design's user-review gate).

If approved:

1. Read the uncommitted diffs of `src/agents/{application.py, config.py, llm/openai.py}` from the original working tree
2. Re-apply them as a clean commit on `worktree-feat-video-score-08`, ensuring no API-key contamination
3. Verify DashScope path: grep for `backend_kind == "dashscope"` behavior around `structured_outputs` injection; if DashScope silently drops the field, document local A/B will not measure this change (online vLLM is the beneficiary)
4. Set `video_structured_outputs: enabled` default in `react.example.yaml`

**Acceptance**: code-review only (no A/B gate). Online vLLM may show effect; local DashScope may not.

If deferred: Phase 4 is no-op, the design's 0.80 target rests entirely on Phase 2 + 3.

### Phase 5 — Cumulative validation (continuous)

After every Phase commit:

- Same 30-task subset run, same seed pair
- Compare cumulative mean to running goal trajectory
- If cumulative regresses by > 0.02, the last Phase is rolled back via toggle
- All phases' attempts (kept and reverted) are appended to an `attempt_log` table in this document on success/end

---

## 6. Testing strategy

### 6.1 Unit tests (`uv run pytest`)

**`tests/verification/test_video_filter_audit.py`** (new):

| group | cases |
|---|---|
| `duplicate_export_rows` | triggers on dup, silent on unique, safe on empty df |
| `boundary_ambiguous` | triggers when row==threshold, silent otherwise, safe when operator missing |
| `row_count_overflow` | triggers when samples N small and df large, silent when N not parseable, safe when samples is list |
| `audit_prediction` aggregator | accumulates multiple findings, swallows internal exceptions as `warn`, safe on `rules=None` / `df.empty` |

**`tests/video/test_rule_templater.py`** (extend):

| group | cases |
|---|---|
| `extract_filter_recipe(question=...)` | `简称` → True; `abbreviation` → True; `abbr` → True; `代号` → True; no match → False; `question=None` → False |
| `render_filter_recipe` | `prefer_abbr_column=True` adds the top hint line; False omits it |

**`tests/verification/test_terminal_policy.py`** (extend):

| group | cases |
|---|---|
| auditor wiring | reject finding → `maybe_reject` returns `TerminalAnswerRejection`; exception in auditor → `maybe_reject` falls through to LLM verifier (no crash); `filter_audit_enabled=False` → auditor skipped; auditor reject still increments `rejections_used` |

### 6.2 Integration tests (no LLM)

**`tests/video/test_explore_video_integration.py`** (extend): mock video sub-agent with fixture returns, assert `task.question` is threaded through to templater and final `filter_recipe` text contains the abbreviation hint line when appropriate.

### 6.3 End-to-end A/B protocol (not in CI)

Per Phase, after implementation:

1. On `worktree-feat-video-score-08`, cherry-pick / commit the Phase change
2. Run `uv run dabench run-benchmark --config configs/react.local.yaml` × 2 (fresh + warm)
3. Score both with `scripts/score_predictions.py`
4. Append result row to the `attempt_log` table at the bottom of this document
5. If `delta_mean ≥ +0.03` AND `walltime_per_task ≤ 800s` → keep; else toggle off and revert commit

### 6.4 Anti-tests (explicitly not written)

- ❌ Per-task hardcoded score assertions (violates generalize policy)
- ❌ Fake "X improvement" mocks
- ❌ Network-dependent tests (use fixtures + dataclass DI)

---

## 7. Error handling

### 7.1 Auditor misfire

| misfire | impact | mitigation |
|---|---|---|
| False reject | wastes 1 retry; worst case retry produces worse answer | shared `rejections_used` budget; after budget exhausted, auditor goes silent |
| False pass | unchanged from current behavior | no regression |
| Auditor internal crash | answer tool fails | `try/except` wraps internals; emit `AuditFinding(code='audit_internal_error', severity='warn')`, fall through |

### 7.2 Abbreviation hint misdirection

| case | effect | mitigation |
|---|---|---|
| Question says "简称" but schema has only full-name column | agent falls back to full-name | hint is advisory; no reject path attached |
| Question says "名称" but means abbreviation | hint silent | no regression |

### 7.3 GCD path divergence (Phase 4)

- vLLM rejects schema → sub-agent retries unconstrained → original behavior
- DashScope silently drops `structured_outputs` → local A/B sees no Phase 4 delta → documented, not blocking

### 7.4 Per-task wall-clock budget

Online ceiling: ~960 s/task at 8 workers (2h / 60 / 8 × 60). Local red line: **800 s/task** averaged.

If Phase 2 retry path pushes wall-clock past red line on the A/B run, the Phase is reverted regardless of score delta.

### 7.5 max_tokens=8192 with thinking=true

Phase 1 must record any task where the model hit `finish_reason='length'`. If ≥ 5 tasks hit the cap, a config-level experiment (raise to 16384 in `react.local.yaml` for testing only — not committed) precedes coding Phase 2.

---

## 8. Risk register

| risk | severity | mitigation |
|---|---|---|
| Phase 1 baseline already ≥ 0.80 (project done) | low | celebrate, write up findings |
| Phase 1 reveals different dominant failure modes than §2 | medium | pause coding, revise Phase 2/3 priorities |
| Score variance dominates measured delta (doc §1.1 lesson) | high | N=2 A/B + +0.03 threshold (~2σ over noise band) |
| max_tokens cap suppresses reasoning quality | medium | Phase 1 instruments cap-hits; raise locally if needed |
| `react.local.yaml` API key leak via PR | high | never commit `react.local.yaml`; only `react.example.yaml` carries toggles |
| Phase 2 auditor regresses 1 task while fixing 2 | medium | A/B mean delta catches net effect; per-task diff in attempt log |
| Auditor reject pushes `rejections_used` past budget on a borderline task | low | budget is shared, not new; existing behavior preserved |
| Phase 4 GCD path not active on DashScope | low | documented, online vLLM still benefits |

---

## 9. Out of scope (this design)

- Self-consistency (N=2/3 sampling on video sub-agent) — compute budget violation
- New video features (sparse keyframe sampling, GenS-style frame selection) — research blocked until vLLM exposes `num_frames` per call
- Boundary inclusive schema field on `VideoReportInput` — reverted previously per doc §8.3
- Expanded templater alias dictionary — reverted previously per doc §8.3
- LLM-based answer self-verifier — reverted previously per doc §8.3 (Park et al. NeurIPS 2024)
- Same-model judge panel — same root cause
- Restructuring `tools/answer.py` — only adds a single check point

---

## 10. ROI estimate

| phase | expected mean Δ | cumulative |
|---|---|---|
| Phase 1 (rebaseline, no code) | — | 0.74-0.78 (new ground truth, not a gain) |
| Phase 2 (filter audit) | +0.03 ~ +0.07 | 0.77-0.85 |
| Phase 3 (abbr preference) | +0.03 ~ +0.06 | 0.80-0.91 |
| Phase 4 (GCD confirm) | local ~0; online +0.01-0.02 | — |

Median path to 0.80: Phase 2 lands ~+0.05, Phase 3 lands ~+0.04 → 0.745 + 0.09 = 0.835. Worst plausible: both halved → +0.045 → 0.79 (one tenth short of target).

Plan B if Phase 2 + 3 fall short: Phase 4 (GCD) is the last unexplored lever; beyond that, the design has exhausted what can be done without violating non-goals.

---

## 11. Attempt log

(Populated by Phase 5 — one row per A/B run.)

| date | phase | description | baseline mean | new mean | Δ | walltime/task | decision |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
