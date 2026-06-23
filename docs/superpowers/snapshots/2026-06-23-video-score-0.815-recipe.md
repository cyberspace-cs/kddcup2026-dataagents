# Video task subset 0.815 — Reproduction recipe

**Tag**: `video-score-0.815`
**Git HEAD**: `46d6c89` (worktree branch `worktree-feat-video-score-08`)
**Run artifact**: `artifacts/runs/20260623-003/`
**Score snapshot**: `docs/superpowers/snapshots/2026-06-23-video-score-0.815-scores.json`

## Result

| metric | value |
|---|---|
| per-video-task mean | **0.815** |
| total_score (60 task slots) | 0.4075 |
| mean_recall | 0.4083 |
| scored_nonzero | 25 / 60 |
| video-only breakdown | 24 perfect / 1 partial / 5 zero |

Remaining failures (5 zeros + 1 partial):

| task | score | gold rows | pred rows | dominant failure mode |
|---|---|---|---|---|
| task_4 | 0.000 | 125 | 137 | duplicate/extra rows (`+12`); Phase 2 `duplicate_export_rows` did not fire — export_fields likely mis-aliased |
| task_25 | 0.000 | 50 | 49 | full-name vs `*Abbr` column choice; abbr hint did not flip the agent |
| task_31 | 0.000 | 10 | 10 | full-name vs `*Abbr` column choice; abbr hint did not flip the agent |
| task_41 | 0.000 | 2 | 3 | mid-baseline variance — was 0.45 in #1/#2, regressed to 0 here |
| task_52 | 0.000 | 1 | 1 | single-cell count value off |
| task_55 | 0.450 | 7 | 7 | partial value match |

## Code state at this tag

Phase 2 (`video_filter_audit`):
- `src/agents/verification/video_filter_audit.py` (new) — `AuditFinding` dataclass + `audit_answer_table()`; three checks: `duplicate_export_rows`, `boundary_ambiguous`, `row_count_overflow`; aggregator swallows internal exceptions as `warn` finding.
- `src/agents/verification/terminal_policy.py` — `_extract_video_rules_and_samples()` helper + auditor call inserted between `evaluate_video_constraints` and the LLM verifier in `maybe_reject`; shares `rejections_used` budget.
- `src/agents/config.py` — `video_filter_audit: Literal["enabled","disabled"] = "enabled"`.
- `src/agents/agent.py` + `src/agents/application.py` — `filter_audit_enabled` threaded from `AgentConfig.video_filter_audit` through `ReActAgent` / `AgentApp`.

Phase 3 (abbreviation column hint):
- `src/agents/video/rule_templater.py` — `extract_filter_recipe(rules, *, question=None)`; `FilterRecipe.prefer_abbr_column: bool = False`; renderer emits `prefer_column_kind: abbreviation` line when set.
- `src/agents/tools/explore_video.py` — passes `task.question` into `synthesize_filter_recipe`.

Scorer fix:
- `scripts/score_predictions.py` — `_read_csv()` tries `utf-8-sig` → `utf-8` → `gb18030` to cover legacy GB18030 gold files (e.g. `data/public/output/task_1/gold.csv`).

Tooling:
- `scripts/triage_failures.py` (new) — markdown bucket of imperfect tasks.

## Configuration (NOT in git; recreate `configs/react.local.yaml`)

```yaml
dataset:
  root_path: data/public/input

agent:
  model: qwen3.5-35b-a3b
  api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
  api_key: <YOUR_DASHSCOPE_KEY_1>
  max_steps: 32
  allow_parallel_tool_calls: true
  max_tokens: 16384
  enable_thinking: false
  video_tool_choice: auto
  video_max_tokens: 16384
  # video_filter_audit: enabled  (default — do NOT add unless you want to flip it)
  backend_kind: null

  temperature: 0.6
  top_p: 0.9
  top_k: 15
  min_p: 0.0
  presence_penalty: 0.0
  seed: 1024
  enable_preact: false

  api_keys:
    - <YOUR_DASHSCOPE_KEY_1>
    - <YOUR_DASHSCOPE_KEY_2>
  rate_limit:
    rpm_per_key: 2000
    tpm_per_key: 2000000
    safety_factor: 0.95
    reserve_output_tokens: 16384
    max_retries: 6
    backoff_initial_seconds: 1.0
    backoff_max_seconds: 30.0

run:
  output_dir: artifacts/runs
  run_id: null
  max_workers: 8
  task_timeout_seconds: 1000
  blocklist:
    - task_2
    - task_3
    - task_5
    - task_8
    - task_9
    - task_10
    - task_13
    - task_14
    - task_15
    - task_16
    - task_18
    - task_19
    - task_28
    - task_30
    - task_32
    - task_33
    - task_38
    - task_40
    - task_42
    - task_43
    - task_44
    - task_45
    - task_46
    - task_47
    - task_49
    - task_50
    - task_51
    - task_53
    - task_58
    - task_59

tracing:
  enabled: true
  db_path: artifacts/traces.db
```

## How to reproduce

```bash
git checkout video-score-0.815
# Populate configs/react.local.yaml as above with your Dashscope keys.
uv run dabench run-benchmark --config configs/react.local.yaml
# Score with the same fixed scorer:
uv run python scripts/score_predictions.py \
    --pred artifacts/runs/<new_run_id> \
    --gold data/public/output \
    --lambda 0.1 \
    --output artifacts/runs/<new_run_id>/scores.json \
    --verbose
```

Expected: per-video mean ≈ 0.78–0.83 (single-run variance ±0.5 total / 30 ≈ ±0.017 mean from the 0.815 anchor).

## Open attribution

The +0.070 net gain vs the 20260621-018 reference (0.745, same scorer) is composed of:
1. `main@746b343` prompt changes (vs old `feat/video-rule-templater@280d63c`)
2. `max_tokens: 8192 → 16384`
3. Phase 2 (filter audit)
4. Phase 3 (abbr hint)

Isolated attribution requires further A/B (e.g. `video_filter_audit: disabled` on this same config) — in flight as run #4 at the time of this snapshot.
