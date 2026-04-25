## Why

The repository has accumulated significant drift between live code and project surface: a ~5-module `workflow/` Plan-Execute subsystem and several test suites were deleted but their `__pycache__`, dead config (`configs/delegate_experiment.yaml`), and ~3,800 lines of design docs (`docs/00`–`docs/11`) describing non-existent modules (`TaskOrchestrator`, `DataProfiler`, `OutputWriter`, `read_pdf`, `read_chart`, etc.) still ship as authoritative artefacts. At the same time, the live code has structural debt that violates Python and agent-design specifications: `ReActAgent.run` (621 LOC file) interleaves protocol parsing, tool dispatch, terminal detection, trace recording, and retry accounting; `AgentConfig.tool_protocol` is `str` despite a `Literal["native","text"]` runtime contract (forcing `# type: ignore[arg-type]` in the runner); `FileTokenBucket._save` is non-atomic; there is no static type checker, no pre-commit, no coverage. New contributors cannot use the docs as reference, and tooling cannot enforce the conventions the code already follows.

This change establishes a single source of truth — a small, accurate `docs/` set generated from the live system; canonical specs for each live capability under `openspec/specs/`; and quality gates (`pyright`, ruff, pytest+coverage, pre-commit) that hold the new boundaries — while doing the structural refactor needed to make the agent layer match those specs without breaking the public CLI/wire-format surface.

## What Changes

**Architecture (live boundaries reshaped — wire-format unchanged)**
- Split `agents/react.py` into three cooperating units: a protocol adapter (native ↔ text → unified `ToolCall` stream), a tool dispatcher (terminal-flag from registry metadata, not from `ToolExecutionResult.is_terminal` discovered post-execution), and a step recorder (typed event kinds replacing the `__error__` sentinel: `model_error`, `parse_error`, `empty_tool_calls`, `tool_error`, `observation`).
- Consolidate `ToolSpec` + handler into a single `ToolDefinition` (name, description, `input_schema`, `json_schema`, handler, `is_terminal`); make `ToolRegistry` immutable post-construction so `NativeToolsOpenAIAdapter`'s schema cache stays valid.
- Introduce a small App-factory boundary in `run/runner.py` (`build_application(config) -> AgentApp`) so the runner stops importing concrete adapters/registry directly; preserves subprocess-timeout/partial-trace behavior verbatim.
- Move `ScriptedModelAdapter` and `ScriptedNativeModelAdapter` from `src/data_agent_baseline/agents/model.py` to `tests/helpers/scripted_adapters.py`. **BREAKING (internal API only — no production caller).**
- Decouple package-root re-exports: `src/data_agent_baseline/{agents,tools,run,benchmark}/__init__.py` either remain as the supported public façade with `__all__` enforced, or are pruned to empty (decision recorded in `design.md` after measuring importer surface; preliminary direction: keep but lock down).

**Configuration & Validation**
- Replace manual YAML parsing in `config.py:152-327` with **Pydantic v2** models that produce byte-compatible `AppConfig`/`AgentConfig`/`RunConfig`/`RateLimitConfig`/`DatasetConfig` instances (same field names, same defaults, same exception types via custom validators). `tool_protocol` becomes `Literal["native","text"]` end-to-end; bool/int coercion becomes strict (`bool('false') == False` no longer); `max_steps`, `max_workers`, `max_tokens` get range validation at load time.
- Existing example YAML (`configs/react_baseline.example.yaml`) parses unchanged.

**Bug fixes (load-bearing semantics)**
- `react.py:527` empty-tool retry uses `>` instead of `>=` → `max_empty_tool_call_retries=1` currently permits two empty turns. Fix to match documented contract.
- `agents/rate_limit.py:144` `_save` writes JSON in place → torn-write window. Switch to write-temp + atomic `os.replace`.
- `tools/python_exec.py:198` `Queue.empty()` is unreliable across processes → switch to `get_nowait()` + `queue.Empty` after `process.join`.
- `runner.py:78` second-resolution UTC run-id collides on rapid back-to-back runs → upgrade default precision (e.g., add micro suffix) while keeping single-directory-name validation.

**Quality gates (new in this change)**
- Add **pyright** with strict mode on `src/data_agent_baseline/` and basic mode on `tests/`. Ship `[tool.pyright]` in `pyproject.toml` and a `pyright-ignore` policy.
- Add `pytest` `addopts = "-ra"` and `--cov=data_agent_baseline --cov-report=term-missing` (coverage as optional dev dep; fail-under threshold proposed but value left to design.md).
- Add `[project.optional-dependencies] dev` += `pyright`, `pytest-cov`, `pre-commit`.
- Add `.pre-commit-config.yaml`: `ruff format` → `ruff check --fix` → `pyright src`. Hook is opt-in; CI is out-of-scope for this change.

**Documentation reset (full overhaul, per user decision)**
- Delete: `docs/00-index.md`, `docs/01-requirements.md`, `docs/02-feasibility.md`, `docs/03-architecture.md`, `docs/04-detailed-design.md`, `docs/05-milestones-testing-risk.md`, `docs/06-interfaces.md`, `docs/07-prompts.md`, `docs/08-local-scorer.md`, `docs/09-deployment.md`, `docs/10-task-playbooks.md`, `docs/11-ops.md`. All describe deleted modules (Plan-Execute, DataProfiler, OutputWriter, vision/PDF tools, `src/dabench` layout) or never-existed referenced files (`baseline_metrics.md`, `submission_log.md`, `experiments.md`, `qna_organizer.md`).
- Rewrite a small honest set under `docs/` against live code:
  - `docs/architecture.md` — current ReAct loop, native+text protocols, key-pool + rate limiter, subprocess isolation.
  - `docs/cli.md` — `dabench` four subcommands and config schema.
  - `docs/agent-protocol.md` — when to use native vs text; tool inventory; trace.json / summary.json schemas.
  - `docs/development.md` — uv workflow, ruff + pyright + pytest gates, pre-commit, contributing rules.
- **README**: delete `README.zh.md`; trim `README.md` to upstream-only content (Quick Start, Dataset, Configuration, CLI, Tools, Outputs, Main Modules). Team-status / milestone content moves into `docs/` only if explicitly retained. **BREAKING for any external bookmark/reference to README.zh.md.**
- `CHANGELOG.md`: rewrite the `[Unreleased]` section to reflect actual recent work (PR-1 baseline, PR-2 dual-key pool + rate limiter, native function calling, partial-trace recovery, this refactor) using the existing template.

**Cleanup (orphans)**
- Delete `src/data_agent_baseline/workflow/` (only stale `__pycache__` remains).
- Delete `scripts/` (only stale `__pycache__`; nothing to preserve).
- Delete `configs/delegate_experiment.yaml` (references `enable_delegate`, `delegate_max_steps_cap` fields that don't exist in `AgentConfig`).
- Delete every stale `.pyc` under `tests/__pycache__/` (13 entries: `test_delegate_*`, `test_workflow_*`, `test_qwen_agent_adapter`, `test_runner_timeout`, `test_sqlite_value_grounding`, `test_final_answer_shaper`, `test_react_native_with_delegate`, `test_prompt_native`).
- Remove unused `DABenchPublicDataset.iter_tasks(task_ids=…, difficulty=…, difficulties=…)` filter parameters (no callers; reintroduce when the CLI gains a `--task-ids` flag).
- Remove `tests/test_runner_key_pinning.py:139 _fake_dataset`, `tests/test_rate_limit_adapter.py:381 _ = Any`, `tests/test_model_adapter_request.py:59 _make_response tool_calls` (dead test helpers).

## Capabilities

### New Capabilities

This change introduces canonical specs for every live capability — `openspec/specs/` is currently empty even though the code implements all of these. Each entry below becomes `openspec/specs/<name>/spec.md`.

- `agent-execution-loop`: ReAct loop with separated protocol adapter, tool dispatcher, and step recorder; typed event kinds; bounded model/empty-tool retries; partial-trace observability via `step_callback`.
- `model-adapter`: OpenAI-compatible `chat.completions` adapter with text and native function-calling protocols, Qwen3 `<think>` stripping, `enable_thinking` extra-body passthrough, `usage` capture (incl. cached/reasoning tokens), and `latency_ms`.
- `tool-registry`: Single `ToolDefinition` (name, description, `input_schema`, `json_schema`, handler, `is_terminal`) with immutable registry post-construction; default 8-tool set (`list_context`, `read_csv`, `read_doc`, `read_json`, `inspect_sqlite_schema`, `execute_context_sql`, `execute_python`, `answer`); strict path sandboxing.
- `task-runner`: Subprocess timeout isolation, partial-trace recovery, key-pool pinning, parallel `ThreadPoolExecutor` execution, deterministic per-task output layout (`trace.json` + `prediction.csv`) and per-run `summary.json`.
- `rate-limited-key-pool`: Cross-process file-token-bucket (`filelock`-protected) with RPM+TPM dual buckets, predeposit/reconcile, Retry-After parsing, exponential backoff with jitter, atomic state writes, and key pinning by `task_index % len(api_keys)`.
- `benchmark-dataset`: Read-only loader for `data/public/input/` (`task_<n>/`), strict `task.json` schema validation, natural-number ordering, difficulty counts.
- `cli-commands`: `dabench {status, inspect-task, run-task, run-benchmark [--limit N]}` with rich progress reporting; YAML config validation; deterministic `run_id` semantics.
- `app-config`: Pydantic v2-based `AppConfig`/`AgentConfig`/`RunConfig`/`RateLimitConfig`/`DatasetConfig` with `Literal` types, range validation, strict bool coercion, key-pool ↔ rate-limit pairing rules, project-root anchored path resolution.
- `quality-gates`: `ruff` (format + lint), `pyright` (strict on src, basic on tests), `pytest` with `-ra` and coverage, `pre-commit` chain; documented exit-code contracts; CI-ready but not CI-wired in this change.
- `project-documentation`: Concise English-only doc set (`architecture`, `cli`, `agent-protocol`, `development`) tied to live code; CHANGELOG with accurate Unreleased section; bilingual README sunset.

### Modified Capabilities

None — `openspec/specs/` is empty pre-change, so there is no spec-level delta. Behavior changes vs. live code (Pydantic v2 substitution, ReAct loop split, ToolDefinition consolidation, retry off-by-one fix) are documented inside each new spec's requirements and exhaustively proven via regression tests on `trace.json` / `summary.json` / config-load fixtures.

## Impact

**Affected source code**
- `src/data_agent_baseline/agents/{react.py, model.py, prompt.py, runtime.py, rate_limit.py, __init__.py}` — split, consolidated, locked-down imports.
- `src/data_agent_baseline/tools/{registry.py, __init__.py}` — `ToolDefinition` consolidation; immutable façade.
- `src/data_agent_baseline/run/{runner.py, __init__.py}` — App factory boundary; subprocess + partial-trace logic preserved verbatim; `Queue.empty()` fix; run-id precision.
- `src/data_agent_baseline/{config.py, cli.py, __init__.py}` — Pydantic v2 migration; CLI imports updated.
- `src/data_agent_baseline/benchmark/{dataset.py, __init__.py}` — drop unused `iter_tasks` filters.
- `src/data_agent_baseline/workflow/` — deleted entirely.

**Affected tests**
- `tests/test_*.py` — updated to import from new module paths; new tests added for typed event kinds, atomic bucket writes, off-by-one fix, Pydantic byte-compat fixtures.
- `tests/helpers/` (new) — houses `ScriptedModelAdapter` / `ScriptedNativeModelAdapter`.
- `tests/__pycache__/*` — purged of all stale `.pyc`.

**Affected configuration & build**
- `pyproject.toml`: add `pyright`, `pytest-cov`, `pre-commit` to `[project.optional-dependencies].dev`; new `[tool.pyright]`, expanded `[tool.pytest.ini_options]` (`addopts`, `cov`); `pydantic>=2.10` added to `dependencies`.
- `uv.lock` — regenerated.
- `.pre-commit-config.yaml` — new file.
- `configs/delegate_experiment.yaml` — deleted.
- `configs/react_baseline.example.yaml` — unchanged in field names; refreshed inline comments to match Pydantic-validated types.

**Affected documentation**
- `docs/*` — entire 12-file set rewritten as the small honest set described above.
- `README.md` — trimmed to upstream-only.
- `README.zh.md` — deleted.
- `CHANGELOG.md` — Unreleased section rewritten.
- `openspec/specs/` — populated with 10 new spec files (one per capability above).
- `openspec/changes/refactor-architecture-and-prune/{proposal.md, design.md, specs/**, tasks.md}` — generated by subsequent OPSX phases.

**Wire-format / external contracts (UNCHANGED)**
- `dabench` CLI command names, options, exit semantics.
- `configs/react_baseline.example.yaml` field names, defaults, validation behavior (input side; outputs identical).
- `trace.json` keys + value shapes; `summary.json` keys + value shapes; `prediction.csv` format.
- Cross-process `ratelimit/bucket_<sha1>.json` schema.

**Dependencies (added)**
- `pydantic>=2.10` (runtime).
- `pyright`, `pytest-cov`, `pre-commit` (dev).

**Breaking changes**
- **BREAKING (internal API only):** `ScriptedModelAdapter` / `ScriptedNativeModelAdapter` move out of `data_agent_baseline.agents.model` to `tests.helpers.scripted_adapters`. No production caller; tests updated in same change.
- **BREAKING (docs/links):** `README.zh.md` deleted; any external bookmark to it 404s.
- **BREAKING (internal helper):** `DABenchPublicDataset.iter_tasks(task_ids=, difficulty=, difficulties=)` filter kwargs removed; `iter_tasks()` (no kwargs) preserved.
- All other public surfaces (CLI, YAML, wire formats) are non-breaking.
