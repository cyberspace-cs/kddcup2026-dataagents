## 1. Characterization & Golden Fixtures (lock-byte-format-before-refactor)

- [x] 1.1 Create `tests/fixtures/golden/` with frozen `trace.json`, `summary.json`, `prediction.csv` produced by the pre-change agent against a deterministic `ScriptedNativeModelAdapter` script + a fixed `PublicTask` fixture
- [x] 1.2 Create `tests/fixtures/configs/valid/` with at least 3 input YAML files plus their `expected.normalized.json` (frozen `dataclasses.asdict` JSON dump from the pre-change loader)
- [x] 1.3 Create `tests/fixtures/configs/invalid/` with input YAML files paired with `expected_error_message` text fixtures for every legacy ValueError raise site in `config.py`
- [x] 1.4 Add `tests/test_golden_trace.py` that runs the agent end-to-end against the golden fixture and asserts byte-equality of `trace.json`, `summary.json`, `prediction.csv`
- [x] 1.5 Add `tests/test_config_byte_compat.py` that loads each `tests/fixtures/configs/valid/<n>.yaml` and asserts the normalized JSON dump byte-equals `expected.normalized.json`
- [x] 1.6 Add `tests/test_config_invalid_messages.py` that loads each `tests/fixtures/configs/invalid/<n>.yaml`, captures the `ValueError` message, and asserts it byte-equals `expected_error_message`
- [x] 1.7 Run the new fixture suite against the current `master` baseline to confirm goldens were captured correctly (this run must pass before any refactor begins)

## 2. Bug fixes (small, isolated, behavior-preserving)

- [x] 2.1 Fix `react.py:527` empty-tool retry guard: change `>` to `>=` so `max_empty_tool_call_retries=N` halts on the Nth empty turn
- [x] 2.2 Update `tests/test_react_native.py` empty-tool tests (currently lines 605-626) to expect failure on the Nth empty turn (matches the new contract)
- [x] 2.3 Fix `agents/rate_limit.py:144` `_save`: write to a tempfile in `state_path.parent`, then `os.replace(tmp, state_path)` for atomic writes
- [x] 2.4 Add `tests/test_rate_limit_atomic_save.py` that fuzzes concurrent reader/writer threads and asserts every observed read parses as valid JSON
- [x] 2.5 Fix `tools/python_exec.py:198` `Queue.empty()` usage: `process.join(timeout)` first, then `try queue.get_nowait() except queue.Empty:` and never call `queue.empty()`
- [x] 2.6 Fix `runner.py:78` second-precision run-id: change strftime format to `%Y%m%dT%H%M%S%fZ` for microsecond precision
- [x] 2.7 Add `tests/test_runner_run_id_precision.py` asserting two adjacent `auto_run_id()` calls within 1 ms produce different strings

## 3. Pydantic v2 config migration (behind compatibility wrapper)

- [x] 3.1 Add `pydantic>=2.10` to `[project.dependencies]` in `pyproject.toml`; run `uv lock --upgrade`
- [x] 3.2 Create `src/data_agent_baseline/config_models.py` (or extend `config.py`) with Pydantic v2 `BaseModel` classes mirroring `AppConfig`/`AgentConfig`/`RunConfig`/`RateLimitConfig`/`DatasetConfig` field-for-field
- [x] 3.3 Make `tool_protocol` `Literal["native","text"]` end-to-end; thread the literal through `AgentConfig` dataclass + Pydantic model
- [x] 3.4 Implement strict bool/int coercion via Pydantic strict types (no string→bool, no string→int)
- [x] 3.5 Implement custom validators that map Pydantic `ValidationError` → legacy `ValueError(...)` with byte-identical messages from the existing `config.py:179-293` raise sites
- [x] 3.6 Refactor `load_app_config(path)` to validate via Pydantic, then materialize the existing frozen dataclasses (do NOT return Pydantic model instances)
- [x] 3.7 Implement project-root anchoring for path fields inside the Pydantic validators (preserve `PROJECT_ROOT / relative_path` semantics)
- [x] 3.8 Implement key-pool ↔ rate-limit pairing rule as a model-level validator
- [x] 3.9 Run `tests/test_config_byte_compat.py` and `tests/test_config_invalid_messages.py` until both pass; iterate validators until messages match
- [x] 3.10 Remove the `# type: ignore[arg-type]` at `runner.py:348`; verify the call type-checks under pyright strict
- [x] 3.11 Document strict bool/int coercion as an intentional behavior change in `CHANGELOG.md` `[Unreleased]`

## 4. ToolDefinition consolidation + immutable ToolRegistry

- [x] 4.1 Define `ToolDefinition` dataclass in `src/data_agent_baseline/tools/registry.py` with `name`, `description`, `input_schema`, `json_schema`, `handler`, `is_terminal`
- [x] 4.2 Refactor `ToolRegistry` to expose `definitions: Mapping[str, ToolDefinition]` as a `MappingProxyType` (immutable post-construction)
- [x] 4.3 Update `create_default_tool_registry` to construct `ToolDefinition` instances; preserve registration order (8 tools as listed in spec)
- [x] 4.4 Set `is_terminal=True` only on the `answer` definition; all other tools `is_terminal=False`
- [x] 4.5 Preserve `to_openai_tools()` sorted-by-name output; add a regression test that two consecutive calls produce byte-identical JSON
- [x] 4.6 Update `NativeToolsOpenAIAdapter` schema cache to read from immutable definitions; verify cache stays valid across calls
- [x] 4.7 Update `tests/test_registry_json_schema.py` to construct via the new `ToolDefinition` API
- [x] 4.8 Add `tests/test_registry_immutable.py` that asserts mutation attempts raise `TypeError` post-construction

## 5. Split `react.py` into protocol adapter + dispatcher + recorder

- [x] 5.1 Create `src/data_agent_baseline/agents/protocol.py`: define `ProtocolTurn` and `adapt_model_response(response, *, protocol, turn_index, allowed_tool_names) -> ProtocolTurn`; move text-fence parsing (currently `react.py:108-172`), native normalization (currently `react.py:391-416`), `<think>` stripping, and reasoning-draft extraction (currently `react.py:486-535`)
- [x] 5.2 Create `src/data_agent_baseline/agents/dispatcher.py`: define `dispatch_tool_call(task, call, registry) -> tuple[ToolExecutionResult, bool]`; move tool execution + terminal short-circuit (currently `react.py:537-584`) and tool exception handling (currently `react.py:585-604`); enforce "terminal only after handler success" (D3)
- [x] 5.3 Create `src/data_agent_baseline/agents/recorder.py`: define internal event-kind `Literal["model_error","parse_error","empty_tool_calls","tool_error","observation"]` plus `record_step_event(state, event, *, step_index, turn_index, callback) -> StepRecord`; preserve serialization back to `action="__error__"` + `error_payload` for `trace.json` byte-stability (D2)
- [x] 5.4 Reduce `react.py` to the loop driver: turn iteration, model invocation, retry counters, terminal halt detection — no protocol parsing, no tool dispatch, no `StepRecord` construction inline
- [x] 5.5 Verify `tests/test_react_native.py`, `tests/test_react_text_fallback.py`, `tests/test_partial_trace.py` all pass without modification (other than the empty-tool retry test from 2.2)
- [x] 5.6 Verify `tests/test_golden_trace.py` (1.4) still passes byte-for-byte after the split
- [x] 5.7 Add `tests/test_protocol_adapter.py` covering: text fence parsing, native arg passthrough, `<think>` stripping, reasoning-draft extraction
- [x] 5.8 Add `tests/test_dispatcher.py` covering: success path, handler-raises path, terminal-only-after-success behavior, sandboxing rejection
- [x] 5.9 Add `tests/test_recorder.py` covering: each event kind round-trips through the recorder and emits the correct `trace.json` shape

## 6. App factory boundary in `runner.py`

- [ ] 6.1 Create `src/data_agent_baseline/run/app.py` defining `AgentApp` (with `dataset`, `registry`, `model_adapter`, `agent_config`, `protocol`, `new_agent(step_callback)`) and `build_application(config: AppConfig) -> AgentApp`
- [ ] 6.2 Move adapter/registry/agent construction from `runner.py:338-351` and `runner.py:653-660` into `build_application`
- [ ] 6.3 Update `runner.py` to import only `build_application` from `run.app`; remove direct imports of `OpenAIModelAdapter`, `NativeToolsOpenAIAdapter`, `RateLimitedAdapter`, `ReActAgent`, `ToolRegistry`, `create_default_tool_registry`
- [ ] 6.4 Ensure `build_application` is invoked **inside** the per-task subprocess child after key pinning at `runner.py:663-688`; do not pickle `AgentApp` across the process boundary
- [ ] 6.5 Preserve the test-injection escape hatch (`model=…, tools=…` kwargs at `runner.py:571-574, 591-601`) so tests can bypass `build_application` for in-process scripted runs
- [ ] 6.6 Verify subprocess timeout + partial-trace recovery (preserved verbatim per D11); `tests/test_partial_trace.py` must pass unchanged
- [ ] 6.7 Add `tests/test_app_factory.py` asserting `build_application(config)` returns an `AgentApp` with all five fields populated and that `new_agent()` returns a fresh `ReActAgent` per call

## 7. Test helper migration: `tests/helpers/scripted_adapters.py`

- [ ] 7.1 Create `tests/helpers/__init__.py` (empty file) to make `tests` an importable package and `helpers` a submodule
- [ ] 7.2 Create `tests/helpers/scripted_adapters.py` containing `ScriptedModelAdapter` and `ScriptedNativeModelAdapter` (copied verbatim from `src/data_agent_baseline/agents/model.py:439-479`)
- [ ] 7.3 Update test imports in `tests/test_react_native.py`, `tests/test_react_text_fallback.py`, `tests/test_partial_trace.py` from `data_agent_baseline.agents.model` to `tests.helpers.scripted_adapters`
- [ ] 7.4 Delete `ScriptedModelAdapter` and `ScriptedNativeModelAdapter` definitions (and their docstring) from `src/data_agent_baseline/agents/model.py`
- [ ] 7.5 Remove docstring references to `ScriptedModelAdapter` from `src/data_agent_baseline/run/runner.py:119,619` and `src/data_agent_baseline/agents/rate_limit.py:20`
- [ ] 7.6 Verify all tests still pass with the new import path

## 8. Package facades + dataset kwargs cleanup

- [ ] 8.1 Add explicit `__all__` lists to `src/data_agent_baseline/{agents,tools,run,benchmark}/__init__.py`; lock to current public re-exports per D13
- [ ] 8.2 Remove `task_ids=`, `difficulty=`, `difficulties=` parameters from `DABenchPublicDataset.iter_tasks` (zero callers verified)
- [ ] 8.3 Remove dead test helpers: `tests/test_runner_key_pinning.py:139 _fake_dataset`, `tests/test_rate_limit_adapter.py:381 _ = Any`, `tests/test_model_adapter_request.py:59 _make_response tool_calls` (confirmed unused)

## 9. Pyright strict fixes (177 errors → 0)

- [ ] 9.1 Add `[tool.pyright]` block to `pyproject.toml` configuring `strict = ["src/data_agent_baseline"]` and `typeCheckingMode = "basic"` for the rest
- [ ] 9.2 Define TypedDicts in `src/data_agent_baseline/agents/model.py` for `json.loads` return shapes used across the OpenAI SDK boundary (request/response payloads)
- [ ] 9.3 Add `cast(...)` annotations at the OpenAI SDK boundary (`response.choices[0].message`, `tool_calls`, `function`, `arguments`) to resolve `reportUnknownMemberType` clusters
- [ ] 9.4 Type the runner injection kwargs (`model: ModelAdapter | None`, `tools: ToolRegistry | None`, `step_callback: StepCallback | None`) precisely
- [ ] 9.5 Type `yaml.safe_load` payloads via TypedDict or a generic `dict[str, Any]` boundary cast
- [ ] 9.6 Narrow `json.loads` payloads in `react.py:133-142` and `runner.py:273-290` via TypedDict definitions
- [ ] 9.7 Annotate `sys.stdout`/`sys.stderr` close calls in `tools/python_exec.py:86-88` to handle `None`-able streams
- [ ] 9.8 Run `uv run pyright` and iterate until exit code `0` (zero strict errors)

## 10. Quality gate wiring (final)

- [ ] 10.1 Add `pyright`, `pytest-cov`, `pre-commit` to `[project.optional-dependencies].dev` in `pyproject.toml`
- [ ] 10.2 Add `[tool.pytest.ini_options]` `addopts = "-ra --cov=data_agent_baseline --cov-report=term-missing --cov-fail-under=70"`
- [ ] 10.3 Create `.pre-commit-config.yaml` with the chain: `ruff format` → `ruff check --fix` → `pyright src`
- [ ] 10.4 Run `uv run ruff format --check .`, `uv run ruff check .`, `uv run pyright`, and `uv run pytest` end-to-end; all exit `0`
- [ ] 10.5 Document the gate locally in `docs/development.md` (uv workflow, `pre-commit install` opt-in, gate exit-code contracts)

## 11. Documentation reset

- [ ] 11.1 Delete `docs/00-index.md`, `docs/01-requirements.md`, `docs/02-feasibility.md`, `docs/03-architecture.md`, `docs/04-detailed-design.md`, `docs/05-milestones-testing-risk.md`, `docs/06-interfaces.md`, `docs/07-prompts.md`, `docs/08-local-scorer.md`, `docs/09-deployment.md`, `docs/10-task-playbooks.md`, `docs/11-ops.md`
- [ ] 11.2 Write `docs/architecture.md` describing the live ReAct loop, native+text protocols, key-pool + rate limiter, subprocess isolation, with links to `openspec/specs/agent-execution-loop`, `openspec/specs/model-adapter`, `openspec/specs/tool-registry`, `openspec/specs/task-runner`, `openspec/specs/rate-limited-key-pool`
- [ ] 11.3 Write `docs/cli.md` documenting the four `dabench` subcommands (`status`, `inspect-task`, `run-task`, `run-benchmark`) and the YAML config schema, linking to `openspec/specs/cli-commands` and `openspec/specs/app-config`
- [ ] 11.4 Write `docs/agent-protocol.md` covering when to use native vs text, the 8-tool default inventory, and the `trace.json` / `summary.json` schemas
- [ ] 11.5 Write `docs/development.md` covering the uv workflow, `ruff + pyright + pytest` gates, `pre-commit install`, and contributing rules
- [ ] 11.6 Verify no doc under `docs/` references a non-existent path (no `src/dabench`, no `TaskOrchestrator`, no `DataProfiler`, no `OutputWriter`, no `read_pdf`, no `read_chart`)
- [ ] 11.7 Delete `README.zh.md`
- [ ] 11.8 Trim `README.md` to upstream-canonical sections (Quick Start, Dataset, Configuration, CLI, Tools, Outputs, Main Modules)
- [ ] 11.9 Rewrite `CHANGELOG.md` `[Unreleased]` section to capture: PR-1 baseline, PR-2 dual-key pool + cross-process rate limiter, native function calling, partial-trace recovery on subprocess timeout, this refactor

## 12. Orphan cleanup

- [ ] 12.1 Delete `src/data_agent_baseline/workflow/` (only stale `__pycache__` remains)
- [ ] 12.2 Delete `scripts/` (only stale `__pycache__`)
- [ ] 12.3 Delete `configs/delegate_experiment.yaml`
- [ ] 12.4 Purge stale `.pyc` files under `tests/__pycache__/` (the 13 entries: `test_delegate_*`, `test_workflow_*`, `test_qwen_agent_adapter`, `test_runner_timeout`, `test_sqlite_value_grounding`, `test_final_answer_shaper`, `test_react_native_with_delegate`, `test_prompt_native`)
- [ ] 12.5 Refresh `configs/react_baseline.example.yaml` inline comments to match the new Pydantic-validated types (no field name or default changes)

## 13. Final acceptance gates

- [ ] 13.1 `uv run ruff format --check .` exits `0`
- [ ] 13.2 `uv run ruff check .` exits `0`
- [ ] 13.3 `uv run pyright` exits `0` (strict on `src/`, basic on `tests/`)
- [ ] 13.4 `uv run pytest` exits `0` with coverage ≥ 70%
- [ ] 13.5 `uv run pre-commit run --all-files` exits `0`
- [ ] 13.6 `tests/test_golden_trace.py`, `tests/test_config_byte_compat.py`, `tests/test_config_invalid_messages.py` all pass byte-for-byte
- [ ] 13.7 `openspec validate refactor-architecture-and-prune --strict` exits `0`
- [ ] 13.8 Visual `git diff` review confirms wire formats (`trace.json`, `summary.json`, `prediction.csv`) are unchanged for the golden fixture
