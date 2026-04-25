## Context

The repository carries three kinds of debt that compound into each other.

1. **Documentation drift** — `docs/00`–`docs/11` (3,800+ lines) describe modules that were deleted upstream (`workflow/` Plan-Execute, `DataProfiler`, `OutputWriter`, `read_pdf`, `read_chart`) and refer to the wrong package path (`src/dabench` instead of the live `src/data_agent_baseline`). `README.zh.md` mirrors stale internal team status. Contributors land on the docs and write code against modules that do not exist.
2. **Architectural debt** — `src/data_agent_baseline/agents/react.py` (621 LOC) interleaves protocol parsing (text + native), tool dispatch, terminal short-circuiting, step recording, retry accounting, and error event synthesis into one method. `AgentConfig.tool_protocol` is typed `str` despite a `Literal["native","text"]` runtime contract — this forces `# type: ignore[arg-type]` at `src/data_agent_baseline/run/runner.py:348` (the **only** type-ignore in `src/`). Manual YAML parsing in `config.py:152-327` reinvents validation that Pydantic v2 already provides. Four distinct correctness bugs are latent in the code (off-by-one retry guard, non-atomic state-file writes, unreliable cross-process `Queue.empty()`, second-precision run-id collisions).
3. **Process debt** — `pyproject.toml` declares `ruff` but no static type checker, no coverage threshold, and no pre-commit hook. The "native function-calling" branch landed without a way to verify byte-stability of `trace.json` / `summary.json` outputs.

Pre-change ground-truth confirmed by codex's run-time verification:
- 1 `# type: ignore` in `src/`, 6 in `tests/`.
- 134 tests pass; coverage is **74%** under `--cov=data_agent_baseline`.
- Pyright strict on `src/` reports **177 errors** (157 `reportUnknown*` from OpenAI SDK Unknowns + json.loads payloads, 5 missing parameter types, etc.). Pyright at default mode reports **11 errors**.
- `ruff check .` is currently clean.
- Nothing in `src/` or `tests/` imports through the package-root façade — every import uses subpackage paths like `from data_agent_baseline.agents.model import …`.
- `ScriptedModelAdapter` / `ScriptedNativeModelAdapter` have no production callers; only the docstring of `runner.py:119` references them by name.
- `iter_tasks(task_ids=, difficulty=, difficulties=)` has zero callers passing kwargs.

Stakeholders: future contributors (grad-student researchers + open-source community), the CI/CD owner (humans), and the public CLI/wire-format surface that downstream submission scripts depend on.

## Goals / Non-Goals

**Goals**

1. Eliminate doc drift: ship a small, accurate doc set tied 1-to-1 to live code.
2. Reshape the agent layer (ReAct loop, tool registry, configuration) so that internal boundaries match the existing wire-format contract — without changing that contract.
3. Replace manual YAML validation with Pydantic v2 and let `Literal` types propagate end-to-end.
4. Fix the four latent correctness bugs while preserving observable behavior under the existing test suite, plus the new characterization tests this change adds.
5. Establish enforceable quality gates (`ruff`, `pyright` strict on `src`, `pytest --cov`, `pre-commit`) so future drift is caught at edit time.
6. Populate `openspec/specs/` with canonical specs for every live capability (none exist today).

**Non-Goals**

- CI wiring (GitHub Actions, etc.). Out of scope for this change; pre-commit is the local gate.
- Adding new capabilities (no new tools, no new CLI subcommands, no new model adapters). The change is **all** internal.
- Changing wire-format outputs: `trace.json` keys + value shapes, `summary.json` keys + value shapes, `prediction.csv` format, cross-process `bucket_<sha1>.json` schema, and `dabench` CLI command names/options remain unchanged.
- Re-litigating PR-1 (baseline) or PR-2 (dual-key pool) decisions — those are accepted as the live contract.
- Bilingual docs. `README.zh.md` is deleted; the project goes English-only.

## Decisions

### D1. Split `react.py` into protocol adapter + tool dispatcher + step recorder

**Choice**: Three cooperating units inside `src/data_agent_baseline/agents/`:
- `protocol.py` — `adapt_model_response(response, *, protocol, turn_index, allowed_tool_names) -> ProtocolTurn` where `ProtocolTurn.calls` is `tuple[ToolCall, ...]`. Owns text-fence parsing, native `tool_calls` normalization, Qwen3 `<think>` stripping, reasoning-draft extraction.
- `dispatcher.py` — `dispatch_tool_call(task, call, registry) -> tuple[ToolExecutionResult, bool]`. Owns terminal-flag resolution (from registry metadata, but only **after** successful handler return — see D3), exception → typed-event mapping, `task.context_dir` sandboxing.
- `recorder.py` — `record_step_event(state, event, *, step_index, turn_index, callback) -> StepRecord`. Owns `StepRecord` construction, internal typed event kinds (`model_error`, `parse_error`, `empty_tool_calls`, `tool_error`, `observation`), and `step_callback` invocation.

The remaining `react.py` keeps only the loop driver: turn iteration, retry counters, model invocation, terminal halt detection.

**Rationale**: The boundaries already exist as comment-delimited phases inside `ReActAgent.run`. Naming them at the module level removes the `if protocol == "native"` branches that pollute the loop, and lets each unit be unit-tested without spinning up the full agent. Codex's risk inventory (native replay first-step assumption, usage/reasoning lead-only attachment, terminal-then-skip semantics, text protocol synthetic-id handling) becomes a code-level invariant of `protocol.py` and `dispatcher.py` rather than a property of the 621-line `run` method.

**Alternatives**: Keep one file, just add helper functions. Rejected — the load-bearing risks above are easy to break silently when everything shares a scope; physical separation makes the contract testable.

### D2. Typed event kinds are **internal only**; `trace.json` wire format is byte-stable

**Choice**: Replace the `__error__` sentinel string in `StepRecord.action` with a Python `Literal` type for internal flow control. Serialize back to the existing `trace.json` schema (`action="__error__"`, `error_payload={...}`) at write time.

**Rationale**: The proposal makes two promises that look like a contract conflict: "typed event kinds replace `__error__`" (proposal §What Changes/Architecture, line 10) and "trace.json keys + value shapes unchanged" (proposal §Impact/Wire-format). Codex flagged this. The resolution is that typed events are an internal refactor — they catch a category of "did I forget an error path?" type bugs at edit time without altering what consumers see.

**Alternatives**: Add an `event_kind` field to `trace.json`. Rejected — wire-format break, no consumer benefit.

### D3. Terminal flag is metadata, but only fires after the handler returns successfully

**Choice**: `ToolDefinition.is_terminal` is set at registry-construction time (only `answer` carries `is_terminal=True`). The dispatcher reads it from the registry, but **only short-circuits the loop after the handler call returns without raising and the result is `ok=True`**. Invalid `answer` payloads continue to raise inside `_resolve_answer_args` (`registry.py:157-175`), and the agent retries on the next turn exactly as today.

**Rationale**: Identity-based termination ("call name is `answer` → halt") would silently break the existing retry path for malformed answers. Codex caught this. The mental model is: terminal-ness describes *intent*; success describes *outcome*; both must hold to halt.

### D4. `ToolRegistry` is immutable post-construction; schema cache is invariant

**Choice**: Replace mutable `ToolRegistry.specs` / `ToolRegistry.handlers` dicts with a single immutable mapping from name to `ToolDefinition`, exposed as `types.MappingProxyType` (or equivalent). `to_openai_tools()` continues to return tools sorted by name (preserves prompt cache stability).

**Rationale**: `NativeToolsOpenAIAdapter` caches `tools.to_openai_tools()` once at construction (`model.py:332-333`). Mutability after that point is a latent bug waiting to happen. No production code mutates the registry today (verified by grep), so freezing it has no migration cost. The one test that constructs a custom registry (`tests/test_registry_json_schema.py:101-126`) uses the constructor only.

### D5. Pydantic v2 migration with a compatibility wrapper around `load_app_config`

**Choice**: Replace the manual validators in `config.py:152-327` with Pydantic v2 `BaseModel` subclasses. `load_app_config(path)` continues to return frozen dataclasses (not Pydantic model instances), and continues to raise `ValueError` with the same string messages it does today. Implementation: validate via Pydantic, catch `ValidationError`, map `loc` + validator code back to the legacy `ValueError(...)` message verbatim, then materialize the frozen dataclasses we already export.

**Rationale**: Codex enumerated all 17 ValueError raise sites with their exact messages (config.py lines 179-293). Existing tests (`test_config_rate_limit.py`, `test_config_new_fields.py`) match on message substrings. Returning Pydantic model instances would break any caller that uses `dataclasses.asdict` or relies on `__slots__`. Wrapper preserves the public contract while moving the validation engine.

### D6. Strict bool/int coercion is an intentional behavior tightening

**Choice (user-resolved)**: `allow_parallel_tool_calls: "false"` (quoted string) **fails** under the new loader. Numeric strings for int fields fail. Pydantic v2's strict mode is the authority. The CHANGELOG `[Unreleased]` documents this as an intentional behavior change.

**Rationale**: Current `bool('false') == True` is a foot-gun: a user writing quoted-`"false"` gets the opposite of what they meant, silently. The new behavior raises clearly. Configs that worked before continue to work as long as they use real YAML scalars (unquoted `false`, unquoted integers).

### D7. Empty-tool-call retry contract: N empty turns total (use `>=`)

**Choice (user-resolved)**: `max_empty_tool_call_retries=N` means "the Nth empty turn is the failure turn." Guard becomes `if empty_tool_call_retries >= self.config.max_empty_tool_call_retries: fail`. Tests at `test_react_native.py:605-626` are rewritten to expect failure on turn N (currently they expect failure on turn N+1).

**Rationale**: The proposal explicitly calls the current `>` behavior a bug. The cleaner mental model is "budget is N total empty turns; on the Nth, we halt." Changes the test fixture, not the public contract (this knob is an internal safeguard, not a CLI flag).

### D8. Microsecond-precision run-id

**Choice**: `run_id` default format becomes `%Y%m%dT%H%M%S%fZ` (UTC microsecond), e.g. `20260424T123456789012Z`. Single directory name; sortable; deterministic for one process clock.

**Rationale**: Microsecond UTC keeps the "single-segment directory name" invariant of `resolve_run_id` (`runner.py:81-95`) while removing the second-resolution collision. A counter suffix would require probing the output dir and racing with `mkdir(exist_ok=False)` retry — more code, more race, no benefit.

### D9. Atomic state-file writes for `FileTokenBucket`

**Choice**: `_save` writes to a tempfile in `self._state_path.parent` (same filesystem as the target — guaranteed because `state_dir` is project-anchored), then `os.replace(tmp, state_path)`.

**Rationale**: `os.replace` is atomic on POSIX and Windows when source and destination are on the same volume. The tempfile-in-parent-dir pattern is the only correct way to guarantee that.

### D10. `python_exec` queue draining order

**Choice**: After `process.join(timeout)`, attempt `queue.get_nowait()` and catch `queue.Empty`. **Do not** call `queue.empty()` (unreliable across processes).

**Rationale**: `multiprocessing.Queue.empty()` is documented as approximate due to background feeder threads. `get_nowait()` is the only authoritative way to check.

### D11. `App` factory boundary in `runner.py`

**Choice**: Introduce `build_application(config: AppConfig) -> AgentApp` in a new `src/data_agent_baseline/run/app.py`. `AgentApp` exposes:
- `dataset: DABenchPublicDataset`
- `registry: ToolRegistry`
- `model_adapter: ModelAdapter`
- `agent_config: ReActAgentConfig`
- `protocol: ToolProtocol`
- `new_agent(step_callback: StepCallback | None = None) -> ReActAgent`

`runner.py` stops importing `OpenAIModelAdapter`, `NativeToolsOpenAIAdapter`, `RateLimitedAdapter`, `ReActAgent`, `ToolRegistry`, `create_default_tool_registry` directly. The factory is **called inside the subprocess child** (after key pinning at `runner.py:663-688`), never in the parent — `AgentApp` instances are not picklable.

**Rationale**: The current direct imports make `runner.py` the de-facto wiring layer. Factoring the wiring out lets the subprocess path, the in-process test path (model+tools injection at `runner.py:571-574`), and the parallel batch path (`runner.py:653-660`) share one construction recipe. Codex's preserved-verbatim list (queue-before-join at 421-423, child callback at 369-390, parent poll at 426-442, terminate-then-kill at 460-467, partial-trace merge at 497-524) stays untouched.

### D12. Move `ScriptedModelAdapter` / `ScriptedNativeModelAdapter` to `tests/helpers/scripted_adapters.py`

**Choice**: Create `tests/helpers/__init__.py` (empty) + `tests/helpers/scripted_adapters.py`. Update tests to import `from tests.helpers.scripted_adapters import …`. Delete the classes from `src/data_agent_baseline/agents/model.py` and remove the docstring references that point to them as `data_agent_baseline.agents.model.ScriptedModelAdapter`.

**Rationale**: Verified zero production callers. Test fixtures don't belong in a runtime package — pyright strict on `src/` would have to type their kwargs as Unknowns from `ModelToolCall` test factories, polluting the strict gate. With `__init__.py` present pytest still collects the rest of `tests/` correctly, and `tests` becomes an importable package for the helper modules.

### D13. Package-root re-export façade: keep, but `__all__`-locked

**Choice**: Keep `src/data_agent_baseline/{agents,tools,run,benchmark}/__init__.py` as the supported public façade with explicit `__all__` lists. Do **not** prune to empty.

**Rationale**: Verified that no current importer uses the façade — every import goes through subpackage paths. So pruning is technically free. But future external scripts (e.g., a researcher writing a one-off comparison harness) will reach for `from data_agent_baseline.agents import ReActAgent` and the façade is cheap to maintain. Locking it down with `__all__` prevents accidental import-leak from the grad-student "import everything from the top" pattern.

### D14. Pyright strict on `src/`, basic on `tests/` (user-resolved)

**Choice**: `[tool.pyright]` configures `strict = ["src/data_agent_baseline"]` and `typeCheckingMode = "basic"` for the rest. The 177 strict errors are addressed before merging this change. Most are mechanical — TypedDicts for `json.loads` payloads, `cast(ChatCompletionMessage, response.choices[0].message)` for the OpenAI SDK boundary, removing implicit `Any` parameters in `runner.py` injection points.

**Rationale**: Codex measured 177 errors — high but mostly in two clusters: OpenAI SDK Unknowns and `json.loads` payload narrowing. Both can be addressed with TypedDict definitions in a single PR-sized commit. Long-term, strict mode catches the exact class of bug that produced the `tool_protocol` ignore in the first place.

### D15. README.zh.md is deleted; project goes English-only (user-resolved)

**Choice**: Delete `README.zh.md`. Trim `README.md` to upstream-only sections. Document the deletion in CHANGELOG.

**Rationale**: User chose alignment with the upstream baseline over maintaining a dual-language doc set. Maintenance friction (every README edit done twice) outweighs the inclusivity benefit for a research codebase whose contributors are expected to read English in practice (paper writeups, GitHub issues, model APIs).

### D16. Coverage fail-under = 70

**Choice**: `pytest --cov=data_agent_baseline --cov-fail-under=70`. Current coverage is 74% — leaves a 4-point margin so adding a new test that incidentally drops coverage doesn't break the gate.

**Rationale**: Codex measured. 75 would already fail (CLI is 0%, `dataset.py` is 29%, `sqlite.py` is 25%, `python_exec.py` is 42%). Raising the gate is a follow-up after CLI snapshot tests + dataset/sqlite/python_exec hardening.

### D17. Capability cardinality: 10

**Choice**: Keep all 10 capabilities listed in the proposal. `quality-gates` and `project-documentation` stay as their own specs.

**Rationale**: Both have testable invariants (`ruff` exits 0; `pyright src` strict exits 0; doc set is exactly four files; no doc references a non-existent module). Spec-as-contract works for them even though they're "process-y" — the spec is the gate the build enforces.

## Risks / Trade-offs

- **Risk: Pydantic byte-compatibility regression** → Mitigation: Add a fixture suite (`tests/fixtures/configs/`) with `input.yaml` + `expected.normalized.json` (frozen dataclass dump) + for invalid inputs `expected_error_message`. Run it under both old (pre-refactor branch) and new loader to validate parity for valid inputs and intentional-tightening behavior for the bool/int edge cases.
- **Risk: Trace.json byte drift from refactor** → Mitigation: Add a regression test that runs the agent end-to-end with `ScriptedNativeModelAdapter` and asserts byte-equality of `trace.json` against a frozen golden file. Add the same for `summary.json` and `prediction.csv`.
- **Risk: Subprocess timeout / partial-trace breakage** → Mitigation: Keep the existing `tests/test_partial_trace.py` green; add a new test that timeouts at step 3 of a 10-step plan and asserts the partial trace contains exactly steps 1-3 + a `__error__` entry with timeout payload.
- **Risk: Pyright strict scope creep** → Mitigation: Land the 177 fixes in this same change. Do not introduce strict mode "later" — the gate must be active at merge time or it never lands.
- **Risk: Pre-commit opt-in means contributors skip it** → Accepted as design constraint: CI is out-of-scope for this change. Document the gate in `docs/development.md`. Future change can wire CI.
- **Trade-off: Strict bool/int coercion breaks any external YAML using quoted bools** → Accepted (intentional tightening; user-resolved). Documented in CHANGELOG `[Unreleased]` under "Behavior changes."
- **Trade-off: README.zh.md deletion alienates Chinese-speaking participants** → Accepted (user-resolved). Mitigated by keeping doc text simple and link-rich for machine translation.

## Migration Plan

This is an internal refactor with no public-facing surface change (per non-goal #3). No deployment-side migration. Contributor-side migration:

1. Run `uv sync --extra dev` to pull `pydantic`, `pyright`, `pytest-cov`, `pre-commit`.
2. (Optional) `pre-commit install` to enable the local hook.
3. Update any local fork's `import` statements if they touched `data_agent_baseline.agents.model.ScriptedModelAdapter` (the only internal-API break — fix is `from tests.helpers.scripted_adapters import ScriptedModelAdapter`).
4. Update any local YAML that used quoted booleans (`"false"`, `"0"`, etc.) to plain YAML scalars.

Rollback: `git revert` the merge commit. The change is one PR, one merge.

## Open Questions

None at the proposal-stage spec. Three classes of open question are deferred to implementation:

- **Pyright strict fix style**: TypedDict per OpenAI SDK return shape, vs. one big `cast(Any, …)` at the boundary. To be decided per-call-site during implementation.
- **Test helper layout**: Whether `tests/helpers/` grows beyond `scripted_adapters.py` in this change or stays minimal. Default is minimal; grow only if a test needs a shared fixture across files.
- **Coverage fail-under bump**: After the dataset/sqlite/python_exec coverage hardening (out of scope here), the gate can rise from 70 → 80. Tracked as a follow-up, not gated by this change.
