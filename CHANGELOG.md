# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- **PR-1 baseline.** ReAct loop with text-fenced JSON action protocol; eight default
  tools (`list_context`, `read_csv`, `read_json`, `read_doc`, `inspect_sqlite_schema`,
  `execute_context_sql`, `execute_python`, `answer`); subprocess timeout isolation.
- **PR-2 dual-key pool + cross-process rate limiter.** `agent.api_keys` enables
  per-task key pinning by `task_index % len(api_keys)`. `agent.rate_limit` wires a
  per-key `FileTokenBucket` (RPM + TPM, `filelock`-protected, atomic state writes,
  retry-after parsing, exponential backoff with jitter).
- **Native function calling.** `agent.tool_protocol = "native"` switches to OpenAI
  native `tool_calls`; legacy `text` protocol preserved as fallback. Qwen3 `<think>`
  blocks are stripped before parsing.
- **Partial-trace recovery on subprocess timeout.** The agent's `step_callback`
  incrementally writes `trace.json`; on subprocess timeout the parent reads back the
  partial trace and synthesizes a terminal `__error__` step with a `timeout` payload.
- **Refactor: agent layer split.** `agents/react.py` decomposed into
  `protocol.py` (text/native → unified `ToolCall`), `dispatcher.py` (tool execution +
  terminal short-circuit on success), and `recorder.py` (typed event kinds with
  byte-stable `trace.json` serialization). `ToolDefinition` consolidates the previous
  `ToolSpec` + handler split; `ToolRegistry` is immutable post-construction.
  `run/app.py` introduces `build_application(config) -> AgentApp`; the runner stops
  importing concrete adapter classes directly.
- **Quality gates.** Pyright strict on `src/data_agent_baseline` (177 → 0 errors).
  `pyright`, `pytest-cov`, `pre-commit` added to dev dependencies.
  `pytest --cov-fail-under=70`. `.pre-commit-config.yaml` chains
  `ruff format` → `ruff check --fix` → `pyright src` (opt-in).
- **Documentation reset.** New concise English-only docs under `docs/`:
  `architecture.md`, `cli.md`, `agent-protocol.md`, `development.md`. Each
  capability has a canonical spec under `openspec/specs/<name>/spec.md`.

### Changed
- **Configuration loader migrated to Pydantic v2.** `load_app_config` now validates
  the YAML through `AppConfigModel`, then materializes the existing frozen dataclasses.
  Public surface (`AppConfig` return type, `ValueError` messages) is byte-compatible.
- **`AgentConfig.tool_protocol` is now `Literal["native", "text"]`** end-to-end. The
  single `# type: ignore[arg-type]` in `runner.py` is removed.

### Fixed
- **`react.py` empty-tool retry off-by-one.** Guard now uses `>=` so
  `max_empty_tool_call_retries=N` halts on the Nth empty turn, matching the
  documented contract.
- **`agents/rate_limit.py` non-atomic state writes.** `_save` writes to a tempfile
  in `state_path.parent` then `os.replace`, eliminating the torn-write window.
- **`tools/python_exec.py` unreliable cross-process queue check.** Replaced
  `Queue.empty()` with `process.join(timeout)` followed by `get_nowait()` /
  `queue.Empty`.
- **`runner.py` second-resolution `run_id` collisions.** Default format upgraded to
  `%Y%m%dT%H%M%S%fZ` (microsecond UTC).

### Removed
- **`README.zh.md`.** The project goes English-only.
- **`src/data_agent_baseline/workflow/`** (only stale `__pycache__` remained).
- **`scripts/`** (only stale `__pycache__`).
- **`configs/delegate_experiment.yaml`** (referenced fields that no longer exist).
- **`docs/00`–`docs/11`** (legacy plan docs describing deleted modules).
- **`DABenchPublicDataset.iter_tasks(task_ids=, difficulty=, difficulties=)`** filter
  kwargs (zero callers).

### Behavior changes (intentional tightening)
- **Strict bool/int coercion.** YAML quoted booleans/integers (`"false"`, `"0"`,
  `"16384"`) are now rejected. Use unquoted YAML scalars (`false`, `0`, `16384`).
  The previous behavior silently flipped quoted bools (`bool("false") == True`).

### Breaking (internal API only)
- `ScriptedModelAdapter` and `ScriptedNativeModelAdapter` moved from
  `data_agent_baseline.agents.model` to `tests.helpers.scripted_adapters`. No
  production caller; tests updated.
