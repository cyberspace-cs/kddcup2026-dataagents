# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- **PR-1 baseline.** ReAct loop with text-fenced JSON action protocol; eight default
  tools (`inspect_files`, `read_csv`, `read_json`, `read_doc`, `inspect_sqlite_schema`,
  `execute_context_sql`, `execute_python`, `answer`); subprocess timeout isolation.
  `inspect_files` is the zero-decision first-turn entry point — it returns columns,
  dtypes, row counts, and sqlite/json schemas for every file under `context/` in one
  call, replacing the previous `list_context` directory walker.
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
- **`run-benchmark --range S-E` task selector.** Filters by `task_<n>` number,
  e.g. `--range 5-10` or `--range task_5-task_10` (both endpoints inclusive,
  numeric / `task_<n>` forms mixable). Composes with `--limit`: range filters
  first, then `--limit` truncates. Empty selection or inverted range exits
  non-zero with a parameter error.

### Changed
- **Configuration loader migrated to Pydantic v2.** `load_app_config` now validates
  the YAML through `AppConfigModel`, then materializes the existing frozen dataclasses.
  Public surface (`AppConfig` return type, `ValueError` messages) is byte-compatible.
- **`AgentConfig.tool_protocol` is now `Literal["native", "text"]`** end-to-end. The
  single `# type: ignore[arg-type]` in `runner.py` is removed.
- **Tool input validation now uses Pydantic v2 models.** Each default tool gets a
  per-tool `BaseModel` (`extra="forbid"`); missing-required, wrong-type, bound, and
  unknown-field errors surface as structured `ValueError("<tool>: <field> <reason>")`
  from `ToolRegistry.execute`. `to_openai_tools()` byte output is locked by
  `tests/fixtures/tool_schemas/*.json` and the in-module `_normalize_for_vllm`
  helper, so no wire-format change ships.

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
- **`read_json` becomes inspect-style.** The `max_chars` parameter is removed
  (`extra='forbid'` rejects it). Output is now a structured `{kind, length |
  key_count, head | keys + value_preview, truncated}` shape with a 5MB
  filesystem guard, parallel to `read_csv`'s SAMPLE preview contract. Top-level
  `array` length and `object` key_count are preserved even under truncation.
  Within `value_preview`, nested arrays additionally expose `first_item` (the
  first element verbatim if ≤ 2KB, otherwise a shape stub) so that wrapper
  shapes like `{table, records: [...]}` still surface a record's schema by
  example without forcing an extra `execute_python` round-trip. Callers needing
  full nested extraction, filtering, or deep traversal must still use
  `execute_python` with `json.load`. Empirical motivation: 0/37 calls in the
  reference build supplied a custom `max_chars`, while 73% of `read_json`
  invocations were already followed by same-file `execute_python` (100% of
  those were cross-file joins). The `first_item` refinement responds to a
  follow-up trace (run 20260429T181710702941Z) where 12/13 read_json calls
  hit the `{table, records}` wrapper and the original depth-1 design hid every
  record field.
- **`read_doc` keyword mode upgraded from "first-match window" to "all matching
  paragraphs".** When `keyword` is provided, `read_doc` now returns every
  paragraph (blank-line-separated) containing the keyword across the whole file
  — capped at 20 matches / 2KB per paragraph / 8KB total. `start_line` and
  `line_count` are ignored in keyword mode. Return shape changes:
  `keyword_match` becomes `{keyword, found, match_count, matches, matches_truncated}`
  (replacing the prior `{keyword, found, line}` + co-located `preview` window);
  `preview`, `returned_range`, and `has_more` are `None` in keyword mode.
  Empirical motivation: in the reference build, `keyword` was used in 1/77
  `read_doc` calls (1.3%) while 88% of calls hit `has_more=True`; task_46 paged
  the 1087-line `superhero.md` 7 times sequentially and still failed with
  `max_steps`. The new contract collapses such serial pagers into a single
  call. Paging mode (no keyword) is unchanged.

### Breaking (internal API only)
- `ScriptedModelAdapter` and `ScriptedNativeModelAdapter` moved from
  `data_agent_baseline.agents.model` to `tests.helpers.scripted_adapters`. No
  production caller; tests updated.
