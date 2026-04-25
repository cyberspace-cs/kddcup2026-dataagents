# task-runner Specification

## Purpose
TBD - created by archiving change refactor-architecture-and-prune. Update Purpose after archive.
## Requirements
### Requirement: Microsecond-precision run-id with single-segment directory name

Auto-generated `run_id` values SHALL use UTC microsecond precision in the format `%Y%m%dT%H%M%S%fZ`. The result MUST be a single-segment directory name (no `/`, no `..`). When a `run_id` is supplied explicitly, it MUST be validated as a single-segment directory name.

#### Scenario: Two adjacent run-ids are distinct
- **WHEN** `auto_run_id()` is invoked twice within the same millisecond on the same process
- **THEN** the two returned strings are not equal

#### Scenario: Explicit run-id with path separator is rejected
- **WHEN** `resolve_run_id("foo/bar")` is called
- **THEN** it raises `ValueError`

### Requirement: Per-task subprocess isolation with timeout

Each `run_task` invocation SHALL execute the agent inside a subprocess so that wall-clock timeouts can be enforced cleanly. On timeout, the subprocess SHALL be terminated then killed if necessary, and a partial trace SHALL be assembled from the events streamed before the timeout.

#### Scenario: Task that exceeds timeout produces a partial trace plus timeout marker
- **WHEN** a task is run with `task_timeout_seconds=1` against an adapter that sleeps for longer
- **THEN** the resulting `trace.json` contains the steps emitted before the timeout, plus a final step with `action="__error__"` and an error payload mentioning timeout

#### Scenario: Successful task within timeout writes a complete trace
- **WHEN** a task completes before timeout
- **THEN** the `trace.json` ends with the answer step and `state.answer` is populated

### Requirement: Partial-trace recovery preserves answer when one was emitted

If a subprocess fails after emitting an answer step, the recovered partial trace SHALL include that answer in `summary.json` (under `final_answer` or equivalent), even though the subprocess did not return cleanly.

#### Scenario: Partial trace recovers answer from last successful step
- **WHEN** a subprocess emits an answer step at index 3, then fails at index 4
- **THEN** the recovered `summary.json` contains the answer payload from step 3

### Requirement: Per-task output layout is deterministic

For each task, the runner SHALL write `trace.json` and `prediction.csv` under the per-task output directory. Per-run aggregate `summary.json` SHALL be written under the run directory.

#### Scenario: Task output directory contains both files
- **WHEN** a task `<id>` completes (success or failure)
- **THEN** the directory contains a `trace.json` file and a `prediction.csv` file (the latter may be empty for failed runs)

#### Scenario: Run summary is written exactly once
- **WHEN** `run_benchmark` finishes processing all tasks
- **THEN** `summary.json` is written exactly once at the run root and contains an entry per task

### Requirement: Key pinning is deterministic by task index

When `agent.api_keys` has length K, task at zero-based index `i` SHALL be pinned to `api_keys[i % K]` for its entire run.

#### Scenario: Task indices map to keys deterministically
- **WHEN** `api_keys=["k0","k1","k2"]` and tasks at indices 0..5 are run
- **THEN** the captured per-task pinned key list is `["k0","k1","k2","k0","k1","k2"]`

### Requirement: ThreadPoolExecutor parallelism honors `max_workers`

When `run.max_workers > 1`, tasks SHALL be executed in parallel via `ThreadPoolExecutor` with at most `max_workers` concurrent worker threads.

#### Scenario: max_workers caps concurrency
- **WHEN** `max_workers=2` and 5 tasks are dispatched
- **THEN** at most 2 tasks are observed running at any instant (verifiable via shared-counter probe)

### Requirement: Wire formats are byte-stable

`trace.json`, `summary.json`, and `prediction.csv` keys, value shapes, and float-formatting MUST match the pre-change baseline byte-for-byte for any fixed input + scripted adapter.

#### Scenario: Golden trace comparison passes
- **WHEN** the benchmark runner runs against a frozen task fixture and a `ScriptedNativeModelAdapter`
- **THEN** the produced `trace.json`, `summary.json`, and `prediction.csv` byte-equal their golden counterparts in `tests/fixtures/golden/`

### Requirement: App factory boundary

The runner SHALL obtain the dataset, registry, model adapter, agent config, and protocol selection through a `build_application(config: AppConfig) -> AgentApp` factory; it MUST NOT import concrete adapter classes (`OpenAIModelAdapter`, `NativeToolsOpenAIAdapter`, `RateLimitedAdapter`) or `create_default_tool_registry` directly. The factory SHALL be invoked inside the per-task subprocess child after key pinning, never in the parent.

#### Scenario: Runner imports stay decoupled
- **WHEN** static analysis is run on `src/data_agent_baseline/run/runner.py`
- **THEN** the file does not import `OpenAIModelAdapter`, `NativeToolsOpenAIAdapter`, `RateLimitedAdapter`, `ReActAgent`, or `create_default_tool_registry`

#### Scenario: AgentApp instances are not pickled
- **WHEN** the parent process dispatches a task to a subprocess
- **THEN** the parent passes only the `AppConfig` (and the pinned key index); the child constructs `AgentApp` itself

