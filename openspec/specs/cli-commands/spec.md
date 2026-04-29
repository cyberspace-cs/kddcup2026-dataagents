# cli-commands Specification

## Purpose
TBD - created by archiving change refactor-architecture-and-prune. Update Purpose after archive.
## Requirements
### Requirement: `dabench` exposes exactly four subcommands

The Typer entrypoint registered as `dabench` SHALL expose exactly these four subcommands: `status`, `inspect-task`, `run-task`, `run-benchmark`. No additional subcommands MAY be registered in this change.

#### Scenario: Help output lists the four subcommands
- **WHEN** `dabench --help` is invoked
- **THEN** the output contains `status`, `inspect-task`, `run-task`, and `run-benchmark`, and no other subcommand entries

### Requirement: All commands accept `--config` and validate it on entry

Every subcommand SHALL accept `--config <path>` and SHALL load + validate the YAML at that path before doing any other work. A YAML parse error or schema violation MUST exit with non-zero status and a human-readable error message naming the offending field.

#### Scenario: Invalid YAML aborts with non-zero exit
- **WHEN** `dabench status --config malformed.yaml` is invoked against a malformed YAML file
- **THEN** the process exits with status 1 (or another non-zero code) and stderr names the offending field or parse error

### Requirement: `run-benchmark --limit N` caps task count

`run-benchmark` SHALL accept `--limit N` and process at most `N` tasks (in dataset natural order). When `--limit` is omitted, all tasks SHALL be processed.

#### Scenario: Limit of 2 processes exactly two tasks
- **WHEN** `run-benchmark --limit 2 --config <cfg>` runs against a dataset of 5 tasks
- **THEN** the resulting `summary.json` contains exactly 2 task entries (the first two in natural order)

### Requirement: `run-benchmark --range S-E` selects a task-id range

`run-benchmark` SHALL accept `--range S-E` and process only tasks whose `task_<n>` number falls in `[S, E]` (both endpoints inclusive). Either endpoint MAY be given as a bare integer (`5`) or as a full task id (`task_5`); the two forms MAY be mixed. The range MAY combine with `--limit`: filtering by range MUST happen before truncation by `--limit`. A range that selects zero tasks SHALL exit non-zero with a parameter error rather than starting an empty run. Inverted (`S > E`), zero, or non-numeric endpoints SHALL exit non-zero with a parameter error.

#### Scenario: Range 3-5 selects task_3..task_5
- **WHEN** `run-benchmark --range 3-5 --config <cfg>` runs against a dataset of 10 tasks
- **THEN** the resulting `summary.json` contains exactly 3 task entries (task_3, task_4, task_5)

#### Scenario: Range combined with limit truncates after filtering
- **WHEN** `run-benchmark --range 2-8 --limit 3 --config <cfg>` runs against a dataset of 10 tasks
- **THEN** the resulting `summary.json` contains exactly 3 task entries (task_2, task_3, task_4)

#### Scenario: Empty selection exits non-zero
- **WHEN** `run-benchmark --range 100-200 --config <cfg>` runs against a dataset of 10 tasks
- **THEN** the process exits non-zero with a message indicating zero tasks selected

### Requirement: Run-id resolution is deterministic given `--run-id`

When `--run-id <X>` is supplied, the runner SHALL use `X` verbatim (after single-segment validation). When omitted, an auto-generated microsecond-precision UTC id is used.

#### Scenario: Explicit run-id is used as the directory name
- **WHEN** `run-task <task> --run-id myrun --config <cfg>` is invoked
- **THEN** outputs are written under `<output_dir>/myrun/`

#### Scenario: Run-id with separator is rejected
- **WHEN** `--run-id "foo/bar"` is passed
- **THEN** the process exits non-zero with a message about a single-segment directory name

### Requirement: Rich progress output is informational, not parseable

The CLI SHALL render progress via `rich.progress` for human consumption. The rendered output is NOT a stable consumer contract — automation SHALL parse `summary.json` instead.

#### Scenario: summary.json is the canonical machine-readable output
- **WHEN** any non-trivial command (`run-task`, `run-benchmark`) completes
- **THEN** `summary.json` exists in the run directory and contains all per-task results in JSON form

### Requirement: Exit codes

`dabench <subcommand>` SHALL exit `0` on success and non-zero on failure. A missing config, missing task, or unrecoverable runtime error MUST produce non-zero exit.

#### Scenario: Successful run exits zero
- **WHEN** `run-task <existing-id> --config <valid-cfg>` completes
- **THEN** the process exits with status `0`

#### Scenario: Missing task exits non-zero
- **WHEN** `inspect-task <nonexistent-id> --config <valid-cfg>` is invoked
- **THEN** the process exits with non-zero status and stderr names the missing task id

