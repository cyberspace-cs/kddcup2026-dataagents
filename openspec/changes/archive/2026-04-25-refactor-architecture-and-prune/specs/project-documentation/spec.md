## ADDED Requirements

### Requirement: Concise English-only doc set

`docs/` SHALL contain exactly four Markdown files, each tied to live code: `architecture.md`, `cli.md`, `agent-protocol.md`, `development.md`. All other prior `docs/00`–`docs/11` files SHALL be deleted.

#### Scenario: docs/ is exactly four files
- **WHEN** `ls docs/*.md` is run after this change
- **THEN** the result is exactly `docs/agent-protocol.md`, `docs/architecture.md`, `docs/cli.md`, and `docs/development.md`

#### Scenario: Deleted docs are gone
- **WHEN** any of `docs/00-index.md`, `docs/01-requirements.md`, `docs/02-feasibility.md`, `docs/03-architecture.md`, `docs/04-detailed-design.md`, `docs/05-milestones-testing-risk.md`, `docs/06-interfaces.md`, `docs/07-prompts.md`, `docs/08-local-scorer.md`, `docs/09-deployment.md`, `docs/10-task-playbooks.md`, `docs/11-ops.md` is checked
- **THEN** the file does not exist

### Requirement: Docs reference only live modules

No file under `docs/` SHALL reference a module path or class name that does not exist in `src/data_agent_baseline/` (no `src/dabench`, no `TaskOrchestrator`, no `DataProfiler`, no `OutputWriter`, no `read_pdf`, no `read_chart`).

#### Scenario: Module-path references resolve
- **WHEN** every Markdown link or fenced reference in `docs/` is grep-collected
- **THEN** each referenced module path or class corresponds to an existing entity in `src/data_agent_baseline/`

### Requirement: README is upstream-canonical, English-only

`README.md` SHALL contain only upstream-canonical sections (Quick Start, Dataset, Configuration, CLI, Tools, Outputs, Main Modules — or equivalent subset). `README.zh.md` SHALL be deleted.

#### Scenario: README.zh.md does not exist
- **WHEN** the repository root is inspected after this change
- **THEN** `README.zh.md` does not exist

#### Scenario: README.md contains only upstream sections
- **WHEN** the headings of `README.md` are extracted
- **THEN** the heading set is a subset of the upstream-canonical section list

### Requirement: CHANGELOG `[Unreleased]` reflects actual recent work

`CHANGELOG.md` `[Unreleased]` section SHALL list the actual changes that have landed since the last release: PR-1 baseline, PR-2 dual-key pool + cross-process rate limiter, native function calling support, partial-trace recovery on subprocess timeout, and this refactor.

#### Scenario: Unreleased section is non-empty and accurate
- **WHEN** `CHANGELOG.md` is parsed
- **THEN** the `[Unreleased]` section contains entries that name PR-1 baseline, PR-2 key pool, native function calling, partial-trace recovery, and the refactor

### Requirement: Capability docs trace to specs

`docs/architecture.md` and `docs/agent-protocol.md` SHALL link to (or summarize) the corresponding `openspec/specs/<capability>/spec.md` for each capability they describe.

#### Scenario: Architecture doc references the canonical specs
- **WHEN** `docs/architecture.md` is read
- **THEN** it references at least the capabilities `agent-execution-loop`, `model-adapter`, `tool-registry`, `task-runner`, `rate-limited-key-pool`
