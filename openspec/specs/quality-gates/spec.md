# quality-gates Specification

## Purpose
TBD - created by archiving change refactor-architecture-and-prune. Update Purpose after archive.
## Requirements
### Requirement: Ruff format + check pass cleanly

`uv run ruff format --check .` SHALL exit `0`, and `uv run ruff check .` SHALL exit `0`, on a clean checkout of the change branch.

#### Scenario: Ruff format check is clean
- **WHEN** `uv run ruff format --check .` is invoked at the repo root
- **THEN** the exit code is `0` and stdout reports no files would be reformatted

#### Scenario: Ruff lint is clean
- **WHEN** `uv run ruff check .` is invoked at the repo root
- **THEN** the exit code is `0` and stdout reports `All checks passed`

### Requirement: Pyright strict on `src/`, basic on `tests/`

The repository SHALL ship a `[tool.pyright]` configuration in `pyproject.toml` with `strict = ["src/data_agent_baseline"]` (or equivalent path) and the rest of the workspace at `typeCheckingMode = "basic"`. `uv run pyright` SHALL exit `0` on the change branch.

#### Scenario: Pyright is configured for strict src/ + basic tests/
- **WHEN** `pyproject.toml` is parsed for `[tool.pyright]`
- **THEN** the configuration includes a strict-mode entry covering `src/data_agent_baseline` and a basic-mode default for the rest

#### Scenario: Pyright run is clean
- **WHEN** `uv run pyright` is invoked at the repo root
- **THEN** the exit code is `0` (zero errors and zero strict-mode violations across the configured scopes)

### Requirement: Pytest with coverage and `-ra`

`[tool.pytest.ini_options]` SHALL set `addopts = "-ra --cov=data_agent_baseline --cov-report=term-missing --cov-fail-under=70"`. `uv run pytest` SHALL exit `0` on the change branch with at least 70% coverage.

#### Scenario: Pytest invocation with default args produces coverage report
- **WHEN** `uv run pytest` is invoked at the repo root
- **THEN** the run completes with exit `0`, and the terminal output includes a `term-missing` coverage report whose total line is at least 70%

#### Scenario: Coverage drop below 70 fails the gate
- **WHEN** the test suite is hypothetically run with coverage at 65%
- **THEN** the pytest exit code is non-zero and the failure message references `cov-fail-under`

### Requirement: Pre-commit chain `ruff format → ruff check --fix → pyright src`

The repository SHALL ship `.pre-commit-config.yaml` defining a chain that runs (in order) `ruff format`, `ruff check --fix`, then `pyright src`. The hook is opt-in (no automatic install).

#### Scenario: Pre-commit run on clean tree exits zero
- **WHEN** a contributor runs `pre-commit run --all-files` after `pre-commit install`
- **THEN** every hook exits `0` and no files are modified

#### Scenario: Pre-commit auto-fixes formatting on dirty tree
- **WHEN** a contributor commits a file with formatting issues
- **THEN** `ruff format` reformats the file; the commit is rejected so the contributor can re-stage and re-commit

### Requirement: Dev dependency declaration

`[project.optional-dependencies].dev` SHALL include `pyright`, `pytest-cov`, and `pre-commit`, in addition to the pre-existing dev dependencies.

#### Scenario: `uv sync --extra dev` installs the new tools
- **WHEN** a fresh checkout runs `uv sync --extra dev`
- **THEN** `uv run pyright --version`, `uv run pytest --co --cov-context=test`, and `uv run pre-commit --version` all exit `0`

