# CLAUDE.md

Project conventions for AI coding assistants (Claude Code, Cursor, etc.) working in this repository. Human contributors should follow the same rules.

## 1. Python execution: always use `uv run`

This project is managed by **uv** (`uv.lock`, `pyproject.toml`). All Python invocations must go through `uv run` so the project's locked environment is used.

**Do:**

```bash
uv run python -c "..."
uv run pytest
uv run ruff format src/
```

**Don't:**

```bash
.venv/bin/python ...   # bypasses uv sync state
python3 ...            # may hit a different interpreter
python ...
pip install ...        # use `uv add` or `uv sync --extra dev` instead
```

Dev dependencies live under `[project.optional-dependencies].dev` — install them with `uv sync --extra dev`.

## 2. Benchmark CLI: drive via `dabench`, not `python -m ...`

The project exposes a Typer CLI through the `dabench` entrypoint (declared in `pyproject.toml` under `[project.scripts]`). Always invoke by the script name:

```bash
uv run dabench status        --config configs/react_baseline.example.yaml
uv run dabench inspect-task <task_id> --config <cfg>
uv run dabench run-task     <task_id> --config <cfg>
uv run dabench run-benchmark         --config <cfg> [--limit N] [--range S-E]
```

**Do not** invoke via `uv run python -m data_agent_baseline.cli ...` — the Typer + module-runner interaction produces no output in this project.

## 3. Shell tools: use `rg` and `fd` instead of `grep` and `find`

- `rg` (ripgrep) replaces `grep` — faster and respects `.gitignore` by default.
- `fd` replaces `find` — same benefits.

## 4. Code contribution rules — see [`CONTRIBUTING.md`](CONTRIBUTING.md)

That file is the canonical source for:

- **Git commit messages** — Conventional Commits + 50/72, English-only, one logical change per commit.
- **CHANGELOG maintenance** — what to record, what to skip, how to format entries, when releases promote `[Unreleased]`.

The rules apply identically to human contributors and AI assistants. If a discrepancy ever arises between this file and `CONTRIBUTING.md`, the latter wins.
