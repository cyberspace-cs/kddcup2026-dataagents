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
uv run dabench run-benchmark         --config <cfg> [--limit N]
```

**Do not** invoke via `uv run python -m data_agent_baseline.cli ...` — the Typer + module-runner interaction produces no output in this project.

## 3. Git commit messages: English only

All commit subjects and bodies must be written in **English**, even when chatting with the tool or reviewer in Chinese. Code comments and docstrings may remain in Chinese — this rule is commit-message specific.

**Format:** Conventional Commits (`type(scope): subject`).

**Do:**

```
docs(baseline): add Chinese annotations across all source files
chore(tooling): integrate ruff and apply first-pass format
fix(runner): prevent zombie subprocess on timeout
```

**Don't:**

```
docs(baseline): 给全量源码添加中文注释
chore(tooling): 集成 ruff 并应用首次格式化
```

Proper nouns that are intrinsically Chinese (e.g. teammate names 华/吴, file names like `docs/12-two-person-plan.md`) may stay as-is inside an otherwise-English message.

---

These three rules are also mirrored in the local AI memory store (`~/.claude/projects/.../memory/`) but are kept here for **team-wide visibility** and for non-Claude tools (Cursor, Copilot, human reviewers) that read repo-root conventions.
