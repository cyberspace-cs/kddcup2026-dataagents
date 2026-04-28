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

## 3. Git commit messages

Follow **Conventional Commits + 50/72**, in **English**.

**Format**

```
<type>(<scope>): <subject>          # imperative, ≤50 chars (hard max 72), no period
                                    # ↑ blank line ↓ (only if a body exists)
<body wrapped at 72 chars>          # optional; only when "why" is non-obvious
```

**Rules**

- **Subject ≤50 chars** (hard max 72 including the `type(scope):` prefix). Imperative mood: `fix`, not `fixed` / `fixes`.
- **Body only when it adds value** — explain *why* or a non-obvious trade-off, never restate the diff. Skip for renames, formatting, trivial fixes.
- **One logical change per commit** — no `feat(a,b,c): X + Y + Z`. Split composite work.
- **English only** for subject and body. Code comments/docstrings may stay in Chinese; intrinsic Chinese proper nouns (file paths, teammate names like 华/吴) may appear inline.
- Valid `<type>`: `feat`, `fix`, `docs`, `refactor`, `style`, `test`, `chore`, `perf`, `ci`, `build`. Lowercase. `<scope>` optional.
- Avoid vague subjects: `update stuff`, `misc`, `WIP`, `fix bug`.

**Examples**

Good:

```
refactor(tools): rename _schema_normalize to schema_normalize
fix(runner): prevent zombie subprocess on timeout
feat(agents): add token-bucket rate limiter
```

Bad:

```
feat(agents,tools,ui): align ... + debug UI + inspect_files     # >72 chars, 3 things in 1
docs(baseline): 给全量源码添加中文注释                            # not English
refactor: updated some files                                    # past tense + vague
```
