# Development

## Toolchain

The project is managed by [uv](https://docs.astral.sh/uv/). All Python
invocations must go through `uv run` so the locked environment is used.

```bash
# Install runtime + dev dependencies
uv sync --extra dev

# Run the CLI / scripts
uv run dabench status --config configs/react_baseline.example.yaml
uv run pytest
uv run ruff format src/
uv run pyright
```

Avoid invoking `python`, `python3`, or `.venv/bin/python` directly — they may bypass
`uv`'s sync state.

## Quality gates

The repo ships four exit-code-clean gates. Each is independent and idempotent.

| Gate | Command | Contract |
| --- | --- | --- |
| Format | `uv run ruff format --check .` | Exit `0`; reports zero files would be reformatted. |
| Lint | `uv run ruff check .` | Exit `0`; `All checks passed`. |
| Types | `uv run pyright` | Exit `0`. Strict mode on `src/data_agent_baseline`, basic on `tests/`. |
| Tests | `uv run pytest` | Exit `0`. Coverage ≥ 70% (gate fails below). |

See the [`quality-gates` spec](../openspec/specs/quality-gates/spec.md) for the
authoritative version.

### Pre-commit (opt-in)

A pre-commit chain is shipped at `.pre-commit-config.yaml` running
`ruff format` → `ruff check --fix` → `pyright src` on changed files. It is opt-in:

```bash
uv run pre-commit install      # one-time, per clone
uv run pre-commit run --all-files
```

CI is intentionally not wired in this repo — the gates run locally only.

## Project layout

```
src/data_agent_baseline/
├── __init__.py
├── cli.py                       # Typer entrypoint (dabench)
├── config.py                    # Frozen dataclasses (public type)
├── config_models.py             # Pydantic v2 validators (internal)
├── agents/
│   ├── react.py                 # Loop driver
│   ├── protocol.py              # Native ↔ text → ToolCall normalization
│   ├── dispatcher.py            # Tool execution + terminal short-circuit
│   ├── recorder.py              # StepRecord + step_callback
│   ├── model.py                 # OpenAI adapters (text + native)
│   ├── rate_limit.py            # FileTokenBucket + RateLimitedAdapter
│   ├── prompt.py                # System / task / observation prompts
│   └── runtime.py               # AgentRuntimeState, AgentRunResult
├── tools/
│   ├── registry.py              # ToolDefinition + immutable ToolRegistry
│   ├── filesystem.py            # list_context, read_csv, read_doc, read_json
│   ├── sqlite.py                # inspect_sqlite_schema, execute_context_sql
│   └── python_exec.py           # execute_python (subprocess + queue)
├── benchmark/
│   ├── dataset.py               # DABenchPublicDataset
│   └── schema.py                # PublicTask
└── run/
    ├── app.py                   # build_application(config) -> AgentApp
    └── runner.py                # Subprocess + timeout + parallel runner
```

Tests live under `tests/`, with shared fixtures in `tests/helpers/` (notably
`scripted_adapters.py`).

## Contributing rules

Commit-message format and `CHANGELOG.md` maintenance live in their own document:
[`CONTRIBUTING.md`](../CONTRIBUTING.md). Repository-specific developer rules below.

- **Don't bypass `uv`.** `pip install`, raw `python`, and `.venv/bin/python` are out.
- **Keep wire formats stable.** `trace.json`, `summary.json`, `prediction.csv`, and
  `bucket_<sha1>.json` are byte-checked by `tests/test_golden_trace.py` and
  `tests/test_config_byte_compat.py`. Any intentional break must update the golden
  fixtures and document the change in `CHANGELOG.md`.
- **Add tests for new code paths.** Coverage gate is 70%; treat it as a floor.
- **Read the relevant spec first.** Capabilities live under `openspec/specs/`. A
  behavior change usually wants a paired spec change in `openspec/changes/`.
