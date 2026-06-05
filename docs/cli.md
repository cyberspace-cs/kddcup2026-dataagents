# CLI

The `dabench` command is the only entrypoint. It is registered via
`pyproject.toml`'s `[project.scripts]` and must be invoked through `uv run`:

```bash
uv run dabench <subcommand> --config <path/to/config.yaml> [options]
```

> Do **not** invoke as `python -m data_agent_baseline.cli` — Typer + the module runner
> produces no output in this project. The supported executable entrypoint is the
> `dabench` script declared in `pyproject.toml`.

## Subcommands

`dabench` exposes five subcommands.

| Command | Purpose |
| --- | --- |
| `status` | Print project paths, dataset root, and public task counts. |
| `inspect-task <task_id>` | Show task metadata and the file tree under `context/`. |
| `run-task <task_id>` | Run the agent on a single task and write outputs. |
| `run-benchmark` | Run the agent across the dataset (optionally `--limit N` and/or `--range S-E`). |
| `view` | Launch the Streamlit trace/prediction viewer. |

All commands require `--config <path>`. The YAML is validated by Pydantic v2 on entry; a
schema violation aborts with non-zero exit before any task runs.

## Examples

```bash
# Smoke-check the dataset and config wiring
uv run dabench status --config configs/react_baseline.example.yaml

# Look at one task without running the agent
uv run dabench inspect-task task_1 --config configs/react_baseline.local.yaml

# Run one task; outputs land under <output_dir>/<run_id>/task_1/
uv run dabench run-task task_1 --config configs/react_baseline.local.yaml

# Run the first 5 tasks
uv run dabench run-benchmark --limit 5 --config configs/react_baseline.local.yaml

# Run a contiguous task subset (inclusive on both ends)
uv run dabench run-benchmark --range 5-10 --config configs/react_baseline.local.yaml
uv run dabench run-benchmark --range task_5-task_10 --config configs/react_baseline.local.yaml

# Combine: range filters first, then --limit truncates → runs task_2..task_4
uv run dabench run-benchmark --range 2-8 --limit 3 --config configs/react_baseline.local.yaml
```

## Config schema

The supported YAML fields are validated by the frozen Pydantic dataclasses in
`src/data_agent_baseline/config.py`; the example below mirrors
`configs/react_baseline.example.yaml`.

```yaml
dataset:
  root_path: data/public/input            # relative paths anchored at PROJECT_ROOT

agent:
  model: YOUR_MODEL_NAME
  api_base: YOUR_API_BASE_URL
  api_key: YOUR_API_KEY                   # ignored when api_keys is set
  max_steps: 16
  temperature: 0.0
  tool_protocol: native                   # "native" | "text"
  allow_parallel_tool_calls: true
  max_tokens: 16384                       # 0 = use endpoint default
  enable_thinking: true                   # Qwen3 extra_body passthrough; null = omit

  # Optional dual-key pool + cross-process rate limiter.
  # Both blocks must be present together or both absent.
  # api_keys:
  #   - sk-key-A
  #   - sk-key-B
  # rate_limit:
  #   rpm_per_key: 600
  #   tpm_per_key: 1000000
  #   safety_factor: 0.85
  #   reserve_output_tokens: 16384
  #   max_retries: 6
  #   backoff_initial_seconds: 1.0
  #   backoff_max_seconds: 30.0
  #   # state_dir: artifacts/ratelimit/<run_id>/  # default

run:
  output_dir: artifacts/runs
  run_id:                                 # null = auto `<UTC YYYYMMDD>-<NNN>`, e.g. 20260430-001
  max_workers: 8
  task_timeout_seconds: 600               # 0 or negative = disable
  retrieval_index:
    enabled: false                        # true builds a task-local SQLite FTS5 index
    cache_dir: artifacts/indexes
    max_chunk_chars: 1600
    chunk_overlap_lines: 2
    max_results: 8
```

### Validation behavior

- Strict bool/int coercion. Quoted YAML scalars like `"false"` or `"16384"` are rejected.
  Use unquoted scalars (`false`, `16384`).
- `tool_protocol` must be `native` or `text` (literal).
- `api_keys` and `rate_limit` are paired: setting one without the other is rejected.
- `agent.rate_limit.reserve_output_tokens` must be ≤ `agent.max_tokens` (when both are set).
- `run.retrieval_index` is optional and defaults to disabled; when enabled, index cache paths
  still resolve relative to the project root.
- All path fields resolve relative to the project root.

## Outputs

```
<output_dir>/<run_id>/
├── summary.json
└── <task_id>/
    ├── trace.json
    └── prediction.csv          # absent if the agent never called `answer`
```

`summary.json` is the canonical machine-readable run output. CLI progress text is
informational only.

## Exit codes

- `0` on success
- non-zero on missing config, missing task, validation error, or unrecoverable runtime error
