# DABench Data-Agent Baseline

ReAct-style baseline for the DABench data-agent benchmark. The runner reads tasks from
`data/public/input/`, drives an OpenAI-compatible LLM through a fixed tool inventory, and
writes per-task `prediction.csv` files plus a per-run `summary.json`. Optional SQLite
tracing writes debug spans to `artifacts/traces.db` for the dashboard.

## Quick Start

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) — on macOS / Linux:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Sync dependencies:

   ```bash
   uv sync
   ```

3. Confirm the dataset is visible:

   ```bash
   uv run dabench status --config configs/react_baseline.example.yaml
   ```

4. Run the baseline:

   ```bash
   uv run dabench run-benchmark --config configs/react_baseline.example.yaml
   ```

For more, see [`docs/`](docs/): `architecture.md`, `cli.md`, `agent-protocol.md`,
`development.md`.

## Dataset

The public demo dataset lives under `data/public/input/`. Each task directory follows:

```text
data/public/input/task_<id>/
├── task.json
└── context/
```

Public answers live separately under `data/public/output/task_<id>/gold.csv`. Hidden
test sets only ship `input/`.

`task.json` carries `task_id`, `difficulty`, and `question`. `context/` may contain CSV,
JSON, SQLite/DB, and text files.

## Configuration

The YAML schema is summarized in [`docs/cli.md`](docs/cli.md), with validation
implemented on the frozen config dataclasses in `src/data_agent_baseline/config.py`.
An example lives at
[`configs/react_baseline.example.yaml`](configs/react_baseline.example.yaml).

```yaml
dataset:
  root_path: data/public/input

agent:
  model: YOUR_MODEL_NAME
  api_base: YOUR_API_BASE_URL
  api_key: YOUR_API_KEY
  max_steps: 16
  temperature: 0.0
  tool_protocol: native

run:
  output_dir: artifacts/runs
  run_id:
  max_workers: 4
  task_timeout_seconds: 600
```

## CLI

```bash
uv run dabench <command> --config PATH [options]
```

| Command | Purpose |
| --- | --- |
| `status` | Show project paths, config path, dataset root, and public task counts. |
| `inspect-task` | Show task metadata and list accessible files under `context/`. |
| `run-task` | Run the agent on one task. |
| `run-benchmark` | Run the agent across the dataset. Accepts `--limit N` and `--range S-E`. |
| `dashboard` | Launch the SQLite traces dashboard. Requires runs created with `tracing.enabled: true`. |

## Tools

The agent calls into the default tool inventory (registered in `src/data_agent_baseline/tools/registry.py`):

| Tool | Inputs |
| --- | --- |
| `inspect_files` | _(none)_ |
| `read_csv` | `path` |
| `read_json` | `path` |
| `read_doc` | `path`, `start_line`, `line_count`, `keyword` |
| `search_context` | `query`, `path`, `limit` |
| `lookup_field` | `term`, `limit` |
| `inspect_sqlite_schema` | `path` |
| `execute_context_sql` | `path`, `sql`, `limit` |
| `execute_python` | `code` |
| `answer` | `columns`, `rows` |

All `path` arguments are relative to the task `context/` directory.

## Outputs

```
artifacts/runs/<run_id>/
├── summary.json
└── <task_id>/
    └── prediction.csv
```

`summary.json` is the canonical machine-readable run output. When a config contains
`tracing.enabled: true`, tracing data is written to `tracing.db_path`
(`artifacts/traces.db` by default) and can be viewed with `uv run dabench dashboard`.

## Main Modules

| Module | Responsibility |
| --- | --- |
| `src/data_agent_baseline/benchmark/dataset.py` | Public dataset loader |
| `src/data_agent_baseline/tools/{filesystem,sqlite,python_exec,registry}.py` | Tool inventory + sandboxed handlers |
| `src/data_agent_baseline/agents/{react,protocol,dispatcher,recorder}.py` | ReAct loop |
| `src/data_agent_baseline/agents/{model,rate_limit}.py` | OpenAI adapters + rate limiter |
| `src/data_agent_baseline/run/{app,runner}.py` | App factory + subprocess runner |
| `src/data_agent_baseline/cli.py` | `dabench` entrypoint |
