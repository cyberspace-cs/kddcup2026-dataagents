# Agent protocol

The agent supports two protocols for getting tool calls out of an LLM. Switch via
`agent.tool_protocol` in the YAML config.

## When to use which

| Protocol | Use when | Notes |
| --- | --- | --- |
| `native` | The endpoint supports OpenAI native function calling. vLLM with `--enable-auto-tool-choice --tool-call-parser qwen3_coder` works. | Default. Multi-call per turn. JSON arguments validated by the SDK. |
| `text` | The endpoint cannot accept a `tools=` payload, or returns `400 Bad Request` when one is sent. | One call per turn. The model emits a single fenced JSON object containing `thought`, `action`, `action_input`. |

In both modes the model adapters strip any `<think>...</think>` block from
`message.content` before parsing.

## Default tool inventory

Registration order is fixed; `to_openai_tools()` sorts by name on output for cache
stability.

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `inspect_files` | _(none)_ | One-shot summary of every file under `context/`: columns, dtypes, row counts, table schemas. |
| `read_csv` | `path` | CSV sample preview (schema + head 20 rows + tail 5 rows). |
| `read_doc` | `path`, `start_line`, `line_count`, `keyword` | Paged text reader; with `keyword`, returns all matching paragraphs (capped at 20 / 8KB). |
| `search_context` | `query`, `path`, `limit` | Search the task-local retrieval index for long-document evidence. |
| `lookup_field` | `term`, `limit` | Search local field, unit, alias, threshold, and schema summaries. |
| `read_json` | `path` | Structured JSON preview (kind + length/keys + sample values, 100MB load cap; >100MB returns streamed schema via ijson). |
| `inspect_sqlite_schema` | `path` | List tables and columns of a SQLite/DB file. |
| `execute_context_sql` | `path`, `sql`, `limit` | Read-only SQL against a SQLite/DB file in `context/`. |
| `execute_python` | `code` | Run arbitrary Python with cwd pinned to the task `context/`. |
| `answer` | `columns`, `rows` | Submit the final answer table. **Terminal**: halts the loop on success. |

All `path` fields are resolved relative to the task's `context_dir`. Path traversal,
absolute paths, and symlinks escaping the sandbox are rejected.

## Loop control

The driver bounds these counters:

- `max_steps` — total recorded steps before the loop halts with a
  `failure_reason` mentioning `max_steps`.
- `max_empty_tool_call_retries` — total empty-call turns tolerated. Halt
  fires when `empty_tool_call_retries >= max_empty_tool_call_retries`
  (i.e. with `N=1`, the first empty turn is the failure turn).

Model request exceptions are recorded as `model_error` steps and consume the normal
`max_steps` budget.

Internal event kinds (`model_error`, `parse_error`, `empty_tool_calls`,
`tool_error`, `observation`) are an internal `Literal`; each recorded error
step serializes as `action="__error__"` plus `error_payload` so downstream
tracing and tests see one stable shape.

## Step Records

`AgentRunResult.to_dict()` returns the in-memory task result shape consumed by the
runner:

```json
{
  "task_id": "task_1",
  "succeeded": true,
  "failure_reason": null,
  "steps": [...],
  "answer": { "columns": [...], "rows": [...] },
  "elapsed_seconds": 12.3,
  "usage": {
    "prompt_tokens": 1234,
    "completion_tokens": 567,
    "total_tokens": 1801,
    "cached_tokens": 0,
    "reasoning_tokens": 12
  }
}
```

Each `steps` entry:

```json
{
  "step_index": 0,
  "turn_index": 0,
  "thought": "...",
  "action": "read_csv",
  "action_input": { "path": "data.csv", "max_rows": 5 },
  "observation": { "ok": true, "content": {...} },
  "error_payload": null,
  "usage": { ... },
  "latency_ms": 412
}
```

Error steps have `action == "__error__"` and a populated `error_payload`:

```json
{
  "step_index": 3,
  "turn_index": 3,
  "action": "__error__",
  "error_payload": { "kind": "tool_error", "tool": "read_csv", "error": "..." }
}
```

## `summary.json`

```json
{
  "run_id": "20260430-001",
  "task_count": 5,
  "succeeded_task_count": 4,
  "max_workers": 4,
  "tasks": [
    {
      "task_id": "task_1",
      "task_output_dir": "artifacts/runs/.../task_1",
      "prediction_csv_path": "artifacts/runs/.../task_1/prediction.csv",
      "succeeded": true,
      "failure_reason": null
    }
  ]
}
```

The `succeeded` flag is purely "did the agent call `answer`?" — it does not assert that
the answer is **correct** against any ground truth.

Debug tracing is stored separately from run artifacts. Set `tracing.enabled: true` in
the config to write SQLite spans to `tracing.db_path` for `dabench dashboard`.
