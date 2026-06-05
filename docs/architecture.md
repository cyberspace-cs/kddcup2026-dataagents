# Architecture

`data_agent_baseline` is a single-process ReAct loop that drives an OpenAI-compatible LLM
through a fixed set of tools, scoped to a per-task `context/` directory. The runner wraps
each task in a subprocess so wall-clock timeouts and crashes can be cleanly bounded.

```
┌────────────────────────────────────────────────────────────────────────┐
│  CLI (Typer)                                                           │
│   dabench {status, inspect-task, run-task, run-benchmark, dashboard}   │
└────────────────────────┬───────────────────────────────────────────────┘
                         │
                ┌────────▼────────┐
                │   AppConfig     │   Pydantic-validated YAML
                └────────┬────────┘
                         │
              ┌──────────▼──────────┐
              │  build_application  │   src/data_agent_baseline/run/app.py
              └──────────┬──────────┘
                         │  AgentApp{ dataset, registry, model_adapter,
                         │            agent_config, protocol }
              ┌──────────▼──────────┐
              │     ReActAgent      │   src/data_agent_baseline/agents/react.py
              │  ┌───────┐ ┌──────┐ │
              │  │protoco│→│dispat│ │
              │  │l adapt│ │ cher │ │
              │  └───┬───┘ └──┬───┘ │
              │      └────┬───┘     │
              │      ┌────▼────┐    │
              │      │recorder │    │
              │      └────┬────┘    │
              └───────────┼─────────┘
                          ▼
                 prediction.csv + summary.json
                 optional traces.db
```

## Live capabilities

These are the current implementation anchors for the main runtime surfaces.

| Capability | Implementation | What lives there |
| --- | --- | --- |
| ReAct loop | `src/data_agent_baseline/agents/react.py` | Bounded loop, message replay, budget guardrails, tool dispatch, answer verifier integration, and step callbacks. |
| Protocol adapter | `src/data_agent_baseline/agents/protocol.py` | Text-fenced JSON and native function-calling responses normalized into a common turn shape. |
| Model I/O | `src/data_agent_baseline/agents/model.py` | OpenAI-compatible `chat.completions` adapters, Qwen3 `<think>` stripping, usage capture, latency capture, and native tool-call parsing. |
| Tool inventory | `src/data_agent_baseline/tools/registry.py` | `ToolDefinition`, immutable `ToolRegistry`, default tool inventory, Pydantic input validation, and OpenAI tool schema export. |
| Subprocess runner | `src/data_agent_baseline/run/runner.py` plus `run/runner_subprocess.py` and `run/runner_artifacts.py` | Per-task subprocess timeout isolation, deterministic key pinning, task artifacts, and `summary.json`. |
| SQLite tracing | `src/data_agent_baseline/tracing/` plus `src/data_agent_baseline/dashboard/` | Opt-in span capture to `tracing.db_path` and the FastAPI/React dashboard that reads it. |
| Rate limiting | `src/data_agent_baseline/agents/rate_limit.py` plus `agents/token_bucket.py` | Cross-process RPM/TPM token buckets with `filelock`, atomic state writes, and Retry-After handling. |

## Two protocols, one loop

The agent supports both legacy text-fenced JSON and OpenAI native function calling. The
response normalization lives in `agents/protocol.py:adapt_model_response`:

- **Text** — model emits a single JSON object (optionally inside a fenced block) with
  `thought`, `action`, `action_input`. The dispatcher picks one tool per turn.
- **Native** — model emits zero-or-more `tool_calls` in `choices[0].message`. The
  dispatcher iterates them in order; the first terminal-and-successful call halts the loop.

Both protocols feed a common tool-call stream into the dispatcher. The loop driver still
has protocol-specific message replay paths because text turns replay as assistant/user
messages, while native turns replay as assistant/tool messages with tool-call IDs.

## Subprocess isolation

`run_single_task` always runs the agent in a child process when no test injection is
active. The parent posts the `AppConfig`; the child calls `build_application` itself
(the `AgentApp` is intentionally not pickled). On timeout or crash, the parent returns a
synthetic failure payload. When `tracing.enabled: true`, the child also writes spans to
the configured SQLite database for dashboard inspection.

## Key pool + cross-process rate limiter

When `agent.api_keys` is set, task index `i` pins to `api_keys[i % len(api_keys)]` for the
entire run. The `RateLimitedAdapter` then guards each request with a `FileTokenBucket`
backed by a `filelock`-protected JSON file under `<output_dir>/../ratelimit/<run_id>/`.
Atomic state writes (write-temp + `os.replace`) prevent torn reads across worker
processes.

## Wire-format contract

The following outputs are treated as stable runtime artifacts and are covered by focused
unit and integration tests:

- `summary.json` — per-run aggregate (run id, task count, succeeded count, per-task entries)
- `prediction.csv` — answer table
- `traces.db` — optional SQLite tracing store when `tracing.enabled: true`
- `ratelimit/bucket_<sha1>.json` — cross-process bucket state
