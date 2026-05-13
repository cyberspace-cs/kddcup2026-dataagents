# Architecture

`data_agent_baseline` is a single-process ReAct loop that drives an OpenAI-compatible LLM
through a fixed set of tools, scoped to a per-task `context/` directory. The runner wraps
each task in a subprocess so wall-clock timeouts and crashes can be cleanly bounded.

```
┌────────────────────────────────────────────────────────────────────────┐
│  CLI (Typer)                                                           │
│   dabench {status, inspect-task, run-task, run-benchmark}              │
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
                 trace.json + answer
```

## Live capabilities

Each capability has a canonical spec under `openspec/specs/<name>/spec.md`.

| Capability | Spec | What lives there |
| --- | --- | --- |
| ReAct loop | [`agent-execution-loop`](../openspec/specs/agent-execution-loop/spec.md) | Bounded loop with separated protocol adapter + tool dispatcher + step recorder; typed event kinds; partial-trace observability |
| Model I/O | [`model-adapter`](../openspec/specs/model-adapter/spec.md) | OpenAI-compatible `chat.completions` for both `text` and `native` function-calling protocols; Qwen3 `<think>` stripping; usage + latency capture |
| Tool inventory | [`tool-registry`](../openspec/specs/tool-registry/spec.md) | Single `ToolDefinition` shape; immutable registry post-construction; default inventory; sandboxed paths |
| Subprocess runner | [`task-runner`](../openspec/specs/task-runner/spec.md) | Per-task subprocess with hard timeout; partial-trace recovery; deterministic key pinning; `summary.json` aggregate |
| Rate limiting | [`rate-limited-key-pool`](../openspec/specs/rate-limited-key-pool/spec.md) | Cross-process file token bucket (RPM + TPM) with `filelock`, atomic state writes, retry-after parsing |

## Two protocols, one loop

The agent supports both legacy text-fenced JSON and OpenAI native function calling. The
distinction lives entirely in `agents/protocol.py:adapt_model_response`:

- **Text** — model emits a single JSON object (optionally inside a fenced block) with
  `thought`, `action`, `action_input`. The dispatcher picks one tool per turn.
- **Native** — model emits zero-or-more `tool_calls` in `choices[0].message`. The
  dispatcher iterates them in order; the first terminal-and-successful call halts the loop.

Both protocols feed a common `ToolCall` stream into the dispatcher, so the loop driver in
`react.py` does not branch on protocol.

## Subprocess isolation

`run_single_task` always runs the agent in a child process when no test injection is
active. The parent posts the `AppConfig` plus a partial-trace path; the child calls
`build_application` itself (the `AgentApp` is intentionally not pickled). On timeout the
parent reads back the partial `trace.json` written incrementally by the `step_callback`
and synthesizes a terminal `__error__` step with a `timeout` payload — see
[`task-runner` spec](../openspec/specs/task-runner/spec.md).

## Key pool + cross-process rate limiter

When `agent.api_keys` is set, task index `i` pins to `api_keys[i % len(api_keys)]` for the
entire run. The `RateLimitedAdapter` then guards each request with a `FileTokenBucket`
backed by a `filelock`-protected JSON file under `<output_dir>/../ratelimit/<run_id>/`.
Atomic state writes (write-temp + `os.replace`) prevent torn reads across worker
processes. See [`rate-limited-key-pool` spec](../openspec/specs/rate-limited-key-pool/spec.md).

## Wire-format contract

The following outputs are byte-stable for any fixed input + scripted adapter, locked by
`tests/test_golden_trace.py`:

- `trace.json` — full step list with `action`, `action_input`, `observation`, `usage`
- `summary.json` — per-run aggregate (run id, task count, succeeded count, per-task entries)
- `prediction.csv` — answer table
- `ratelimit/bucket_<sha1>.json` — cross-process bucket state
