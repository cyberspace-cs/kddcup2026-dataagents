## ADDED Requirements

### Requirement: Bounded ReAct loop with separated protocol adapter, tool dispatcher, and step recorder

The system SHALL implement the ReAct execution loop as three cooperating units — a protocol adapter, a tool dispatcher, and a step recorder — coordinated by a thin loop driver in `ReActAgent.run`. The driver SHALL bound total turns by `max_steps`, model-call retries by `max_model_call_retries`, and empty-tool-call retries by `max_empty_tool_call_retries`. The loop MUST always halt for any input that respects these bounds.

#### Scenario: Loop halts when max_steps reached
- **WHEN** the agent runs against a `ScriptedNativeModelAdapter` whose script never emits the `answer` tool and `max_steps=5`
- **THEN** the agent records exactly 5 step entries, sets `state.failure_reason` to a string mentioning `max_steps`, and exits with `state.answer is None`

#### Scenario: Loop halts after Nth empty turn (N total budget, not retries)
- **WHEN** `max_empty_tool_call_retries=2` and the model returns no tool calls for two consecutive turns
- **THEN** the first empty turn is recorded as an `empty_tool_calls` event with `empty_tool_call_retries=1` and the loop continues, and on the second empty turn (`empty_tool_call_retries=2`) the loop halts with a failure step indicating empty-call exhaustion (i.e. the guard fires when `empty_tool_call_retries >= max_empty_tool_call_retries`, not when `>` exceeds it; with `N=1` the very first empty turn is the failure turn)

#### Scenario: Model-call retries are bounded
- **WHEN** the model adapter raises an exception on every call and `max_model_call_retries=2`
- **THEN** the loop records exactly 2 `model_error` events plus a final terminating `model_error` step, and `state.failure_reason` mentions `max_model_call_retries`

### Requirement: Typed internal event kinds with byte-stable trace.json output

The agent SHALL produce internal step events with a closed `Literal` set of kinds: `model_error`, `parse_error`, `empty_tool_calls`, `tool_error`, `observation`. Wire-format `trace.json` keys and value shapes MUST remain byte-stable with the pre-change baseline; the legacy `action="__error__"` sentinel and `error_payload` shape SHALL be preserved at serialization time.

#### Scenario: trace.json round-trip is byte-stable for a fixed scripted run
- **WHEN** the agent runs end-to-end against a fixed `ScriptedNativeModelAdapter` script and writes `trace.json`
- **THEN** the output bytes equal a frozen golden file checked into `tests/fixtures/golden/trace.json`

#### Scenario: Internal event kinds do not leak into trace.json keys
- **WHEN** the agent emits a `model_error` event internally
- **THEN** the resulting step in `trace.json` carries `action == "__error__"` (not `action == "model_error"`) and the structured detail lives in `error_payload`

### Requirement: Determinism under scripted adapters

When driven by a deterministic adapter (e.g. `ScriptedNativeModelAdapter`), the agent SHALL produce a byte-identical trace on repeated runs given the same task input.

#### Scenario: Two consecutive runs produce identical traces
- **WHEN** the same `ScriptedNativeModelAdapter` script and the same `PublicTask` are run twice through `ReActAgent.run`
- **THEN** both runs produce equal `StepRecord` sequences (same `step_index`, `turn_index`, `action`, `arguments`, `observation`, and `usage` values)

### Requirement: Terminal short-circuit only fires after handler success

A tool definition's `is_terminal=True` flag SHALL halt the loop only after the tool's handler returns successfully (`ok=True`). A handler that raises (e.g. invalid `answer` payload) MUST NOT halt the loop; the failure SHALL be recorded as a `tool_error` event and the loop MUST continue (subject to existing retry budgets).

#### Scenario: Invalid answer payload retries instead of halting
- **WHEN** the model emits an `answer` tool call with a payload that fails `_resolve_answer_args` validation
- **THEN** the step is recorded as `tool_error`, `state.answer` remains `None`, and the loop continues to the next turn

#### Scenario: Successful answer halts the loop on the same turn
- **WHEN** the model emits a valid `answer` tool call
- **THEN** the loop sets `state.answer`, records the answer step, skips any remaining tool calls in the same native turn, and exits

### Requirement: Partial trace observability via `step_callback`

The agent SHALL invoke a caller-provided `step_callback(step: StepRecord) -> None` after recording each step, before advancing to the next turn. The callback MUST be invoked exactly once per recorded step.

#### Scenario: Callback receives every step in order
- **WHEN** a callback that appends each step to a list is passed to `ReActAgent.run`
- **THEN** after the run, the list equals `state.steps` in order
