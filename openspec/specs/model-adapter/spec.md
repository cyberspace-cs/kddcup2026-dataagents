# model-adapter Specification

## Purpose
TBD - created by archiving change refactor-architecture-and-prune. Update Purpose after archive.
## Requirements
### Requirement: OpenAI-compatible chat.completions adapter with text and native protocols

The system SHALL provide an `OpenAIModelAdapter` that performs chat-completions requests against any OpenAI-compatible endpoint. A separate `NativeToolsOpenAIAdapter` SHALL extend it to send native function-calling `tools` payloads. Both adapters MUST honor a `protocol: Literal["native", "text"]` selector and MUST accept the same `ModelMessage` input shape.

#### Scenario: Text protocol returns ModelResponse with content only
- **WHEN** an `OpenAIModelAdapter` is invoked with `protocol="text"` against a stub endpoint that returns `{ "choices": [{ "message": { "content": "...```json {...} ```..." } }] }`
- **THEN** the returned `ModelResponse.content` is the raw string and `ModelResponse.tool_calls` is `None` or empty

#### Scenario: Native protocol returns parsed ModelToolCall list
- **WHEN** a `NativeToolsOpenAIAdapter` is invoked against a stub endpoint that returns one `tool_calls` entry with `function.name="answer"` and `function.arguments='{"answer":[]}'`
- **THEN** the returned `ModelResponse.tool_calls` is a list of one `ModelToolCall` with `name="answer"` and `arguments == {"answer": []}`

### Requirement: Qwen3 reasoning content is stripped before parse

The text-protocol adapter SHALL strip any `<think>...</think>` block from the message content before returning it. The stripped content MUST NOT appear in `ModelResponse.content` nor in any downstream tool-call argument.

#### Scenario: Think block is removed from content
- **WHEN** the endpoint returns content of the form `<think>internal reasoning</think>actual answer json fence`
- **THEN** `ModelResponse.content` equals `actual answer json fence` (with the `<think>` block removed) and contains no `<think>` substring

#### Scenario: Think block does not contaminate native tool arguments
- **WHEN** a native-protocol response has `tool_calls[0].function.arguments` set to a JSON string and a separate `reasoning_content` payload
- **THEN** the parsed `ModelToolCall.arguments` dict contains only the keys present in the JSON string and no fragment of the reasoning content

### Requirement: enable_thinking is passed through extra_body when configured

When `AgentConfig.enable_thinking` is non-`None`, the adapter SHALL include `extra_body={"enable_thinking": <value>}` in the chat-completions request.

#### Scenario: enable_thinking=False is forwarded
- **WHEN** the agent is configured with `enable_thinking=False` and issues a model call
- **THEN** the captured request payload includes `extra_body == {"enable_thinking": False}`

#### Scenario: enable_thinking=None is omitted
- **WHEN** the agent is configured with `enable_thinking=None`
- **THEN** the captured request payload contains no `extra_body` key (or an empty dict)

### Requirement: Token usage and latency are captured on every response

The adapter SHALL populate `ModelResponse.usage` with a `TokenUsage` instance carrying `prompt_tokens`, `completion_tokens`, and (when reported by the endpoint) `cached_tokens` and `reasoning_tokens`. `ModelResponse.latency_ms` SHALL be set to the wall-clock duration of the request in milliseconds, and MUST be `>= 0`.

#### Scenario: Usage fields are non-negative integers
- **WHEN** any model response is returned
- **THEN** `usage.prompt_tokens >= 0`, `usage.completion_tokens >= 0`, and `latency_ms >= 0`

#### Scenario: Cached and reasoning tokens are passed through when reported
- **WHEN** an endpoint response includes `usage.prompt_tokens_details.cached_tokens=10` and `usage.completion_tokens_details.reasoning_tokens=5`
- **THEN** the resulting `ModelResponse.usage.cached_tokens == 10` and `usage.reasoning_tokens == 5`

### Requirement: Adapters must not mutate caller-supplied messages

Calling `adapter.complete(messages, ...)` SHALL NOT modify the caller's `messages` list or any item it contains.

#### Scenario: Caller's message list is unchanged after a call
- **WHEN** the caller passes a `messages` list and inspects it after the call returns
- **THEN** the list and every item compare equal to the pre-call deep copy

