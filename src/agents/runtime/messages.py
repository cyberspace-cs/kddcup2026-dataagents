"""Message replay and serialization helpers for the ReAct runtime."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from agents.llm.protocol import ToolProtocol
from agents.llm.types import ModelAdapter, ModelMessage, ModelResponse, ModelToolCall
from agents.prompts import (
    build_native_system_prompt,
    build_observation_prompt,
    build_system_prompt,
)
from agents.runtime.state import AgentRuntimeState, StepRecord
from agents.tools import ToolRegistry


def build_system_message(
    *,
    model: ModelAdapter,
    tools: ToolRegistry,
    protocol: ToolProtocol,
    system_prompt: str | None,
) -> ModelMessage:
    """Build the protocol-specific system message."""
    if protocol == "native":
        raw_enable_thinking = getattr(model, "enable_thinking", None)
        enable_thinking = raw_enable_thinking if isinstance(raw_enable_thinking, bool) else None
        content = build_native_system_prompt(system_prompt, enable_thinking=enable_thinking)
    else:
        content = build_system_prompt(
            tools.describe_for_prompt(),
            system_prompt=system_prompt,
        )
    return ModelMessage(role="system", content=content)


def build_messages(
    *,
    state: AgentRuntimeState,
    initial_content: str | list[dict[str, Any]],
    model: ModelAdapter,
    tools: ToolRegistry,
    protocol: ToolProtocol,
    system_prompt: str | None,
) -> list[ModelMessage]:
    """Build model request messages by dispatching to the active tool protocol."""
    system_message = build_system_message(
        model=model,
        tools=tools,
        protocol=protocol,
        system_prompt=system_prompt,
    )
    if protocol == "native":
        return build_messages_native(
            state=state,
            initial_content=initial_content,
            system_message=system_message,
        )
    return build_messages_text(
        state=state,
        initial_content=initial_content,
        system_message=system_message,
    )


def build_messages_text(
    *,
    state: AgentRuntimeState,
    initial_content: str | list[dict[str, Any]],
    system_message: ModelMessage,
) -> list[ModelMessage]:
    """Build text-protocol history: assistant raw response plus user observation."""
    messages = [system_message, ModelMessage(role="user", content=initial_content)]
    for step in state.steps:
        messages.append(ModelMessage(role="assistant", content=step.raw_response))
        messages.append(
            ModelMessage(
                role="user",
                content=build_observation_prompt(step.observation, ok=step.ok),
            )
        )
    return messages


def build_messages_native(
    *,
    state: AgentRuntimeState,
    initial_content: str | list[dict[str, Any]],
    system_message: ModelMessage,
) -> list[ModelMessage]:
    """Build native-tool history by replaying assistant/tool message pairs per turn."""
    messages = [system_message, ModelMessage(role="user", content=initial_content)]

    steps_by_turn: dict[int, list[StepRecord]] = {}
    for step in state.steps:
        steps_by_turn.setdefault(step.turn_index, []).append(step)

    for turn_index in sorted(steps_by_turn):
        steps_in_turn = steps_by_turn[turn_index]
        first_step = steps_in_turn[0]

        if all(step.action == "__error__" for step in steps_in_turn):
            messages.append(ModelMessage(role="assistant", content=first_step.raw_response))
            messages.append(
                ModelMessage(
                    role="user",
                    content=build_observation_prompt(first_step.observation, ok=first_step.ok),
                )
            )
            continue

        messages.append(
            ModelMessage(
                role="assistant",
                content=first_step.thought or "",
                tool_calls=first_step.raw_tool_calls or None,
            )
        )
        for step in steps_in_turn:
            tool_content = json.dumps(step.observation, ensure_ascii=False)
            messages.append(
                ModelMessage(
                    role="tool",
                    content=tool_content,
                    tool_call_id=step.tool_call_id or "",
                )
            )
    return messages


def serialize_message(message: ModelMessage) -> dict[str, Any]:
    """Serialize ``ModelMessage`` into a JSON-friendly trace payload."""
    content: str | list[dict[str, Any]] = message.content
    if isinstance(content, list):
        content = [
            {**block, "video_url": {"url": "<base64-stripped>"}}
            if block.get("type") == "video_url"
            else block
            for block in content
        ]
    payload: dict[str, Any] = {"role": message.role, "content": content}
    if message.tool_calls is not None:
        payload["tool_calls"] = message.tool_calls
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    return payload


def serialize_response(response: ModelResponse) -> dict[str, Any]:
    """Serialize ``ModelResponse`` into a JSON-friendly trace payload."""
    return {
        "content": response.content,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in response.tool_calls
        ],
        "raw_response": response.raw_response,
        "raw_tool_calls": list(response.raw_tool_calls),
        "reasoning_content": response.reasoning_content,
        "usage": response.usage.to_dict(),
        "latency_ms": response.latency_ms,
    }


def synthesize_raw_tool_calls(calls: Sequence[ModelToolCall]) -> list[dict[str, object]]:
    """Build replay-compatible raw tool call payloads for recovered native calls."""
    return [
        {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(call.arguments, ensure_ascii=False),
            },
        }
        for call in calls
    ]
