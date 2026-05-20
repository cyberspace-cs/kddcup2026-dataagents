"""协议适配层 `protocol.adapt_model_response` + 内部解析助手测试。

覆盖：
- text 协议的 ```json fence 解析（成功 / 解析失败 → ProtocolTurn.parse_error）
- native 协议下原始 tool_calls 透传 + thought 来自 response.content
- native 协议空 tool_calls 时从 reasoning 提取草稿（含已知工具 / 未知工具）
- `<think>` 剥离已发生在 model 层，但本层 thought/raw_response 行为不依赖该剥离
- `parse_model_step` 三件套校验（thought 非字符串 / action 空 / action_input 非对象）
"""

from __future__ import annotations

import pytest

from data_agent_baseline.agents.model import ModelResponse, ModelToolCall
from data_agent_baseline.agents.protocol import (
    ProtocolTurn,
    adapt_model_response,
    extract_reasoning_tool_call_draft,
    parse_model_step,
)


def _resp(
    *,
    content: str = "",
    tool_calls: list[ModelToolCall] | None = None,
    reasoning: str = "",
    raw_response: str = "",
) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tool_calls or [],
        raw_response=raw_response or content,
        raw_tool_calls=[],
        reasoning_content=reasoning,
    )


# ---------- text 协议 ----------


def test_text_protocol_fence_parses_to_synthetic_call() -> None:
    response = _resp(
        content=(
            '```json\n{"thought": "list files", "action": "inspect_files", "action_input": {}}\n```'
        )
    )

    turn = adapt_model_response(
        response,
        protocol="text",
        turn_index=3,
        allowed_tool_names=("inspect_files", "answer"),
    )

    assert isinstance(turn, ProtocolTurn)
    assert turn.parse_error is None
    assert turn.thought == "list files"
    assert len(turn.calls) == 1
    call = turn.calls[0]
    assert call.id == "text:3"
    assert call.name == "inspect_files"
    assert call.arguments == {}


def test_text_protocol_parse_failure_returns_parse_error() -> None:
    response = _resp(content="garbage non-json content")

    turn = adapt_model_response(
        response,
        protocol="text",
        turn_index=1,
        allowed_tool_names=("answer",),
    )

    assert turn.calls == ()
    assert turn.thought == ""
    assert turn.parse_error is not None
    # 原始 ValueError 信息应原样透传（json.JSONDecodeError 的 "Expecting value..." 文案）
    assert "Expecting value" in turn.parse_error


@pytest.mark.parametrize(
    ("raw_payload", "expected_substring"),
    [
        # action 必须非空
        ('{"thought":"t","action":"","action_input":{}}', "action must be a non-empty"),
        # action_input 必须是 dict
        ('{"thought":"t","action":"x","action_input":[]}', "action_input must be a JSON object"),
        # thought 必须是字符串
        ('{"thought":1,"action":"x","action_input":{}}', "thought must be a string"),
    ],
)
def test_parse_model_step_rejects_malformed(raw_payload: str, expected_substring: str) -> None:
    fenced = "```json\n" + raw_payload + "\n```"
    with pytest.raises(ValueError, match=expected_substring):
        parse_model_step(fenced)


def test_parse_model_step_accepts_unfenced_json() -> None:
    # 没有围栏时直接当 JSON 解析也能工作（再降级路径）
    step = parse_model_step('{"thought":"t","action":"answer","action_input":{}}')
    assert step.action == "answer"
    assert step.thought == "t"


# ---------- native 协议 ----------


def test_native_protocol_passes_through_tool_calls() -> None:
    call = ModelToolCall(id="call_42", name="inspect_files", arguments={})
    response = _resp(content="thinking", tool_calls=[call])

    turn = adapt_model_response(
        response,
        protocol="native",
        turn_index=1,
        allowed_tool_names=("inspect_files",),
    )

    assert turn.parse_error is None
    assert turn.calls == (call,)
    assert turn.thought == "thinking"
    assert turn.reasoning_draft is None


def test_native_protocol_empty_calls_recovers_from_reasoning() -> None:
    reasoning = (
        "<tool_call>\n"
        "<function=read_doc>\n"
        "<parameter=path>knowledge.md</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
    )
    response = _resp(content="", reasoning=reasoning)

    turn = adapt_model_response(
        response,
        protocol="native",
        turn_index=1,
        allowed_tool_names=("read_doc", "answer"),
    )

    assert turn.parse_error is None
    assert len(turn.calls) == 1
    assert turn.calls[0].id == "recovered:1"
    assert turn.calls[0].name == "read_doc"
    assert turn.calls[0].arguments == {"path": "knowledge.md"}


def test_native_protocol_empty_calls_recovers_from_content() -> None:
    content = (
        "<tool_call><function=execute_python>"
        "<parameter=code>print(1)</parameter>"
        "</function></tool_call>"
    )
    response = _resp(content=content, reasoning="")

    turn = adapt_model_response(
        response,
        protocol="native",
        turn_index=5,
        allowed_tool_names=("execute_python", "answer"),
    )

    assert turn.parse_error is None
    assert len(turn.calls) == 1
    assert turn.calls[0].id == "recovered:5"
    assert turn.calls[0].name == "execute_python"
    assert turn.calls[0].arguments == {"code": "print(1)"}


def test_native_protocol_content_takes_priority_over_reasoning() -> None:
    content = (
        "<tool_call><function=inspect_files><parameter=path>.</parameter></function></tool_call>"
    )
    reasoning = (
        "<tool_call><function=execute_python>"
        "<parameter=code>print(1)</parameter>"
        "</function></tool_call>"
    )
    response = _resp(content=content, reasoning=reasoning)

    turn = adapt_model_response(
        response,
        protocol="native",
        turn_index=2,
        allowed_tool_names=("inspect_files", "execute_python", "answer"),
    )

    assert len(turn.calls) == 1
    assert turn.calls[0].name == "inspect_files"


def test_native_protocol_empty_calls_skips_unknown_tool_draft() -> None:
    # function name 不在 allowed_tool_names → 草稿被丢弃
    reasoning = (
        "<tool_call><function=unknown_tool><parameter=x>1</parameter></function></tool_call>"
    )
    response = _resp(content="", reasoning=reasoning)

    turn = adapt_model_response(
        response,
        protocol="native",
        turn_index=1,
        allowed_tool_names=("inspect_files",),
    )

    assert turn.calls == ()
    assert turn.reasoning_draft is None


def test_native_protocol_empty_calls_no_reasoning() -> None:
    response = _resp(content="", reasoning="")

    turn = adapt_model_response(
        response,
        protocol="native",
        turn_index=1,
        allowed_tool_names=("answer",),
    )

    assert turn.calls == ()
    assert turn.reasoning_draft is None


def test_extract_reasoning_draft_parses_code_param_verbatim() -> None:
    # `code` 参数永远当成多行字符串保留，不走 JSON 解析
    reasoning = (
        "<tool_call><function=execute_python>"
        "<parameter=code>import os\nprint(os.listdir('.'))</parameter>"
        "</function></tool_call>"
    )
    draft = extract_reasoning_tool_call_draft(reasoning, ("execute_python",))
    assert draft is not None
    assert draft["arguments"]["code"] == "import os\nprint(os.listdir('.'))"
