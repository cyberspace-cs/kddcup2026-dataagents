"""模型协议适配层（native + text → 统一 ProtocolTurn）。

把 `ModelResponse` 里两种协议的差异规整成同一个 `ProtocolTurn`，让 ReAct 循环驱动器
不再关心协议分支：
- native：原样使用 `response.tool_calls`；空时尝试从 reasoning 提取草稿，便于下一轮纠偏
- text：从 ```json fence 解析出 `thought/action/action_input`，合成单个 `ModelToolCall`

`adapt_model_response` 是该层的唯一入口；其他符号都是内部解析助手或常量。

设计要点（D1）：把 text-fence 解析、native tool_calls 规整、Qwen3 `<think>` 剥离与
reasoning 草稿提取统一收口在本模块，driver 只看 `ProtocolTurn`。
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Literal, cast

from data_agent_baseline.agents.model import ModelResponse, ModelStep, ModelToolCall

# 支持的协议字面量：在类型层阻止拼错
ToolProtocol = Literal["native", "text"]


_REASONING_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call\b[^>]*>(.*?)</tool_call>",
    flags=re.IGNORECASE | re.DOTALL,
)
_REASONING_FUNCTION_PATTERN = re.compile(
    r"<function\s*=\s*([^>\s]+)\s*>(.*?)</function>",
    flags=re.IGNORECASE | re.DOTALL,
)
_REASONING_PARAMETER_PATTERN = re.compile(
    r"<parameter\s*=\s*([^>\s]+)\s*>(.*?)</parameter>",
    flags=re.IGNORECASE | re.DOTALL,
)


# native 协议下，模型某轮没产出可执行 tool_call 时回灌给模型的纠偏文案。
# 长版本：模型甚至没有把调用塞进 reasoning（或拿不到草稿），需要更明确的引导。
EMPTY_TOOL_CALL_ERROR = (
    "Your previous turn did not produce an executable tool call. "
    "Any tool call written inside reasoning/thinking text was ignored, "
    "even if it looked syntactically complete. In this turn: do not "
    "draft <tool_call>, <function=...>, JSON actions, or executable "
    "code in reasoning. If you already decided which tool to call last "
    "turn, emit that call now as a structured tool call. If final answer "
    "is ready, call answer directly. You must call a tool."
)

# 短版本：reasoning 里捕到了草稿，附上草稿后只需提醒模型 "把它升格成真正的 tool_call"。
REASONING_DRAFT_RELAY_INSTRUCTION = (
    "A tool-call draft was found in reasoning_content, but reasoning drafts are not "
    "executable and were not run. On the next turn, emit the drafted function and "
    "arguments as a real structured tool_call through the native tool-calling "
    "interface. Do not put the call in reasoning/thinking text. If the final answer "
    "is ready, call answer directly."
)


@dataclass(frozen=True, slots=True)
class ProtocolTurn:
    """协议规整后的一轮模型输出。

    四种状态由 `parse_error` / `calls` 联合判别：
    - `parse_error is not None`：text 协议解析失败（calls 必为空）
    - `parse_error is None and calls`：可派发的 tool_call 列表
    - `parse_error is None and not calls`：native 协议拿到空 tool_calls
        - `reasoning_draft is not None`：从 reasoning 抽到了草稿，可作为纠偏提示
        - `reasoning_draft is None`：完全空轮，需要长版纠偏文案

    `thought` 是要展示的"模型思考"字段：
    - native：原样回填 `response.content`（可能是空串）
    - text：取 JSON 解析出来的 `thought` 字段
    """

    calls: tuple[ModelToolCall, ...]
    thought: str
    parse_error: str | None = None
    reasoning_draft: dict[str, Any] | None = None


def _strip_json_fence(raw_response: str) -> str:
    """从模型响应里抽出 JSON 正文。

    兼容顺序：
    1. ` ```json ... ``` `（首选，prompt 明确要求）
    2. 任意 ``` ... ```（降级兼容，有些模型会漏写 json 语言标记）
    3. 无围栏时原样返回（再降级，由 JSON 解析阶段决定是否可接受）
    """
    text = raw_response.strip()
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match is not None:
        return fence_match.group(1).strip()
    generic_fence_match = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    if generic_fence_match is not None:
        return generic_fence_match.group(1).strip()
    return text


def _load_single_json_object(text: str) -> dict[str, Any]:
    """解析一段文本，要求它恰好是一个 JSON 对象。

    使用 `raw_decode` 而非 `loads`，以便拿到 "第一个 JSON 之后的剩余文本"。
    剩余文本经过转义字符清洗（部分模型会追加 `\\n`）后若仍非空，
    说明模型输出了多个对象/尾随垃圾，拒收以免语义不清。
    """
    payload, end = json.JSONDecoder().raw_decode(text)
    remainder = text[end:].strip()
    if remainder:
        # 清掉纯转义序列的尾巴（\\n \\r \\t 的反复出现），再判断
        cleaned_remainder = re.sub(r"(?:\\[nrt])+", "", remainder).strip()
        if cleaned_remainder:
            raise ValueError("Model response must contain only one JSON object.")
    if not isinstance(payload, dict):
        raise ValueError("Model response must be a JSON object.")
    return cast(dict[str, Any], payload)


def parse_model_step(raw_response: str) -> ModelStep:
    """把模型原始输出解析成结构化的 `ModelStep`（text 协议）。

    校验三要素：
    - thought:      字符串（允许空串）
    - action:       非空字符串
    - action_input: 字典（允许空字典，但必须是对象而非数组/标量）
    任何格式问题都抛 ValueError，由 ReAct 循环吞入，记录为错误步并继续。
    """
    normalized = _strip_json_fence(raw_response)
    payload = _load_single_json_object(normalized)

    thought = payload.get("thought", "")
    action = payload.get("action")
    action_input = payload.get("action_input", {})
    if not isinstance(thought, str):
        raise ValueError("thought must be a string.")
    if not isinstance(action, str) or not action:
        raise ValueError("action must be a non-empty string.")
    if not isinstance(action_input, dict):
        raise ValueError("action_input must be a JSON object.")

    return ModelStep(
        thought=thought,
        action=action,
        action_input=cast(dict[str, Any], action_input),
        raw_response=raw_response,
    )


def _looks_like_json_value(value: str) -> bool:
    """Return whether a parameter value is worth trying to parse as JSON."""
    stripped = value.lstrip()
    if not stripped:
        return False
    return (
        stripped[0] in {'"', "{", "[", "-"}
        or stripped[0].isdigit()
        or stripped in {"true", "false", "null"}
    )


def _parse_reasoning_parameter_value(name: str, value: str) -> Any:
    """Parse one Qwen-style reasoning parameter conservatively."""
    stripped = value.strip()
    if name == "code":
        return stripped
    if _looks_like_json_value(stripped):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return stripped


def extract_reasoning_tool_call_draft(
    reasoning: str,
    allowed_tool_names: Collection[str],
) -> dict[str, Any] | None:
    """Extract the first complete Qwen-style tool-call draft from reasoning.

    This is deliberately best-effort and non-authoritative: it only helps the next
    model turn recover from a reasoning-only draft. The returned draft must never
    be executed by the runner.
    """
    if not reasoning:
        return None

    tool_call_match = _REASONING_TOOL_CALL_PATTERN.search(reasoning)
    if tool_call_match is None:
        return None

    tool_call_body = tool_call_match.group(1)
    function_match = _REASONING_FUNCTION_PATTERN.search(tool_call_body)
    if function_match is None:
        return None

    function_name = function_match.group(1).strip()
    if function_name not in allowed_tool_names:
        return None

    function_body = function_match.group(2)
    arguments: dict[str, Any] = {}
    for parameter_match in _REASONING_PARAMETER_PATTERN.finditer(function_body):
        parameter_name = parameter_match.group(1).strip()
        if not parameter_name:
            return None
        arguments[parameter_name] = _parse_reasoning_parameter_value(
            parameter_name,
            parameter_match.group(2),
        )

    unparsed_body = _REASONING_PARAMETER_PATTERN.sub("", function_body).strip()
    if unparsed_body:
        return None

    return {
        "function": function_name,
        "arguments": arguments,
        "source": "reasoning_content",
        "executed": False,
    }


def adapt_model_response(
    response: ModelResponse,
    *,
    protocol: ToolProtocol,
    turn_index: int,
    allowed_tool_names: Collection[str],
) -> ProtocolTurn:
    """把 `ModelResponse` 规整成统一的 `ProtocolTurn`。

    协议分叉：
    - native：直接使用 `response.tool_calls`；空时尝试从 reasoning 提取草稿
    - text：从 ```json fence 解析出 `ModelStep`，合成单个 `ModelToolCall`（id=`text:{turn}`）

    text 协议解析失败 → 返回 `ProtocolTurn(parse_error=...)`，driver 据此记一条 parse_error。
    """
    if protocol == "native":
        calls = tuple(response.tool_calls)
        if not calls:
            draft = extract_reasoning_tool_call_draft(
                response.reasoning_content,
                allowed_tool_names,
            )
            return ProtocolTurn(calls=(), thought=response.content, reasoning_draft=draft)
        return ProtocolTurn(calls=calls, thought=response.content)

    # text 协议
    try:
        step = parse_model_step(response.content)
    except ValueError as exc:
        return ProtocolTurn(calls=(), thought="", parse_error=str(exc))

    synthetic_call = ModelToolCall(
        id=f"text:{turn_index}",
        name=step.action,
        arguments=step.action_input,
    )
    return ProtocolTurn(calls=(synthetic_call,), thought=step.thought)
