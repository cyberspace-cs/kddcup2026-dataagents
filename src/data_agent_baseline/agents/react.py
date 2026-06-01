"""ReAct Agent 主循环驱动器。

本模块只剩三件事：循环预算（`ReActAgentConfig`）、消息构造（按协议拼 system/user/
assistant/tool 序列）、和驱动主循环本身（调模型、把响应转成事件、记账、判终止）。
具体的协议解析、工具派发、事件序列化分别下沉到：
- `protocol.adapt_model_response`：text/native → 统一 `ProtocolTurn`
- `dispatcher.dispatch_tool_call`：执行单个 tool_call，返回 `(result, should_terminate)`
- `recorder.record_step_event`：把事件序列化成 `StepRecord` 并通知 step_callback

为兼容现有 import（`from data_agent_baseline.agents.react import parse_model_step,
_extract_reasoning_tool_call_draft`），保留薄薄的再导出层。
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from data_agent_baseline.agents.answer_verifier import AnswerVerifier
from data_agent_baseline.agents.dispatcher import dispatch_tool_call
from data_agent_baseline.agents.model import (
    ModelAdapter,
    ModelMessage,
    ModelResponse,
    TokenUsage,
    ToolCallParseError,
)
from data_agent_baseline.agents.prompt import (
    build_native_system_prompt,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.protocol import (
    ToolProtocol,
    adapt_model_response,
    parse_model_step,
)
from data_agent_baseline.agents.protocol import (
    extract_reasoning_tool_call_draft as _extract_reasoning_tool_call_draft,
)
from data_agent_baseline.agents.recorder import (
    EmptyToolCallsEvent,
    ModelErrorEvent,
    ObservationEvent,
    ParseErrorEvent,
    StepCallback,
    ToolErrorEvent,
    record_step_event,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.handlers import (
    build_answer_table_without_side_effects,
    cleanup_answer_artifacts,
)
from data_agent_baseline.tools.registry import ToolExecutionResult, ToolRegistry

logger = logging.getLogger(__name__)

# 公开再导出（向后兼容现有 tests/agents/__init__ 的 import 路径）
__all__ = [
    "ReActAgent",
    "ReActAgentConfig",
    "StepCallback",
    "ToolProtocol",
    "_extract_reasoning_tool_call_draft",
    "parse_model_step",
]

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def _find_first_video(context_dir: Path) -> Path | None:
    """Recurse into ``context/`` and return the first video file (sorted for determinism)."""
    candidates = sorted(
        p for p in context_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if len(candidates) > 1:
        logger.warning(
            "Multiple videos in %s; using %s, ignoring %s",
            context_dir.name,
            candidates[0].name,
            [c.name for c in candidates[1:]],
        )
    return candidates[0] if candidates else None


_VIDEO_MAX_BYTES = 100 * 1024 * 1024  # 100 MB


def _build_initial_user_content(task: PublicTask) -> str | list[dict[str, Any]]:
    """Build the initial user message content, attaching a video if one exists."""
    text = build_task_prompt(task)
    video_path = _find_first_video(task.context_dir)
    if video_path is None:
        return text
    size = video_path.stat().st_size
    if size > _VIDEO_MAX_BYTES:
        logger.warning(
            "Video %s skipped: %d bytes exceeds %d byte limit",
            video_path.name,
            size,
            _VIDEO_MAX_BYTES,
        )
        return text
    mime_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
    video_b64 = base64.standard_b64encode(video_path.read_bytes()).decode("ascii")
    data_url = f"data:{mime_type};base64,{video_b64}"
    return [
        {"type": "text", "text": text},
        {"type": "video_url", "video_url": {"url": data_url}},
    ]


def _serialize_message(message: ModelMessage) -> dict[str, Any]:
    """把 `ModelMessage` 序列化成 JSON 友好的 dict（保留 role/content/tool_calls/tool_call_id）。

    与 `model._message_to_payload` 的差异：本函数为 trace.json 落盘服务，保留所有
    非 None 字段，便于离线调试时完整还原"当时发给 LLM 的请求"；payload 那边则是
    专门给 OpenAI SDK 看的，会按角色裁剪字段。
    """
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


def _serialize_response(response: ModelResponse) -> dict[str, Any]:
    """把 `ModelResponse` 序列化成 JSON 友好的 dict。

    保留 content / tool_calls / raw_response / reasoning_content / usage / latency_ms：
    覆盖离线回放与性能分析所需的全部字段。`tool_calls` 中的每项展开成
    `{id, name, arguments}`，与 `ModelToolCall` 一致。
    """
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


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    """ReAct 循环运行参数。

    仅暴露循环预算，其余行为（prompt / 工具集）通过构造参数注入，
    让本类对评测脚本/单元测试都易替换。

    注意：`max_steps` 在两种协议下的语义不同：
    - text 模式：每轮模型回复恰好对应一次工具调用 → max_steps ≈ 工具调用次数
    - native 模式：每轮非空 tool_calls 可能并行发出多个工具调用 → max_steps 是
      **有效模型轮数**上限，实际工具调用次数可能更高

    `max_empty_tool_call_retries` 只用于 native 模式下 response.tool_calls 为空的情况。
    这类响应只产生纠偏 observation，不应消耗业务 max_steps；单独设上限避免无限重试。
    """

    max_steps: int = 16
    max_empty_tool_call_retries: int = 8
    enable_preact: bool = False


class ReActAgent:
    """ReAct 范式的 Agent 实例。

    构造时注入 model 和 tools 允许在不同场景复用同一套循环逻辑：
    - 线上（native）：NativeToolsOpenAIAdapter + 默认 ToolRegistry
    - 线上（text）：OpenAIModelAdapter + 默认 ToolRegistry（降级路径）
    - 离线测试：Scripted[Native]ModelAdapter + mock ToolRegistry
    """

    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
        protocol: ToolProtocol = "native",
        step_callback: StepCallback | None = None,
        answer_verifier: AnswerVerifier | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        # 缺省配置：max_steps=16
        self.config = config or ReActAgentConfig()
        self.protocol: ToolProtocol = protocol
        # 允许外部替换 prompt 基座用于 A/B，缺省根据协议选合适的模板
        self.system_prompt = system_prompt
        # 每步回调：runner 在子进程里会注入 "把 state 落到 trace.json" 的 writer，
        # 这样即便子进程因超时被硬 kill，磁盘上仍保留最后一次 append 后的部分 trace
        self.step_callback = step_callback
        self.answer_verifier = answer_verifier

    # ----------------------------- 消息构造 -----------------------------

    def _system_message(self) -> ModelMessage:
        """按协议选合适的 system prompt。

        native 协议下额外读取 `model.enable_thinking`：仅在显式 False 时
        会跳过 `_NATIVE_OUTPUT_PROTOCOL` 段（参见 `build_native_system_prompt`）。
        `getattr` + isinstance 让 ModelAdapter Protocol 可以无该属性，未来
        加入非 bool 字段也不会被误读。
        """
        if self.protocol == "native":
            raw_enable_thinking = getattr(self.model, "enable_thinking", None)
            enable_thinking = raw_enable_thinking if isinstance(raw_enable_thinking, bool) else None
            content = build_native_system_prompt(
                self.system_prompt,
                enable_thinking=enable_thinking,
            )
        else:
            content = build_system_prompt(
                self.tools.describe_for_prompt(),
                system_prompt=self.system_prompt,
            )
        return ModelMessage(role="system", content=content)

    def _build_messages_text(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        initial_content: str | list[dict[str, Any]],
    ) -> list[ModelMessage]:
        """text 协议的消息拼装：回放 assistant 原始响应 + user observation。

        顺序：
        1. system: 基础规则 + 工具表 + 响应示例
        2. user:   任务问题（首次提供）
        3. 对每个已完成步骤，依次追加：
           - assistant: 原始响应（保留 ```json fence，便于模型延续风格）
           - user:      observation（工具返回）
        """
        messages = [self._system_message()]
        messages.append(ModelMessage(role="user", content=initial_content))
        if self.config.enable_preact and not state.steps:
            messages.append(
                ModelMessage(
                    role="user",
                    content=(
                        "Before acting, outline a brief plan: "
                        "(1) which files/tables contain the needed data, "
                        "(2) what joins/filters/aggregations are needed, "
                        "(3) what the final answer columns should be. "
                        "Then start with inspect_files."
                    ),
                )
            )
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(
                    role="user",
                    content=build_observation_prompt(step.observation, ok=step.ok),
                )
            )
        return messages

    def _build_messages_native(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        initial_content: str | list[dict[str, Any]],
    ) -> list[ModelMessage]:
        """native 协议的消息拼装：把同一 turn 的并行 tool_calls 合并成一条 assistant 消息。

        结构：
        1. system
        2. user(task)
        3. 对每个 turn：
           - assistant(content=thought, tool_calls=raw_tool_calls)  ← 一条
           - tool(tool_call_id=..., content=observation_json)       ← N 条（N = 并行数）
        """
        messages = [self._system_message()]
        messages.append(ModelMessage(role="user", content=initial_content))

        if self.config.enable_preact and not state.steps:
            messages.append(
                ModelMessage(
                    role="user",
                    content=(
                        "Before acting, outline a brief plan: "
                        "(1) which files/tables contain the needed data, "
                        "(2) what joins/filters/aggregations are needed, "
                        "(3) what the final answer columns should be. "
                        "Then start with inspect_files."
                    ),
                )
            )

        # 按 turn_index 分组：同一轮的所有 StepRecord 共享一个 assistant 消息
        steps_by_turn: dict[int, list[StepRecord]] = {}
        for step in state.steps:
            steps_by_turn.setdefault(step.turn_index, []).append(step)

        for turn_index in sorted(steps_by_turn):
            steps_in_turn = steps_by_turn[turn_index]
            first_step = steps_in_turn[0]

            # __error__ 步记录的是"模型输出无法解析"类错误（不含 tool_calls）；
            # 此时回放为纯文本 assistant + 对应 user 反馈，让模型能看到纠错观察
            if all(step.action == "__error__" for step in steps_in_turn):
                # 单条 __error__：回放 assistant 原文 + user 观察
                messages.append(ModelMessage(role="assistant", content=first_step.raw_response))
                messages.append(
                    ModelMessage(
                        role="user",
                        content=build_observation_prompt(first_step.observation, ok=first_step.ok),
                    )
                )
                continue

            # 正常轮次：assistant 带上原始 tool_calls；每个工具结果走 tool 消息
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

    def _build_messages(
        self,
        task: PublicTask,
        state: AgentRuntimeState,
        initial_content: str | list[dict[str, Any]],
    ) -> list[ModelMessage]:
        """按协议分派具体的消息构造策略。"""
        if self.protocol == "native":
            return self._build_messages_native(task, state, initial_content)
        return self._build_messages_text(task, state, initial_content)

    # ----------------------------- 主循环 -----------------------------

    def run(self, task: PublicTask) -> AgentRunResult:
        """驱动 ReAct 循环解决单个任务。

        循环不变式：
        - 每轮至少 append 一条 StepRecord（成功 / 工具异常 / 解析 __error__），用于 trace 可观测
        - 调用 answer 工具会设置 `state.answer` 并立刻 break
        - native 空 tool_calls 只消耗 empty retry budget，不消耗业务 max_steps
        - 达到 max_steps 未 answer 时，写入 failure_reason 并正常返回
        """
        state = AgentRuntimeState()
        initial_content = _build_initial_user_content(task)
        step_counter = 0
        effective_turns = 0
        empty_tool_call_retries = 0
        turn_index = 0
        terminated = False
        budget_warning_fired = False
        last_step_retries = 0
        verifier_rejections_used = 0
        fallback_answer: AnswerTable | None = None
        fallback_answer_arguments: dict[str, Any] | None = None

        while effective_turns < self.config.max_steps:
            turn_index += 1

            # ---- 1. 调用模型 ----
            messages = self._build_messages(task, state, initial_content)

            budget_ratio = (
                effective_turns / self.config.max_steps if self.config.max_steps > 0 else 0
            )
            remaining = self.config.max_steps - effective_turns
            if budget_ratio >= 0.90:
                messages.append(
                    ModelMessage(
                        role="user",
                        content=(
                            f"CRITICAL: {remaining} step(s) left. "
                            "You MUST call `answer` NOW with whatever result you have. "
                            "Failure to answer = score 0."
                        ),
                    )
                )
            elif not budget_warning_fired and budget_ratio >= 0.70:
                budget_warning_fired = True
                messages.append(
                    ModelMessage(
                        role="user",
                        content=(
                            f"BUDGET WARNING: {effective_turns}/{self.config.max_steps} steps used, "
                            f"{remaining} remaining. "
                            "If you have computed ANY answer, call `answer` NOW — "
                            "an imperfect answer scores higher than no answer. "
                            "If not, do ONE final computation and submit."
                        ),
                    )
                )

            # 提前把 messages 序列化好：即便 model.complete 抛错，也能把当时发出的
            # prompt_messages 落进 ModelErrorEvent，方便复盘"哪条请求把 LLM 炸了"
            serialized_prompt = [_serialize_message(message) for message in messages]
            try:
                response = self.model.complete(messages)
            except ToolCallParseError as exc:
                # tool_call arguments JSON 解析失败：携带部分响应，保留
                # raw_tool_calls / usage / reasoning 供 trace 和回放
                partial = exc.partial_response
                effective_turns += 1
                step_counter += 1
                record_step_event(
                    state,
                    ParseErrorEvent(
                        error=f"model.complete failed: {exc}",
                        raw_response=partial.raw_response,
                        reasoning=partial.reasoning_content,
                        usage=partial.usage,
                        prompt_messages=serialized_prompt,
                        response_message=_serialize_response(partial),
                    ),
                    step_index=step_counter,
                    turn_index=turn_index,
                    callback=self.step_callback,
                )
                continue
            except Exception as exc:
                # 网络错误 / API 失败：记一条 model_error，让循环在下一轮重试
                effective_turns += 1
                step_counter += 1
                record_step_event(
                    state,
                    ModelErrorEvent(
                        error=f"model.complete failed: {exc}",
                        prompt_messages=serialized_prompt,
                    ),
                    step_index=step_counter,
                    turn_index=turn_index,
                    callback=self.step_callback,
                )
                continue

            # ---- 2. 协议规整 ----
            serialized_response = _serialize_response(response)
            turn = adapt_model_response(
                response,
                protocol=self.protocol,
                turn_index=turn_index,
                allowed_tool_names=self.tools.definitions,
            )

            if turn.parse_error is not None:
                # text 协议的 fence 解析失败
                effective_turns += 1
                step_counter += 1
                record_step_event(
                    state,
                    ParseErrorEvent(
                        error=turn.parse_error,
                        raw_response=response.raw_response,
                        reasoning=response.reasoning_content,
                        usage=response.usage,
                        prompt_messages=serialized_prompt,
                        response_message=serialized_response,
                    ),
                    step_index=step_counter,
                    turn_index=turn_index,
                    callback=self.step_callback,
                )
                continue

            if not turn.calls:
                # native 协议下模型回了空 tool_calls：记纠偏 observation，独立 retry budget
                empty_tool_call_retries += 1
                step_counter += 1
                record_step_event(
                    state,
                    EmptyToolCallsEvent(
                        thought=turn.thought,
                        raw_response=response.raw_response,
                        reasoning=response.reasoning_content,
                        usage=response.usage,
                        drafted_tool_call=turn.reasoning_draft,
                        prompt_messages=serialized_prompt,
                        response_message=serialized_response,
                    ),
                    step_index=step_counter,
                    turn_index=turn_index,
                    callback=self.step_callback,
                )
                if empty_tool_call_retries >= self.config.max_empty_tool_call_retries:
                    state.failure_reason = (
                        "Agent exceeded max_empty_tool_call_retries "
                        f"({self.config.max_empty_tool_call_retries}) without an "
                        "executable tool call."
                    )
                    terminated = True
                    break
                continue

            # ---- 3. 依次派发 tool_calls ----

            # Last-step guardrail: on the final step of a long-running task
            # (past the 90% CRITICAL threshold), block non-answer tools and
            # give the model one free retry to call `answer`. The blocked
            # turn does NOT consume effective_turns budget.
            if (
                remaining <= 1
                and budget_ratio >= 0.90
                and last_step_retries < 2
                and not any(c.name == "answer" for c in turn.calls)
            ):
                last_step_retries += 1
                step_counter += 1
                record_step_event(
                    state,
                    ToolErrorEvent(
                        thought=turn.thought,
                        arguments={},
                        error=(
                            "BLOCKED: final step must be an `answer` call. "
                            "Call `answer` with your best result immediately."
                        ),
                        raw_response=response.raw_response,
                        tool_call_id=turn.calls[0].id if turn.calls else None,
                        raw_tool_calls=response.raw_tool_calls,
                        reasoning=response.reasoning_content,
                        usage=response.usage,
                        prompt_messages=serialized_prompt,
                        response_message=serialized_response,
                    ),
                    step_index=step_counter,
                    turn_index=turn_index,
                    callback=self.step_callback,
                )
                continue

            effective_turns += 1
            # text 协议下 StepRecord 不承载原生调用 id / 原始 tool_calls（语义上它们都是
            # 合成的占位值），保持 None 可让 trace.json 读者清楚区分两种协议。
            is_native = self.protocol == "native"
            # recovered tool call（从 content/reasoning 解析）没有原始 raw_tool_calls，
            # 需要合成以便 native 协议 message 回放时 assistant.tool_calls 与 tool.tool_call_id 匹配
            effective_raw_tool_calls: list[dict[str, object]] = response.raw_tool_calls or []
            if is_native and not effective_raw_tool_calls and turn.calls:
                effective_raw_tool_calls = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        },
                    }
                    for call in turn.calls
                ]
            replay_raw_tool_calls = effective_raw_tool_calls
            verifier_rejected_this_turn = False
            turn_first_state_index = len(state.steps)
            for call_idx, tool_call in enumerate(turn.calls):
                step_counter += 1
                is_lead = call_idx == 0
                # 仅首条 StepRecord 带 raw_tool_calls（用于 native 模式 message 回放）
                if is_native:
                    step_tool_call_id: str | None = tool_call.id or None
                    step_raw_tool_calls: list[dict[str, object]] | None = (
                        replay_raw_tool_calls if is_lead else None
                    )
                else:
                    step_tool_call_id = None
                    step_raw_tool_calls = None
                # usage / reasoning / prompt_messages / response_message 与 raw_tool_calls
                # 同约定：仅 lead 承载，避免双计 + 节省体积
                step_reasoning = response.reasoning_content if is_lead else ""
                step_usage = response.usage if is_lead else TokenUsage()
                step_prompt_messages = serialized_prompt if is_lead else None
                step_response_message = serialized_response if is_lead else None

                # 终止工具预验证：在 handler 执行（会删 CSV artifact）之前跑 verifier
                defn = self.tools.definitions.get(tool_call.name)
                if (
                    self.answer_verifier is not None
                    and defn is not None
                    and defn.is_terminal
                    and verifier_rejections_used < self.answer_verifier.max_rejections
                    # On the last remaining step, rejecting would leave no model turn to fix it.
                    and remaining > 1
                ):
                    rejection = self.answer_verifier.verify(task.question, tool_call.arguments)
                    if rejection is not None:
                        verifier_rejections_used += 1
                        candidate = build_answer_table_without_side_effects(
                            task,
                            tool_call.arguments,
                        )
                        if candidate is not None and candidate.columns and candidate.rows:
                            fallback_answer = candidate
                            fallback_answer_arguments = dict(tool_call.arguments)
                        if is_native:
                            # Native replay requires one tool message for every assistant
                            # tool_call. If a rejection cuts the turn short, omit later
                            # unexecuted tool calls from replay metadata.
                            replay_raw_tool_calls = effective_raw_tool_calls[: call_idx + 1]
                            if is_lead:
                                step_raw_tool_calls = replay_raw_tool_calls
                            elif turn_first_state_index < len(state.steps):
                                lead_step = state.steps[turn_first_state_index]
                                state.steps[turn_first_state_index] = replace(
                                    lead_step,
                                    raw_tool_calls=replay_raw_tool_calls,
                                )
                        result = ToolExecutionResult(
                            ok=True,
                            content={"answer_rejected": rejection},
                        )
                        record_step_event(
                            state,
                            ObservationEvent(
                                thought=turn.thought,
                                tool_name=tool_call.name,
                                arguments=tool_call.arguments,
                                content=result.content,
                                ok=result.ok,
                                raw_response=response.raw_response,
                                tool_call_id=step_tool_call_id,
                                raw_tool_calls=step_raw_tool_calls,
                                reasoning=step_reasoning,
                                usage=step_usage,
                                prompt_messages=step_prompt_messages,
                                response_message=step_response_message,
                            ),
                            step_index=step_counter,
                            turn_index=turn_index,
                            callback=self.step_callback,
                        )
                        verifier_rejected_this_turn = True
                        break

                try:
                    result, should_terminate = dispatch_tool_call(task, tool_call, self.tools)
                except Exception as exc:
                    record_step_event(
                        state,
                        ToolErrorEvent(
                            thought=turn.thought,
                            arguments=tool_call.arguments,
                            error=str(exc),
                            raw_response=response.raw_response,
                            tool_call_id=step_tool_call_id,
                            raw_tool_calls=step_raw_tool_calls,
                            reasoning=step_reasoning,
                            usage=step_usage,
                            prompt_messages=step_prompt_messages,
                            response_message=step_response_message,
                        ),
                        step_index=step_counter,
                        turn_index=turn_index,
                        callback=self.step_callback,
                    )
                    continue

                if should_terminate:
                    state.answer = result.answer
                record_step_event(
                    state,
                    ObservationEvent(
                        thought=turn.thought,
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                        content=result.content,
                        ok=result.ok,
                        raw_response=response.raw_response,
                        tool_call_id=step_tool_call_id,
                        raw_tool_calls=step_raw_tool_calls,
                        reasoning=step_reasoning,
                        usage=step_usage,
                        prompt_messages=step_prompt_messages,
                        response_message=step_response_message,
                    ),
                    step_index=step_counter,
                    turn_index=turn_index,
                    callback=self.step_callback,
                )
                if should_terminate:
                    terminated = True
                    break

            if terminated:
                break
            if verifier_rejected_this_turn:
                continue

        if not terminated and effective_turns >= self.config.max_steps:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        # 循环正常结束但没拿到 answer：判为 max_steps 耗尽
        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        # Fallback: use the latest valid verifier-rejected answer rather than scoring 0.
        if state.answer is None and fallback_answer is not None:
            if fallback_answer_arguments is not None:
                cleanup_answer_artifacts(task, fallback_answer_arguments)
            state.answer = fallback_answer
            state.failure_reason = None

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            # list(...) 拷贝切断 state.steps 与返回值之间的引用，防止后续修改串扰
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
