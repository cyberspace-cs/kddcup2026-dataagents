"""ReAct Agent 核心循环。

支持两种模型协议（通过 `ReActAgent(protocol=...)` 切换）：

- **text**：模型以 ```json fence 返回 `thought/action/action_input`；
  ReAct 层靠 `parse_model_step` 解析单步。
- **native**：模型以原生 OpenAI `tool_calls` 返回（一轮可含多个工具调用）；
  ReAct 层把每个 tool_call 映射成 `StepRecord`，同一轮共享 `turn_index`。

流程概览（`ReActAgent.run`）：
  loop (turn_index 从 1 到 max_steps):
    1. 拼接对话消息（`_build_messages_text` / `_build_messages_native`）
    2. 调 model.complete 拿 `ModelResponse`
    3. 把 response 规整成 `ModelToolCall` 列表（text 模式靠 parse；native 直接拿）
    4. 依次执行每个 tool_call，追加 `StepRecord`
    5. 任一 tool_call 命中 terminal（answer）则短路退出本轮并终结循环
    任何异常 → 记录一条 action="__error__" 的 StepRecord，进入下一轮（不终止循环）

返回 `AgentRunResult`。未在 max_steps 内提交答案，会写入 failure_reason 但仍返回。
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from data_agent_baseline.agents.model import (
    ModelAdapter,
    ModelMessage,
    ModelResponse,
    ModelStep,
    ModelToolCall,
    TokenUsage,
)
from data_agent_baseline.agents.prompt import (
    build_native_system_prompt,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry

# 支持的协议字面量：在类型层阻止拼错
ToolProtocol = Literal["native", "text"]

# 每次 state.steps 发生变化时回调；供 runner 做增量 trace 落盘（防子进程被硬 kill 时 trace 全丢）
StepCallback = Callable[[AgentRuntimeState], None]


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    """ReAct 循环运行参数。

    仅暴露 max_steps，其余行为（prompt / 工具集）通过构造参数注入，
    让本类对评测脚本/单元测试都易替换。

    注意：`max_steps` 在两种协议下的语义不同：
    - text 模式：每轮模型回复恰好对应一次工具调用 → max_steps ≈ 工具调用次数
    - native 模式：每轮模型回复可能并行发出多个工具调用 → max_steps 是**模型轮数**
      上限，实际工具调用次数可能更高
    """

    max_steps: int = 16


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


def _load_single_json_object(text: str) -> dict[str, object]:
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
    return payload


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
        action_input=action_input,
        raw_response=raw_response,
    )


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
    ) -> None:
        self.model = model
        self.tools = tools
        # 缺省配置：max_steps=16
        self.config = config or ReActAgentConfig()
        self.protocol = protocol
        # 允许外部替换 prompt 基座用于 A/B，缺省根据协议选合适的模板
        self.system_prompt = system_prompt
        # 每步回调：runner 在子进程里会注入 "把 state 落到 trace.json" 的 writer，
        # 这样即便子进程因超时被硬 kill，磁盘上仍保留最后一次 append 后的部分 trace
        self.step_callback = step_callback

    # ----------------------------- 步骤落定辅助 -----------------------------

    def _record_step(self, state: AgentRuntimeState, record: StepRecord) -> None:
        """统一的"落定一步"通道：append 到 state 并通知外部 callback。

        callback 的异常绝对不允许污染 Agent 主循环——任何失败都吞掉。
        """
        state.steps.append(record)
        if self.step_callback is None:
            return
        # 观测回调挂掉不应该炸任务；静默忽略
        with contextlib.suppress(Exception):
            self.step_callback(state)

    # ----------------------------- 消息构造 -----------------------------

    def _system_message(self) -> ModelMessage:
        """按协议选合适的 system prompt。"""
        if self.protocol == "native":
            content = build_native_system_prompt(self.system_prompt)
        else:
            content = build_system_prompt(
                self.tools.describe_for_prompt(),
                system_prompt=self.system_prompt,
            )
        return ModelMessage(role="system", content=content)

    def _build_messages_text(
        self, task: PublicTask, state: AgentRuntimeState
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
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))
        for step in state.steps:
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        return messages

    def _build_messages_native(
        self, task: PublicTask, state: AgentRuntimeState
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
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))

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
                        content=build_observation_prompt(first_step.observation),
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

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        """按协议分派具体的消息构造策略。"""
        if self.protocol == "native":
            return self._build_messages_native(task, state)
        return self._build_messages_text(task, state)

    # ----------------------------- 主循环 -----------------------------

    def _resolve_tool_calls(
        self, response: ModelResponse, turn_index: int
    ) -> tuple[list[ModelToolCall], str]:
        """把 `ModelResponse` 规整成 `(tool_calls, thought)` 对。

        返回值语义：
        - tool_calls 非空：可以执行；thought 作为展示字段
        - tool_calls 为空：上层应记一条 __error__ 步（调用方自己构造）
        该函数本身会把文本协议里的 fence 解析错误通过抛 ValueError 向上传递。

        协议分叉：
        - native 模式：只看 `response.tool_calls`；空则直接返回空，让上层记 __error__
        - text 模式：永远尝试 parse_model_step，把 ```json fence 解析成合成 ModelToolCall
        """
        if self.protocol == "native":
            return list(response.tool_calls), response.content

        # text 协议：把 ```json fence 解析成 ModelStep
        step = parse_model_step(response.content)
        synthetic_id = f"text:{turn_index}"
        synthetic_call = ModelToolCall(
            id=synthetic_id,
            name=step.action,
            arguments=step.action_input,
        )
        return [synthetic_call], step.thought

    def run(self, task: PublicTask) -> AgentRunResult:
        """驱动 ReAct 循环解决单个任务。

        循环不变式：
        - 每轮至少 append 一条 StepRecord（成功 / 工具异常 / 解析 __error__），用于 trace 可观测
        - 调用 answer 工具会设置 `state.answer` 并立刻 break
        - 达到 max_steps 未 answer 时，写入 failure_reason 并正常返回
        """
        state = AgentRuntimeState()
        step_counter = 0
        terminated = False

        for turn_index in range(1, self.config.max_steps + 1):
            # ---- 1. 调用模型 ----
            try:
                response = self.model.complete(self._build_messages(task, state))
            except Exception as exc:
                # 网络错误 / API 失败：记一条 __error__，让循环在下一轮重试
                step_counter += 1
                self._record_step(
                    state,
                    StepRecord(
                        step_index=step_counter,
                        turn_index=turn_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response="",
                        observation={"ok": False, "error": f"model.complete failed: {exc}"},
                        ok=False,
                        tool_call_id=None,
                        raw_tool_calls=None,
                    ),
                )
                continue

            # ---- 2. 规整响应为 tool_calls ----
            try:
                tool_calls, thought = self._resolve_tool_calls(response, turn_index)
            except Exception as exc:
                # text 协议的 fence 解析失败，或 native 协议下缺 tool_calls
                step_counter += 1
                self._record_step(
                    state,
                    StepRecord(
                        step_index=step_counter,
                        turn_index=turn_index,
                        thought="",
                        action="__error__",
                        action_input={},
                        raw_response=response.raw_response,
                        observation={"ok": False, "error": str(exc)},
                        ok=False,
                        tool_call_id=None,
                        raw_tool_calls=None,
                        reasoning=response.reasoning_content,
                        usage=response.usage,
                    ),
                )
                continue

            if not tool_calls:
                # native 协议下模型回了空 tool_calls（纯文本 content）：
                # 给一条 observation 强制纠偏，下一轮让模型重试
                step_counter += 1
                self._record_step(
                    state,
                    StepRecord(
                        step_index=step_counter,
                        turn_index=turn_index,
                        thought=thought,
                        action="__error__",
                        action_input={},
                        raw_response=response.raw_response,
                        observation={
                            "ok": False,
                            "error": (
                                "Your previous turn did not produce an executable tool call. "
                                "Any tool call written inside reasoning/thinking text was ignored, "
                                "even if it looked syntactically complete. In this turn: do not "
                                "draft <tool_call>, <function=...>, JSON actions, or executable "
                                "code in reasoning. If you already decided which tool to call last "
                                "turn, emit that call now as a structured tool call. If final answer "
                                "is ready, call answer directly. You must call a tool."
                            ),
                        },
                        ok=False,
                        tool_call_id=None,
                        raw_tool_calls=None,
                        reasoning=response.reasoning_content,
                        usage=response.usage,
                    ),
                )
                continue

            # ---- 3. 依次执行 tool_calls ----
            # text 协议下 StepRecord 不承载原生调用 id / 原始 tool_calls（语义上它们都是
            # 合成的占位值），保持 None 可让 trace.json 读者清楚区分两种协议。
            is_native = self.protocol == "native"
            for call_idx, tool_call in enumerate(tool_calls):
                step_counter += 1
                is_lead = call_idx == 0
                # 仅首条 StepRecord 带 raw_tool_calls（用于 native 模式 message 回放）
                if is_native:
                    step_tool_call_id: str | None = tool_call.id or None
                    step_raw_tool_calls: list[dict[str, Any]] | None = (
                        response.raw_tool_calls if is_lead else None
                    )
                else:
                    step_tool_call_id = None
                    step_raw_tool_calls = None
                # usage / reasoning 与 raw_tool_calls 同约定：仅 lead 承载，避免双计
                step_reasoning = response.reasoning_content if is_lead else ""
                step_usage = response.usage if is_lead else TokenUsage()
                try:
                    tool_result = self.tools.execute(task, tool_call.name, tool_call.arguments)
                    observation: dict[str, Any] = {
                        "ok": tool_result.ok,
                        "tool": tool_call.name,
                        "content": tool_result.content,
                    }
                    record = StepRecord(
                        step_index=step_counter,
                        turn_index=turn_index,
                        thought=thought,
                        action=tool_call.name,
                        action_input=tool_call.arguments,
                        raw_response=response.raw_response,
                        observation=observation,
                        ok=tool_result.ok,
                        tool_call_id=step_tool_call_id,
                        raw_tool_calls=step_raw_tool_calls,
                        reasoning=step_reasoning,
                        usage=step_usage,
                    )
                    # 终止工具（目前仅 answer）：记录答案后短路，**跳过同 turn 后续 tool_calls**
                    if tool_result.is_terminal:
                        state.answer = tool_result.answer
                    self._record_step(state, record)
                    if tool_result.is_terminal:
                        terminated = True
                        break
                except Exception as exc:
                    # 工具执行异常：未知工具名 / 参数非法 / handler 抛错等
                    self._record_step(
                        state,
                        StepRecord(
                            step_index=step_counter,
                            turn_index=turn_index,
                            thought=thought,
                            action="__error__",
                            action_input=tool_call.arguments,
                            raw_response=response.raw_response,
                            observation={"ok": False, "error": str(exc)},
                            ok=False,
                            tool_call_id=step_tool_call_id,
                            raw_tool_calls=step_raw_tool_calls,
                            reasoning=step_reasoning,
                            usage=step_usage,
                        ),
                    )

            if terminated:
                break

        # 循环正常结束但没拿到 answer：判为 max_steps 耗尽
        if state.answer is None and state.failure_reason is None:
            state.failure_reason = "Agent did not submit an answer within max_steps."

        return AgentRunResult(
            task_id=task.task_id,
            answer=state.answer,
            # list(...) 拷贝切断 state.steps 与返回值之间的引用，防止后续修改串扰
            steps=list(state.steps),
            failure_reason=state.failure_reason,
        )
