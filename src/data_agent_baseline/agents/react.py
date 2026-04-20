"""ReAct Agent 核心循环。

流程概览（`ReActAgent.run`）：
  loop (step_index 从 1 到 max_steps):
    1. 拼接对话消息（system + task + 历次 assistant/observation）
    2. 调 model.complete 拿原始响应
    3. 解析 JSON（parse_model_step）→ 得到 thought/action/action_input
    4. 经 ToolRegistry 执行工具，得到 ToolExecutionResult
    5. 记录 StepRecord 到 state.steps
    6. 若 tool_result.is_terminal（即调用了 answer），保存 answer 并退出循环
    任何异常 → 记录一条 action="__error__" 的 StepRecord，进入下一步（不终止循环）

返回 `AgentRunResult`。未在 max_steps 内提交答案，会写入 failure_reason 但仍返回。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage, ModelStep
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ReActAgentConfig:
    """ReAct 循环运行参数。

    仅暴露 max_steps，其余行为（prompt / 工具集）通过构造参数注入，
    让本类对评测脚本/单元测试都易替换。
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
    """把模型原始输出解析成结构化的 `ModelStep`。

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
    - 线上：OpenAIModelAdapter + 默认 ToolRegistry
    - 离线测试：ScriptedModelAdapter + mock ToolRegistry
    """

    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        config: ReActAgentConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        # 缺省配置：max_steps=16
        self.config = config or ReActAgentConfig()
        # 允许外部替换 prompt 基座用于 A/B，默认用标准 ReAct 提示
        self.system_prompt = system_prompt or REACT_SYSTEM_PROMPT

    def _build_messages(self, task: PublicTask, state: AgentRuntimeState) -> list[ModelMessage]:
        """按 ReAct 对话规范拼装历史消息。

        顺序：
        1. system: 基础规则 + 工具表 + 响应示例
        2. user:   任务问题（首次提供）
        3. 对每个已完成步骤，依次追加：
           - assistant: 原始响应（保留 ```json fence，便于模型延续风格）
           - user:      observation（工具返回）
        """
        system_content = build_system_prompt(
            self.tools.describe_for_prompt(),
            system_prompt=self.system_prompt,
        )
        messages = [ModelMessage(role="system", content=system_content)]
        # 任务问题以 user 角色注入；不并入 system 是为了让模型把它当"当前请求"
        messages.append(ModelMessage(role="user", content=build_task_prompt(task)))
        for step in state.steps:
            # 回放 assistant 的原始响应，保留模型自己的格式偏好
            messages.append(ModelMessage(role="assistant", content=step.raw_response))
            # observation 以 user 身份回传，形成 ReAct 的观察循环
            messages.append(
                ModelMessage(role="user", content=build_observation_prompt(step.observation))
            )
        return messages

    def run(self, task: PublicTask) -> AgentRunResult:
        """驱动 ReAct 循环解决单个任务。

        循环不变式：
        - 每轮至少 append 一条 StepRecord（成功或 __error__），用于 trace 可观测
        - 调用 answer 工具会设置 `state.answer` 并立刻 break
        - 达到 max_steps 未 answer 时，写入 failure_reason 并正常返回
        """
        state = AgentRuntimeState()
        # step_index 从 1 起算：人类阅读 trace.json 时更直观
        for step_index in range(1, self.config.max_steps + 1):
            # 每轮都重新构造 messages：历史全量，代价 O(n) 但胜在实现简单
            raw_response = self.model.complete(self._build_messages(task, state))
            try:
                # 解析 + 执行工具：任一环节抛错都会跳进下面的 except
                model_step = parse_model_step(raw_response)
                tool_result = self.tools.execute(task, model_step.action, model_step.action_input)

                # observation 给下一轮模型看，需要包含 ok 标志、工具名和 content
                observation = {
                    "ok": tool_result.ok,
                    "tool": model_step.action,
                    "content": tool_result.content,
                }
                step_record = StepRecord(
                    step_index=step_index,
                    thought=model_step.thought,
                    action=model_step.action,
                    action_input=model_step.action_input,
                    raw_response=raw_response,
                    observation=observation,
                    ok=tool_result.ok,
                )
                state.steps.append(step_record)

                # 终止工具（目前仅 answer）：记录答案后跳出循环
                if tool_result.is_terminal:
                    state.answer = tool_result.answer
                    break
            except Exception as exc:
                # 典型原因：JSON 解析失败、未知工具名、工具入参不合法
                # 不终止循环，让模型在下一轮看到错误观察并自我修正
                observation = {
                    "ok": False,
                    "error": str(exc),
                }
                state.steps.append(
                    StepRecord(
                        step_index=step_index,
                        thought="",  # 异常分支拿不到原始 thought
                        action="__error__",  # 哨兵值：trace 上便于筛选
                        action_input={},
                        raw_response=raw_response,
                        observation=observation,
                        ok=False,
                    )
                )

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
