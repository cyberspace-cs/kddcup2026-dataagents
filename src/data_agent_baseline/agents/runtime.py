"""Agent 运行时状态与结果对象。

区分三个层次：
- `StepRecord`:       单步产物（frozen，表示 "这步已经完成"）
- `AgentRuntimeState`: 循环过程中的可变状态容器（`slots=True` 但非 frozen）
- `AgentRunResult`:    整次运行的对外快照（frozen，供 runner 序列化到 trace.json）

保持 frozen 与 mutable 的区分，是为了避免 ReAct 循环中误改已经落定的历史步骤。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable


@dataclass(frozen=True, slots=True)
class StepRecord:
    """一次 ReAct 循环的落定步骤。

    - step_index:    从 1 开始的步号（human-friendly）
    - thought:       模型输出的自然语言思考（JSON payload 里的 `thought`）
    - action:        被调用的工具名；错误分支使用哨兵值 `__error__`
    - action_input:  工具入参（来自模型输出的 `action_input`）
    - raw_response:  模型原始输出字符串（含 ```json fence），保留以便排错
    - observation:   工具执行后回填给下一轮的观察结果
    - ok:            本步是否成功（失败时 observation 里带 error 字段）
    """

    step_index: int
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str
    observation: dict[str, Any]
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        """asdict 把 dataclass 递归转成普通 dict（用于 JSON 序列化）。"""
        return asdict(self)


@dataclass(slots=True)
class AgentRuntimeState:
    """ReAct 循环的可变状态。

    - steps:           已完成的步骤列表（按完成顺序）
    - answer:          Agent 调用 `answer` 工具后填入；有值即视为成功终止
    - failure_reason:  异常终止或 max_steps 耗尽时写入
    非 frozen 是刻意为之：循环内需要不断 append/写入字段。
    """

    steps: list[StepRecord] = field(default_factory=list)
    answer: AnswerTable | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """单个任务整轮 ReAct 运行的最终产物。

    由 `ReActAgent.run` 构造；runner 会把它序列化为 trace.json。
    """

    task_id: str
    answer: AnswerTable | None
    steps: list[StepRecord]
    failure_reason: str | None

    @property
    def succeeded(self) -> bool:
        """成功的充要条件：`answer` 已提交且没有记录失败原因。

        注意这里的 "成功" 仅指 Agent 正常走到终止工具，不代表答案正确。
        """
        return self.answer is not None and self.failure_reason is None

    def to_dict(self) -> dict[str, Any]:
        """序列化为 runner 可直接落盘的 dict。"""
        return {
            "task_id": self.task_id,
            "answer": self.answer.to_dict() if self.answer is not None else None,
            "steps": [step.to_dict() for step in self.steps],
            "failure_reason": self.failure_reason,
            "succeeded": self.succeeded,
        }
