"""Agents 子包：ReAct 循环及模型/提示相关的公共导出。

把常用符号聚合到包根，上层代码可写 `from data_agent_baseline.agents import ReActAgent`，
而不必深入每个子模块。具体实现在 `model.py` / `prompt.py` / `react.py` / `runtime.py` 中。
"""

from data_agent_baseline.agents.model import (
    ModelAdapter,
    ModelMessage,
    ModelStep,
    OpenAIModelAdapter,
)
from data_agent_baseline.agents.prompt import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord

# 显式 __all__ 控制 `from ... import *` 的暴露面
__all__ = [
    "REACT_SYSTEM_PROMPT",
    "AgentRunResult",
    "AgentRuntimeState",
    "ModelAdapter",
    "ModelMessage",
    "ModelStep",
    "OpenAIModelAdapter",
    "ReActAgent",
    "ReActAgentConfig",
    "StepRecord",
    "build_observation_prompt",
    "build_system_prompt",
    "build_task_prompt",
]
