"""Data-agent package public exports.

把常用符号聚合到包根，上层代码可写 `from agents import ReActAgent`，
而不必深入每个子模块。具体实现分布在 `agent.py`、`runtime/`、`llm/` 与 `tools/` 中。
"""

from agents.agent import ReActAgent, ReActAgentConfig
from agents.llm import ModelAdapter, ModelMessage, ModelStep, OpenAIModelAdapter
from agents.prompts import (
    REACT_SYSTEM_PROMPT,
    build_observation_prompt,
    build_system_prompt,
    build_task_prompt,
)
from agents.runtime import AgentRunResult, AgentRuntimeState, StepRecord

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
