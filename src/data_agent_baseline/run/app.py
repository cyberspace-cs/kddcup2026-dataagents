"""App 工厂：把 dataset/registry/model adapter/agent 的 wiring 收敛到一处。

`runner.py` 不再直接持有具体的 `OpenAIModelAdapter` / `ToolRegistry` / `ReActAgent`
依赖；它只调用 `build_application(config)` 拿到 `AgentApp`，再用 `app.new_agent()` 驱动循环。

线上路径
    runner 子进程在 key 固定后调 `build_application(pinned_config)`，构造完整 AgentApp。
    `AgentApp` 不会被 pickle 跨进程——只有 `AppConfig` 跨进程，`AgentApp` 始终在子进程内构造。

测试路径
    pytest 通过 `build_application(config, model=ScriptedModelAdapter(...), tools=mock_registry)`
    跳过真实 OpenAI 客户端构造；override 完全保留原有 model/tools 注入语义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_agent_baseline.agents.model import (
    ModelAdapter,
    NativeToolsOpenAIAdapter,
    OpenAIModelAdapter,
)
from data_agent_baseline.agents.protocol import ToolProtocol
from data_agent_baseline.agents.rate_limit import RateLimitedAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.agents.recorder import StepCallback
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AgentConfig, AppConfig
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry

# Qwen3 model card Best Practices 推荐的两套采样组合：
# - thinking 模式：温度低、top_p 高、不抑制 presence
# - non-thinking 模式：温度更高、top_p 收紧、用 presence_penalty 抑制复读
# `seed` 在两套里都不下发（保留服务端默认行为）。
_THINKING_DEFAULTS: dict[str, float | int] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
}
_NON_THINKING_DEFAULTS: dict[str, float | int] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
}
_SAMPLING_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "seed",
)


def _resolve_sampling_defaults(agent_config: AgentConfig) -> dict[str, Any]:
    """按 `enable_thinking` 给未显式设置的采样字段填 Qwen3 官方推荐兜底。

    规则：
    - `enable_thinking=True`  且字段为 None → 从 _THINKING_DEFAULTS 取
    - `enable_thinking=False` 且字段为 None → 从 _NON_THINKING_DEFAULTS 取
    - `enable_thinking=None`                → 不填，全部 None 透传（不下发）；
      与 `enable_thinking=None → 不下发 extra_body` 的语义对偶
    - 用户显式值（非 None）原样保留，不会被默认值覆盖

    `seed` 不在任一套兜底里——两个 mode 都让其保持用户原始意图（None=不下发）。

    返回值是 ``{field: resolved_value_or_None}``，调用方逐字段下发到 Adapter。
    """
    resolved: dict[str, Any] = {field: getattr(agent_config, field) for field in _SAMPLING_FIELDS}

    if agent_config.enable_thinking is None:
        return resolved

    defaults = _THINKING_DEFAULTS if agent_config.enable_thinking else _NON_THINKING_DEFAULTS
    for field, default_value in defaults.items():
        if resolved[field] is None:
            resolved[field] = default_value
    return resolved


def build_model_adapter(config: AppConfig, tools: ToolRegistry | None = None) -> ModelAdapter:
    """根据 `config.agent` 构造模型适配器。

    按 `tool_protocol` 分发：
    - "native": 需要传入 `tools` 以把工具 schema 下推到 OpenAI `tools=[...]` 字段
    - "text":   传统文本协议，不需要 tools 参数

    当 `config.agent.rate_limit` 非 None 且 `api_key` 非空时，外面再套一层
    `RateLimitedAdapter`（PR-2）。线上 vLLM 路径留空 rate_limit 即跳过。

    `_resolve_sampling_defaults` 在此处而非 `AgentConfig` 内调用：保持 AgentConfig
    存 "用户原始意图"，Adapter 拿到 "实际下发值"，与 `enable_thinking is None →
    不下发` 的语义对偶。

    独立成函数主要是为了测试：可以 monkeypatch 掉这里注入 ScriptedModelAdapter。
    """
    sampling = _resolve_sampling_defaults(config.agent)

    inner: NativeToolsOpenAIAdapter | OpenAIModelAdapter
    if config.agent.tool_protocol == "native":
        if tools is None:
            # 给一个防御性错误：native 协议必须提供 tools
            raise ValueError("NativeToolsOpenAIAdapter requires a ToolRegistry; pass `tools=...`.")
        inner = NativeToolsOpenAIAdapter(
            model=config.agent.model,
            api_base=config.agent.api_base,
            api_key=config.agent.api_key,
            tools=tools,
            allow_parallel_tool_calls=config.agent.allow_parallel_tool_calls,
            max_tokens=config.agent.max_tokens,
            enable_thinking=config.agent.enable_thinking,
            backend_kind=config.agent.backend_kind,
            **sampling,
        )
    else:
        inner = OpenAIModelAdapter(
            model=config.agent.model,
            api_base=config.agent.api_base,
            api_key=config.agent.api_key,
            max_tokens=config.agent.max_tokens,
            enable_thinking=config.agent.enable_thinking,
            backend_kind=config.agent.backend_kind,
            **sampling,
        )

    # PR-2：有 rate_limit 就包一层限速壳；没有就返回裸 adapter（线上路径不变）
    if config.agent.rate_limit is not None and config.agent.api_key:
        return RateLimitedAdapter(
            inner=inner,
            api_key=config.agent.api_key,
            rate_limit=config.agent.rate_limit,
        )
    return inner


@dataclass(frozen=True, slots=True)
class AgentApp:
    """ReAct 循环运行所需的全部依赖集合。

    五个字段都是构造完毕的实例：
    - dataset:        从 `config.dataset.root_path` 加载的只读 task 集合
    - registry:       不可变工具注册表（默认 8 工具）
    - model_adapter:  按 tool_protocol + rate_limit 包装好的模型适配器
    - agent_config:   ReAct 循环预算（max_steps / empty retry）
    - protocol:       字面量 "native" | "text"，决定消息构造策略

    `new_agent` 每次产出一个独立 `ReActAgent` 实例：共享 model/tools/config，
    但循环状态相互隔离，便于同一进程串行跑多个任务。
    """

    dataset: DABenchPublicDataset
    registry: ToolRegistry
    model_adapter: ModelAdapter
    agent_config: ReActAgentConfig
    protocol: ToolProtocol

    def new_agent(self, step_callback: StepCallback | None = None) -> ReActAgent:
        """生成一个使用本 App 共享依赖的新 ReAct 实例。

        子进程在每步结束时通过 step_callback 增量落盘 trace.json，主进程超时硬 kill 后
        仍可读到死前最后一次 append 的状态。其它场景（测试 / 同步跑）传 None 即可。
        """
        return ReActAgent(
            model=self.model_adapter,
            tools=self.registry,
            config=self.agent_config,
            protocol=self.protocol,
            step_callback=step_callback,
        )


def build_application(
    config: AppConfig,
    *,
    model: ModelAdapter | None = None,
    tools: ToolRegistry | None = None,
) -> AgentApp:
    """从 `AppConfig` 一次性构造 ReAct 运行所需的全部依赖。

    `model` / `tools` 是给测试预留的 override：
    - 传 `model=ScriptedModelAdapter(...)` 跳过真实 OpenAI 客户端构造
    - 传 `tools=mock_registry`（或一个裁剪过的 ToolRegistry）跳过默认 8 工具

    线上路径两个参数都为 None，走完整的 build_model_adapter + create_default_tool_registry。

    注意：`AgentApp` 实例不可 pickle（`OpenAI` 客户端含线程锁），所以本函数必须在
    子进程内调用——主进程只跨进程传递 `AppConfig`。
    """
    dataset = DABenchPublicDataset(config.dataset.root_path)
    registry = tools if tools is not None else create_default_tool_registry()
    model_adapter = model if model is not None else build_model_adapter(config, tools=registry)
    agent_config = ReActAgentConfig(
        max_steps=config.agent.max_steps,
        max_empty_tool_call_retries=config.agent.max_empty_tool_call_retries,
    )
    return AgentApp(
        dataset=dataset,
        registry=registry,
        model_adapter=model_adapter,
        agent_config=agent_config,
        protocol=config.agent.tool_protocol,
    )


__all__ = [
    "AgentApp",
    "build_application",
    "build_model_adapter",
]

# `_resolve_sampling_defaults` 故意不进 __all__：保持私有 API，但允许测试 import。
