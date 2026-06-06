"""`agents.application.build_application` 工厂单测。

覆盖点：
- `build_application(config)` 返回 `AgentApp`，五个字段全部赋值且类型正确
- `new_agent()` 每次调用返回独立的 `ReActAgent` 实例（不共享循环状态）
- `model=...` / `tools=...` override 跳过真实 OpenAI 客户端构造
- `protocol` 字段透传 `config.agent.tool_protocol`
- `agent_config` 透传 `config.agent.max_steps` / `max_empty_tool_call_retries`
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.agent import ReActAgent, ReActAgentConfig
from agents.application import AgentApp, build_application, build_no_tools_model_adapter
from agents.benchmark.dataset import DABenchPublicDataset
from agents.config import (
    AgentConfig,
    AppConfig,
    DatasetConfig,
    RateLimitConfig,
    RunConfig,
)
from agents.llm.openai import OpenAIModelAdapter
from agents.llm.rate_limit import RateLimitedAdapter
from agents.tools.registry import ToolRegistry, create_default_tool_registry
from agents.verification.answer import AnswerVerifier
from tests.helpers.scripted_adapters import ScriptedModelAdapter


def _make_config(
    *,
    tmp_path: Path,
    api_key: str = "primary-key",
    tool_protocol: str = "native",
    max_steps: int = 13,
    max_empty_tool_call_retries: int = 5,
    rate_limit: RateLimitConfig | None = None,
) -> AppConfig:
    """构造一个最小可用的 AppConfig，避免真实数据集依赖。"""
    return AppConfig(
        dataset=DatasetConfig(root_path=tmp_path / "data"),
        agent=AgentConfig(
            model="m",
            api_base="http://x",
            api_key=api_key,
            tool_protocol=tool_protocol,  # type: ignore[arg-type]
            max_steps=max_steps,
            max_empty_tool_call_retries=max_empty_tool_call_retries,
            rate_limit=rate_limit,
        ),
        run=RunConfig(
            output_dir=tmp_path / "runs",
            run_id=None,
            max_workers=1,
            task_timeout_seconds=0,
        ),
    )


def test_build_application_populates_all_fields(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path=tmp_path)
    app = build_application(cfg)

    assert isinstance(app, AgentApp)
    assert isinstance(app.dataset, DABenchPublicDataset)
    assert app.dataset.root_dir == cfg.dataset.root_path
    assert isinstance(app.registry, ToolRegistry)
    # 默认 8 工具都注册到位
    assert {"answer", "execute_python", "execute_context_sql"} <= set(app.registry.definitions)
    assert isinstance(app.agent_config, ReActAgentConfig)
    assert app.agent_config.max_steps == 13
    assert app.agent_config.max_empty_tool_call_retries == 5
    assert app.protocol == "native"
    # model_adapter 实现 ModelAdapter 协议（鸭子类型）：有 complete 方法
    assert hasattr(app.model_adapter, "complete")
    assert isinstance(app.answer_verifier, AnswerVerifier)


def test_build_application_text_protocol(tmp_path: Path) -> None:
    """text 协议路径不需要 tools 参数；构造的 model_adapter 是 OpenAIModelAdapter。"""
    cfg = _make_config(tmp_path=tmp_path, tool_protocol="text")
    app = build_application(cfg)
    assert app.protocol == "text"
    assert hasattr(app.model_adapter, "complete")


def test_build_application_rejects_placeholder_api_key(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path=tmp_path, api_key="YOUR_API_KEY")

    with pytest.raises(ValueError, match=r"agent\.api_key.*placeholder"):
        build_application(cfg)


def test_new_agent_returns_fresh_instance_per_call(tmp_path: Path) -> None:
    """每次 new_agent() 都是一个独立的 ReActAgent，循环状态不共享。"""
    cfg = _make_config(tmp_path=tmp_path)
    app = build_application(cfg)

    agent_a = app.new_agent()
    agent_b = app.new_agent()

    assert isinstance(agent_a, ReActAgent)
    assert isinstance(agent_b, ReActAgent)
    assert agent_a is not agent_b
    # 共享底层依赖（model/tools）减少重复构造，但实例本身独立
    assert agent_a.model is agent_b.model
    assert agent_a.tools is agent_b.tools
    assert agent_a.config is agent_b.config
    assert agent_a.answer_verifier is agent_b.answer_verifier


def test_new_agent_wires_step_callback(tmp_path: Path) -> None:
    """传入的 step_callback 应该被透传到 ReActAgent.step_callback。"""
    cfg = _make_config(tmp_path=tmp_path)
    app = build_application(cfg)

    captured: list[str] = []

    def cb(state: object) -> None:
        del state
        captured.append("called")

    agent = app.new_agent(step_callback=cb)
    assert agent.step_callback is cb


def test_build_application_model_override_skips_real_adapter(tmp_path: Path) -> None:
    """传 model=ScriptedModelAdapter 时不应触发 build_model_adapter。"""
    cfg = _make_config(tmp_path=tmp_path)
    scripted = ScriptedModelAdapter(["dummy"])
    app = build_application(cfg, model=scripted)
    # 注入的 model 直接被用作 model_adapter，不再被 RateLimitedAdapter / OpenAIModelAdapter 包裹
    assert app.model_adapter is scripted
    assert app.answer_verifier is None


def test_build_no_tools_model_adapter_uses_plain_openai_adapter(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path=tmp_path, tool_protocol="native")

    adapter = build_no_tools_model_adapter(cfg)

    assert isinstance(adapter, OpenAIModelAdapter)
    assert adapter.enable_thinking is False
    assert not hasattr(adapter, "_openai_tools")


def test_build_no_tools_model_adapter_preserves_rate_limit_wrapper(tmp_path: Path) -> None:
    cfg = _make_config(
        tmp_path=tmp_path,
        rate_limit=RateLimitConfig(state_dir=tmp_path / "rl"),
    )

    adapter = build_no_tools_model_adapter(cfg)

    assert isinstance(adapter, RateLimitedAdapter)
    assert isinstance(adapter.inner, OpenAIModelAdapter)
    assert adapter.inner.enable_thinking is False


def test_build_application_tools_override(tmp_path: Path) -> None:
    """传 tools=custom_registry 时跳过 create_default_tool_registry。"""
    cfg = _make_config(tmp_path=tmp_path, tool_protocol="text")  # text 协议下无须真实工具集
    custom = create_default_tool_registry()
    app = build_application(cfg, tools=custom)
    assert app.registry is custom


def test_agent_config_threads_through_protocol(tmp_path: Path) -> None:
    """`config.agent.tool_protocol` 字面量被透传到 AgentApp.protocol；同样的字面量再下沉到
    `new_agent()` 创建的 ReActAgent 实例。"""
    cfg = _make_config(tmp_path=tmp_path, tool_protocol="text")
    app = build_application(cfg)
    assert app.protocol == "text"
    agent = app.new_agent()
    assert agent.protocol == "text"
