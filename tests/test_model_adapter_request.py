"""对 `OpenAIModelAdapter` / `NativeToolsOpenAIAdapter` 请求体与响应解析的单元测试。

真正的 OpenAI/vLLM 在 CI 上不可达，用 `monkeypatch` 把 `openai.OpenAI` 替换成
返回 `MagicMock` 的伪客户端，专注断言：
- 请求 kwargs 随配置正确组装（max_tokens / extra_body 的条件下发）
- 响应侧把 `reasoning_content` 与 `usage`（含嵌套字段）解析到 `ModelResponse`

使用 `types.SimpleNamespace` 充当 SDK pydantic 对象的鸭子替身 —— 我们只读几个字段，
不需要真正的 pydantic 行为。
"""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from data_agent_baseline.agents import model as model_mod
from data_agent_baseline.agents.model import (
    ModelMessage,
    NativeToolsOpenAIAdapter,
    OpenAIModelAdapter,
    TokenUsage,
    _extract_reasoning_content,
    _extract_usage,
)
from data_agent_baseline.tools.registry import create_default_tool_registry


def _make_usage(
    *,
    prompt: int = 0,
    completion: int = 0,
    total: int = 0,
    cached: int | None = None,
    reasoning: int | None = None,
) -> types.SimpleNamespace:
    """构造伪 `response.usage`，可选填嵌套 details 对象。

    details 缺省为 None，用于测试"后端不返回嵌套字段时应归零"的分支。
    """
    usage_ns = types.SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )
    if cached is not None:
        usage_ns.prompt_tokens_details = types.SimpleNamespace(cached_tokens=cached)
    if reasoning is not None:
        usage_ns.completion_tokens_details = types.SimpleNamespace(reasoning_tokens=reasoning)
    return usage_ns


def _make_response(
    *,
    content: str = "",
    reasoning_content: str | None = None,
    usage: Any = None,
) -> types.SimpleNamespace:
    """构造伪 `ChatCompletion` 响应：message + usage 两个关键字段。"""
    message = types.SimpleNamespace(content=content, tool_calls=[])
    if reasoning_content is not None:
        message.reasoning_content = reasoning_content
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=message)],
        usage=usage,
    )


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, response: Any) -> MagicMock:
    """把 `model.OpenAI` 换成返回 MagicMock 的伪类，拦截 `chat.completions.create`。

    返回那个 MagicMock，测试用 `create_mock.call_args.kwargs` 断言请求 kwargs。
    """
    create_mock = MagicMock(return_value=response)
    client_ns = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create_mock),
        )
    )
    monkeypatch.setattr(model_mod, "OpenAI", lambda **_: client_ns)
    return create_mock


# ---------- 基础 helper 的小单测：先保证 _extract_* 行为可靠 ----------


def test_extract_reasoning_uses_direct_attribute() -> None:
    message = types.SimpleNamespace(reasoning_content="pondering")
    assert _extract_reasoning_content(message) == "pondering"


def test_extract_reasoning_falls_back_to_model_extra() -> None:
    # 没声明 reasoning_content，但底层 JSON 挂在 model_extra 上
    message = types.SimpleNamespace(model_extra={"reasoning_content": "hidden"})
    assert _extract_reasoning_content(message) == "hidden"


def test_extract_reasoning_returns_empty_when_absent() -> None:
    message = types.SimpleNamespace()  # 两条路径都没有
    assert _extract_reasoning_content(message) == ""


def test_extract_usage_handles_full_nested_payload() -> None:
    response = types.SimpleNamespace(
        usage=_make_usage(prompt=1200, completion=450, total=1650, cached=800, reasoning=300)
    )
    assert _extract_usage(response) == TokenUsage(1200, 450, 1650, 800, 300)


def test_extract_usage_defaults_missing_details_to_zero() -> None:
    response = types.SimpleNamespace(usage=_make_usage(prompt=100, completion=50, total=150))
    # 没有 details 嵌套对象 → cached / reasoning 归零，不抛错
    assert _extract_usage(response) == TokenUsage(100, 50, 150, 0, 0)


def test_extract_usage_handles_missing_usage() -> None:
    response = types.SimpleNamespace(usage=None)
    assert _extract_usage(response) == TokenUsage()


# ---------- OpenAIModelAdapter（文本协议）---------- #


def test_openai_adapter_sends_max_tokens_when_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_response(content="hi"))
    adapter = OpenAIModelAdapter(
        model="m", api_base="http://x", api_key="k", temperature=0.0, max_tokens=4096
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["max_tokens"] == 4096


def test_openai_adapter_omits_max_tokens_when_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_response(content="hi"))
    adapter = OpenAIModelAdapter(
        model="m", api_base="http://x", api_key="k", temperature=0.0, max_tokens=0
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert "max_tokens" not in kwargs


def test_openai_adapter_sends_extra_body_when_enable_thinking_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_response(content="hi"))
    adapter = OpenAIModelAdapter(
        model="m", api_base="http://x", api_key="k", temperature=0.0, enable_thinking=True
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["extra_body"] == {"enable_thinking": True}


def test_openai_adapter_sends_extra_body_when_enable_thinking_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_response(content="hi"))
    adapter = OpenAIModelAdapter(
        model="m", api_base="http://x", api_key="k", temperature=0.0, enable_thinking=False
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    # False 也要显式下发 —— 避免服务端默认 on 时用户的关闭意图被吞掉
    assert kwargs["extra_body"] == {"enable_thinking": False}


def test_openai_adapter_omits_extra_body_when_enable_thinking_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_response(content="hi"))
    adapter = OpenAIModelAdapter(
        model="m", api_base="http://x", api_key="k", temperature=0.0, enable_thinking=None
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert "extra_body" not in kwargs


def test_openai_adapter_captures_reasoning_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _make_response(
        content="final answer",
        reasoning_content="step-by-step thought",
        usage=_make_usage(prompt=100, completion=50, total=150, cached=40, reasoning=20),
    )
    _install_fake_openai(monkeypatch, response)
    adapter = OpenAIModelAdapter(model="m", api_base="http://x", api_key="k", temperature=0.0)

    result = adapter.complete([ModelMessage(role="user", content="hello")])

    assert result.content == "final answer"
    assert result.reasoning_content == "step-by-step thought"
    assert result.usage == TokenUsage(100, 50, 150, 40, 20)
    assert result.latency_ms >= 0


# ---------- NativeToolsOpenAIAdapter（原生 function calling）---------- #


def _native_adapter(**overrides: Any) -> NativeToolsOpenAIAdapter:
    """构造 native adapter，复用默认工具注册表，支持部分字段覆盖。"""
    defaults: dict[str, Any] = {
        "model": "m",
        "api_base": "http://x",
        "api_key": "k",
        "temperature": 0.0,
        "tools": create_default_tool_registry(),
        "max_tokens": 0,
        "enable_thinking": None,
    }
    defaults.update(overrides)
    return NativeToolsOpenAIAdapter(**defaults)


def test_native_adapter_sends_max_tokens_and_extra_body(monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_response(content=""))
    adapter = _native_adapter(max_tokens=8192, enable_thinking=True)
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["max_tokens"] == 8192
    assert kwargs["extra_body"] == {"enable_thinking": True}
    # 保留原有 native 字段确保没被改动回归
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is True
    assert isinstance(kwargs["tools"], list) and len(kwargs["tools"]) > 0


def test_native_adapter_omits_optional_fields_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_response(content=""))
    adapter = _native_adapter()  # max_tokens=0, enable_thinking=None
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert "max_tokens" not in kwargs
    assert "extra_body" not in kwargs


def test_native_adapter_captures_reasoning_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _make_response(
        content="",
        reasoning_content="I should call list_context",
        usage=_make_usage(prompt=500, completion=80, total=580, cached=200, reasoning=60),
    )
    _install_fake_openai(monkeypatch, response)
    adapter = _native_adapter()

    result = adapter.complete([ModelMessage(role="user", content="hello")])

    assert result.reasoning_content == "I should call list_context"
    assert result.usage == TokenUsage(500, 80, 580, 200, 60)
    assert result.latency_ms >= 0
