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

from agents.llm import openai as model_mod
from agents.llm.openai import (
    NativeToolsOpenAIAdapter,
    OpenAIModelAdapter,
    ToolCallParseError,
    _extract_reasoning_content,
    _extract_usage,
)
from agents.llm.rate_limit import estimate_prompt_tokens
from agents.llm.types import ModelMessage, TokenUsage
from agents.tools.registry import create_default_tool_registry


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


def _make_stream_chunks(
    *,
    content: str = "",
    reasoning_content: str | None = None,
    usage: Any = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> list[types.SimpleNamespace]:
    """构造伪流式 chunk 列表，模拟 `stream=True` 的逐块返回。"""
    chunks: list[types.SimpleNamespace] = []

    if reasoning_content:
        delta = types.SimpleNamespace(content=None, tool_calls=None)
        delta.reasoning_content = reasoning_content
        chunks.append(
            types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)], usage=None)
        )

    if content:
        delta = types.SimpleNamespace(content=content, tool_calls=None)
        chunks.append(
            types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)], usage=None)
        )

    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            tc_delta = types.SimpleNamespace(
                index=tc.get("_index", 0),
                id=tc.get("id"),
                type=tc.get("type"),
                function=types.SimpleNamespace(
                    name=fn.get("name"),
                    arguments=fn.get("arguments"),
                ),
            )
            delta = types.SimpleNamespace(content=None, tool_calls=[tc_delta])
            chunks.append(
                types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)], usage=None)
            )

    if not chunks:
        delta = types.SimpleNamespace(content="", tool_calls=None)
        chunks.append(
            types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)], usage=None)
        )

    final_chunk = types.SimpleNamespace(choices=[], usage=usage)
    chunks.append(final_chunk)

    return chunks


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, chunks: Any) -> MagicMock:
    """把 `model.OpenAI` 换成返回 MagicMock 的伪类，拦截 `chat.completions.create`。

    `chunks` 应为 `_make_stream_chunks(...)` 返回的列表。create 返回其迭代器，
    模拟流式响应。测试用 `create_mock.call_args.kwargs` 断言请求 kwargs。
    """
    create_mock = MagicMock(side_effect=lambda **_kw: iter(chunks))
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


# ---------- OpenAIModelAdapter（无工具）---------- #


def test_openai_adapter_sends_max_tokens_when_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(
        model="m", api_base="http://x", api_key="k", temperature=0.0, max_tokens=4096
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["max_tokens"] == 4096


def test_openai_adapter_omits_max_tokens_when_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(
        model="m", api_base="http://x", api_key="k", temperature=0.0, max_tokens=0
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert "max_tokens" not in kwargs


def test_openai_adapter_sends_extra_body_when_enable_thinking_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    # api_base="http://x" 不含 dashscope hint → 自动推断为 vllm → 嵌套形式
    adapter = OpenAIModelAdapter(
        model="m", api_base="http://x", api_key="k", temperature=0.0, enable_thinking=True
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}


def test_openai_adapter_sends_extra_body_when_enable_thinking_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(
        model="m", api_base="http://x", api_key="k", temperature=0.0, enable_thinking=False
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    # False 也要显式下发 —— 避免服务端默认 on 时用户的关闭意图被吞掉；vLLM 推断走嵌套
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_openai_adapter_omits_extra_body_when_enable_thinking_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(
        model="m", api_base="http://x", api_key="k", temperature=0.0, enable_thinking=None
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert "extra_body" not in kwargs


def test_openai_adapter_omits_temperature_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(model="m", api_base="http://x", api_key="k", temperature=None)
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert "temperature" not in kwargs


def test_openai_adapter_sends_top_p_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(
        model="m", api_base="http://x", api_key="k", temperature=0.6, top_p=0.95
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["top_p"] == 0.95


def test_openai_adapter_omits_top_p_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(model="m", api_base="http://x", api_key="k", temperature=0.6)
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert "top_p" not in kwargs


def test_openai_adapter_sends_seed_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(
        model="m", api_base="http://x", api_key="k", temperature=0.6, seed=42
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["seed"] == 42


def test_openai_adapter_sends_presence_penalty_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(
        model="m", api_base="http://x", api_key="k", temperature=0.7, presence_penalty=1.5
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["presence_penalty"] == 1.5


def test_openai_adapter_extra_body_merges_top_k_and_enable_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关键回归测试：top_k / min_p 必须与 enable_thinking 在 extra_body 里共存而非覆盖。

    vLLM 推断路径下 enable_thinking 走 chat_template_kwargs 嵌套，
    与顶层的 top_k / min_p 互不冲突。
    """
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(
        model="m",
        api_base="http://x",
        api_key="k",
        temperature=0.6,
        enable_thinking=True,
        top_k=20,
        min_p=0.0,
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True},
        "top_k": 20,
        "min_p": 0.0,
    }


def test_openai_adapter_extra_body_contains_only_top_k_when_thinking_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(
        model="m",
        api_base="http://x",
        api_key="k",
        temperature=0.6,
        enable_thinking=None,
        top_k=20,
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["extra_body"] == {"top_k": 20}


def test_openai_adapter_captures_reasoning_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = _make_stream_chunks(
        content="final answer",
        reasoning_content="step-by-step thought",
        usage=_make_usage(prompt=100, completion=50, total=150, cached=40, reasoning=20),
    )
    _install_fake_openai(monkeypatch, chunks)
    adapter = OpenAIModelAdapter(model="m", api_base="http://x", api_key="k", temperature=0.0)

    result = adapter.complete([ModelMessage(role="user", content="hello")])

    assert result.content == "final answer"
    assert result.reasoning_content == "step-by-step thought"
    assert result.usage == TokenUsage(100, 50, 150, 40, 20)
    assert result.latency_ms >= 0


def test_openai_adapter_prompt_estimate_is_message_only() -> None:
    messages = [ModelMessage(role="user", content="hello")]
    adapter = OpenAIModelAdapter(model="m", api_base="http://x", api_key="k", temperature=0.0)

    assert adapter.estimate_prompt_tokens(messages) == estimate_prompt_tokens(messages)


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
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content=""))
    adapter = _native_adapter(max_tokens=8192, enable_thinking=True)
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["max_tokens"] == 8192
    # api_base="http://x" → 推断 vllm → 嵌套形式
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    # 保留原有 native 字段确保没被改动回归
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is True
    assert isinstance(kwargs["tools"], list) and len(kwargs["tools"]) > 0


def test_native_adapter_omits_optional_fields_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content=""))
    adapter = _native_adapter()  # max_tokens=0, enable_thinking=None
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert "max_tokens" not in kwargs
    assert "extra_body" not in kwargs


def test_native_adapter_captures_reasoning_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = _make_stream_chunks(
        content="",
        reasoning_content="I should call inspect_files",
        usage=_make_usage(prompt=500, completion=80, total=580, cached=200, reasoning=60),
    )
    _install_fake_openai(monkeypatch, chunks)
    adapter = _native_adapter()

    result = adapter.complete([ModelMessage(role="user", content="hello")])

    assert result.reasoning_content == "I should call inspect_files"
    assert result.usage == TokenUsage(500, 80, 580, 200, 60)
    assert result.latency_ms >= 0


def test_native_adapter_malformed_tool_arguments_preserve_partial_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_tool_call = {
        "id": "bad_answer",
        "type": "function",
        "function": {
            "name": "answer",
            "arguments": '{"columns": ["x"], "rows": [["unterminated"]',
        },
    }
    chunks = _make_stream_chunks(
        content="submitting answer",
        reasoning_content="answer json got truncated",
        usage=_make_usage(prompt=10, completion=5, total=15, cached=3, reasoning=2),
        tool_calls=[{**raw_tool_call, "_index": 0}],
    )
    _install_fake_openai(monkeypatch, chunks)
    adapter = _native_adapter()

    with pytest.raises(ToolCallParseError) as exc_info:
        adapter.complete([ModelMessage(role="user", content="hello")])

    partial = exc_info.value.partial_response
    assert "Failed to parse tool_call arguments for 'answer'" in str(exc_info.value)
    assert partial.content == "submitting answer"
    assert partial.raw_response == "submitting answer"
    assert partial.raw_tool_calls == [raw_tool_call]
    assert partial.reasoning_content == "answer json got truncated"
    assert partial.usage == TokenUsage(10, 5, 15, 3, 2)
    assert partial.tool_calls == []
    assert partial.latency_ms >= 0


def test_native_adapter_omits_temperature_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content=""))
    adapter = _native_adapter(temperature=None)
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert "temperature" not in kwargs


def test_native_adapter_sends_top_p_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content=""))
    adapter = _native_adapter(temperature=0.6, top_p=0.95)
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["top_p"] == 0.95


def test_native_adapter_omits_top_p_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content=""))
    adapter = _native_adapter(temperature=0.6)
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert "top_p" not in kwargs


def test_native_adapter_sends_seed_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content=""))
    adapter = _native_adapter(seed=7)
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["seed"] == 7


def test_native_adapter_sends_presence_penalty_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content=""))
    adapter = _native_adapter(presence_penalty=1.5)
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["presence_penalty"] == 1.5


def test_native_adapter_extra_body_merges_top_k_and_enable_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关键回归测试：top_k / min_p 必须与 enable_thinking 在 extra_body 里共存而非覆盖。

    vLLM 推断路径下 enable_thinking 嵌入 chat_template_kwargs，与顶层 top_k/min_p 并存。
    """
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content=""))
    adapter = _native_adapter(enable_thinking=True, top_k=20, min_p=0.0)
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True},
        "top_k": 20,
        "min_p": 0.0,
    }


def test_native_adapter_extra_body_contains_only_top_k_when_thinking_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content=""))
    adapter = _native_adapter(enable_thinking=None, top_k=20)
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["extra_body"] == {"top_k": 20}


def test_native_adapter_prompt_estimate_includes_tool_schema() -> None:
    messages = [ModelMessage(role="user", content="hello")]
    adapter = _native_adapter()

    message_only = estimate_prompt_tokens(messages)
    estimated = adapter.estimate_prompt_tokens(messages)

    assert estimated > message_only
    assert estimated - message_only > 1_000


# ---------- backend_kind 矩阵：vllm vs dashscope，自动推断 vs 显式覆盖 ----------


def test_openai_adapter_dashscope_api_base_auto_infers_dashscope_top_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """api_base 含 aliyuncs.com → 自动推断 dashscope → enable_thinking 走顶层。"""
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(
        model="m",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="k",
        temperature=0.0,
        enable_thinking=True,
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["extra_body"] == {"enable_thinking": True}


def test_openai_adapter_explicit_dashscope_overrides_localhost_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式 backend_kind="dashscope" 覆盖 localhost api_base 的 vllm 推断。"""
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(
        model="m",
        api_base="http://localhost:8000/v1",
        api_key="k",
        temperature=0.0,
        enable_thinking=True,
        backend_kind="dashscope",
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["extra_body"] == {"enable_thinking": True}


def test_openai_adapter_explicit_vllm_overrides_dashscope_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """显式 backend_kind="vllm" 覆盖 dashscope api_base 的自动推断 → 仍走嵌套。"""
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(
        model="m",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="k",
        temperature=0.0,
        enable_thinking=True,
        backend_kind="vllm",
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}


def test_native_adapter_explicit_vllm_uses_chat_template_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NativeToolsOpenAIAdapter 与 OpenAIModelAdapter 共用 _thinking_extra_body 工具，
    显式 backend_kind="vllm" 行为应当一致：走嵌套。"""
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content=""))
    adapter = _native_adapter(enable_thinking=True, backend_kind="vllm")
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}


def test_native_adapter_explicit_dashscope_uses_top_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content=""))
    adapter = _native_adapter(enable_thinking=False, backend_kind="dashscope")
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["extra_body"] == {"enable_thinking": False}


def test_backend_kind_does_not_emit_extra_body_when_thinking_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """enable_thinking=None 时无论 backend_kind 是什么，都不应下发对应字段；
    避免 backend_kind 单独下发误触发未定义 chat template 行为。"""
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(
        model="m",
        api_base="http://x",
        api_key="k",
        temperature=0.0,
        enable_thinking=None,
        backend_kind="vllm",
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert "extra_body" not in kwargs


def test_resolve_backend_kind_explicit_value_wins() -> None:
    """单独验证 resolve_backend_kind helper 的显式优先语义。"""
    from agents.llm.openai import resolve_backend_kind

    assert resolve_backend_kind("vllm", "https://dashscope.aliyuncs.com/v1") == "vllm"
    assert resolve_backend_kind("dashscope", "http://localhost:8000/v1") == "dashscope"
    assert resolve_backend_kind("deepseek", "http://localhost:8000/v1") == "deepseek"


def test_resolve_backend_kind_inference_fallback() -> None:
    """None 时按 host 子串匹配；命中 aliyuncs.com 或 dashscope → dashscope，否则 vllm。"""
    from agents.llm.openai import resolve_backend_kind

    assert resolve_backend_kind(None, "https://dashscope.aliyuncs.com/v1") == "dashscope"
    assert resolve_backend_kind(None, "https://api.dashscope.com/v1") == "dashscope"
    assert resolve_backend_kind(None, "http://localhost:8000/v1") == "vllm"
    assert resolve_backend_kind(None, "http://10.0.0.5:8001") == "vllm"
    # 大小写不敏感（resolve_backend_kind 内做 lower）
    assert resolve_backend_kind(None, "HTTPS://DASHSCOPE.ALIYUNCS.COM/v1") == "dashscope"
    # DeepSeek API 自动推断
    assert resolve_backend_kind(None, "https://api.deepseek.com/v1") == "deepseek"
    assert resolve_backend_kind(None, "https://API.DEEPSEEK.COM/v1") == "deepseek"


# ---------- DeepSeek V4 thinking mode 矩阵 ----------


def test_openai_adapter_deepseek_thinking_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """thinking_mode="enabled" → extra_body 含 thinking 对象。"""
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(
        model="deepseek-v4-flash",
        api_base="https://api.deepseek.com/v1",
        api_key="k",
        thinking_mode="enabled",
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


def test_openai_adapter_deepseek_thinking_enabled_with_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """thinking_mode="enabled" + reasoning_effort="max"。"""
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(
        model="deepseek-v4-pro",
        api_base="https://api.deepseek.com/v1",
        api_key="k",
        thinking_mode="enabled",
        reasoning_effort="max",
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["extra_body"] == {
        "thinking": {"type": "enabled", "reasoning_effort": "max"},
    }


def test_openai_adapter_deepseek_thinking_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """thinking_mode="disabled" → thinking 对象只含 type，不含 reasoning_effort。"""
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(
        model="deepseek-v4-flash",
        api_base="https://api.deepseek.com/v1",
        api_key="k",
        thinking_mode="disabled",
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_openai_adapter_deepseek_thinking_disabled_ignores_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """thinking_mode="disabled" 时 reasoning_effort 不写入 thinking 对象。"""
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(
        model="deepseek-v4-flash",
        api_base="https://api.deepseek.com/v1",
        api_key="k",
        thinking_mode="disabled",
        reasoning_effort="max",
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_openai_adapter_deepseek_no_thinking_mode_no_extra_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """thinking_mode=None 且 enable_thinking=None → 不下发 extra_body。"""
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(
        model="deepseek-v4-flash",
        api_base="https://api.deepseek.com/v1",
        api_key="k",
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert "extra_body" not in kwargs


def test_native_adapter_deepseek_thinking_enabled_with_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NativeToolsOpenAIAdapter: thinking + top_k 在 extra_body 里共存。"""
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content=""))
    adapter = _native_adapter(
        api_base="https://api.deepseek.com/v1",
        thinking_mode="enabled",
        reasoning_effort="high",
        top_k=20,
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["extra_body"] == {
        "thinking": {"type": "enabled", "reasoning_effort": "high"},
        "top_k": 20,
    }


def test_deepseek_thinking_takes_precedence_over_enable_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """若同时设置了 thinking_mode 和 enable_thinking，thinking_mode 优先。"""
    create_mock = _install_fake_openai(monkeypatch, _make_stream_chunks(content="hi"))
    adapter = OpenAIModelAdapter(
        model="deepseek-v4-flash",
        api_base="https://api.deepseek.com/v1",
        api_key="k",
        thinking_mode="enabled",
        enable_thinking=True,
    )
    adapter.complete([ModelMessage(role="user", content="hello")])

    kwargs = create_mock.call_args.kwargs
    assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


# ---------- DeepSeek usage 兼容 ----------


def test_extract_usage_deepseek_cache_hit_tokens() -> None:
    """DeepSeek V4 返回 prompt_cache_hit_tokens 在 usage 顶层。"""
    usage_ns = types.SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=200,
        total_tokens=1200,
        prompt_cache_hit_tokens=600,
        prompt_cache_miss_tokens=400,
        completion_tokens_details=types.SimpleNamespace(reasoning_tokens=50),
    )
    response = types.SimpleNamespace(usage=usage_ns)
    result = _extract_usage(response)

    assert result.prompt_tokens == 1000
    assert result.completion_tokens == 200
    assert result.cached_tokens == 600
    assert result.reasoning_tokens == 50


def test_extract_usage_prefers_larger_cached_value() -> None:
    """同时存在嵌套 cached_tokens 和顶层 prompt_cache_hit_tokens 时取 max。"""
    usage_ns = types.SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=200,
        total_tokens=1200,
        prompt_tokens_details=types.SimpleNamespace(cached_tokens=300),
        prompt_cache_hit_tokens=600,
    )
    response = types.SimpleNamespace(usage=usage_ns)
    result = _extract_usage(response)

    assert result.cached_tokens == 600
