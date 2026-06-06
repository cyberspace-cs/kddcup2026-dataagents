"""`application.py:_resolve_sampling_defaults` 的解析测试。

覆盖三个 mode 分支：
- enable_thinking=True  → 填 Qwen3 thinking 兜底（0.6 / 0.95 / 20 / 0 / 0）
- enable_thinking=False → 填 non-thinking 兜底（0.7 / 0.8 / 20 / 0 / 1.5）
- enable_thinking=None  → 全部保持 None（与 "不下发" 语义对偶）

以及显式值优先：用户显式值不会被默认值覆盖，无论哪种 mode。
"""

from __future__ import annotations

from agents.application import _resolve_sampling_defaults
from agents.config import AgentConfig


def _agent(**overrides: object) -> AgentConfig:
    """构造 AgentConfig，可选覆盖任意字段。"""
    return AgentConfig(api_key="k", **overrides)  # type: ignore[arg-type]


def test_thinking_mode_fills_official_recommendation() -> None:
    cfg = _agent(enable_thinking=True)
    resolved = _resolve_sampling_defaults(cfg)
    assert resolved == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "seed": None,
    }


def test_non_thinking_mode_fills_official_recommendation() -> None:
    cfg = _agent(enable_thinking=False)
    resolved = _resolve_sampling_defaults(cfg)
    assert resolved == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "seed": None,
    }


def test_enable_thinking_none_keeps_all_fields_none() -> None:
    # mode 未确定时不能瞎填——保持与 enable_thinking=None → 不下发的语义对偶
    cfg = _agent(enable_thinking=None)
    resolved = _resolve_sampling_defaults(cfg)
    assert resolved == {
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "min_p": None,
        "presence_penalty": None,
        "seed": None,
    }


def test_explicit_values_override_thinking_defaults() -> None:
    # 用户显式 temperature=0.3：thinking 模式下也保留 0.3，不被 0.6 覆盖
    cfg = _agent(enable_thinking=True, temperature=0.3, top_p=0.5)
    resolved = _resolve_sampling_defaults(cfg)
    assert resolved["temperature"] == 0.3
    assert resolved["top_p"] == 0.5
    # 未显式的字段仍然走兜底
    assert resolved["top_k"] == 20
    assert resolved["min_p"] == 0.0
    assert resolved["presence_penalty"] == 0.0


def test_explicit_values_override_non_thinking_defaults() -> None:
    cfg = _agent(enable_thinking=False, presence_penalty=0.0, top_k=40)
    resolved = _resolve_sampling_defaults(cfg)
    assert resolved["presence_penalty"] == 0.0  # 显式 0.0 不被 1.5 覆盖
    assert resolved["top_k"] == 40  # 显式 40 不被 20 覆盖
    # 未显式的仍走兜底
    assert resolved["temperature"] == 0.7
    assert resolved["top_p"] == 0.8


def test_explicit_seed_passes_through_in_thinking_mode() -> None:
    cfg = _agent(enable_thinking=True, seed=42)
    resolved = _resolve_sampling_defaults(cfg)
    assert resolved["seed"] == 42  # seed 永远透传，与默认无关


def test_explicit_seed_passes_through_in_none_mode() -> None:
    # enable_thinking=None 时 seed 显式值仍然透传
    cfg = _agent(enable_thinking=None, seed=7)
    resolved = _resolve_sampling_defaults(cfg)
    assert resolved["seed"] == 7
    # 其它字段仍保持 None
    assert resolved["temperature"] is None
    assert resolved["top_p"] is None
