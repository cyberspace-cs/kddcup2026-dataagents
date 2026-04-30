"""`config.load_app_config` 对新加字段的解析测试。

覆盖：
- max_tokens 默认值 / 显式数字 / 字符串数字转 int
- max_empty_tool_call_retries 默认值 / 显式数字 / 非法负数抛错
- enable_thinking 的四种情形：缺省 / null / true / false / 非 bool 抛错
- temperature 默认 None；采样字段（top_p/top_k/min_p/presence_penalty/seed）的边界与默认
- thinking + greedy decoding 的 UserWarning 触发
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from data_agent_baseline.config import load_app_config


def _write_config(path: Path, body: str) -> Path:
    """写一个最小可解析 YAML；body 会被插入到 agent: 下面作为额外字段。"""
    path.write_text(f"agent:\n  model: m\n  api_base: http://x\n  api_key: k\n{body}")
    return path


def test_defaults_when_fields_absent(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.max_tokens == 16384  # 默认值
    assert cfg.agent.max_empty_tool_call_retries == 8
    assert cfg.agent.enable_thinking is None


def test_max_tokens_int_round_trip(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  max_tokens: 8192\n")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.max_tokens == 8192


def test_max_tokens_zero_is_allowed(tmp_path: Path) -> None:
    # 0 = 不下发，是合法值
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  max_tokens: 0\n")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.max_tokens == 0


def test_max_empty_tool_call_retries_round_trip(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "cfg.yaml",
        "  max_empty_tool_call_retries: 3\n",
    )
    cfg = load_app_config(cfg_path)
    assert cfg.agent.max_empty_tool_call_retries == 3


def test_max_empty_tool_call_retries_rejects_negative(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "cfg.yaml",
        "  max_empty_tool_call_retries: -1\n",
    )
    with pytest.raises(ValueError, match="max_empty_tool_call_retries"):
        load_app_config(cfg_path)


def test_enable_thinking_true(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  enable_thinking: true\n")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.enable_thinking is True


def test_enable_thinking_false(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  enable_thinking: false\n")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.enable_thinking is False


def test_enable_thinking_explicit_null(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  enable_thinking: null\n")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.enable_thinking is None


def test_enable_thinking_rejects_non_bool(tmp_path: Path) -> None:
    # `1` 不应被静默当作 True —— 明确报错，避免运维踩坑
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  enable_thinking: 1\n")
    with pytest.raises(ValueError, match="enable_thinking"):
        load_app_config(cfg_path)


# ---------- 采样字段：temperature / top_p / top_k / min_p / presence_penalty / seed ----------


def test_temperature_default_is_none(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "")
    cfg = load_app_config(cfg_path)
    # 旧默认 0.0 已废弃；mode-aware 兜底由 _resolve_sampling_defaults 完成
    assert cfg.agent.temperature is None


def test_temperature_zero_is_allowed(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  temperature: 0.0\n")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.temperature == 0.0


def test_temperature_rejects_negative(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  temperature: -0.1\n")
    with pytest.raises(ValueError, match="temperature"):
        load_app_config(cfg_path)


def test_top_p_default_is_none(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.top_p is None


def test_top_p_round_trip(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  top_p: 0.95\n")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.top_p == 0.95


def test_top_p_rejects_zero(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  top_p: 0.0\n")
    with pytest.raises(ValueError, match="top_p"):
        load_app_config(cfg_path)


def test_top_p_rejects_above_one(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  top_p: 1.1\n")
    with pytest.raises(ValueError, match="top_p"):
        load_app_config(cfg_path)


def test_top_k_round_trip(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  top_k: 20\n")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.top_k == 20


def test_top_k_rejects_zero(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  top_k: 0\n")
    with pytest.raises(ValueError, match="top_k"):
        load_app_config(cfg_path)


def test_top_k_rejects_negative(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  top_k: -1\n")
    with pytest.raises(ValueError, match="top_k"):
        load_app_config(cfg_path)


def test_min_p_round_trip(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  min_p: 0.0\n")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.min_p == 0.0


def test_min_p_rejects_above_one(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  min_p: 1.5\n")
    with pytest.raises(ValueError, match="min_p"):
        load_app_config(cfg_path)


def test_min_p_rejects_negative(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  min_p: -0.1\n")
    with pytest.raises(ValueError, match="min_p"):
        load_app_config(cfg_path)


def test_presence_penalty_round_trip(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  presence_penalty: 1.5\n")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.presence_penalty == 1.5


def test_presence_penalty_rejects_out_of_range(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  presence_penalty: 2.5\n")
    with pytest.raises(ValueError, match="presence_penalty"):
        load_app_config(cfg_path)


def test_presence_penalty_rejects_below_negative_two(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  presence_penalty: -2.5\n")
    with pytest.raises(ValueError, match="presence_penalty"):
        load_app_config(cfg_path)


def test_seed_default_is_none(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.seed is None


def test_seed_accepts_arbitrary_int(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  seed: 42\n")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.seed == 42


def test_seed_accepts_negative_int(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  seed: -1\n")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.seed == -1


# ---------- thinking + greedy decoding 的 UserWarning ----------


def test_warning_emitted_for_thinking_with_zero_temperature(tmp_path: Path) -> None:
    # Qwen3 model card 反推荐 thinking 模式下 greedy decoding；显式 0.0 仍允许，
    # 但加载时必须发 UserWarning，避免运维静默踩坑
    cfg_path = _write_config(
        tmp_path / "cfg.yaml",
        "  enable_thinking: true\n  temperature: 0.0\n",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_app_config(cfg_path)
    assert cfg.agent.temperature == 0.0
    assert cfg.agent.enable_thinking is True
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert any("greedy" in str(w.message).lower() for w in user_warnings)


def test_no_warning_when_temperature_unset_with_thinking(tmp_path: Path) -> None:
    # 未显式 temperature → mode-aware 兜底会填 0.6，不应触发 warning
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  enable_thinking: true\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_app_config(cfg_path)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert not any("greedy" in str(w.message).lower() for w in user_warnings)


def test_no_warning_when_thinking_disabled_with_greedy(tmp_path: Path) -> None:
    # non-thinking 模式不在 Qwen3 反推荐范围
    cfg_path = _write_config(
        tmp_path / "cfg.yaml",
        "  enable_thinking: false\n  temperature: 0.0\n",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_app_config(cfg_path)
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert not any("greedy" in str(w.message).lower() for w in user_warnings)
