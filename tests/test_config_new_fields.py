"""`config.load_app_config` 对新加字段的解析测试。

覆盖：
- max_tokens 默认值 / 显式数字 / 字符串数字转 int
- max_empty_tool_call_retries 默认值 / 显式数字 / 非法负数抛错
- enable_thinking 的四种情形：缺省 / null / true / false / 非 bool 抛错
"""

from __future__ import annotations

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
