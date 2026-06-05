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
from pydantic import ValidationError

from data_agent_baseline.config import AgentConfig, RunConfig, load_app_config


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
    assert cfg.run.retrieval_index.enabled is False
    assert cfg.run.retrieval_index.max_chunk_chars == 1600
    assert cfg.run.retrieval_index.chunk_overlap_lines == 2
    assert cfg.run.retrieval_index.max_results == 8


def test_load_app_config_rejects_unknown_agent_field(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  does_not_exist: 123\n")
    with pytest.raises(ValueError, match="Unexpected keyword argument"):
        load_app_config(cfg_path)


def test_config_constructors_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="does_not_exist"):
        AgentConfig(does_not_exist=123)  # type: ignore[call-arg]
    with pytest.raises(ValidationError, match="does_not_exist"):
        RunConfig(does_not_exist=123)  # type: ignore[call-arg]


def test_retrieval_index_config_round_trip(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "agent:",
                "  model: m",
                "  api_base: http://x",
                "  api_key: k",
                "run:",
                "  retrieval_index:",
                "    enabled: true",
                "    cache_dir: local-indexes",
                "    max_chunk_chars: 2000",
                "    chunk_overlap_lines: 3",
                "    max_results: 12",
            ]
        )
    )
    cfg = load_app_config(cfg_path)
    assert cfg.run.retrieval_index.enabled is True
    assert cfg.run.retrieval_index.cache_dir.name == "local-indexes"
    assert cfg.run.retrieval_index.max_chunk_chars == 2000
    assert cfg.run.retrieval_index.chunk_overlap_lines == 3
    assert cfg.run.retrieval_index.max_results == 12


def test_retrieval_index_rejects_bad_limits(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "agent:",
                "  model: m",
                "  api_base: http://x",
                "  api_key: k",
                "run:",
                "  retrieval_index:",
                "    max_results: 0",
            ]
        )
    )
    with pytest.raises(ValueError, match=r"retrieval_index\.max_results"):
        load_app_config(cfg_path)


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


# ---------- backend_kind ----------


def test_backend_kind_default_is_none(tmp_path: Path) -> None:
    """缺省字段 → None，由 adapter 按 api_base 自动推断。"""
    cfg_path = _write_config(tmp_path / "cfg.yaml", "")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.backend_kind is None


def test_backend_kind_vllm_explicit(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  backend_kind: vllm\n")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.backend_kind == "vllm"


def test_backend_kind_dashscope_explicit(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  backend_kind: dashscope\n")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.backend_kind == "dashscope"


def test_backend_kind_normalizes_case(tmp_path: Path) -> None:
    """与 tool_protocol 一致：大小写规范化，保留 strict 字段集合校验。"""
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  backend_kind: VLLM\n")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.backend_kind == "vllm"


def test_backend_kind_explicit_null(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  backend_kind: null\n")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.backend_kind is None


def test_backend_kind_rejects_unknown_value(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  backend_kind: ollama\n")
    with pytest.raises(ValueError, match="backend_kind"):
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


# ---------- enable_preact ----------


def test_enable_preact_default_is_false(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.enable_preact is False


def test_enable_preact_true(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  enable_preact: true\n")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.enable_preact is True


def test_enable_preact_false_explicit(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  enable_preact: false\n")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.enable_preact is False
