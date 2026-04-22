"""`config.load_app_config` 对 PR-2 新加字段（api_keys / rate_limit）的解析测试。

覆盖：
- 两者都缺省 → 默认值（api_keys=() / rate_limit=None），与 PR-1 单 Key 路径行为一致
- 两者都显式配置 → 各字段正确解析，state_dir 规整成绝对 Path
- 只配其一（api_keys 无 rate_limit / 反之）→ 拒收
- api_keys 中重复、空串、非字符串 → 拒收
- 数值字段越界 / 非法（rpm<=0、safety 不在 (0,1]、reserve>max_tokens …）→ 拒收
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_agent_baseline.config import load_app_config


def _write_config(path: Path, body: str) -> Path:
    """写一个最小可解析 YAML；body 追加到 agent: 块下面，作为额外字段。"""
    path.write_text(f"agent:\n  model: m\n  api_base: http://x\n  api_key: k\n{body}")
    return path


# --- 默认 / 禁用场景 --------------------------------------------------------------


def test_defaults_when_both_absent(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.api_keys == ()
    assert cfg.agent.rate_limit is None


# --- 完整块解析 -------------------------------------------------------------------


def test_full_block_round_trip(tmp_path: Path) -> None:
    body = """
  api_keys:
    - key-A
    - key-B
  rate_limit:
    rpm_per_key: 600
    tpm_per_key: 1000000
    safety_factor: 0.8
    reserve_output_tokens: 8192
    max_retries: 4
    backoff_initial_seconds: 0.5
    backoff_max_seconds: 15.0
    state_dir: /tmp/ratelimit-test
"""
    cfg_path = _write_config(tmp_path / "cfg.yaml", body)
    cfg = load_app_config(cfg_path)
    assert cfg.agent.api_keys == ("key-A", "key-B")
    assert cfg.agent.rate_limit is not None
    rl = cfg.agent.rate_limit
    assert rl.rpm_per_key == 600
    assert rl.tpm_per_key == 1_000_000
    assert rl.safety_factor == 0.8
    assert rl.reserve_output_tokens == 8192
    assert rl.max_retries == 4
    assert rl.backoff_initial_seconds == 0.5
    assert rl.backoff_max_seconds == 15.0
    assert rl.state_dir == Path("/tmp/ratelimit-test")


def test_rate_limit_state_dir_none_when_absent(tmp_path: Path) -> None:
    body = """
  api_keys: [key-A, key-B]
  rate_limit:
    rpm_per_key: 600
    tpm_per_key: 1000000
"""
    cfg_path = _write_config(tmp_path / "cfg.yaml", body)
    cfg = load_app_config(cfg_path)
    assert cfg.agent.rate_limit is not None
    # state_dir 缺省时交给 runner 在 pin 时填充；此处保留 None
    assert cfg.agent.rate_limit.state_dir is None


# --- 两者必须成对存在 -------------------------------------------------------------


def test_rejects_api_keys_without_rate_limit(tmp_path: Path) -> None:
    body = "  api_keys: [key-A, key-B]\n"
    cfg_path = _write_config(tmp_path / "cfg.yaml", body)
    with pytest.raises(ValueError, match=r"api_keys and agent\.rate_limit"):
        load_app_config(cfg_path)


def test_rejects_rate_limit_without_api_keys(tmp_path: Path) -> None:
    body = """
  rate_limit:
    rpm_per_key: 600
    tpm_per_key: 1000000
"""
    cfg_path = _write_config(tmp_path / "cfg.yaml", body)
    with pytest.raises(ValueError, match=r"api_keys and agent\.rate_limit"):
        load_app_config(cfg_path)


# --- api_keys 校验 ---------------------------------------------------------------


def test_rejects_api_keys_not_list(tmp_path: Path) -> None:
    body = "  api_keys: not-a-list\n"
    cfg_path = _write_config(tmp_path / "cfg.yaml", body)
    with pytest.raises(ValueError, match="api_keys must be a list"):
        load_app_config(cfg_path)


def test_rejects_non_string_api_key(tmp_path: Path) -> None:
    body = """
  api_keys:
    - key-A
    - 42
"""
    cfg_path = _write_config(tmp_path / "cfg.yaml", body)
    with pytest.raises(ValueError, match=r"api_keys\[1\]"):
        load_app_config(cfg_path)


def test_rejects_empty_string_api_key(tmp_path: Path) -> None:
    body = """
  api_keys:
    - key-A
    - ""
"""
    cfg_path = _write_config(tmp_path / "cfg.yaml", body)
    with pytest.raises(ValueError, match="non-empty"):
        load_app_config(cfg_path)


def test_rejects_duplicate_api_keys(tmp_path: Path) -> None:
    body = """
  api_keys: [key-A, key-A]
  rate_limit:
    rpm_per_key: 600
    tpm_per_key: 1000000
"""
    cfg_path = _write_config(tmp_path / "cfg.yaml", body)
    with pytest.raises(ValueError, match="duplicate"):
        load_app_config(cfg_path)


# --- rate_limit 数值字段 ---------------------------------------------------------


def _rate_limited_body(**overrides: object) -> str:
    """构造一个"基本合法"的 rate_limit 块，然后按需覆写字段。"""
    defaults: dict[str, object] = {
        "rpm_per_key": 600,
        "tpm_per_key": 1_000_000,
        "safety_factor": 0.85,
        "reserve_output_tokens": 4096,
        "max_retries": 3,
        "backoff_initial_seconds": 1.0,
        "backoff_max_seconds": 10.0,
    }
    defaults.update(overrides)
    lines = ["  api_keys: [key-A, key-B]", "  rate_limit:"]
    for key, value in defaults.items():
        lines.append(f"    {key}: {value}")
    return "\n".join(lines) + "\n"


def test_rejects_non_positive_rpm(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", _rate_limited_body(rpm_per_key=0))
    with pytest.raises(ValueError, match="rpm_per_key"):
        load_app_config(cfg_path)


def test_rejects_non_positive_tpm(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", _rate_limited_body(tpm_per_key=-1))
    with pytest.raises(ValueError, match="tpm_per_key"):
        load_app_config(cfg_path)


def test_rejects_safety_factor_out_of_range(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", _rate_limited_body(safety_factor=0))
    with pytest.raises(ValueError, match="safety_factor"):
        load_app_config(cfg_path)
    cfg_path2 = _write_config(tmp_path / "cfg2.yaml", _rate_limited_body(safety_factor=1.2))
    with pytest.raises(ValueError, match="safety_factor"):
        load_app_config(cfg_path2)


def test_rejects_non_positive_reserve(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", _rate_limited_body(reserve_output_tokens=0))
    with pytest.raises(ValueError, match="reserve_output_tokens"):
        load_app_config(cfg_path)


def test_rejects_reserve_larger_than_max_tokens(tmp_path: Path) -> None:
    # max_tokens 默认 16384；reserve 超过它时应报错
    body = _rate_limited_body(reserve_output_tokens=32768)
    cfg_path = _write_config(tmp_path / "cfg.yaml", "  max_tokens: 8192\n" + body)
    with pytest.raises(ValueError, match="reserve_output_tokens"):
        load_app_config(cfg_path)


def test_rejects_negative_max_retries(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", _rate_limited_body(max_retries=-1))
    with pytest.raises(ValueError, match="max_retries"):
        load_app_config(cfg_path)


def test_rejects_backoff_max_lt_initial(tmp_path: Path) -> None:
    cfg_path = _write_config(
        tmp_path / "cfg.yaml",
        _rate_limited_body(backoff_initial_seconds=5.0, backoff_max_seconds=1.0),
    )
    with pytest.raises(ValueError, match="backoff_max_seconds"):
        load_app_config(cfg_path)
