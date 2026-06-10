"""`config.load_app_config` 对 PR-2 新加字段（api_keys / rate_limit）的解析测试。

覆盖：
- 两者都缺省 → 默认值（api_keys=() / rate_limit=None），与 PR-1 单 Key 路径行为一致
- 两者都显式配置 → 各字段正确解析，state_dir 规整成绝对 Path
- 拒收参数化表：只配其一 / api_keys 形状与内容非法 / 数值字段越界
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agents.config import load_app_config


def _write_config(path: Path, body: str) -> Path:
    """写一个最小可解析 YAML；body 追加到 agent: 块下面，作为额外字段。"""
    path.write_text(f"agent:\n  model: m\n  api_base: http://x\n  api_key: k\n{body}")
    return path


def test_defaults_when_both_absent(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", "")
    cfg = load_app_config(cfg_path)
    assert cfg.agent.api_keys == ()
    assert cfg.agent.rate_limit is None


@pytest.mark.skipif(
    sys.platform == "win32",
    # `state_dir: /tmp/ratelimit-test` resolves to drive-relative `E:/tmp/...`
    # on Windows; the assertion compares against `Path('/tmp/ratelimit-test')`
    # which becomes a different WindowsPath. Behaviour is correct on the Linux
    # production target.
    reason="POSIX-only path semantics; verified on Linux",
)
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


@pytest.mark.parametrize(
    ("body", "match"),
    [
        # --- 两者必须成对存在 ---
        pytest.param(
            "  api_keys: [key-A, key-B]\n",
            r"api_keys and agent\.rate_limit",
            id="api-keys-without-rate-limit",
        ),
        pytest.param(
            "\n  rate_limit:\n    rpm_per_key: 600\n    tpm_per_key: 1000000\n",
            r"api_keys and agent\.rate_limit",
            id="rate-limit-without-api-keys",
        ),
        # --- api_keys 形状与内容 ---
        pytest.param("  api_keys: not-a-list\n", "api_keys must be a list", id="api-keys-not-list"),
        pytest.param(
            "\n  api_keys:\n    - key-A\n    - 42\n",
            r"api_keys\[1\]",
            id="api-key-non-string",
        ),
        pytest.param(
            '\n  api_keys:\n    - key-A\n    - ""\n',
            "non-empty",
            id="api-key-empty-string",
        ),
        pytest.param(
            "\n  api_keys: [key-A, key-A]\n  rate_limit:\n    rpm_per_key: 600\n    tpm_per_key: 1000000\n",
            "duplicate",
            id="api-keys-duplicate",
        ),
        # --- rate_limit 数值字段 ---
        pytest.param(_rate_limited_body(rpm_per_key=0), "rpm_per_key", id="rpm-zero"),
        pytest.param(_rate_limited_body(tpm_per_key=-1), "tpm_per_key", id="tpm-negative"),
        pytest.param(_rate_limited_body(safety_factor=0), "safety_factor", id="safety-zero"),
        pytest.param(_rate_limited_body(safety_factor=1.2), "safety_factor", id="safety-above-one"),
        pytest.param(
            _rate_limited_body(reserve_output_tokens=0),
            "reserve_output_tokens",
            id="reserve-zero",
        ),
        # max_tokens 显式 8192；reserve 超过它时应报错
        pytest.param(
            "  max_tokens: 8192\n" + _rate_limited_body(reserve_output_tokens=32768),
            "reserve_output_tokens",
            id="reserve-above-max-tokens",
        ),
        pytest.param(_rate_limited_body(max_retries=-1), "max_retries", id="max-retries-negative"),
        pytest.param(
            _rate_limited_body(backoff_initial_seconds=5.0, backoff_max_seconds=1.0),
            "backoff_max_seconds",
            id="backoff-max-lt-initial",
        ),
    ],
)
def test_load_app_config_rejects_invalid_rate_limit(tmp_path: Path, body: str, match: str) -> None:
    cfg_path = _write_config(tmp_path / "cfg.yaml", body)
    with pytest.raises(ValueError, match=match):
        load_app_config(cfg_path)
