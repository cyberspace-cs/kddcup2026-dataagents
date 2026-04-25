"""每个 invalid YAML fixture 必须使 `load_app_config` 抛 ValueError，
错误消息字节等于配对的 `<n>.expected_error.txt`。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_agent_baseline.config import load_app_config

INVALID_DIR = Path(__file__).parent / "fixtures" / "configs" / "invalid"


@pytest.mark.parametrize(
    "yaml_path",
    sorted(INVALID_DIR.glob("*.yaml")),
    ids=lambda p: p.stem,
)
def test_invalid_yaml_raises_expected_message(yaml_path: Path) -> None:
    expected_path = yaml_path.with_suffix(".expected_error.txt")
    assert expected_path.exists(), f"missing fixture: {expected_path.name}"

    expected_message = expected_path.read_text().rstrip("\n")
    with pytest.raises(ValueError) as exc_info:
        load_app_config(yaml_path)
    assert str(exc_info.value) == expected_message
