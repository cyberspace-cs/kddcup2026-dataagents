"""配置加载 byte-compat 回归：每个 valid YAML 经 `load_app_config` 后，
`dataclasses.asdict` 序列化结果必须与冻结的 `expected.normalized.json` 字节一致。

PROJECT_ROOT 路径在 fixture 中以 `<PROJECT_ROOT>` 占位符存储，比对时替换为实际值。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from data_agent_baseline.config import PROJECT_ROOT, load_app_config

VALID_DIR = Path(__file__).parent / "fixtures" / "configs" / "valid"


def _serialize(obj: object, project_root_str: str) -> object:
    """把 Path 转成字符串；PROJECT_ROOT 前缀替换成 `<PROJECT_ROOT>` 占位符。"""
    if isinstance(obj, Path):
        s = str(obj)
        if s.startswith(project_root_str):
            return s.replace(project_root_str, "<PROJECT_ROOT>", 1)
        return s
    if isinstance(obj, dict):
        return {k: _serialize(v, project_root_str) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x, project_root_str) for x in obj]
    return obj


@pytest.mark.parametrize(
    "yaml_path",
    sorted(VALID_DIR.glob("*.yaml")),
    ids=lambda p: p.stem,
)
def test_normalized_json_byte_compat(yaml_path: Path) -> None:
    expected_path = yaml_path.with_suffix(".normalized.json")
    assert expected_path.exists(), f"missing fixture: {expected_path.name}"

    cfg = load_app_config(yaml_path)
    payload = _serialize(asdict(cfg), str(PROJECT_ROOT))
    actual_text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    assert actual_text == expected_path.read_text()
