"""配置加载 byte-compat 回归：每个 valid YAML 经 `load_app_config` 后，
`dataclasses.asdict` 序列化结果必须与冻结的 `expected.normalized.json` 字节一致。

PROJECT_ROOT 路径在 fixture 中以 `<PROJECT_ROOT>` 占位符存储，比对时替换为实际值。
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from data_agent_baseline.config import PROJECT_ROOT, load_app_config

VALID_DIR = Path(__file__).parent / "fixtures" / "configs" / "valid"


def _serialize(obj: object, project_root: Path) -> object:
    """把 Path 转成 posix 字符串；PROJECT_ROOT 前缀替换成 `<PROJECT_ROOT>` 占位符。

    用 `as_posix()` 而不是 `str()`，否则 Windows 下分隔符是反斜杠，与冻结的
    forward-slash fixture 字节不一致。
    """
    if isinstance(obj, Path):
        s = obj.as_posix()
        root = project_root.as_posix()
        if s.startswith(root):
            return s.replace(root, "<PROJECT_ROOT>", 1)
        return s
    if isinstance(obj, dict):
        return {k: _serialize(v, project_root) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x, project_root) for x in obj]
    return obj


@pytest.mark.parametrize(
    "yaml_path",
    sorted(VALID_DIR.glob("*.yaml")),
    ids=lambda p: p.stem,
)
def test_normalized_json_byte_compat(yaml_path: Path) -> None:
    expected_path = yaml_path.with_suffix(".normalized.json")
    assert expected_path.exists(), f"missing fixture: {expected_path.name}"

    # Fixtures with POSIX absolute paths (e.g. `/tmp/...`) can't byte-match on
    # Windows: `Path('/tmp/foo')` resolves to drive-relative `E:/tmp/foo` here.
    # Skip those cases on win32; the production target is the Linux Docker
    # image, where the byte-compat property is meaningful and CI verifies it.
    if sys.platform == "win32" and "/tmp/" in yaml_path.read_text():
        pytest.skip("POSIX-only paths in fixture; byte-stable on Linux only")

    cfg = load_app_config(yaml_path)
    payload = _serialize(asdict(cfg), PROJECT_ROOT)
    actual_text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    assert actual_text == expected_path.read_text()
