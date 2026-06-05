"""Tracing config parsing — enabled/disabled, db_path defaults."""

from __future__ import annotations

from pathlib import Path

import yaml

from data_agent_baseline.config import TracingConfig, load_app_config


def _write_config(path: Path, extra: dict | None = None) -> Path:
    base: dict = {}
    if extra:
        base.update(extra)
    cfg_path = path / "config.yaml"
    cfg_path.write_text(yaml.dump(base))
    return cfg_path


def test_tracing_disabled_by_default(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    config = load_app_config(cfg_path)
    assert config.tracing is None


def test_tracing_enabled(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path, {"tracing": {"enabled": True}})
    config = load_app_config(cfg_path)
    assert config.tracing is not None
    assert config.tracing.enabled is True
    assert isinstance(config.tracing.db_path, Path)


def test_tracing_custom_db_path(tmp_path: Path) -> None:
    custom = str(tmp_path / "custom.db")
    cfg_path = _write_config(tmp_path, {"tracing": {"enabled": True, "db_path": custom}})
    config = load_app_config(cfg_path)
    assert config.tracing is not None
    assert str(config.tracing.db_path) == custom


def test_tracing_config_frozen() -> None:
    tc = TracingConfig(enabled=True)
    import pytest

    with pytest.raises(AttributeError):
        tc.enabled = False  # type: ignore[misc]
