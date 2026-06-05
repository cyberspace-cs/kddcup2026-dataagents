"""Dashboard FastAPI app factory tests."""

from __future__ import annotations

from pathlib import Path


def test_dashboard_root_returns_404_when_static_index_is_missing(tmp_path: Path) -> None:
    from starlette.testclient import TestClient

    from data_agent_baseline.dashboard.app import create_app

    app = create_app(tmp_path / "traces.db", static_dir=tmp_path / "missing-static")

    resp = TestClient(app).get("/")

    assert resp.status_code == 404
    assert "Dashboard static index not found" in resp.json()["detail"]
