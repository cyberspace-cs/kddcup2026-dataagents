from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_tokenizer_loader_state() -> None:
    """Reset tokenizer singleton before each test so cross-file ordering is safe."""
    import data_agent_baseline.agents.tokenizer_loader as mod

    mod._loaded = False
    mod._tokenizer = None
