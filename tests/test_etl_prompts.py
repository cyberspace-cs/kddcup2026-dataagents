from __future__ import annotations

from typing import Any, cast

from data_agent_baseline.agents.model import ModelResponse
from data_agent_baseline.etl.extractor import _compress_chunk


class _CapturingAdapter:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    def complete(self, messages: list[Any]) -> ModelResponse:
        self.messages = messages
        return ModelResponse(content="ID: 1 | field: value")


def test_compress_prompt_does_not_name_specific_filler_domains() -> None:
    """Compression should not pre-delete specific domains like hobbies or catering."""
    adapter = _CapturingAdapter()

    _compress_chunk(
        cast(Any, adapter),
        "ID 1. Avery has score 7 and a chess club role.",
        schema_columns=["ID", "score", "club_role"],
        primary_key="ID",
    )

    prompt = adapter.messages[1].content
    lower = prompt.lower()

    assert "hobbies" not in lower
    assert "catering" not in lower
    assert "anecdotes" not in lower
    assert "question-relevant fact" in lower
    assert "structured field" in lower
    assert "schema field" in lower
