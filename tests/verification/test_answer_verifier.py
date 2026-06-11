from __future__ import annotations

from agents.llm.types import ModelMessage, ModelResponse
from agents.verification.answer import verify_answer


class _ScriptedVerifier:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(
        self, messages: list[ModelMessage], *, tools: object | None = None
    ) -> ModelResponse:
        return ModelResponse(content=self.response, raw_response=self.response)


def test_verifier_rejects_global_superlative_multiple_rows() -> None:
    adapter = _ScriptedVerifier(
        '{"verdict": "reject", "check": "row_count", '
        '"reason": "Global superlative answers should have one row."}'
    )

    rejection = verify_answer(
        adapter,
        "Who is the fastest driver?",
        ["driver"],
        [["Alice"], ["Bob"]],
    )

    assert rejection is not None
    assert "row_count" in rejection


def test_verifier_allows_grouped_superlative_multiple_rows() -> None:
    adapter = _ScriptedVerifier('{"verdict": "ok"}')

    rejection = verify_answer(
        adapter,
        "Which driver is fastest in each team?",
        ["team", "driver"],
        [["red", "Alice"], ["blue", "Bob"]],
    )

    assert rejection is None
