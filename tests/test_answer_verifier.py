from __future__ import annotations

from data_agent_baseline.agents.answer_verifier import _VERIFIER_SYSTEM, verify_answer
from data_agent_baseline.agents.model import ModelMessage, ModelResponse


class _PromptCapturingVerifier:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[list[ModelMessage]] = []

    def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        self.messages.append(messages)
        return ModelResponse(content=self.response, raw_response=self.response)


def test_verifier_prompt_rejects_global_superlative_multiple_rows() -> None:
    adapter = _PromptCapturingVerifier(
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
    system_prompt = adapter.messages[0][0].content
    user_prompt = adapter.messages[0][1].content
    assert "Global singular superlative questions" in system_prompt
    assert "ONE entity at the global extreme" in system_prompt
    assert "TOTAL ROWS: 2" in user_prompt


def test_verifier_prompt_allows_grouped_superlative_multiple_rows() -> None:
    adapter = _PromptCapturingVerifier('{"verdict": "ok"}')

    rejection = verify_answer(
        adapter,
        "Which driver is fastest in each team?",
        ["team", "driver"],
        [["red", "Alice"], ["blue", "Bob"]],
    )

    assert rejection is None
    system_prompt = adapter.messages[0][0].content
    assert "Do NOT apply this rule" in system_prompt
    assert "for each" in system_prompt
    assert "per group/category/region" in system_prompt
    assert "one row per" in system_prompt
    assert "group" in system_prompt


def test_verifier_system_documents_grouped_superlative_exception() -> None:
    assert "Global singular superlative questions" in _VERIFIER_SYSTEM
    assert "which X has the highest Y for each Z" in _VERIFIER_SYSTEM
