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


class _CapturingVerifier:
    def __init__(self, response: str = '{"verdict": "ok"}') -> None:
        self.response = response
        self.seen_content: object = None

    def complete(
        self, messages: list[ModelMessage], *, tools: object | None = None
    ) -> ModelResponse:
        del tools
        self.seen_content = messages[-1].content
        return ModelResponse(content=self.response, raw_response=self.response)


def test_verifier_sees_producing_code_when_provided() -> None:
    adapter = _CapturingVerifier()
    code = "SELECT EndDate, TotalAssets FROM t WHERE TotalAssets IS NOT NULL"

    verify_answer(adapter, "查一下总资产", ["TotalAssets"], [[1.0]], producing_code=code)

    assert adapter.seen_content is not None
    assert isinstance(adapter.seen_content, str)
    assert code in adapter.seen_content

    adapter2 = _CapturingVerifier()
    verify_answer(adapter2, "查一下总资产", ["TotalAssets"], [[1.0]])
    assert adapter2.seen_content is not None
    assert isinstance(adapter2.seen_content, str)
    assert "PRODUCING CODE" not in adapter2.seen_content


def test_verifier_includes_producing_code_for_count_questions() -> None:
    adapter = _CapturingVerifier()
    code = "SELECT Education, PersonalCode, ChineseName, TotalAUM FROM joined"

    verify_answer(
        adapter,
        "根据视频中方案 Q3-FM-07 的配置，满足该方案筛选规则的基金经理共有多少人？",
        ["基金经理人数"],
        [[6]],
        producing_code=code,
    )

    assert adapter.seen_content is not None
    assert isinstance(adapter.seen_content, str)
    assert "PRODUCING CODE" in adapter.seen_content
    assert code in adapter.seen_content


def test_verifier_rejects_null_filtered_listing() -> None:
    adapter = _ScriptedVerifier(
        '{"verdict": "reject", "check": "null_filter", '
        '"reason": "Final query drops NULL TotalAssets rows; rebuild without the filter."}'
    )

    rejection = verify_answer(
        adapter,
        "查一下总资产的金额大小",
        ["TotalAssets"],
        [[40622900.0]],
        producing_code="SELECT TotalAssets FROM t WHERE TotalAssets IS NOT NULL",
    )

    assert rejection is not None
    assert "null_filter" in rejection
