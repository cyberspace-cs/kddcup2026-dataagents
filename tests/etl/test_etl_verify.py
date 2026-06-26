"""Regression tests for ETL value retry and field verification."""

from __future__ import annotations

from typing import Any, cast

from agents.etl._verify import retry_missing_values, verify_field_values
from agents.llm.types import ModelResponse


class _ReplyAdapter:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list[object] = []

    def complete(self, messages: object) -> ModelResponse:
        self.messages.append(messages)
        return ModelResponse(content=self.reply)


def test_retry_missing_values_matches_mixed_case_reply_keys() -> None:
    text = "RecordID: 1 | PersonalCode: "

    result = retry_missing_values(
        cast(Any, _ReplyAdapter("personalcode: 123")),
        text,
        schema_columns=["RecordID", "PersonalCode"],
        primary_key="RecordID",
        anchor_keys=["RecordID"],
        entity_groups={"1": ["RecordID 1 has PersonalCode 123."]},
        forced_gaps={"1": ["PersonalCode"]},
    )

    assert result == "RecordID: 1 | PersonalCode: 123 "


def test_verify_field_values_matches_mixed_case_reply_keys() -> None:
    text = "RecordID: 1 | PersonalCode: 123"

    result = verify_field_values(
        cast(Any, _ReplyAdapter("personalcode: 456")),
        text,
        schema_columns=["RecordID", "PersonalCode"],
        primary_key="RecordID",
        anchor_keys=["RecordID"],
        entity_groups={"1": ["RecordID 1 has PersonalCode 456."]},
    )

    assert result == "RecordID: 1 | PersonalCode: 456 "


def test_verify_field_values_matches_metadata_case_insensitively() -> None:
    adapter = _ReplyAdapter("ALL_CORRECT")

    verify_field_values(
        cast(Any, adapter),
        "RecordID: 1 | TradingDay: 2021-01-08 | Amount: 10",
        schema_columns=["RecordID", "TradingDay", "Amount"],
        primary_key="RecordID",
        anchor_keys=["RecordID"],
        entity_groups={"1": ["RecordID 1 traded on 2021-01-08 with amount 10."]},
        schema_field_types={"tradingday": "date", "amount": "number"},
        schema_field_defs={"amount": "Reported amount."},
    )

    assert adapter.messages
    messages = cast(list[Any], adapter.messages[0])
    prompt = messages[0].content
    assert "Fields to verify: Amount" in prompt
    assert "TradingDay" not in prompt.split("CURRENT EXTRACTION:", 1)[0]
    assert "Amount: Reported amount." in prompt


def test_verify_field_values_rejects_none_when_value_exists() -> None:
    """NONE correction must not clear an already-populated field."""
    text = "RecordID: 1 | TotalAssets: 123"

    result = verify_field_values(
        cast(Any, _ReplyAdapter("totalassets: NONE")),
        text,
        schema_columns=["RecordID", "TotalAssets"],
        primary_key="RecordID",
        anchor_keys=["RecordID"],
        entity_groups={"1": ["RecordID 1 does not mention TotalAssets."]},
        schema_field_types={"TotalAssets": "number"},
    )

    assert result == "RecordID: 1 | TotalAssets: 123"


def test_verify_field_values_allows_none_when_value_empty() -> None:
    """NONE on an already-empty field is a no-op (no change)."""
    text = "RecordID: 1 | TotalAssets: "

    result = verify_field_values(
        cast(Any, _ReplyAdapter("totalassets: NONE")),
        text,
        schema_columns=["RecordID", "TotalAssets"],
        primary_key="RecordID",
        anchor_keys=["RecordID"],
        entity_groups={"1": ["RecordID 1 does not mention TotalAssets."]},
        schema_field_types={"TotalAssets": "number"},
    )

    assert result == "RecordID: 1 | TotalAssets: "


def test_verify_field_values_accepts_string_corrections() -> None:
    text = "RecordID: 1 | Status: open"

    result = verify_field_values(
        cast(Any, _ReplyAdapter("status: closed")),
        text,
        schema_columns=["RecordID", "Status"],
        primary_key="RecordID",
        anchor_keys=["RecordID"],
        entity_groups={"1": ["RecordID 1 has status closed."]},
        schema_field_types={"Status": "category"},
    )

    assert result == "RecordID: 1 | Status: closed "
