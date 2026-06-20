"""Regression tests for ETL value retry and field verification."""

from __future__ import annotations

from typing import Any, cast

from agents.etl._verify import retry_missing_values, verify_field_values
from agents.llm.types import ModelResponse


class _ReplyAdapter:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, _messages: object) -> ModelResponse:
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
