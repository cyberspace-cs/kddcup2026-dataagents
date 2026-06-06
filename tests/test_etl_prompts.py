from __future__ import annotations

from typing import Any, cast

from agents.etl._compress import _clean_compressed_entity_lines, compress_chunk
from agents.etl.extractor import _ensure_local_record_anchor
from agents.llm.types import ModelResponse


class _CapturingAdapter:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    def complete(self, messages: list[Any]) -> ModelResponse:
        self.messages = messages
        return ModelResponse(content="ID: 1 | field: value")


def test_compress_prompt_does_not_name_specific_filler_domains() -> None:
    """Compression should not pre-delete specific domains like hobbies or catering."""
    adapter = _CapturingAdapter()

    compress_chunk(
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


def test_compress_prompt_leaves_unsuitable_schema_fields_empty() -> None:
    """Schema-guided compression must not coerce prose into the nearest field."""
    adapter = _CapturingAdapter()

    compress_chunk(
        cast(Any, adapter),
        "ID 1. Acme has a conservative investment culture.",
        schema_columns=["ID", "entity_name", "fund_type"],
        primary_key="ID",
    )

    prompt = adapter.messages[1].content
    lower = prompt.lower()

    assert "only when it explicitly matches" in lower
    assert "do not force narrative text into the closest field" in lower
    assert "no explicit, suitable value" in lower
    assert "omitting that non-identity field" in lower
    assert "field type constraints" in lower
    assert "missing/not provided/unavailable" in lower
    assert "skip that paragraph entirely" in lower
    assert "only identity fields and empty data fields" in lower
    assert "field-name qualifiers are binding" in lower
    assert "overall/global" in lower
    assert "qualified/category-specific" in lower


def test_compress_prompt_distinguishes_local_record_ids_from_personalcode() -> None:
    adapter = _CapturingAdapter()

    compress_chunk(
        cast(Any, adapter),
        "档案 353 的经理人个人代码为 101016352。",
        schema_columns=["record_id", "personalcode", "manager_name"],
        primary_key="record_id",
        anchor_keys=["record_id", "personalcode", "manager_name"],
    )

    prompt = adapter.messages[1].content
    lower = prompt.lower()

    assert "local document labels" in lower
    assert "record identifiers" in lower
    assert "record_id: integer_scalar local_record_id" in lower
    assert "convert labels" in lower
    assert "scalar numeric portion" in lower
    assert "never output ranges" in lower
    assert "never put a local" in lower
    assert "personalcode/personalcodes" not in lower
    assert "personalcode/personalcode fields are person identifiers" in lower
    assert "manager name" in lower
    assert "local document record number" in lower


def test_compress_clean_drops_grouped_record_ids_before_clean_markdown() -> None:
    cleaned = _clean_compressed_entity_lines(
        "\n".join(
            [
                (
                    "record_id: 档案 197至217 | province_name: 黑龙江、甘肃、广东、陕西、"
                    "湖南和云南 | enddate: | gdp:"
                ),
                ("record_id: 档案 286 | province_name: 山西省 | primary_industry_gdp: 302.48"),
            ]
        ),
        schema_columns=["record_id", "province_name", "enddate", "gdp", "primary_industry_gdp"],
        schema_field_types={
            "record_id": "integer_scalar local_record_id",
            "province_name": "string",
            "enddate": "date",
            "gdp": "number",
            "primary_industry_gdp": "number",
        },
    )

    assert "197至217" not in cleaned
    assert "record_id: 286 | province_name: 山西省 | primary_industry_gdp: 302.48" in cleaned


def test_schema_adds_record_id_for_repeated_local_record_labels() -> None:
    columns, anchors = _ensure_local_record_anchor(
        "档案 21 记录一个经理。\n\n战略单元 353 记录另一个经理。\n\n记录 418 继续。",
        ["personalcode", "manager_name"],
        ["personalcode", "manager_name"],
    )

    assert columns == ["record_id", "personalcode", "manager_name"]
    assert anchors == ["record_id", "personalcode", "manager_name"]


def test_compress_prompt_multi_anchor() -> None:
    """When anchor_keys has >1 entry, prompt should list all identity fields."""
    adapter = _CapturingAdapter()

    compress_chunk(
        cast(Any, adapter),
        "PersonalCode 101001149. 张三 has AUM 1000.",
        schema_columns=["personalcode", "chinesename", "aum"],
        primary_key="personalcode",
        anchor_keys=["personalcode", "chinesename"],
    )

    prompt = adapter.messages[1].content
    lower = prompt.lower()

    assert "personalcode, chinesename" in lower
    assert "identity fields" in lower
    assert "include all of them" in lower


def test_compress_prompt_single_anchor_unchanged() -> None:
    """With a single anchor key, prompt should use original single-PK wording."""
    adapter = _CapturingAdapter()

    compress_chunk(
        cast(Any, adapter),
        "ID 1. Avery has score 7.",
        schema_columns=["ID", "score"],
        primary_key="ID",
        anchor_keys=["ID"],
    )

    prompt = adapter.messages[1].content
    lower = prompt.lower()

    assert "start each line with id as stated" in lower
    assert "anchors every line" in lower
