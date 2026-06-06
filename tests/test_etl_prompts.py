from __future__ import annotations

from typing import Any, cast

from agents.etl._compress import (
    _clean_compressed_entity_lines,
    compress_chunk,
    extract_csv_from_text,
)
from agents.etl.extractor import _ensure_local_record_anchor, _parse_schema_response
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
    assert "no structured field" in lower


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
    assert "do not invent synonyms" in lower
    assert "do not substitute" in lower
    assert "omitting that non-identity field" in lower
    assert "missing/not provided/unavailable" in lower
    assert "skip that paragraph" in lower
    assert "only identity fields and empty data fields" in lower
    assert "qualifiers are binding" in lower
    assert "overall/global" in lower
    assert "qualified/category-specific" in lower


def test_compress_prompt_has_short_absolute_type_rules() -> None:
    adapter = _CapturingAdapter()

    compress_chunk(
        cast(Any, adapter),
        "fundcode 159001. The fund name is 万家上证50交易型开放式指数证券投资基金.",
        schema_columns=[
            "fundcode",
            "lowestsumsubscribing",
            "establishmentdate",
            "iffof",
        ],
        primary_key="fundcode",
        schema_field_types={
            "fundcode": "scalar_id",
            "lowestsumsubscribing": "number",
            "establishmentdate": "date",
            "iffof": "boolean",
        },
    )

    prompt = adapter.messages[1].content
    lower = prompt.lower()

    assert "hard type rules" in lower
    assert "wrong type" in lower
    assert "leave the value blank" in lower
    assert "not put names, descriptions, or labels into number/date/boolean" in lower
    assert "numeric values only (integers or decimals)" in lower
    assert "fields: lowestsumsubscribing" in lower
    assert "convert to yyyy-mm-dd" in lower
    assert "fields: establishmentdate" in lower
    assert "normalize to true or false" in lower
    assert "fields: iffof" in lower


def test_direct_extract_prompt_repeats_hard_type_rules() -> None:
    adapter = _CapturingAdapter()

    extract_csv_from_text(
        cast(Any, adapter),
        "fundcode: 159001 | lowestsumsubscribing: 万家上证50交易型开放式指数证券投资基金",
        ["fundcode", "lowestsumsubscribing"],
        "show rows",
        schema_field_types={
            "fundcode": "scalar_id",
            "lowestsumsubscribing": "number",
        },
    )

    prompt = adapter.messages[1].content
    lower = prompt.lower()

    assert "hard type rules" in lower
    assert "wrong type" in lower
    assert "numeric values only (integers or decimals)" in lower
    assert "fields: lowestsumsubscribing" in lower


def test_compress_prompt_distinguishes_local_record_ids_from_personalcode() -> None:
    adapter = _CapturingAdapter()

    compress_chunk(
        cast(Any, adapter),
        "档案 353 的经理人个人代码为 101016352。",
        schema_columns=["record_id", "personalcode", "manager_name"],
        primary_key="record_id",
        anchor_keys=["record_id", "personalcode", "manager_name"],
        schema_field_types={
            "record_id": "integer_scalar local_record_id",
            "personalcode": "scalar_id",
            "manager_name": "string",
        },
    )

    prompt = adapter.messages[1].content
    lower = prompt.lower()

    assert "record_id: integer_scalar local_record_id" in lower
    assert '"353"' in lower
    assert "skip the paragraph" in lower
    assert "personalcode/personalcode fields are person identifiers" in lower
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


def test_compress_clean_blanks_values_with_wrong_schema_type() -> None:
    cleaned = _clean_compressed_entity_lines(
        (
            "fundcode: 159001 | lowestsumsubscribing: 万家上证50交易型开放式指数证券投资基金 "
            "| lowestsumredemption: 900000.0 | foundedsize: 1,234.50 "
            "| establishmentdate: 股票型 | iffof: 万家上证50交易型开放式指数证券投资基金 "
            "| ifinitiatingfund: 否"
        ),
        schema_columns=[
            "fundcode",
            "lowestsumsubscribing",
            "lowestsumredemption",
            "foundedsize",
            "establishmentdate",
            "iffof",
            "ifinitiatingfund",
        ],
        schema_field_types={
            "fundcode": "scalar_id",
            "lowestsumsubscribing": "number",
            "lowestsumredemption": "number",
            "foundedsize": "number",
            "establishmentdate": "date",
            "iffof": "boolean",
            "ifinitiatingfund": "boolean",
        },
    )

    assert "lowestsumsubscribing:  | lowestsumredemption: 900000.0" in cleaned
    assert "foundedsize: 1234.50" in cleaned
    assert "establishmentdate:  | iffof:  | ifinitiatingfund: false" in cleaned


def test_schema_adds_record_id_for_repeated_local_record_labels() -> None:
    columns, anchors = _ensure_local_record_anchor(
        "档案 21 记录一个经理。\n\n战略单元 353 记录另一个经理。\n\n记录 418 继续。",
        ["personalcode", "manager_name"],
        ["personalcode", "manager_name"],
    )

    assert columns == ["record_id", "personalcode", "manager_name"]
    assert anchors == ["record_id", "personalcode", "manager_name"]


def test_parse_schema_response_normal_order() -> None:
    result = _parse_schema_response(
        "PK: fundcode\n"
        "ANCHORS: fundcode, fundname\n"
        "TYPES: fundcode=scalar_id, fundname=string, nav=number\n"
        "fundcode, fundname, nav"
    )
    assert result is not None
    pk, anchors, types, cols = result
    assert pk == "fundcode"
    assert anchors == ["fundcode", "fundname"]
    assert types == {"fundcode": "scalar_id", "fundname": "string", "nav": "number"}
    assert cols == ["fundcode", "fundname", "nav"]


def test_parse_schema_response_out_of_order() -> None:
    result = _parse_schema_response(
        "TYPES: id=integer, score=number\nANCHORS: id\nPK: id\nid, score, label"
    )
    assert result is not None
    pk, anchors, _types, cols = result
    assert pk == "id"
    assert anchors == ["id"]
    assert cols == ["id", "score", "label"]


def test_parse_schema_response_missing_anchors_and_types() -> None:
    result = _parse_schema_response("PK: patient_id\npatient_id, age, weight")
    assert result is not None
    pk, anchors, types, cols = result
    assert pk == "patient_id"
    assert anchors == ["patient_id"]
    assert types == {}
    assert cols == ["patient_id", "age", "weight"]


def test_parse_schema_response_extra_commentary_ignored() -> None:
    result = _parse_schema_response(
        "Here is the schema:\n"
        "PK: record_id\n"
        "ANCHORS: record_id, province_name\n"
        "TYPES: record_id=integer_scalar local_record_id, gdp=number\n"
        "record_id, province_name, gdp\n"
        "Note: gdp is in billions."
    )
    assert result is not None
    pk, anchors, types, cols = result
    assert pk == "record_id"
    assert anchors == ["record_id", "province_name"]
    assert "gdp" in types
    assert cols == ["record_id", "province_name", "gdp"]


def test_parse_schema_response_returns_none_on_garbage() -> None:
    assert _parse_schema_response("NONE") is None
    assert _parse_schema_response("I cannot determine the schema.") is None
    assert _parse_schema_response("PK: x\nonly_one_column") is None


def test_parse_schema_response_anchor_added_to_columns() -> None:
    result = _parse_schema_response("PK: id\nANCHORS: id, name\nid, score")
    assert result is not None
    _, _, _, cols = result
    assert "name" in cols


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

    assert "start each line with id as stated in the paragraph" in lower
    assert "anchors every line" in lower
