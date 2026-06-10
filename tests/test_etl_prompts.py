from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from agents.etl._compress import _clean_compressed_entity_lines
from agents.etl.extractor import (
    _deterministic_factor,
    _ensure_local_record_anchor,
    _parse_schema_response,
    _parse_unit_factor_response,
)
from agents.llm.types import ModelResponse


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
    pk, anchors, types, cols, units, defs = result
    assert pk == "fundcode"
    assert anchors == ["fundcode", "fundname"]
    assert types == {"fundcode": "scalar_id", "fundname": "string", "nav": "number"}
    assert cols == ["fundcode", "fundname", "nav"]
    assert units == {}
    assert defs == {}


def test_parse_schema_response_out_of_order() -> None:
    result = _parse_schema_response(
        "TYPES: id=integer, score=number\nANCHORS: id\nPK: id\nid, score, label"
    )
    assert result is not None
    pk, anchors, _, cols, _, _ = result
    assert pk == "id"
    assert anchors == ["id"]
    assert cols == ["id", "score", "label"]


def test_parse_schema_response_missing_anchors_and_types() -> None:
    result = _parse_schema_response("PK: patient_id\npatient_id, age, weight")
    assert result is not None
    pk, anchors, types, cols, units, defs = result
    assert pk == "patient_id"
    assert anchors == ["patient_id"]
    assert types == {}
    assert cols == ["patient_id", "age", "weight"]
    assert units == {}
    assert defs == {}


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
    pk, anchors, types, cols, _, _ = result
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
    _, _, _, cols, _, _ = result
    assert "name" in cols


def test_deterministic_factor_cny_magnitudes() -> None:
    """Within the CNY family the factor is scale(source)/scale(target)."""
    assert _deterministic_factor("亿元", "万元") == 10000.0
    assert _deterministic_factor("元", "万元") == 0.0001
    assert _deterministic_factor("万元", "元") == 10000.0
    assert _deterministic_factor("万元", "亿元") == 0.0001
    assert _deterministic_factor("元", "亿元") == 1e-8


def test_deterministic_factor_percent_and_unknown() -> None:
    assert _deterministic_factor("%", "ratio") == 0.01
    assert _deterministic_factor("ratio", "%") == 100.0
    # Different families or unknown units are not computable.
    assert _deterministic_factor("元", "%") is None
    assert _deterministic_factor("万吨", "吨") is None
    assert _deterministic_factor("万元", "USD") is None


def test_unit_factor_response_corrects_inverted_direction() -> None:
    """元→万元 must divide by 10000 even when the LLM echoes the *10000 example.

    Regression: ed_chinafibalancesheetrmb (task_47) — the LLM pattern-matched
    the 亿元→万元 example and emitted *10000 for every 元-sourced field.
    """
    source = {"corporatesavings": "元", "totalloans": "元"}
    parsed = _parse_unit_factor_response(
        "corporatesavings=万元*10000, totalloans=万元*10000",
        source,
    )
    assert parsed == {
        "corporatesavings": ("万元", 0.0001),
        "totalloans": ("万元", 0.0001),
    }


def test_unit_factor_response_fills_missing_factor_for_known_pair() -> None:
    """A known unit pair no longer depends on the LLM emitting *factor."""
    parsed = _parse_unit_factor_response("gdp=万元", {"gdp": "亿元"})
    assert parsed == {"gdp": ("万元", 10000.0)}


def test_unit_factor_response_keeps_llm_factor_for_unknown_pair() -> None:
    parsed = _parse_unit_factor_response("weight=吨*10000", {"weight": "万吨"})
    assert parsed == {"weight": ("吨", 10000.0)}


def test_unit_factor_response_drops_equal_scale_spellings() -> None:
    """Same-scale units (factor 1) must not produce a conversion entry."""
    parsed = _parse_unit_factor_response("rate=ratio*0.01", {"rate": "%"})
    assert parsed == {"rate": ("ratio", 0.01)}
    parsed = _parse_unit_factor_response("gdp=万元*1", {"gdp": "万元"})
    assert parsed == {}


_KM_FIXTURE = """\
# Governance

### 2.1 Balance Sheet (`ed_bs`)

| Column | Semantic Definition |
|---|---|
| `totalassets` | Aggregate assets |
| `totalliabilities` | Aggregate liabilities |

Key Relationship: identity.

### 2.9 Cross-Table Disambiguation

| Pair | Rule |
|---|---|
| `totalassets` vs `netabroadassets` | Never substitute |

### 3.1 Example Queries on `ed_bs`

No field table here, only prose.
"""


def test_km_table_fields_parses_backticked_table() -> None:
    from agents.etl.extractor import _km_table_fields

    fields = _km_table_fields(_KM_FIXTURE, "ed_bs")
    assert fields == {
        "totalassets": "Aggregate assets",
        "totalliabilities": "Aggregate liabilities",
    }


def test_km_table_fields_parses_chinese_table() -> None:
    from agents.etl.extractor import _km_table_fields

    km = (
        "### 2.2 国内生产总值 (Gross Domestic Product)\n\n"
        "| Column | Semantic Definition |\n"
        "|--------|---------------------|\n"
        "| 国内生产总值(百万元) | China's GDP in millions of RMB. |\n"
        "| 人均国内生产总值(元/人) | Per capita GDP. |\n"
    )
    fields = _km_table_fields(km, "国内生产总值")
    assert "国内生产总值(百万元)" in fields
    assert "人均国内生产总值(元/人)" in fields


def test_km_table_fields_rejects_substring_stem_anchor() -> None:
    """`mf_netvalue` must not anchor inside `mf_netvalueperformancehis`."""
    from agents.etl.extractor import _km_table_fields

    km = (
        "### 2.8 Net Value Performance History (`mf_netvalueperformancehis`)\n\n"
        "| Column | Semantic Definition |\n"
        "|---|---|\n"
        "| `nvdailygrowthrate` | Daily growth rate |\n"
    )
    assert _km_table_fields(km, "mf_netvalue") == {}


def test_km_table_fields_retries_next_heading_and_skips_junk_cells() -> None:
    """A table-less heading falls through to the next matching one; prose-like
    cells (`a` vs `b`) never become columns."""
    from agents.etl.extractor import _km_table_fields

    km = "### 1.1 Overview of `ed_bs`\n\nProse only here.\n\n" + _KM_FIXTURE
    fields = _km_table_fields(km, "ed_bs")
    assert sorted(fields) == ["totalassets", "totalliabilities"]
    # the disambiguation table's "`totalassets` vs `netabroadassets`" cell
    # must never parse as a column even if that section were matched
    assert not any("vs" in c for c in fields)


def test_apply_km_schema_floor_readds_omitted_columns() -> None:
    from agents.etl.extractor import _apply_km_schema_floor

    columns = ["record_id", "totalassets"]
    raw_defs = {"totalassets": "from llm"}
    km_fields = {
        "totalassets": "Aggregate assets",
        "totalliabilities": "Aggregate liabilities",
    }

    readded = _apply_km_schema_floor(columns, raw_defs, km_fields)

    assert readded == ["totalliabilities"]
    assert columns == ["record_id", "totalassets", "totalliabilities"]
    # existing LLM definition wins; the km definition fills the gap
    assert raw_defs["totalassets"] == "from llm"
    assert raw_defs["totalliabilities"] == "Aggregate liabilities"


class _FailingAdapter:
    """Asserts the knowledge path is skipped before any LLM call."""

    def complete(self, messages: list[Any]) -> ModelResponse:
        raise AssertionError("LLM must not be called when km lacks the table")


class _SequenceAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def complete(self, messages: list[Any]) -> ModelResponse:
        self.calls += 1
        return ModelResponse(content=self._responses.pop(0))


def test_parse_knowledge_schema_skips_unmentioned_table(tmp_path) -> None:
    """Gate A: km never mentions the table → no LLM call, fall to inference.

    Regression: task_59 ed_consumerpriceindex — the entity matcher
    force-matched ed_retailvalueofscgoods and the schema inherited a wrong
    table's columns, blanking every CPI value.
    """
    from types import SimpleNamespace

    from agents.etl.extractor import _parse_knowledge_schema

    (tmp_path / "knowledge.md").write_text(_KM_FIXTURE, encoding="utf-8")
    task = SimpleNamespace(context_dir=tmp_path)

    result = _parse_knowledge_schema(
        cast(Any, task),
        Path("doc/ed_consumerpriceindex.md"),
        cast(Any, _FailingAdapter()),
        "Strategic Unit 3 reported a QoqCPI of 100.1 for the period.",
    )

    assert result is None


def test_parse_knowledge_schema_rejects_filter_output_without_stem(tmp_path) -> None:
    """Gate B: the pre-filter returning another entity's section is no-match."""
    from types import SimpleNamespace

    from agents.etl.extractor import _parse_knowledge_schema

    km = _KM_FIXTURE + "\nCross-reference: see also `ed_cpi` for deflators.\n"
    (tmp_path / "knowledge.md").write_text(km, encoding="utf-8")
    task = SimpleNamespace(context_dir=tmp_path)
    adapter = _SequenceAdapter(
        ["### 2.1 Balance Sheet (`ed_bs`)\n\n| Column | Def |\n|---|---|\n| `totalassets` | x |"]
    )

    result = _parse_knowledge_schema(
        cast(Any, task),
        Path("doc/ed_cpi.md"),
        cast(Any, adapter),
        "Unit 3 reported values for the period in question here.",
    )

    assert result is None
    assert adapter.calls == 1  # only the pre-filter ran; main call skipped


def test_parse_knowledge_schema_applies_km_floor(tmp_path) -> None:
    """Happy path: LLM omits a governance column; the floor re-adds it."""
    from types import SimpleNamespace

    from agents.etl.extractor import _parse_knowledge_schema

    (tmp_path / "knowledge.md").write_text(_KM_FIXTURE, encoding="utf-8")
    task = SimpleNamespace(context_dir=tmp_path)
    section = (
        "### 2.1 Balance Sheet (`ed_bs`)\n\n"
        "| Column | Semantic Definition |\n|---|---|\n"
        "| `totalassets` | Aggregate assets |\n"
        "| `totalliabilities` | Aggregate liabilities |\n"
    )
    schema_reply = (
        "PK: record_id\n"
        "ANCHORS: record_id\n"
        "TYPES: record_id=integer_scalar local_record_id, totalassets=number\n"
        "record_id, totalassets"
    )
    adapter = _SequenceAdapter([section, schema_reply])

    result = _parse_knowledge_schema(
        cast(Any, task),
        Path("doc/ed_bs.md"),
        cast(Any, adapter),
        "Record 3 of ed_bs held assets. Record 4 held more. Record 5 held some.",
    )

    assert result is not None
    columns, _anchors, _types, _units, defs = result
    assert "totalliabilities" in columns  # floor re-added the omitted column
    assert defs.get("totalliabilities") == "Aggregate liabilities"
