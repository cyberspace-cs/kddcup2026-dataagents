from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from agents.etl._compress import _clean_compressed_entity_lines
from agents.etl._schema import (
    _ensure_local_record_anchor,
    _parse_schema_response,
)
from agents.etl._units import _deterministic_factor, _parse_unit_factor_response
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


@pytest.mark.parametrize(
    ("raw", "pk", "anchors", "types", "cols"),
    [
        pytest.param(
            "PK: fundcode\n"
            "ANCHORS: fundcode, fundname\n"
            "TYPES: fundcode=scalar_id, fundname=string, nav=number\n"
            "fundcode, fundname, nav",
            "fundcode",
            ["fundcode", "fundname"],
            {"fundcode": "scalar_id", "fundname": "string", "nav": "number"},
            ["fundcode", "fundname", "nav"],
            id="normal-order",
        ),
        # 各前缀行乱序出现也按前缀识别，不按位置
        pytest.param(
            "TYPES: id=integer, score=number\nANCHORS: id\nPK: id\nid, score, label",
            "id",
            ["id"],
            {"id": "integer", "score": "number"},
            ["id", "score", "label"],
            id="out-of-order",
        ),
        # 只有 PK 行：anchors 回退为 [pk]，types 为空
        pytest.param(
            "PK: patient_id\npatient_id, age, weight",
            "patient_id",
            ["patient_id"],
            {},
            ["patient_id", "age", "weight"],
            id="missing-anchors-and-types",
        ),
        # 前后的 prose 行被忽略，不污染列清单
        pytest.param(
            "Here is the schema:\n"
            "PK: record_id\n"
            "ANCHORS: record_id, province_name\n"
            "TYPES: record_id=integer_scalar local_record_id, gdp=number\n"
            "record_id, province_name, gdp\n"
            "Note: gdp is in billions.",
            "record_id",
            ["record_id", "province_name"],
            {"record_id": "integer_scalar local_record_id", "gdp": "number"},
            ["record_id", "province_name", "gdp"],
            id="extra-commentary-ignored",
        ),
        # 声明为 ANCHOR 但缺席列清单的字段被补到 columns 尾部
        pytest.param(
            "PK: id\nANCHORS: id, name\nid, score",
            "id",
            ["id", "name"],
            {},
            ["id", "score", "name"],
            id="anchor-appended-to-columns",
        ),
    ],
)
def test_parse_schema_response_variants(
    raw: str, pk: str, anchors: list[str], types: dict[str, str], cols: list[str]
) -> None:
    result = _parse_schema_response(raw)
    assert result is not None
    got_pk, got_anchors, got_types, got_cols, units, defs = result
    assert (got_pk, got_anchors, got_types, got_cols) == (pk, anchors, types, cols)
    assert units == {}
    assert defs == {}


@pytest.mark.parametrize(
    "raw",
    [
        "NONE",
        "I cannot determine the schema.",
        "PK: x\nonly_one_column",
    ],
)
def test_parse_schema_response_returns_none_on_garbage(raw: str) -> None:
    assert _parse_schema_response(raw) is None


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        # Within the CNY family the factor is scale(source)/scale(target).
        ("亿元", "万元", 10000.0),
        ("元", "万元", 0.0001),
        ("万元", "元", 10000.0),
        ("万元", "亿元", 0.0001),
        ("元", "亿元", 1e-8),
        ("%", "ratio", 0.01),
        ("ratio", "%", 100.0),
        # Different families or unknown units are not computable.
        ("元", "%", None),
        ("万吨", "吨", None),
        ("万元", "USD", None),
    ],
)
def test_deterministic_factor(source: str, target: str, expected: float | None) -> None:
    assert _deterministic_factor(source, target) == expected


@pytest.mark.parametrize(
    ("raw", "source_units", "expected"),
    [
        # 元→万元 must divide by 10000 even when the LLM echoes the *10000 example.
        # Regression: ed_chinafibalancesheetrmb (task_47) — the LLM pattern-matched
        # the 亿元→万元 example and emitted *10000 for every 元-sourced field.
        pytest.param(
            "corporatesavings=万元*10000, totalloans=万元*10000",
            {"corporatesavings": "元", "totalloans": "元"},
            {"corporatesavings": ("万元", 0.0001), "totalloans": ("万元", 0.0001)},
            id="corrects-inverted-direction",
        ),
        # A known unit pair no longer depends on the LLM emitting *factor.
        pytest.param(
            "gdp=万元",
            {"gdp": "亿元"},
            {"gdp": ("万元", 10000.0)},
            id="fills-missing-factor-for-known-pair",
        ),
        pytest.param(
            "weight=吨*10000",
            {"weight": "万吨"},
            {"weight": ("吨", 10000.0)},
            id="keeps-llm-factor-for-unknown-pair",
        ),
        pytest.param(
            "rate=ratio*0.01",
            {"rate": "%"},
            {"rate": ("ratio", 0.01)},
            id="percent-spelling-converts",
        ),
        # Same-scale units (factor 1) must not produce a conversion entry.
        pytest.param("gdp=万元*1", {"gdp": "万元"}, {}, id="equal-scale-spelling-dropped"),
    ],
)
def test_parse_unit_factor_response(
    raw: str, source_units: dict[str, str], expected: dict[str, tuple[str, float]]
) -> None:
    assert _parse_unit_factor_response(raw, source_units) == expected


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
    from agents.etl.knowledge import km_table_fields

    fields = km_table_fields(_KM_FIXTURE, "ed_bs")
    assert fields == {
        "totalassets": "Aggregate assets",
        "totalliabilities": "Aggregate liabilities",
    }


def test_km_table_fields_parses_chinese_table() -> None:
    from agents.etl.knowledge import km_table_fields

    km = (
        "### 2.2 国内生产总值 (Gross Domestic Product)\n\n"
        "| Column | Semantic Definition |\n"
        "|--------|---------------------|\n"
        "| 国内生产总值(百万元) | China's GDP in millions of RMB. |\n"
        "| 人均国内生产总值(元/人) | Per capita GDP. |\n"
    )
    fields = km_table_fields(km, "国内生产总值")
    assert "国内生产总值(百万元)" in fields
    assert "人均国内生产总值(元/人)" in fields


def test_km_table_fields_rejects_substring_stem_anchor() -> None:
    """`mf_netvalue` must not anchor inside `mf_netvalueperformancehis`."""
    from agents.etl.knowledge import km_table_fields

    km = (
        "### 2.8 Net Value Performance History (`mf_netvalueperformancehis`)\n\n"
        "| Column | Semantic Definition |\n"
        "|---|---|\n"
        "| `nvdailygrowthrate` | Daily growth rate |\n"
    )
    assert km_table_fields(km, "mf_netvalue") == {}


def test_km_table_fields_retries_next_heading_and_skips_junk_cells() -> None:
    """A table-less heading falls through to the next matching one; prose-like
    cells (`a` vs `b`) never become columns."""
    from agents.etl.knowledge import km_table_fields

    km = "### 1.1 Overview of `ed_bs`\n\nProse only here.\n\n" + _KM_FIXTURE
    fields = km_table_fields(km, "ed_bs")
    assert sorted(fields) == ["totalassets", "totalliabilities"]
    # the disambiguation table's "`totalassets` vs `netabroadassets`" cell
    # must never parse as a column even if that section were matched
    assert not any("vs" in c for c in fields)


def test_apply_km_schema_floor_readds_omitted_columns() -> None:
    from agents.etl._schema import _apply_km_schema_floor

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

    def complete(self, messages: list[Any], *, tools: object | None = None) -> ModelResponse:
        raise AssertionError("LLM must not be called when km lacks the table")


class _SequenceAdapter:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def complete(self, messages: list[Any], *, tools: object | None = None) -> ModelResponse:
        self.calls += 1
        return ModelResponse(content=self._responses.pop(0))


def test_parse_knowledge_schema_skips_unmentioned_table(tmp_path) -> None:
    """Gate A: km never mentions the table → no LLM call, fall to inference.

    Regression: task_59 ed_consumerpriceindex — the entity matcher
    force-matched ed_retailvalueofscgoods and the schema inherited a wrong
    table's columns, blanking every CPI value.
    """
    from types import SimpleNamespace

    from agents.etl._schema import parse_knowledge_schema

    (tmp_path / "knowledge.md").write_text(_KM_FIXTURE, encoding="utf-8")
    task = SimpleNamespace(context_dir=tmp_path)

    result = parse_knowledge_schema(
        cast(Any, task),
        Path("doc/ed_consumerpriceindex.md"),
        cast(Any, _FailingAdapter()),
        "Strategic Unit 3 reported a QoqCPI of 100.1 for the period.",
    )

    assert result is None


def test_parse_knowledge_schema_rejects_filter_output_without_stem(tmp_path) -> None:
    """Gate B: the pre-filter returning another entity's section is no-match."""
    from types import SimpleNamespace

    from agents.etl._schema import parse_knowledge_schema

    km = _KM_FIXTURE + "\nCross-reference: see also `ed_cpi` for deflators.\n"
    (tmp_path / "knowledge.md").write_text(km, encoding="utf-8")
    task = SimpleNamespace(context_dir=tmp_path)
    adapter = _SequenceAdapter(
        ["### 2.1 Balance Sheet (`ed_bs`)\n\n| Column | Def |\n|---|---|\n| `totalassets` | x |"]
    )

    result = parse_knowledge_schema(
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

    from agents.etl._schema import parse_knowledge_schema

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

    result = parse_knowledge_schema(
        cast(Any, task),
        Path("doc/ed_bs.md"),
        cast(Any, adapter),
        "Record 3 of ed_bs held assets. Record 4 held more. Record 5 held some.",
    )

    assert result is not None
    columns, _anchors, _types, _units, defs = result
    assert "totalliabilities" in columns  # floor re-added the omitted column
    assert defs.get("totalliabilities") == "Aggregate liabilities"
