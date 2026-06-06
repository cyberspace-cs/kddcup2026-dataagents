from __future__ import annotations

import csv
import io

from agents.etl._merge import (
    deterministic_kv_to_csv,
    normalize_entity_lines,
    pre_merge_entity_lines,
)


def test_pre_merge_uses_secondary_anchor_when_primary_key_is_blank() -> None:
    text = "\n".join(
        [
            "personalcode: 101001204 | chinesename: 柳军 | equityfundnv: 851.187055",
            "personalcode:  | chinesename: 韩海平 | totalfundnv: 182.488480",
            "personalcode:  | chinesename: 柳军 | totalfundnv: 883.586211",
        ]
    )

    merged = pre_merge_entity_lines(
        text,
        primary_key="personalcode",
        anchor_keys=["personalcode", "chinesename"],
    )

    assert (
        "personalcode: 101001204 | chinesename: 柳军 | equityfundnv: 851.187055 "
        "| totalfundnv: 883.586211"
    ) in merged
    assert "chinesename: 韩海平 | totalfundnv: 182.488480" in merged


def test_pre_merge_does_not_merge_distinct_primary_keys_by_shared_secondary_anchor() -> None:
    text = "\n".join(
        [
            ("record_id: 150 | province_name: 黑龙江省 | primary_industry_gdp: 684.60"),
            "record_id: 197 | province_name: 黑龙江省 | gdp: 10235.00",
        ]
    )

    merged = pre_merge_entity_lines(
        text,
        primary_key="record_id",
        anchor_keys=["record_id", "province_name", "enddate"],
    )

    assert ("record_id: 150 | province_name: 黑龙江省 | primary_industry_gdp: 684.60") in merged
    assert "record_id: 197 | province_name: 黑龙江省 | gdp: 10235.00" in merged
    assert "record_id: 197 | province_name: 黑龙江省 | gdp: 10235.00 | primary" not in merged


def test_pre_merge_blocks_anchor_merge_when_another_anchor_conflicts() -> None:
    """Temp group shares one secondary anchor but conflicts on another — must not merge."""
    text = "\n".join(
        [
            "personalcode: 101001204 | chinesename: 柳军 | dept: A",
            "personalcode:  | chinesename: 柳军 | dept: B | bonus: 500",
        ]
    )

    merged = pre_merge_entity_lines(
        text,
        primary_key="personalcode",
        anchor_keys=["personalcode", "chinesename", "dept"],
    )

    lines = [row.strip() for row in merged.splitlines() if row.strip()]
    assert len(lines) == 2, f"Expected 2 separate groups, got {len(lines)}"
    real_line = next(row for row in lines if "personalcode: 101001204" in row)
    assert "bonus" not in real_line, "bonus should not leak into the real group"
    assert "dept: A" in real_line


def test_deterministic_kv_to_csv_matches_schema_keys_case_insensitively() -> None:
    text = (
        "ID: 1688094 | pid: 017-30133 | height_cm: 173.0 "
        "| admission_weight_kg: 86.10 | weight_loss_kg: 1.2 "
        "| discharge_weight_kg: 84.90 | origin: Emergency Department "
        "| discharge_status: Alive"
    )
    schema = [
        "id",
        "pid",
        "height_cm",
        "admission_weight_kg",
        "weight_loss_kg",
        "discharge_weight_kg",
        "origin",
        "discharge_status",
    ]

    normalized = normalize_entity_lines(text, schema, primary_key="id")
    csv_text = deterministic_kv_to_csv(normalized, schema, primary_key="id")
    assert csv_text is not None

    row = next(csv.DictReader(io.StringIO(csv_text)))
    assert row == {
        "id": "1688094",
        "pid": "017-30133",
        "height_cm": "173.0",
        "admission_weight_kg": "86.10",
        "weight_loss_kg": "1.2",
        "discharge_weight_kg": "84.90",
        "origin": "Emergency Department",
        "discharge_status": "Alive",
    }


def test_primary_key_detection_does_not_match_key_substrings() -> None:
    schema = ["id", "pid", "height_cm"]
    csv_text = deterministic_kv_to_csv(
        "pid: 017-30133 | height_cm: 173.0",
        schema,
        primary_key="id",
    )

    assert csv_text is None


def test_deterministic_kv_to_csv_preserves_rows_without_semantic_filtering() -> None:
    schema = [
        "record_id",
        "province_name",
        "enddate",
        "gdp",
        "primary_industry_gdp",
        "second_industry_gdp",
        "third_industry_gdp",
    ]
    text = "\n".join(
        [
            (
                "record_id: 档案 197至217 | province_name: 黑龙江、甘肃、广东、陕西、"
                "湖南和云南 | enddate: | gdp: | primary_industry_gdp: | "
                "second_industry_gdp: | third_industry_gdp:"
            ),
            (
                "record_id: 档案 286 | province_name: 山西省 | enddate: | gdp: | "
                "primary_industry_gdp: 302.48 | second_industry_gdp: 4265.77 | "
                "third_industry_gdp: 2370.48"
            ),
        ]
    )

    csv_text = deterministic_kv_to_csv(
        text,
        schema,
        primary_key="record_id",
    )
    assert csv_text is not None

    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert len(rows) == 2
    assert rows[0]["record_id"] == "档案 197至217"
    assert rows[0]["province_name"] == "黑龙江、甘肃、广东、陕西、湖南和云南"
    assert rows[1]["record_id"] == "档案 286"
    assert rows[1]["primary_industry_gdp"] == "302.48"
