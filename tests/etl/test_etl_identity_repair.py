"""Behavior tests for additive-identity mining and approximate-numeral repair."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from agents.etl._merge import (
    clean_cell_for_type,
    mine_additive_identities,
    repair_numeric_identities,
)
from agents.etl.extractor import _repair_csv_numeric_identities

# Six exact rows satisfying before == after + transfer (task_15 shape), plus
# auxiliary columns: a constant code, a date, and dusty percentage columns
# whose stated values do NOT satisfy any exact additive identity.
HEADER = [
    "record_id",
    "secucode",
    "sumbeforetran",
    "transfer_shares",
    "sumaftertran",
    "pctbeforetran",
    "pctaftertran",
    "tran_date",
]

EXACT_ROWS = [
    ["29", "300707", "2441732", "300000", "2141732", "1.55", "1.36", "2018-11-14"],
    ["30", "300707", "5469268", "785000", "4684268", "3.48", "2.981", "2018-11-14"],
    ["31", "300707", "3527568", "40000", "3487568", "2.244", "2.2186", "2019-10-10"],
    ["48", "300707", "3487568", "20000", "3467568", "2.2186", "2.2058", "2019-10-11"],
    ["49", "300707", "2763268", "430000", "2333268", "1.7578", "1.4843", "2019-10-24"],
    ["51", "300707", "3013268", "100000", "2913268", "1.9168", "1.8532", "2019-11-01"],
]

SUM_IDENT = (
    HEADER.index("sumbeforetran"),
    HEADER.index("transfer_shares"),
    HEADER.index("sumaftertran"),
)


def _rows(*extra: list[str]) -> list[list[str]]:
    return [list(r) for r in EXACT_ROWS] + [list(r) for r in extra]


def _fuzzy_row(before: str) -> list[str]:
    # true before = 6723434 + 250000 = 6973434; stated coarsely as 6.97M
    return ["1400", "300707", before, "250000", "6723434", "4.436", "4.277", "2019-12-31"]


# ---------------------------------------------------------------------------
# clean_cell_for_type — approx tag survival rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "field_type", "expected"),
    [
        ("~6970000", "integer_count", "~6970000"),
        ("~6970000 股", "integer_count", "~6970000"),
        ("~2,441,732", "number", "~2441732"),
        ("～6970000", "integer_count", "~6970000"),  # fullwidth marker normalized
        ("6973434", "integer_count", "6973434"),  # exact values untouched
        ("~abc", "number", ""),  # tag does not rescue a non-number
        ("~2019-12-31", "date", "2019-12-31"),  # approximate dates out of scope
        ("~6970000", None, "~6970000"),  # untyped numeric keeps the tag
        ("~stake holder", None, "~stake holder"),  # untyped non-numeric: raw form
    ],
)
def test_clean_cell_for_type_approx_tags(raw: str, field_type: str | None, expected: str) -> None:
    assert clean_cell_for_type(raw, field_type) == expected


# ---------------------------------------------------------------------------
# mine_additive_identities — invariant gates
# ---------------------------------------------------------------------------


def test_mining_finds_share_identity_and_rejects_dusty_pcts() -> None:
    identities = mine_additive_identities(HEADER, _rows())

    assert SUM_IDENT in identities
    pct_cols = {HEADER.index("pctbeforetran"), HEADER.index("pctaftertran")}
    assert not [ident for ident in identities if pct_cols & set(ident)]


def test_mining_tolerates_coarse_violation_rows() -> None:
    identities = mine_additive_identities(HEADER, _rows(_fuzzy_row("6970000")))
    assert SUM_IDENT in identities


def test_mining_kills_identity_on_hard_violation() -> None:
    # off by 1 from a full-precision numeral: not coarse, not float dust
    bad = ["52", "300707", "3013269", "100000", "2913268", "1.9", "1.85", "2019-11-02"]
    identities = mine_additive_identities(HEADER, _rows(bad))
    assert SUM_IDENT not in identities


def test_mining_requires_minimum_support() -> None:
    identities = mine_additive_identities(HEADER, _rows()[:4])
    assert SUM_IDENT not in identities


def test_mining_requires_support_to_dominate_suspects() -> None:
    # 6 exact rows vs 7 coarse-violation rows: invariant evidence does not dominate
    suspects = [_fuzzy_row("6970000") for _ in range(7)]
    identities = mine_additive_identities(HEADER, _rows(*suspects))
    assert SUM_IDENT not in identities


# ---------------------------------------------------------------------------
# repair_numeric_identities — repair and tag-strip behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stated", ["~6970000", "6970000"])
def test_repair_recomputes_coarse_numeral(stated: str) -> None:
    rows = _rows(_fuzzy_row(stated))

    repaired, repairs = repair_numeric_identities(HEADER, rows)

    assert repaired[-1][HEADER.index("sumbeforetran")] == "6973434"
    assert [(r[1], r[2], r[3]) for r in repairs] == [("sumbeforetran", stated, "6973434")]


def test_repair_strips_tag_when_terms_missing() -> None:
    row = _fuzzy_row("~6970000")
    row[HEADER.index("sumaftertran")] = ""
    repaired, repairs = repair_numeric_identities(HEADER, _rows(row))

    assert repaired[-1][HEADER.index("sumbeforetran")] == "6970000"
    assert repairs == []


def test_repair_rejects_granularity_inconsistent_solution() -> None:
    # solved = 6723434 + 400000 = 7123434, which does not quantize back to 697万
    row = _fuzzy_row("~6970000")
    row[HEADER.index("transfer_shares")] = "400000"
    repaired, repairs = repair_numeric_identities(HEADER, _rows(row))

    assert repaired[-1][HEADER.index("sumbeforetran")] == "6970000"
    assert repairs == []


def test_repair_skips_row_with_two_coarse_candidates() -> None:
    # both sumbeforetran (7000000, 1e6 grain) and transfer_shares (2000000,
    # 1e6 grain) could absorb the discrepancy — ambiguous, repair nothing
    row = ["60", "300707", "7000000", "2000000", "4973434", "4.4", "3.1", "2020-01-01"]
    repaired, repairs = repair_numeric_identities(HEADER, _rows(row))

    assert repaired[-1][HEADER.index("sumbeforetran")] == "7000000"
    assert repaired[-1][HEADER.index("transfer_shares")] == "2000000"
    assert repairs == []


def test_repair_imputes_single_missing_term_of_validated_identity() -> None:
    # compression dropped the stated value entirely; siblings determine it
    row = _fuzzy_row("")
    repaired, repairs = repair_numeric_identities(HEADER, _rows(row))

    assert repaired[-1][HEADER.index("sumbeforetran")] == "6973434"
    assert [(r[1], r[2], r[3]) for r in repairs] == [("sumbeforetran", "", "6973434")]


def test_repair_leaves_genuine_null_pattern_untouched() -> None:
    # source-null rows miss 2+ terms ("转让方信息缺失"): unsolvable, stay null
    row = _fuzzy_row("")
    row[HEADER.index("sumaftertran")] = ""
    repaired, repairs = repair_numeric_identities(HEADER, _rows(row))

    assert repaired[-1][HEADER.index("sumbeforetran")] == ""
    assert repaired[-1][HEADER.index("sumaftertran")] == ""
    assert repairs == []


def test_repair_strips_tags_even_without_any_identity() -> None:
    header = ["record_id", "amount"]
    rows = [["1", "~500"], ["2", "300"]]
    repaired, repairs = repair_numeric_identities(header, rows)

    assert repaired[0][1] == "500"
    assert repairs == []


# ---------------------------------------------------------------------------
# _repair_csv_numeric_identities — file-level wrapper
# ---------------------------------------------------------------------------


def _write(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)


def test_csv_repair_rewrites_file_and_reports(tmp_path: Path) -> None:
    csv_path = tmp_path / "lc_sharetransfer.csv"
    _write(csv_path, _rows(_fuzzy_row("~6970000")))

    repairs = _repair_csv_numeric_identities(csv_path)

    assert [(r[1], r[3]) for r in repairs] == [("sumbeforetran", "6973434")]
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    assert rows[-1][HEADER.index("sumbeforetran")] == "6973434"
    assert not any("~" in cell for row in rows for cell in row)


def test_csv_repair_strips_orphan_tags_without_repairs(tmp_path: Path) -> None:
    csv_path = tmp_path / "t.csv"
    row = _fuzzy_row("~6970000")
    row[HEADER.index("sumaftertran")] = ""
    _write(csv_path, _rows(row))

    repairs = _repair_csv_numeric_identities(csv_path)

    assert repairs == []
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    assert rows[-1][HEADER.index("sumbeforetran")] == "6970000"
