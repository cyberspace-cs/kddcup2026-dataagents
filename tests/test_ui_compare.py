"""UI 比对层（`data_agent_baseline.ui.compare.compare`）的纯函数单测。

覆盖 6 种 status：
- match / column_subset_match / value_mismatch / shape_mismatch / no_gold / no_prediction
"""

from __future__ import annotations

import pandas as pd

from data_agent_baseline.ui.compare import compare


def test_compare_match() -> None:
    gold = pd.DataFrame({"v": [1, 2, 3]})
    pred = pd.DataFrame({"v": [1, 2, 3]})

    result = compare(gold, pred)
    assert result.status == "match"
    assert result.gold_shape == (3, 1)
    assert result.pred_shape == (3, 1)
    assert result.cell_diff_count == 0


def test_compare_no_gold() -> None:
    pred = pd.DataFrame({"v": [1]})
    result = compare(None, pred)
    assert result.status == "no_gold"
    assert result.gold_shape is None
    assert result.pred_shape == (1, 1)


def test_compare_no_prediction() -> None:
    gold = pd.DataFrame({"v": [1, 2]})
    result = compare(gold, None)
    assert result.status == "no_prediction"
    assert result.gold_shape == (2, 1)
    assert result.pred_shape is None


def test_compare_both_missing_reports_no_gold() -> None:
    """两边都缺时倾向把 status 设为 no_gold，避免误导用户以为只是 prediction 没生成。"""
    result = compare(None, None)
    assert result.status == "no_gold"
    assert result.gold_shape is None
    assert result.pred_shape is None


def test_compare_shape_mismatch_rows() -> None:
    gold = pd.DataFrame({"v": [1, 2, 3]})
    pred = pd.DataFrame({"v": [1, 2]})
    result = compare(gold, pred)
    assert result.status == "shape_mismatch"
    assert result.gold_shape == (3, 1)
    assert result.pred_shape == (2, 1)


def test_compare_shape_mismatch_cols_no_overlap() -> None:
    """列数不同 + 一列也匹配不上：归到 shape_mismatch。"""
    gold = pd.DataFrame({"a": [1], "b": [2]})
    pred = pd.DataFrame({"x": [99]})
    result = compare(gold, pred)
    assert result.status == "shape_mismatch"
    assert result.gold_shape == (1, 2)
    assert result.pred_shape == (1, 1)
    # extras / missing 仍要正确反映
    assert result.matched_columns == 0
    assert result.extra_pred_columns == 1
    assert result.missing_gold_columns == 2


def test_compare_value_mismatch_counts_diff_cells() -> None:
    """shape 一致但有单元格不等：返回精确 diff_count。"""
    gold = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pred = pd.DataFrame({"a": [1, 9, 3], "b": [4, 5, 99]})  # 2 个不等
    result = compare(gold, pred)
    assert result.status == "value_mismatch"
    assert result.cell_diff_count == 2


def test_compare_treats_nan_as_equal() -> None:
    """两边同位置都为 NaN 不应记入 diff_count。"""
    gold = pd.DataFrame({"v": [1.0, float("nan"), 3.0]})
    pred = pd.DataFrame({"v": [1.0, float("nan"), 3.0]})
    result = compare(gold, pred)
    assert result.status == "match"


def test_compare_match_ignores_column_names() -> None:
    """列名不同但单元格值完全相等：应直接报 match，列名差异不进入判定。"""
    gold = pd.DataFrame({"a": [1, 2]})
    pred = pd.DataFrame({"x": [1, 2]})
    result = compare(gold, pred)
    assert result.status == "match"
    assert result.cell_diff_count == 0


def test_compare_value_mismatch_ignores_column_names() -> None:
    """列名不同但有值不等：应只算被改动的单元格，不把列名差异计入 diff_count。"""
    gold = pd.DataFrame({"a": [1, 2, 3]})
    pred = pd.DataFrame({"x": [1, 9, 3]})  # 一个不等
    result = compare(gold, pred)
    assert result.status == "value_mismatch"
    assert result.cell_diff_count == 1


def test_compare_match_ignores_index_differences() -> None:
    """gold 与 pred index 不一致（例如一个 RangeIndex 一个含 label）但值相同 → match。"""
    gold = pd.DataFrame({"v": [1, 2]}, index=[10, 20])
    pred = pd.DataFrame({"v": [1, 2]})  # 默认 RangeIndex(0..1)
    result = compare(gold, pred)
    assert result.status == "match"


# ============================================================================
# column_subset_match：benchmark 评分判定为正确（≥1 列匹配）但有偏差（extras/missing/shape）
# ============================================================================


def test_compare_match_when_columns_swapped() -> None:
    """同形状但列序换：每个 gold 列都能在 pred 中找到等值列 → match（不是 column_subset_match）。"""
    gold = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    pred = pd.DataFrame({"b": [10, 20, 30], "a": [1, 2, 3]})  # 列序换
    result = compare(gold, pred)
    assert result.status == "match"
    assert result.matched_columns == 2
    assert result.extra_pred_columns == 0
    assert result.missing_gold_columns == 0


def test_compare_column_subset_match_extras_only() -> None:
    """gold 1 列，pred 含此列 + 2 个无关多余列：应 column_subset_match，extras=2。"""
    gold = pd.DataFrame({"answer": [1, 2, 3]})
    pred = pd.DataFrame(
        {
            "answer": [1, 2, 3],
            "noise_a": [99, 99, 99],
            "noise_b": ["x", "y", "z"],
        }
    )
    result = compare(gold, pred)
    assert result.status == "column_subset_match"
    assert result.matched_columns == 1
    assert result.extra_pred_columns == 2
    assert result.missing_gold_columns == 0
    # gold/pred shape 都正确暴露给 UI
    assert result.gold_shape == (3, 1)
    assert result.pred_shape == (3, 3)


def test_compare_column_subset_match_missing_only() -> None:
    """gold 2 列，pred 只命中 1 列且不带多余：should 是 column_subset_match（缺一列也是偏差）。"""
    gold = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
    pred = pd.DataFrame({"a": [1, 2]})
    result = compare(gold, pred)
    assert result.status == "column_subset_match"
    assert result.matched_columns == 1
    assert result.extra_pred_columns == 0
    assert result.missing_gold_columns == 1


def test_compare_column_subset_match_extras_and_missing() -> None:
    """同形状但 1 列匹配 + 1 列被替换：matched=1, extras=1, missing=1。"""
    gold = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pred = pd.DataFrame({"a": [1, 2, 3], "x": [99, 99, 99]})  # x 是多余 / b 缺失
    result = compare(gold, pred)
    assert result.status == "column_subset_match"
    assert result.matched_columns == 1
    assert result.extra_pred_columns == 1
    assert result.missing_gold_columns == 1


def test_compare_column_subset_match_ignores_column_names_for_matching() -> None:
    """匹配只看值序列，列名不参与：a 列被找到不论它在 pred 里叫什么。"""
    gold = pd.DataFrame({"a": [1, 2, 3]})
    # pred 列名故意都不同，但有一列值与 gold.a 相同
    pred = pd.DataFrame({"foo": [99, 99, 99], "bar": [1, 2, 3]})
    result = compare(gold, pred)
    assert result.status == "column_subset_match"
    assert result.matched_columns == 1
    assert result.extra_pred_columns == 1


def test_compare_greedy_one_to_one_matching() -> None:
    """一对一贪心匹配：gold 里有 2 个相同列时，pred 中只有 1 个匹配列只能认领一次。

    场景：gold = [v, v]（两列值都是 [1,2,3]），pred = [v]（一列 [1,2,3]）。
    期望：matched=1（gold 第一列认领了 pred 的唯一列），missing=1（gold 第二列找不到），
          extras=0。最终 status = column_subset_match（因为缺了一列）。
    """
    gold = pd.DataFrame({"v1": [1, 2, 3], "v2": [1, 2, 3]})
    pred = pd.DataFrame({"v": [1, 2, 3]})
    result = compare(gold, pred)
    assert result.status == "column_subset_match"
    assert result.matched_columns == 1
    assert result.missing_gold_columns == 1
    assert result.extra_pred_columns == 0


def test_compare_greedy_does_not_double_count_pred_column() -> None:
    """与上一个互补：gold = [v]，pred = [v, v]（重复）。

    pred 第一列被 gold 唯一列认领；pred 第二列没人要 → extras=1。
    status = column_subset_match（gold 命中但 pred 有多余）。
    """
    gold = pd.DataFrame({"v": [1, 2, 3]})
    pred = pd.DataFrame({"v1": [1, 2, 3], "v2": [1, 2, 3]})
    result = compare(gold, pred)
    assert result.status == "column_subset_match"
    assert result.matched_columns == 1
    assert result.extra_pred_columns == 1
    assert result.missing_gold_columns == 0


def test_compare_shape_mismatch_when_no_columns_match() -> None:
    """列数不同 + 没有任一列匹配上：仍归 shape_mismatch（与原行为一致）。"""
    gold = pd.DataFrame({"a": [1, 2, 3]})
    pred = pd.DataFrame({"x": [9, 9, 9], "y": [8, 8, 8]})
    result = compare(gold, pred)
    assert result.status == "shape_mismatch"
    assert result.matched_columns == 0
    assert result.extra_pred_columns == 2
    assert result.missing_gold_columns == 1


def test_compare_row_count_mismatch_skips_column_matching() -> None:
    """行数不同：列长度都对不齐，任何列都不可能等值——直接 shape_mismatch，不进入列匹配。"""
    gold = pd.DataFrame({"v": [1, 2, 3]})
    pred = pd.DataFrame({"v": [1, 2, 3, 4, 5]})  # 行数不同
    result = compare(gold, pred)
    assert result.status == "shape_mismatch"
    # 行数不同时不计算 matched/extras/missing（保持 0 的 default）
    assert result.matched_columns == 0


# ============================================================================
# multiset 语义：行序无关 + 重复度敏感（leaderboard 口径）
# ============================================================================


def test_compare_match_when_rows_reordered_single_column() -> None:
    """task_22 回归：gold=[2019-10-02, 2019-09-12]，pred 把同两个日期换序。
    位置比对会判 2 格不等；multiset 比对应直接 match。
    """
    gold = pd.DataFrame({"date_received": ["2019-10-02", "2019-09-12"]})
    pred = pd.DataFrame({"date_paid": ["2019-09-12", "2019-10-02"]})
    result = compare(gold, pred)
    assert result.status == "match"
    assert result.cell_diff_count == 0
    assert result.matched_columns == 1


def test_compare_match_when_rows_reordered_with_duplicates() -> None:
    """multiset 重复度敏感：[A, A, B] vs [A, B, A] 应 match（重复度都是 A:2 B:1）。"""
    gold = pd.DataFrame({"v": ["A", "A", "B"]})
    pred = pd.DataFrame({"v": ["A", "B", "A"]})
    result = compare(gold, pred)
    assert result.status == "match"


def test_compare_value_mismatch_when_multiplicity_differs() -> None:
    """multiset 重复度敏感的另一面：[A, B] vs [A, A] 不等价（A:1 B:1 vs A:2）。"""
    gold = pd.DataFrame({"v": ["A", "B"]})
    pred = pd.DataFrame({"v": ["A", "A"]})
    result = compare(gold, pred)
    assert result.status == "value_mismatch"
    # gold 比 pred 多一个 B → diff 1
    assert result.cell_diff_count == 1


def test_compare_match_when_multi_column_rows_reordered() -> None:
    """多列时每列各自做 multiset 比对：列内行序变 + 列间联动重排都允许，仍判 match。

    口径取自 leaderboard："each answer column is treated as one unordered
    column-value vector"——列之间的行级关联不参与匹配。
    """
    gold = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    # pred 列内重排，且两列重排得不同步——只要每列 multiset 与 gold 一致就算 match
    pred = pd.DataFrame({"a": [3, 1, 2], "b": [20, 30, 10]})
    result = compare(gold, pred)
    assert result.status == "match"


def test_compare_diff_cells_counts_multiset_symmetric_diff() -> None:
    """value_mismatch 路径下 cell_diff_count 走 multiset 对称差累加。

    gold.a={1,2,3} vs pred.a={1,9,3}: gold 多出 {2}，pred 多出 {9} → 单边 diff=1
    gold.b={4,5,6} vs pred.b={4,5,99}: 单边 diff=1
    总 cell_diff_count = 2（与原 positional 实现的数量在此例巧合相同，但语义已变）。
    """
    gold = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pred = pd.DataFrame({"a": [3, 1, 9], "b": [99, 4, 5]})  # 两列都打乱顺序
    result = compare(gold, pred)
    assert result.status == "value_mismatch"
    assert result.cell_diff_count == 2
