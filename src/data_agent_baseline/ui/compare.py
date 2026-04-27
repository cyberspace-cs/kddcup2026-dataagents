"""gold vs prediction 比对（纯函数）。

按"列值匹配"思路评估 prediction 对 gold 的覆盖度，状态六态：

- `match`:                同形状 AND 每个 gold 列都能在 pred 中找到等值多重集列（列名/位置/行序都不限）
- `column_subset_match`:  pred 至少匹配 1 个 gold 列，但存在偏差（extras / missing / 形状差异）—
                          benchmark 评分判定为"正确但带惩罚"
- `value_mismatch`:       同形状但一列也匹配不上
- `shape_mismatch`:       行数不同；或行数同但 pred 一列也匹配不上且列数不同
- `no_gold` / `no_prediction`: 任一边缺失

匹配规则：贪心一对一——遍历 gold 每一列，在 pred 中找首个未被占用且**值多重集**相等的列；
NaN 视作可比相等（两边都缺等价于都"有同一种缺失值"）。同 pred 列不会同时认领多个 gold 列。

口径与 leaderboard 一致："each answer column is treated as one unordered column-value
vector"——列被视作行序无关的 multiset。所以 gold=[A, B] 与 pred=[B, A] 应判 match，
位置相同的位反而需要重排到 multiset 维度才有意义。

`extra_pred_columns` 字段是"多余的 pred 列数"——评测脚本通常按此计入惩罚因子。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

CompareStatus = Literal[
    "match",
    "column_subset_match",
    "value_mismatch",
    "shape_mismatch",
    "no_gold",
    "no_prediction",
]


@dataclass(frozen=True, slots=True)
class CompareResult:
    """gold/prediction 比对的结果摘要。

    - status:                上述 6 个状态之一
    - gold_shape:            gold.csv 的 (rows, cols)，缺失时为 None
    - pred_shape:            prediction.csv 的 (rows, cols)，缺失时为 None
    - cell_diff_count:       value_mismatch 路径下按列对位的多重集对称差累加；其它状态下为 0
                             （直观含义："要把 pred 改成 gold，至少需要替换几格值"）
    - matched_columns:       gold 中能在 pred 找到等值多重集列的列数（贪心一对一）
    - extra_pred_columns:    pred 中没匹配上任何 gold 列的"多余列"数（惩罚信号）
    - missing_gold_columns:  gold 中没被 pred 覆盖到的列数
    """

    status: CompareStatus
    gold_shape: tuple[int, int] | None
    pred_shape: tuple[int, int] | None
    cell_diff_count: int = 0
    matched_columns: int = 0
    extra_pred_columns: int = 0
    missing_gold_columns: int = 0


def compare(
    gold: pd.DataFrame | None,
    pred: pd.DataFrame | None,
) -> CompareResult:
    """比对 gold 与 prediction，返回 CompareResult。

    设计取舍：
    - 列匹配按"值序列相等"判定，**列名 / 位置都不参与**——评测口径与之一致
    - 行数不同直接 shape_mismatch（列长度不同则任何列都不可能等值）
    - "至少 1 列匹配 + 任一种偏差（extras/missing/形状差异）" → column_subset_match，
      该状态对应 benchmark 中"答对了但有冗余列"的扣分场景
    - NaN 同位视为相等；不抛异常，所有边缘情况都映射到一个 status
    """
    gold_shape = (int(gold.shape[0]), int(gold.shape[1])) if gold is not None else None
    pred_shape = (int(pred.shape[0]), int(pred.shape[1])) if pred is not None else None

    if gold is None and pred is None:
        # 两边都缺：把"无 gold"作为更优先的提示——用户通常更关心的是"无法比对"
        return CompareResult(status="no_gold", gold_shape=None, pred_shape=None)
    if gold is None:
        return CompareResult(status="no_gold", gold_shape=None, pred_shape=pred_shape)
    if pred is None:
        return CompareResult(status="no_prediction", gold_shape=gold_shape, pred_shape=None)

    # 行数不同：列长度都不一致，没法做值序列比对——直接归为 shape_mismatch
    if gold.shape[0] != pred.shape[0]:
        return CompareResult(
            status="shape_mismatch",
            gold_shape=gold_shape,
            pred_shape=pred_shape,
        )

    matched, extras, missing = _match_columns_by_value(gold, pred)
    same_shape = gold.shape == pred.shape

    if matched == 0:
        # 一列都匹配不上：同形状 → value_mismatch（再附带 cell_diff_count 帮助调试）；
        # 不同形状 → shape_mismatch
        if same_shape:
            return CompareResult(
                status="value_mismatch",
                gold_shape=gold_shape,
                pred_shape=pred_shape,
                cell_diff_count=_count_multiset_diff_cells(gold, pred),
                matched_columns=0,
                extra_pred_columns=extras,
                missing_gold_columns=missing,
            )
        return CompareResult(
            status="shape_mismatch",
            gold_shape=gold_shape,
            pred_shape=pred_shape,
            extra_pred_columns=extras,
            missing_gold_columns=missing,
        )

    # matched >= 1：按 benchmark 口径已经"判定为对"，再看是否完美
    is_perfect = same_shape and extras == 0 and missing == 0
    if is_perfect:
        return CompareResult(
            status="match",
            gold_shape=gold_shape,
            pred_shape=pred_shape,
            matched_columns=matched,
        )
    return CompareResult(
        status="column_subset_match",
        gold_shape=gold_shape,
        pred_shape=pred_shape,
        matched_columns=matched,
        extra_pred_columns=extras,
        missing_gold_columns=missing,
    )


def _match_columns_by_value(gold: pd.DataFrame, pred: pd.DataFrame) -> tuple[int, int, int]:
    """贪心一对一列匹配，返回 `(matched, extras, missing)`。

    遍历 gold 每一列，在 pred 中找首个未被占用且**值多重集**相等的列；同 pred 列不会被
    认领多次（避免 1 个 pred 列同时算作多个 gold 列的等价物）。

    Pre-condition：调用方已确保 `gold.shape[0] == pred.shape[0]`，否则多重集判定无意义。
    """
    n_gold = int(gold.shape[1])
    n_pred = int(pred.shape[1])
    used: set[int] = set()
    matched = 0
    for gi in range(n_gold):
        gcol = gold.iloc[:, gi]
        for pi in range(n_pred):
            if pi in used:
                continue
            if _series_value_equal(gcol, pred.iloc[:, pi]):
                used.add(pi)
                matched += 1
                break
    return matched, n_pred - len(used), n_gold - matched


# Counter 里给 NaN 用的哨兵：直接放 float('nan') 会因 NaN != NaN 让两边总是不等。
# 哨兵走 `is`-equality，把"两边都缺值"规约成"同一种缺失"，与评测口径一致。
_NAN_SENTINEL = "__compare_nan_sentinel__"


def _series_to_counter(s: pd.Series) -> Counter[Any]:
    """把一列变成 Counter（multiset）；NaN 统一替换为哨兵字符串。

    `pd.isna` 同时兜住 float('nan')、np.nan、pd.NaT、None——pandas 读 CSV 出来的
    各种"缺值"形态都能命中。逐元素而非向量是因为 Counter 需要 hashable scalar。
    """
    return Counter(_NAN_SENTINEL if pd.isna(v) else v for v in s)


def _series_value_equal(a: pd.Series, b: pd.Series) -> bool:
    """两个等长 Series 是否值多重集相等（行序无关；NaN 视作可比相等）。

    评测口径："each answer column is treated as one unordered column-value vector"
    ——所以 [A, B] 与 [B, A] 等价，[A, A] 与 [A] 不等价。Counter 比较把两边规约到
    multiset 后比对元素重复度，刚好对应这条语义。
    """
    if len(a) != len(b):
        return False
    return _series_to_counter(a) == _series_to_counter(b)


def _count_multiset_diff_cells(gold: pd.DataFrame, pred: pd.DataFrame) -> int:
    """按列对位计算多重集对称差，再求和；同形状下等价于"最少需要替换几格"。

    以 Counter 减法为基元：`gc - pc` 给出 "gold 多出的元素及重复度"；同形状下
    `sum((gc - pc).values()) == sum((pc - gc).values())`，所以只取一边累加即可。
    哨兵规约的 NaN 之间相互抵消，不会被算成差异。

    仅给 value_mismatch 路径用（提供"差几格"的人类可读 hint）；多重集语义下，
    重排顺序的 prediction 不会再被错判成"全格不等"——这正是 task_22 的回归点。
    """
    n_cols = min(int(gold.shape[1]), int(pred.shape[1]))
    total = 0
    for ci in range(n_cols):
        gc = _series_to_counter(gold.iloc[:, ci])
        pc = _series_to_counter(pred.iloc[:, ci])
        # multiset 减法：负计数会被丢弃，只剩 "gc 比 pc 多" 的元素总数
        total += sum((gc - pc).values())
    return total
