"""Synthesize a candidate pandas/SQL filter recipe from explore_video's rules dict.

The video sub-agent's ``findings.extracted_data.rules`` arrives in heterogeneous
shapes across runs — keys observed in production include ``filter_field``,
``threshold_value``, ``ranking_limit``, ``output_fields``, ``aggregation_field``,
``data_source_table``, and many Chinese-suffixed variants. Each run the agent
re-writes the same filter logic from scratch in ``execute_python`` /
``execute_context_sql`` and frequently mistranslates:

- threshold boundary inclusion (``>`` vs ``>=``)
- column choice (e.g. ``ChiName`` vs ``ChiNameAbbr``)
- forgetting top-N, group-by, batch_id, snapshot multiplicity

This module performs a deterministic pre-translation so the downstream agent
starts from a structured recipe instead of free-form rewriting. Output is a
plaintext hint attached to the explore_video observation; agents are explicitly
told to verify exact column names against the actual table schema before use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, cast

# 视频抽出的 rules 字段在不同 run 里键名漂得很厉害,用别名 tuple 兜底。
# 排在前面的优先;一旦命中即停。
_FIELD_KEYS: tuple[str, ...] = (
    "field",
    "filter_field",
    "metric",
    "column",
    "attribute",
    "aggregation_field",
)
_FIELD_LABEL_KEYS: tuple[str, ...] = ("field_label", "field_cn", "label", "field_name_cn")
_OPERATOR_KEYS: tuple[str, ...] = ("operator", "comparator", "op", "condition_type")
_THRESHOLD_KEYS: tuple[str, ...] = (
    "threshold",
    "threshold_value",
    "value",
    "condition_value",
)
_THRESHOLD_TEXT_KEYS: tuple[str, ...] = ("threshold_text", "value_text")
_UNIT_KEYS: tuple[str, ...] = ("unit", "threshold_unit", "value_unit")
_SORT_FIELD_KEYS: tuple[str, ...] = (
    "sort_field",
    "sort_by",
    "order_by",
    "ranking_field",
)
_SORT_DIR_KEYS: tuple[str, ...] = (
    "sort_direction",
    "sort_order",
    "order",
    "ranking_order",
    "sort_cn",
)
_TOPN_KEYS: tuple[str, ...] = ("top_n", "ranking_limit", "limit", "n", "top")
_EXPORT_KEYS: tuple[str, ...] = (
    "export_field",
    "export_fields",
    "output_field",
    "output_fields",
    "select",
    "fields_to_export",
    "fields_for_result",
    "result_fields",
    "fields",
)
_GROUPBY_KEYS: tuple[str, ...] = ("group_by", "grouping_dimension", "groupby", "group")
_TABLE_KEYS: tuple[str, ...] = (
    "table",
    "source_table",
    "data_source_table",
    "source_tables",
)
_BATCH_KEYS: tuple[str, ...] = (
    "batch_id",
    "batch_no",
    "archive_id",
    "snapshot_id",
    "scheme_id",
)
_SNAPSHOT_KEYS: tuple[str, ...] = ("snapshot_dates", "reporting_dates", "quarter_ends")

# Operator 归一化:把英文/中文/符号都映射到 pandas/SQL 表达式里能直接拼的符号串。
# 顺序无关;映射目标永远是 Python 比较运算符。
_OPERATOR_NORMALIZATION: dict[str, str] = {
    ">": ">",
    "greater than": ">",
    "above": ">",
    "exceeds": ">",
    "more than": ">",
    "高于": ">",
    "大于": ">",
    "≥": ">=",
    ">=": ">=",
    "at or above": ">=",
    "no less than": ">=",
    "not less than": ">=",
    "at least": ">=",
    "gte": ">=",
    "大于等于": ">=",
    "不低于": ">=",
    "不少于": ">=",
    "<": "<",
    "less than": "<",
    "below": "<",
    "under": "<",
    "fewer than": "<",
    "低于": "<",
    "小于": "<",
    "≤": "<=",
    "<=": "<=",
    "at or below": "<=",
    "no more than": "<=",
    "not more than": "<=",
    "at most": "<=",
    "lte": "<=",
    "小于等于": "<=",
    "不超过": "<=",
    "不多于": "<=",
    "=": "==",
    "==": "==",
    "equal to": "==",
    "equals": "==",
    "等于": "==",
    "!=": "!=",
    "≠": "!=",
    "not equal to": "!=",
    "not equals": "!=",
    "不等于": "!=",
}

# 排序方向:大小写不敏感的子串匹配。
_DESCENDING_TOKENS: tuple[str, ...] = (
    "desc",
    "descending",
    "降序",
    "由高到低",
    "highest first",
    "high to low",
)
_ASCENDING_TOKENS: tuple[str, ...] = (
    "asc",
    "ascending",
    "升序",
    "由低到高",
    "lowest first",
    "low to high",
)

# 视频里的字段名常带括号注释:"流通A股股本 (AFLOATS)" / "本日基准增长率 (dailybenchgr)"。
# 括号里的 ASCII 标识符通常就是源表里的列名 — 优先用它。
_PAREN_IDENT_RE = re.compile(r"\(([A-Za-z_][A-Za-z0-9_]*)\)")
# 不带 ASCII 标识符的情况下,先 trim 括号注释再返回原文。
_PAREN_ANY_RE = re.compile(r"\s*\([^)]*\)\s*$")


@dataclass(frozen=True, slots=True)
class FilterRecipe:
    """Canonical filter intent extracted from heterogeneous video rules."""

    field: str | None = None
    field_label: str | None = None  # Chinese label retained for context
    operator: str | None = None  # normalized to one of >, >=, <, <=, ==, !=
    threshold: str | None = None  # raw string; agent decides numeric coercion
    threshold_unit: str | None = None
    sort_field: str | None = None
    sort_descending: bool | None = None
    top_n: int | None = None
    export_fields: tuple[str, ...] = ()
    group_by: str | None = None
    table_hint: str | None = None
    batch_id: str | None = None
    snapshot_dates: tuple[str, ...] = ()

    def is_sufficient(self) -> bool:
        """True when the recipe has enough structure to be worth surfacing.

        Two anchor patterns count:
        1. A filter pair (``field`` + ``operator``) — most common rule shape.
        2. A ranking pair (``sort_field`` + ``top_n``) — covers "Top N by X" videos.
        Plus a fallback: any of group_by / batch_id / snapshot_dates is enough
        on its own because the downstream agent gains structure even without
        a comparator.
        """
        has_filter = self.field is not None and self.operator is not None
        has_ranking = self.sort_field is not None and self.top_n is not None
        has_structural = bool(self.group_by or self.batch_id or self.snapshot_dates)
        return has_filter or has_ranking or has_structural


def _first_value(source: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first non-empty value from ``source`` for any of ``keys``."""
    for key in keys:
        if key not in source:
            continue
        value: Any = source[key]
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, tuple)) and not cast(list[Any], value):
            continue
        return cast(Any, value)
    return None


def _clean_field_name(raw: Any) -> tuple[str | None, str | None]:
    """Extract ``(field, label)`` from a possibly-decorated field string.

    Examples:
        "流通A股股本 (AFLOATS)" → ("AFLOATS", "流通A股股本")
        "dailybenchgr"          → ("dailybenchgr", None)
        "本日基准增长率"        → ("本日基准增长率", None)
    """
    if not isinstance(raw, str):
        return None, None
    text = raw.strip()
    if not text:
        return None, None
    match = _PAREN_IDENT_RE.search(text)
    if match:
        ident = match.group(1)
        label = _PAREN_ANY_RE.sub("", text).strip() or None
        if label == ident:
            label = None
        return ident, label
    cleaned = _PAREN_ANY_RE.sub("", text).strip()
    return (cleaned or text), None


def _normalize_operator(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    key = raw.strip().lower()
    if not key:
        return None
    if key in _OPERATOR_NORMALIZATION:
        return _OPERATOR_NORMALIZATION[key]
    # 也尝试去掉常见尾词:"greater than (strict)" → "greater than"
    for marker in (" (strict)", " (inclusive)", " (exclusive)"):
        if key.endswith(marker):
            stripped = key[: -len(marker)]
            if stripped in _OPERATOR_NORMALIZATION:
                return _OPERATOR_NORMALIZATION[stripped]
    return None


def _normalize_sort_direction(raw: Any) -> bool | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    for token in _DESCENDING_TOKENS:
        if token in text:
            return True
    for token in _ASCENDING_TOKENS:
        if token in text:
            return False
    return None


def _coerce_int(raw: Any) -> int | None:
    if isinstance(raw, bool):  # bool is subclass of int — exclude
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    if isinstance(raw, str):
        text = raw.strip().replace(",", "").replace("，", "")
        if text.isdigit():
            value = int(text)
            return value if value > 0 else None
    return None


def _coerce_export_fields(raw: Any) -> tuple[str, ...]:
    """Normalize a single string or a list of strings into a clean field tuple."""
    if raw is None:
        return ()
    items: list[Any]
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(cast(list[Any], raw))
    else:
        return ()
    out: list[str] = []
    for item in items:
        cleaned, _ = _clean_field_name(item)
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return tuple(out)


def _coerce_str(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        return text or None
    if isinstance(raw, (int, float)):
        return str(raw)
    return None


def _coerce_snapshot_dates(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    items = list(cast(list[Any], raw))
    return tuple(str(item) for item in items if item is not None and str(item).strip())


def _extract_threshold(rules: dict[str, Any]) -> tuple[str | None, str | None]:
    """Pull threshold value + unit from either flat or nested rule shapes."""
    raw = _first_value(rules, _THRESHOLD_KEYS)
    if isinstance(raw, dict):
        # task_4 shape: threshold = {"value": 2000000, "unit": "shares", ...}
        nested = cast(dict[str, Any], raw)
        value = _coerce_str(_first_value(nested, ("value", *_THRESHOLD_KEYS)))
        unit = _coerce_str(_first_value(nested, _UNIT_KEYS))
        return value, unit
    value = _coerce_str(raw) or _coerce_str(_first_value(rules, _THRESHOLD_TEXT_KEYS))
    unit = _coerce_str(_first_value(rules, _UNIT_KEYS))
    return value, unit


def _extract_operator(rules: dict[str, Any]) -> str | None:
    """Try top-level then nested-threshold for the operator string."""
    direct = _normalize_operator(_first_value(rules, _OPERATOR_KEYS))
    if direct:
        return direct
    raw_threshold = _first_value(rules, _THRESHOLD_KEYS)
    if isinstance(raw_threshold, dict):
        nested = cast(dict[str, Any], raw_threshold)
        return _normalize_operator(_first_value(nested, _OPERATOR_KEYS))
    return None


def extract_filter_recipe(rules: dict[str, Any] | None) -> FilterRecipe | None:
    """Walk a video rules dict, return a canonical FilterRecipe or None if empty.

    The function is deliberately tolerant: any single key may be missing. The
    caller decides whether the result is meaningful via ``recipe.is_sufficient()``.
    """
    if not isinstance(rules, dict):
        return None
    if not rules:
        return None

    field_value, field_label_from_paren = _clean_field_name(_first_value(rules, _FIELD_KEYS))
    explicit_label = _coerce_str(_first_value(rules, _FIELD_LABEL_KEYS))
    field_label = field_label_from_paren or explicit_label

    operator = _extract_operator(rules)
    threshold, threshold_unit = _extract_threshold(rules)

    sort_field_raw = _first_value(rules, _SORT_FIELD_KEYS)
    sort_field, _ = _clean_field_name(sort_field_raw)
    sort_descending = _normalize_sort_direction(_first_value(rules, _SORT_DIR_KEYS))

    top_n = _coerce_int(_first_value(rules, _TOPN_KEYS))
    export_fields = _coerce_export_fields(_first_value(rules, _EXPORT_KEYS))

    group_by_raw = _first_value(rules, _GROUPBY_KEYS)
    group_by, _ = _clean_field_name(group_by_raw)

    table_hint_raw = _first_value(rules, _TABLE_KEYS)
    table_hint: str | None
    if isinstance(table_hint_raw, (list, tuple)):
        first = next(iter(cast(list[Any], table_hint_raw)), None)
        table_hint = _coerce_str(first)
    else:
        table_hint = _coerce_str(table_hint_raw)

    batch_id = _coerce_str(_first_value(rules, _BATCH_KEYS))
    snapshot_dates = _coerce_snapshot_dates(_first_value(rules, _SNAPSHOT_KEYS))

    recipe = FilterRecipe(
        field=field_value,
        field_label=field_label,
        operator=operator,
        threshold=threshold,
        threshold_unit=threshold_unit,
        sort_field=sort_field,
        sort_descending=sort_descending,
        top_n=top_n,
        export_fields=export_fields,
        group_by=group_by,
        table_hint=table_hint,
        batch_id=batch_id,
        snapshot_dates=snapshot_dates,
    )
    return (
        recipe
        if any(
            (
                recipe.field,
                recipe.operator,
                recipe.threshold,
                recipe.sort_field,
                recipe.top_n,
                recipe.export_fields,
                recipe.group_by,
                recipe.batch_id,
                recipe.snapshot_dates,
            )
        )
        else None
    )


def render_filter_recipe(recipe: FilterRecipe) -> str:
    """Format a FilterRecipe into the plaintext hint surfaced to the agent.

    Output is intentionally machine-readable-ish: a structured bullet list of
    the recipe fields followed by a pandas template skeleton. Every line
    reminds the agent that field/table names must be verified before use.
    """
    lines: list[str] = [
        "[video-rule recipe — VERIFY each name against actual schema before use]",
    ]
    if recipe.field:
        label_tail = f" (label: {recipe.field_label})" if recipe.field_label else ""
        lines.append(f"  field:        {recipe.field}{label_tail}")
    elif recipe.field_label:
        lines.append(f"  field_label:  {recipe.field_label} [no ASCII column name extracted]")
    if recipe.operator:
        lines.append(f"  operator:     {recipe.operator}")
    if recipe.threshold is not None:
        unit_tail = f" (unit: {recipe.threshold_unit})" if recipe.threshold_unit else ""
        lines.append(f"  threshold:    {recipe.threshold}{unit_tail}")
    if recipe.sort_field:
        direction = (
            "DESC"
            if recipe.sort_descending
            else ("ASC" if recipe.sort_descending is False else "?")
        )
        lines.append(f"  sort:         {recipe.sort_field} {direction}")
    if recipe.top_n is not None:
        lines.append(f"  top_n:        {recipe.top_n}")
    if recipe.group_by:
        lines.append(f"  group_by:     {recipe.group_by}")
    if recipe.export_fields:
        lines.append(f"  output_cols:  {list(recipe.export_fields)}")
    if recipe.table_hint:
        lines.append(f"  table_hint:   {recipe.table_hint}")
    if recipe.batch_id:
        lines.append(f"  batch_id:     {recipe.batch_id}")
    if recipe.snapshot_dates:
        lines.append(f"  snapshot_dates: {list(recipe.snapshot_dates)}  (emit one row per date)")

    template_lines = _render_pandas_template(recipe)
    if template_lines:
        lines.append("")
        lines.append("Pandas template skeleton (adapt column names from inspect_files):")
        lines.extend(template_lines)
    return "\n".join(lines)


def _render_pandas_template(recipe: FilterRecipe) -> list[str]:
    """Emit a minimal pandas snippet that mirrors the recipe's intent."""
    if not recipe.is_sufficient():
        return []
    out: list[str] = ["```python"]
    if recipe.table_hint:
        out.append(f"  df = pd.read_csv('csv/{recipe.table_hint}.csv')  # verify exact path")
    else:
        out.append("  df = pd.read_csv(...)  # use the source table identified by inspect_files")
    if recipe.field and recipe.operator and recipe.threshold is not None:
        op = recipe.operator
        out.append(f"  col = '{recipe.field}'  # verify exact column name in df.columns")
        out.append(
            f"  mask = df[col] {op} {recipe.threshold!r}  # coerce threshold to the column's dtype"
        )
        out.append("  result = df[mask]")
    elif recipe.field and recipe.operator:
        out.append(f"  col = '{recipe.field}'  # verify exact column name in df.columns")
        out.append(f"  # operator: {recipe.operator}  (threshold not specified by video)")
        out.append("  result = df  # add the comparison once the threshold is resolved")
    else:
        out.append("  result = df  # no explicit filter rule extracted")
    if recipe.sort_field:
        ascending = recipe.sort_descending is False
        out.append(f"  result = result.sort_values('{recipe.sort_field}', ascending={ascending})")
    if recipe.top_n is not None:
        out.append(f"  result = result.head({recipe.top_n})")
    if recipe.group_by:
        out.append(
            f"  # group_by: {recipe.group_by} — adjust to groupby('{recipe.group_by}').agg(...) "
            "if the question asks per-group output"
        )
    if recipe.export_fields:
        cols_repr = ", ".join(f"'{c}'" for c in recipe.export_fields)
        out.append(f"  answer_df = result[[{cols_repr}]]  # verify columns are the WH-target")
    else:
        out.append("  answer_df = result[[...]]  # pick the WH-target columns explicitly")
    out.append("```")
    return out


def synthesize_filter_recipe(rules: dict[str, Any] | None) -> str | None:
    """One-shot convenience: rules dict → rendered hint or None.

    Returns None when no recipe can be extracted OR when the extracted recipe
    is too sparse to be worth surfacing — callers can use truthiness to gate.
    """
    recipe = extract_filter_recipe(rules)
    if recipe is None or not recipe.is_sufficient():
        return None
    return render_filter_recipe(recipe)
