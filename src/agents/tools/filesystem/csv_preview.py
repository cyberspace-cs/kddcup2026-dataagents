"""CSV 预览与列级 profiling。"""

from __future__ import annotations

import contextlib
import csv
from pathlib import Path
from typing import Any

from agents.benchmark.schema import PublicTask
from agents.tools.filesystem import constants
from agents.tools.filesystem.path import resolve_context_path


def classify_dtype(values: list[str]) -> str:
    """根据列前 N 个非空样本推断粗 dtype（int/float/bool/str/empty）。"""
    non_empty = [v for v in values if v != ""]
    if not non_empty:
        return "empty"
    bool_set = {"true", "false"}
    if all(v.lower() in bool_set for v in non_empty):
        return "bool"
    all_int = True
    for v in non_empty:
        candidate = v.lstrip("-+")
        if not candidate.isdigit():
            all_int = False
            break
    if all_int:
        return "int"
    all_float = True
    for v in non_empty:
        try:
            float(v)
        except ValueError:
            all_float = False
            break
    if all_float:
        return "float"
    return "str"


def profile_columns(
    columns: list[str],
    sample_rows: list[list[str]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """从字符串化的样本行推断 dtypes 和列级 profile。返回 (dtypes, profile)。"""
    column_count = len(columns)
    dtype_sample = sample_rows[: constants.INSPECT_FILES_DTYPE_SAMPLE]
    dtypes: list[str] = []
    for col_idx in range(column_count):
        col_vals = [row[col_idx] if col_idx < len(row) else "" for row in dtype_sample]
        dtypes.append(classify_dtype(col_vals))

    profile: list[dict[str, Any]] = []
    profile_count = len(sample_rows)
    for col_idx in range(column_count):
        col_values = [row[col_idx] if col_idx < len(row) else "" for row in sample_rows]
        null_count = sum(1 for v in col_values if v.strip() == "")
        null_rate = round(null_count / profile_count, 3) if profile_count > 0 else 0.0
        col_profile: dict[str, Any] = {"null_rate": null_rate}
        col_dtype = dtypes[col_idx] if col_idx < len(dtypes) else "str"

        if col_dtype in ("int", "float"):
            numeric_vals: list[float] = []
            for v in col_values:
                v_stripped = v.strip()
                if v_stripped:
                    with contextlib.suppress(ValueError):
                        numeric_vals.append(float(v_stripped))
            if numeric_vals:
                if col_dtype == "int":
                    col_profile["min"] = int(min(numeric_vals))
                    col_profile["max"] = int(max(numeric_vals))
                else:
                    col_profile["min"] = min(numeric_vals)
                    col_profile["max"] = max(numeric_vals)
        else:
            non_empty = sorted({v.strip() for v in col_values if v.strip()})
            if len(non_empty) <= 20:
                col_profile["values"] = non_empty
            else:
                col_profile["unique_count"] = len(non_empty)
                col_profile["sample_values"] = non_empty[:5]

        profile.append(col_profile)

    return dtypes, profile


def read_csv_preview(task: PublicTask, relative_path: str) -> dict[str, Any]:
    """返回 CSV 的 schema + head/tail 样本（**不是**取数据工具）。"""
    path = resolve_context_path(task, relative_path)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        return {
            "path": relative_path,
            "columns": [],
            "dtypes": [],
            "row_count": 0,
            "head": [],
            "tail": [],
        }

    header = rows[0]
    data_rows = rows[1:]
    row_count = len(data_rows)
    head_rows = data_rows[: constants.READ_CSV_HEAD_ROWS]
    tail_start = max(constants.READ_CSV_HEAD_ROWS, row_count - constants.READ_CSV_TAIL_ROWS)
    tail_rows = data_rows[tail_start:]

    column_count = len(header)
    dtypes: list[str] = []
    sample_for_dtype = head_rows[: constants.INSPECT_FILES_DTYPE_SAMPLE]
    for column_index in range(column_count):
        column_values = [
            row[column_index] if column_index < len(row) else "" for row in sample_for_dtype
        ]
        dtypes.append(classify_dtype(column_values))

    return {
        "path": relative_path,
        "columns": header,
        "dtypes": dtypes,
        "row_count": row_count,
        "head": head_rows,
        "tail": tail_rows,
    }


def summarize_csv(path: Path) -> dict[str, Any]:
    """流式读取 CSV：拿 header + 前 N 行做 dtype 推断 + 列级 profiling + 累加总行数。"""
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return {
                "columns": [],
                "dtypes": [],
                "row_count": 0,
                "dtypes_confidence": "approx",
            }
        sample_rows: list[list[str]] = []
        row_count = 0
        for row in reader:
            row_count += 1
            if len(sample_rows) < constants.INSPECT_FILES_PROFILE_SAMPLE:
                sample_rows.append(row)
    dtypes, profile = profile_columns(header, sample_rows)

    return {
        "columns": header,
        "dtypes": dtypes,
        "row_count": row_count,
        "dtypes_confidence": "approx",
        "profile": profile,
        "profile_sample_rows": min(len(sample_rows), row_count),
    }
