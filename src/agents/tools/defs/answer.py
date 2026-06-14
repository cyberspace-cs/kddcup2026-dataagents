"""answer tool: terminal tool for submitting the final answer table."""

from __future__ import annotations

import csv
import pathlib
import shutil
from contextlib import suppress
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agents.benchmark.schema import AnswerTable, PublicTask
from agents.tools.contracts import (
    INLINE_ANSWER_CELL_LIMIT,
    INLINE_ANSWER_ROW_LIMIT,
    answer_artifact_expression,
    artifact_answer_size_phrase,
    inline_answer_size_phrase,
)
from agents.tools.decorator import function_tool
from agents.tools.filesystem import classify_dtype
from agents.tools.registry import ToolExecutionResult

_ANSWER_SCRATCH_ROOT = pathlib.Path("/tmp/dabench")
_FROM_CSV_SIZE_LIMIT_BYTES = 5 * 1024 * 1024


def _answer_dir_for(task: PublicTask) -> pathlib.Path:
    return _ANSWER_SCRATCH_ROOT / task.task_id / "_answer"


class AnswerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "Column header names. Leave empty when using from_csv. "
                "Inline is rejected for 10+ rows — use from_csv instead."
            ),
        ),
    ]
    rows: Annotated[
        list[list[str | int | float | bool | None]],
        Field(
            default_factory=list,
            description=(
                "Answer rows; each row's length must match the columns array. "
                "Leave empty when using from_csv. "
                "Inline is rejected for 10+ rows — use from_csv instead."
            ),
        ),
    ]
    from_csv: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Absolute path to a CSV file containing the answer table "
                "(header row + data rows). REQUIRED for 10+ rows or 50+ cells "
                "(inline is rejected above this threshold). When set, `columns` "
                "and `rows` MUST be empty. Convention: write the CSV from "
                "`execute_python` under `os.environ['DABENCH_ANSWER_DIR']`."
            ),
        ),
    ] = None

    @model_validator(mode="after")
    def _check_payload(self) -> AnswerInput:
        if self.from_csv is not None:
            if self.columns or self.rows:
                raise ValueError("provide either from_csv or columns+rows, not both")
            return self
        if not self.columns:
            raise ValueError("columns must be non-empty when from_csv is not set")
        n_rows = len(self.rows)
        n_cells = n_rows * len(self.columns)
        if n_rows >= INLINE_ANSWER_ROW_LIMIT or n_cells >= INLINE_ANSWER_CELL_LIMIT:
            raise ValueError(
                f"inline rows payload exceeds artifact-handoff threshold "
                f"(got {n_rows} rows, {n_cells} cells; threshold = "
                f"{INLINE_ANSWER_ROW_LIMIT} rows or {INLINE_ANSWER_CELL_LIMIT} cells); "
                f"write the table to "
                f"{answer_artifact_expression()} from "
                f"execute_python and resubmit with from_csv (no columns/rows)"
            )
        width = len(self.columns)
        for index, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(f"row {index} has {len(row)} cells but columns has {width}")
        return self


def _coerce_row(row: list[str], dtypes: list[str]) -> list[Any]:
    out: list[Any] = []
    for cell, dtype in zip(row, dtypes, strict=True):
        if cell == "":
            out.append(None)
        elif dtype == "int":
            out.append(int(cell))
        elif dtype == "float":
            out.append(float(cell))
        elif dtype == "bool":
            out.append(cell.lower() == "true")
        else:
            out.append(cell)
    return out


def parse_answer_csv(path: pathlib.Path) -> tuple[list[str], list[list[Any]], list[str]]:
    """Read and type-cast an answer artifact CSV.

    Returns ``(header, typed_rows, dtypes)``.  All failures raise
    ``ValueError``; the dispatcher translates to ``ToolErrorEvent``.
    """
    if not path.exists():
        raise ValueError(f"from_csv file not found: {path}")
    if path.stat().st_size > _FROM_CSV_SIZE_LIMIT_BYTES:
        raise ValueError("from_csv exceeds 5MB; split the table or use inline columns/rows")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            raw = list(csv.reader(handle))
    except UnicodeDecodeError as exc:
        raise ValueError(f"from_csv is not valid UTF-8: {exc}") from exc
    except csv.Error as exc:
        raise ValueError(f"failed to parse from_csv: {exc}") from exc
    if not raw:
        raise ValueError("from_csv is empty; expected header row + data rows")
    header, *data = raw
    if not header:
        raise ValueError("from_csv header row is empty")
    width = len(header)
    for index, row in enumerate(data):
        if len(row) != width:
            raise ValueError(f"from_csv row {index} has {len(row)} cells but header has {width}")
    dtypes: list[str] = []
    for col_idx in range(width):
        column_values = [row[col_idx] for row in data]
        dtypes.append(classify_dtype(column_values))
    typed = [_coerce_row(row, dtypes) for row in data]
    return header, typed, dtypes


def _prune_null_columns(
    columns: list[str], rows: list[list[Any]]
) -> tuple[list[str], list[list[Any]], list[str]]:
    if not rows or not columns:
        return columns, rows, []

    width = len(columns)
    keep_indices: list[int] = []
    removed: list[str] = []

    for col_idx in range(width):
        all_null = all(
            row[col_idx] is None or (isinstance(row[col_idx], str) and row[col_idx].strip() == "")
            for row in rows
            if col_idx < len(row)
        )
        if all_null and len(rows) > 0:
            removed.append(columns[col_idx])
        else:
            keep_indices.append(col_idx)

    if not removed:
        return columns, rows, []

    pruned_columns = [columns[i] for i in keep_indices]
    pruned_rows = [[row[i] for i in keep_indices] for row in rows]
    return pruned_columns, pruned_rows, removed


@function_tool(
    description=(
        "Use this when the final answer table is determined. This is the only "
        "way to submit; plain text answers are ignored. "
        f"For small answers ({inline_answer_size_phrase()}): pass inline columns/rows. "
        f"For {artifact_answer_size_phrase()}: write CSV in execute_python first, "
        "then pass from_csv (inline is rejected at this threshold). "
        "IMPORTANT: for content-entity questions "
        '("what is the comment/post/message/title"), columns must contain '
        "the text body (Text/Body/Title), NOT a surrogate key (Id/Code)."
    ),
    input_model=AnswerInput,
    is_terminal=True,
)
def answer(task: PublicTask, args: AnswerInput) -> ToolExecutionResult:
    artifact_path = _answer_dir_for(task) / "answer.csv"
    if args.from_csv is not None:
        path = pathlib.Path(args.from_csv).expanduser()
        if not path.is_absolute():
            raise ValueError(
                f"from_csv must be an absolute path, got '{args.from_csv}' "
                "(hint: use os.environ['DABENCH_ANSWER_DIR'] to build the path)"
            )
        columns, rows, dtypes = parse_answer_csv(path)
        columns, rows, pruned_cols = _prune_null_columns(columns, rows)
        answer_table = AnswerTable(columns=columns, rows=rows)
        path.unlink(missing_ok=True)
        shutil.rmtree(artifact_path.parent, ignore_errors=True)
        content: dict[str, Any] = {
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(rows),
            "from_csv": {
                "path": str(path),
                "dtypes": dtypes,
                "head_preview": rows[:5],
            },
        }
        if pruned_cols:
            content["pruned_null_columns"] = pruned_cols
        return ToolExecutionResult(
            ok=True,
            content=content,
            is_terminal=True,
            answer=answer_table,
        )
    if artifact_path.exists() and artifact_path.stat().st_size > 0:
        raise ValueError(
            f"answer artifact already exists at {artifact_path} but inline "
            "columns/rows were sent; the artifact is the canonical answer "
            "and inline cannot override it. Resubmit as "
            f'answer({{"from_csv": "{artifact_path}"}}) with no columns/rows.'
        )
    columns = list(args.columns)
    rows = [list(row) for row in args.rows]
    columns, rows, pruned_cols = _prune_null_columns(columns, rows)
    answer_table = AnswerTable(columns=columns, rows=rows)
    shutil.rmtree(artifact_path.parent, ignore_errors=True)
    content: dict[str, Any] = {
        "status": "submitted",
        "column_count": len(columns),
        "row_count": len(rows),
    }
    if pruned_cols:
        content["pruned_null_columns"] = pruned_cols
    return ToolExecutionResult(
        ok=True,
        content=content,
        is_terminal=True,
        answer=answer_table,
    )


def build_answer_table_without_side_effects(
    task: PublicTask,
    arguments: dict[str, Any],
) -> AnswerTable | None:
    """Validate and normalize raw answer arguments without committing."""
    try:
        args = AnswerInput.model_validate(arguments)
    except ValidationError:
        return None

    if args.from_csv is not None:
        path = pathlib.Path(args.from_csv).expanduser()
        if not path.is_absolute():
            return None
        try:
            columns, rows, _dtypes = parse_answer_csv(path)
        except (ValueError, OSError):
            return None
        columns, rows, _pruned_cols = _prune_null_columns(columns, rows)
        return AnswerTable(columns=columns, rows=rows)

    artifact_path = _answer_dir_for(task) / "answer.csv"
    try:
        if artifact_path.exists() and artifact_path.stat().st_size > 0:
            return None
    except OSError:
        return None

    columns = list(args.columns)
    rows = [list(row) for row in args.rows]
    columns, rows, _pruned_cols = _prune_null_columns(columns, rows)
    return AnswerTable(columns=columns, rows=rows)


def cleanup_answer_artifacts(task: PublicTask, arguments: dict[str, Any]) -> None:
    """Best-effort cleanup after promoting a fallback answer."""
    try:
        args = AnswerInput.model_validate(arguments)
    except ValidationError:
        return

    if args.from_csv is not None:
        path = pathlib.Path(args.from_csv).expanduser()
        if path.is_absolute():
            with suppress(OSError):
                path.unlink(missing_ok=True)

    shutil.rmtree(_answer_dir_for(task), ignore_errors=True)
