"""Sub-agent answer verifier (MAC-SQL Refiner pattern).

A single lightweight LLM call that sanity-checks the answer table before
the answer handler commits (deletes CSV artifacts, marks terminal).

The caller (ReAct loop) invokes ``preview_answer_table`` to extract
columns/rows from raw tool-call arguments **without side effects**, then
passes them to ``verify_answer``.  If the verifier rejects, the handler
is never called — the CSV artifact stays on disk and the agent gets
a rejection observation to fix and resubmit.
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Any, cast

from agents.llm.types import ModelAdapter, ModelMessage, ModelResponse
from agents.tools import parse_answer_csv

logger = logging.getLogger(__name__)

_VERIFIER_SYSTEM = """\
You are an answer-table auditor.  You will receive:
1. The original QUESTION the agent was asked.
2. The COLUMNS the agent is about to submit.
3. A SAMPLE of rows (up to 5) and the TOTAL row count.

Run ALL checks and report ALL failures found:

Check 1 — EXTRA columns:
NEVER look at PRODUCING CODE for this check.  The ONLY input is the
submitted COLUMNS array.  SQL/Python may SELECT dozens of columns for
intermediate processing — those are irrelevant.  A column is "extra"
ONLY if it appears in the COLUMNS array AND the question did not ask
for it.  Common mistakes:
- Row-ID / primary-key / code / rank columns (id, CustomerID, PostId,
  基金代码, fund_code, ticker, stock_code, 排名, rank, 序号, …) when
  the question does not ask for identifiers or rankings.
- Filter / WHERE columns whose value is constant across all rows.
- Join-key columns needed for computation but not for the answer.
- Source-table dimension columns (Province, Date, EndDate, Year, Region,
  Category, …) that provide context but are NOT the attribute the question
  asks about.  When the question says "show me X" / "X的记录/数据", only
  X is the result column.  Date/region/category columns are extra unless
  the question explicitly names them (e.g., "show X by province", "X and
  their dates", "for each year").

Check 2 — MISSING columns:
NEVER look at PRODUCING CODE for this check.  Does the question
EXPLICITLY name output attributes that the COLUMNS array does NOT
include?  Only count attributes the question asks you to RETURN, not
values used solely to filter or rank.  Words like "记录"/"records"/
"数据"/"data" do NOT imply additional columns — they refer to rows of
whatever attribute the question names.
Examples:
  "list their ID, sex and diagnosis" → 3 output columns required.
  "give the name and reference" → 2 output columns required.
  "which event has the highest cost" → 1 column (the event); the cost
    is a ranking criterion, NOT a requested output — do NOT flag it
    as missing.
  "who is the fastest driver" → 1 column (the driver name).
When in doubt about whether a value is a requested output or just a
filter/ranking criterion, say "ok".

Check 3 — ROW COUNT sanity:
- Empty result (0 rows) → always flag.
- "How many" / "what percentage" / "what is the average/ratio/count"
  → expect exactly 1 row.  Flag if row_count != 1.
- Global singular superlative questions → expect exactly 1 row.  Flag
  if row_count != 1.  Patterns: "who/what/which is the most/least/best/
  worst/dumbest/fastest/tallest/shortest/oldest/youngest/highest/
  lowest/…", "the most X", "the Nth …" (ordinal).  These ask for
  ONE entity at the global extreme — the answer must be a single row.
  Do NOT apply this rule when the question asks for one extreme per
  group, such as "for each", "in each", "per group/category/region",
  "by category/group", or "which X has the highest Y for each Z"; those
  expect one row per group.
- Only flag row_count == 1 as too few when the question EXPLICITLY
  uses plural list language ("list all", "who are the", "which ones").
  Do NOT flag single-row answers when the question uses singular
  phrasing ("the player's", "which driver", "who is the", "calculate
  for the") — those naturally expect 1 row.
- Data listing / retrieval questions ("show the data", "list the records",
  "查一下数据", "记录是什么样的", "展示/显示/列出", "找一下数据",
  "瞅瞅", "看看", "是多少") expect ALL qualifying source rows.  Multiple
  rows sharing the same date or key value is NORMAL when the source is
  multi-dimensional (e.g., per-province, per-product, per-category).
  Do NOT reject for "duplicate" dates or repeated key values in this
  case — the rows represent different entities at the same time point,
  not true duplicates.
- IMPORTANT: do NOT assume that a question asking about a national-level
  metric (e.g., "国内生产总值", "our country's GDP") requires exactly 1
  row.  The source data may be per-province with no national aggregate
  row, in which case the correct answer is ALL provincial rows.  The
  agent decides whether to aggregate — the verifier must NOT reject a
  multi-row answer just because the question sounds like a single value.
- When in doubt about expected plurality, say "ok".

Check 4 — VALUE sanity (only flag MATHEMATICAL impossibilities):
- A percentage that clearly represents a share/proportion of a whole
  must be in [0, 100] (0% and 100% are valid).
- Growth rates / change-rate percentages may be negative or exceed 100%;
  when unsure whether a percentage is a proportion or a change rate, say "ok".
- A count that clearly represents a discrete item total is impossible
  if negative or fractional.
- A ratio "X times" is impossible if negative.
NEVER judge whether a value is "too high", "too low", or "unlikely"
based on domain knowledge or common sense — you do not have access to
the data.  Values like 100%, 0%, or 99.95% are perfectly valid.
Only flag values that are MATHEMATICALLY impossible for the requested measure.

Check 5 — NULL-filter leakage (only when a PRODUCING CODE section is present):
The PRODUCING CODE section is provided ONLY for this null-filter check.
Do not use producing code for extra, missing, row-count, or value checks.
FIRST check: does the QUESTION itself ask for non-null / non-empty /
non-blank records (e.g., "非空数据", "非空记录", "不为空", "non-null",
"not null", "not empty", "有数据的记录")?  If YES → the NULL filter is
correct and required by the question.  This check PASSES — do NOT
include "null_filter" in the checks array.
Only if the question does NOT ask for non-null records, reject when a
data-listing answer's producing code DROPS NULL rows of a SUBMITTED
column (`WHERE <col> IS NOT NULL`, `.dropna(`, `.notna()`).
Say "ok" in ALL of these cases — they are NOT null-filter leakage:
- `.notna()` / `.isna()` / `IS NOT NULL` used in print/logging/counting
  (e.g., `print(df.notna().sum())`, `COUNT(*)...WHERE x IS NOT NULL`).
  Only reject when the filter controls which rows are WRITTEN to the
  output file or returned as the answer.
- NULL filters on non-submitted columns and inside aggregates
  (AVG/SUM/COUNT...).
NEVER reject an answer for INCLUDING NULL rows.

Check 6 — UNIT SUFFIX in cell values:
Scan the SAMPLE ROWS for numeric cells that carry a unit suffix —
common suffixes: %, 万, 亿, 元, USD, kg, km, m², etc.  A cell like
"48.706%" or "3.5万" is WRONG — the column name may contain the unit
indicator (e.g., "收益率(%)"), but every cell value MUST be a bare
number (e.g., 48.706).  The grader compares normalized numeric values;
a trailing suffix causes a hash mismatch and scores 0.
If ANY numeric cell in the sample contains a unit suffix, reject with
check="unit_suffix" and instruct: "Strip the unit suffix from cell
values — column names keep the unit, cells must be bare numbers."

Check 7 — TRANSPOSED / PIVOTED layout:
The grader matches columns by hashed value lists.  A transposed answer
(metrics as rows instead of columns) produces the WRONG column hashes
and scores 0.  Detect this pattern:
- Exactly 2 columns where one is a label/metric-name column and the
  other is a value column, AND multiple rows whose label column holds
  names that SHOULD be separate result columns (e.g., "总资产最大值",
  "max_revenue", "count", …).
- A "wide" question asking for several named statistics/metrics at once
  ("最大值和最小值", "max and min", "mean, median, and std") is a
  strong signal the answer should have one column per statistic.
If detected, reject with check="transposed" and instruct: "Pivot the
table so each metric is its own column and values fill a single row."

OUTPUT FORMAT — follow EXACTLY:
Your entire response MUST be a single JSON object.  NEVER output
reasoning, analysis, chain-of-thought, or any text before or after
the JSON.  "checks" MUST list ONLY checks that FAILED — NEVER include
a check that passed.

{"verdict": "ok"}
{"verdict": "reject", "checks": ["extra"], "reason": "<25 words per failed check>"}

Be conservative: when in doubt, say "ok"."""

_MAX_SAMPLE_ROWS = 5


@dataclass(frozen=True, slots=True)
class AnswerVerifier:
    """No-tools sub-agent wrapper for pre-submit answer checks.

    ``max_rejections`` is a loop-safety policy: one rejection gives the main
    agent a chance to fix obvious answer-shape errors, while later submissions
    are allowed through so the run cannot get stuck in verifier churn.
    """

    model: ModelAdapter
    max_rejections: int = 2

    def verify(
        self,
        question: str,
        arguments: dict[str, Any],
        producing_code: str | None = None,
    ) -> str | None:
        """Return a rejection message for this answer tool call, or None."""
        result = preview_answer_table(arguments)
        if result is None:
            return None
        columns, rows = result
        return verify_answer(self.model, question, columns, rows, producing_code)


def preview_answer_table(
    arguments: dict[str, Any],
) -> tuple[list[str], list[list[Any]]] | None:
    """Extract columns/rows from raw answer tool-call arguments without side effects.

    Returns ``(columns, rows)`` on success, or ``None`` if the arguments are
    malformed (in which case the normal handler will raise a proper error).
    """
    from_csv = arguments.get("from_csv")
    if from_csv is not None:
        if not isinstance(from_csv, str):
            return None
        path = pathlib.Path(from_csv).expanduser()
        if not path.is_absolute() or not path.exists():
            return None
        try:
            columns, rows, _dtypes = parse_answer_csv(path)
        except (ValueError, OSError):
            return None
        return columns, rows

    columns = arguments.get("columns")
    rows = arguments.get("rows")
    if not columns or not isinstance(columns, list):
        return None
    if rows is None or not isinstance(rows, list):
        return None
    return cast(list[str], columns), cast(list[list[Any]], rows)


def verify_answer(
    adapter: ModelAdapter,
    question: str,
    columns: list[str],
    rows: list[list[Any]],
    producing_code: str | None = None,
) -> str | None:
    """Run the sub-agent verifier.  Returns rejection reason or None."""
    if not columns:
        return None

    n = len(rows)
    if n <= _MAX_SAMPLE_ROWS:
        sample = rows
    else:
        step = max(1, n // _MAX_SAMPLE_ROWS)
        sample = [rows[i * step] for i in range(_MAX_SAMPLE_ROWS) if i * step < n]
    user_content = (
        f"QUESTION: {question}\n\n"
        f"COLUMNS: {json.dumps(columns)}\n\n"
        f"TOTAL ROWS: {len(rows)}\n\n"
        f"SAMPLE ROWS ({len(sample)} of {len(rows)}):\n"
        f"{json.dumps(sample, ensure_ascii=False, default=str)}"
    )
    if producing_code:
        user_content += (
            f"\n\nPRODUCING CODE (for Check 5 null-filter analysis ONLY — "
            f"do NOT use for extra/missing/row-count/value checks; "
            f"SQL may SELECT columns not in the submitted COLUMNS list above, "
            f"that is normal intermediate processing and NOT an extra-column issue):\n{producing_code}"
        )

    messages = [
        ModelMessage(role="system", content=_VERIFIER_SYSTEM),
        ModelMessage(role="user", content=user_content),
    ]

    try:
        response: ModelResponse = adapter.complete(messages)
    except Exception:
        logger.debug("answer verifier LLM call failed, allowing submission", exc_info=True)
        return None

    text = response.content.strip()
    # Strip markdown fence if present
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        logger.debug("answer verifier returned unparseable response: %s", text[:200])
        return None

    verdict = result.get("verdict")
    if verdict == "ok":
        return None

    if verdict == "reject":
        checks_raw: object = result.get("checks", result.get("check", "unknown"))
        if isinstance(checks_raw, list):
            checks = ", ".join(str(c) for c in cast(list[Any], checks_raw))
        else:
            checks = str(checks_raw)
        reason = result.get("reason", "")
        parts = [f"ANSWER REJECTED by verifier (checks={checks}): {reason}"]
        if "null_filter" in str(checks):
            parts.append(
                "NULL-FILTER FIX: unless the question explicitly asks for "
                "non-null/non-empty records, REMOVE every `WHERE col IS NOT NULL`, "
                "`.dropna()`, `.notna()` on submitted columns — "
                "the grader expects NULL rows as empty cells."
            )
        parts.append("Fix the issue and resubmit with `answer`.")
        return " ".join(parts)

    return None
