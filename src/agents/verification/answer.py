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

Run these checks IN ORDER and report the FIRST failure found:

Check 1 — EXTRA columns:
Is any column NOT requested by the question?  Common mistakes:
- Row-ID / primary-key columns (id, CustomerID, PostId, …) when the
  question does not ask for identifiers.
- Filter / WHERE columns whose value is constant across all rows.
- Join-key columns needed for computation but not for the answer.
- Source-table dimension columns (Province, Date, EndDate, Year, Region,
  Category, …) that provide context but are NOT the attribute the question
  asks about.  When the question says "show me X" / "X的记录/数据", only
  X is the result column.  Date/region/category columns are extra unless
  the question explicitly names them (e.g., "show X by province", "X and
  their dates", "for each year").

Check 2 — MISSING columns:
Does the question EXPLICITLY ask the user to output attributes that
the answer does NOT include?  Only count attributes the question asks
you to RETURN, not values used solely to filter or rank.
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
Reject a data-listing answer whose producing code drops NULL rows of a
SUBMITTED column (`WHERE <col> IS NOT NULL`, `.dropna(`, `.notna()`)
unless the question explicitly excludes empty values.  Instruction:
rebuild keeping NULL rows; filter only on identity/key columns.
NULL filters on non-submitted columns and inside aggregates
(AVG/SUM/COUNT...) are fine — say "ok".

Reply with a JSON object (no markdown fence):
{"verdict": "ok"} — all checks pass.
{"verdict": "reject", "check": "extra|missing|row_count|value|null_filter", \
"reason": "one-sentence diagnosis with concrete fix instruction"}

Be conservative: when in doubt, say "ok".  Only reject when you are
confident the answer violates one of the checks above: extra/missing
columns, row-count expectations, or mathematical value constraints.
Never reject merely because a value seems surprising, high, low, or unlikely."""

_MAX_SAMPLE_ROWS = 5


@dataclass(frozen=True, slots=True)
class AnswerVerifier:
    """No-tools sub-agent wrapper for pre-submit answer checks.

    ``max_rejections`` is a loop-safety policy: one rejection gives the main
    agent a chance to fix obvious answer-shape errors, while later submissions
    are allowed through so the run cannot get stuck in verifier churn.
    """

    model: ModelAdapter
    max_rejections: int = 1

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

    sample = rows[:_MAX_SAMPLE_ROWS]
    user_content = (
        f"QUESTION: {question}\n\n"
        f"COLUMNS: {json.dumps(columns)}\n\n"
        f"TOTAL ROWS: {len(rows)}\n\n"
        f"SAMPLE ROWS ({len(sample)} of {len(rows)}):\n"
        f"{json.dumps(sample, ensure_ascii=False, default=str)}"
    )
    if producing_code:
        user_content += (
            f"\n\nPRODUCING CODE (the final execute steps that built this table):\n{producing_code}"
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
        check = result.get("check", "unknown")
        reason = result.get("reason", "")
        return (
            f"ANSWER REJECTED by verifier (check={check}): {reason} "
            "Fix the issue and resubmit with `answer`."
        )

    return None
