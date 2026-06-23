#!/usr/bin/env python
"""Triage failing video tasks in a run by comparing prediction.csv vs gold.csv.

Usage:
    uv run python scripts/triage_failures.py --run artifacts/runs/<run_id> [--gold data/public/output]

Emits a markdown table with one row per imperfect task and a coarse failure-mode
guess (column_count_mismatch, value_mismatch_short, value_mismatch_large) so
the engineer can quickly bucket failures before deeper diff work.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    if not path.exists():
        return [], []
    # Gold/pred CSVs are usually UTF-8 but some legacy gold files are GBK on Windows.
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as fh:
                rows = list(csv.reader(fh))
            break
        except UnicodeDecodeError:
            continue
    else:
        return [], []
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _classify(gold_rows: list[list[str]], pred_rows: list[list[str]]) -> str:
    g, p = len(gold_rows), len(pred_rows)
    if g == 0 and p == 0:
        return "both_empty"
    if g == 0 or p == 0:
        return "one_side_empty"
    if g == p:
        return "value_mismatch_same_count"
    if p > g * 2:
        return "value_mismatch_large_excess"
    if g > p * 2:
        return "value_mismatch_large_shortfall"
    return "value_mismatch_small_diff"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="artifacts/runs/<run_id>")
    parser.add_argument("--gold", type=Path, default=Path("data/public/output"))
    args = parser.parse_args()

    scores_path = args.run / "scores.json"
    if not scores_path.exists():
        print(f"ERROR: {scores_path} not found", flush=True)
        return 2
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    per_task = scores.get("per_task", {})

    print("| task | score | status | gold_rows | pred_rows | classification |")
    print("|---|---|---|---|---|---|")
    for tid, info in sorted(per_task.items(), key=lambda kv: kv[1].get("score", 0.0)):
        score = info.get("score", 0.0)
        status = info.get("status", "")
        if score >= 0.99 or status == "missing_prediction":
            continue
        gold_path = args.gold / tid / "gold.csv"
        pred_path = args.run / tid / "prediction.csv"
        _, gold_rows = _read_csv(gold_path)
        _, pred_rows = _read_csv(pred_path)
        classification = _classify(gold_rows, pred_rows)
        print(
            f"| {tid} | {score:.3f} | {status} | {len(gold_rows)} | {len(pred_rows)} | {classification} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
