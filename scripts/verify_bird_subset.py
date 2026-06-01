"""End-to-end verification for data/bird_dev_subset.

Checks (each task):
  1. directory layout: input/task_N/{task.json, context/{db/*.sqlite, knowledge.md}}
                       output/task_N/gold.csv
  2. task.json schema: keys exactly {task_id, difficulty, question}; task_id == dir name
  3. difficulty in {medium, hard}
  4. slim sqlite opens read-only and contains all tables listed in the manifest
  5. gold SQL re-executes against the slim DB and the result matches gold.csv byte-for-byte
  6. manifest aggregates match: counts, db distribution, all 11 dbs covered

Exits non-zero on any failure.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBSET = PROJECT_ROOT / "data/bird_dev_subset"
BIRD_JSON = PROJECT_ROOT / ".scratch/bird/mini_dev_sqlite.json"

EXPECTED_DBS = {
    "debit_card_specializing",
    "student_club",
    "thrombosis_prediction",
    "european_football_2",
    "formula_1",
    "superhero",
    "codebase_community",
    "card_games",
    "toxicology",
    "california_schools",
    "financial",
}
EXPECTED_DIFFICULTIES = {"medium", "hard"}
EXPECTED_TASK_KEYS = {"task_id", "difficulty", "question"}

# Bird difficulty → project difficulty mapping (mirrors build_bird_subset.py)
BIRD_TO_PROJECT_DIFFICULTY = {"moderate": "medium", "challenging": "hard"}


def fail(task_id: str, msg: str, errors: list[str]) -> None:
    errors.append(f"[{task_id}] {msg}")


def csv_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def sql_result_rows(db: Path, sql: str) -> tuple[list[str], list[list[str]]]:
    """Run SQL and return (header, rows) matching csv writer output."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cur = conn.execute(sql)
        header = [d[0] for d in cur.description] if cur.description else []
        rows = []
        for row in cur.fetchall():
            rows.append(["" if cell is None else str(cell) for cell in row])
    finally:
        conn.close()
    return header, rows


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    if not SUBSET.exists():
        print(f"FATAL: subset root not found: {SUBSET}")
        return 2

    manifest_path = SUBSET / "manifest.json"
    if not manifest_path.exists():
        print(f"FATAL: manifest not found: {manifest_path}")
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_task = {t["task_id"]: t for t in manifest["tasks"]}

    bird_records = {
        int(r["question_id"]): r for r in json.loads(BIRD_JSON.read_text(encoding="utf-8"))
    }

    input_root = SUBSET / "input"
    output_root = SUBSET / "output"

    input_dirs = sorted(
        [p for p in input_root.iterdir() if p.is_dir() and p.name.startswith("task_")],
        key=lambda p: int(p.name.removeprefix("task_")),
    )
    output_dirs = sorted(
        [p for p in output_root.iterdir() if p.is_dir() and p.name.startswith("task_")],
        key=lambda p: int(p.name.removeprefix("task_")),
    )

    input_ids = [p.name for p in input_dirs]
    output_ids = [p.name for p in output_dirs]
    if input_ids != output_ids:
        errors.append(
            f"input/output dir mismatch: only-in-input={set(input_ids) - set(output_ids)}, "
            f"only-in-output={set(output_ids) - set(input_ids)}"
        )

    if set(input_ids) != set(by_task):
        errors.append(
            f"manifest/dirs mismatch: dirs_only={set(input_ids) - set(by_task)}, "
            f"manifest_only={set(by_task) - set(input_ids)}"
        )

    counts: Counter[str] = Counter()
    db_counts: Counter[str] = Counter()

    for d in input_dirs:
        tid = d.name

        task_json = d / "task.json"
        if not task_json.exists():
            fail(tid, "missing task.json", errors)
            continue
        try:
            payload = json.loads(task_json.read_text(encoding="utf-8"))
        except Exception as e:
            fail(tid, f"task.json invalid JSON: {e}", errors)
            continue

        if set(payload) != EXPECTED_TASK_KEYS:
            fail(tid, f"task.json keys = {sorted(payload)}", errors)
        if payload.get("task_id") != tid:
            fail(tid, f"task_id field {payload.get('task_id')!r} != dir name", errors)
        diff = payload.get("difficulty")
        if diff not in EXPECTED_DIFFICULTIES:
            fail(tid, f"unexpected difficulty {diff!r}", errors)
        if not isinstance(payload.get("question"), str) or not payload["question"].strip():
            fail(tid, "question missing/empty", errors)
        counts[diff] += 1

        ctx = d / "context"
        if not ctx.is_dir():
            fail(tid, "missing context dir", errors)
            continue
        km = ctx / "knowledge.md"
        if not km.exists() or km.stat().st_size == 0:
            fail(tid, "knowledge.md missing/empty", errors)

        db_dir = ctx / "db"
        if not db_dir.is_dir():
            fail(tid, "missing context/db", errors)
            continue
        sqlites = sorted(db_dir.glob("*.sqlite"))
        if len(sqlites) != 1:
            fail(tid, f"expected 1 .sqlite, got {len(sqlites)}", errors)
            continue
        slim_db = sqlites[0]

        mentry = by_task.get(tid)
        if mentry is None:
            fail(tid, "absent from manifest", errors)
            continue
        db_counts[mentry["db_id"]] += 1
        if slim_db.stem != mentry["db_id"]:
            fail(
                tid, f"slim db name {slim_db.stem!r} != manifest db_id {mentry['db_id']!r}", errors
            )

        # Open slim DB and verify tables present
        try:
            conn = sqlite3.connect(f"file:{slim_db}?mode=ro", uri=True)
            try:
                got = {
                    r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                }
            finally:
                conn.close()
        except Exception as e:
            fail(tid, f"slim db unreadable: {e}", errors)
            continue
        wanted = set(mentry["tables"])
        if not wanted.issubset(got):
            fail(tid, f"slim db missing tables {wanted - got} (has {sorted(got)})", errors)

        # Verify gold.csv matches re-executing the gold SQL
        gold_csv = output_root / tid / "gold.csv"
        if not gold_csv.exists():
            fail(tid, "missing gold.csv", errors)
            continue
        try:
            written = csv_rows(gold_csv)
        except Exception as e:
            fail(tid, f"gold.csv unreadable: {e}", errors)
            continue

        bird = bird_records.get(mentry["bird_question_id"])
        if bird is None:
            fail(
                tid,
                f"bird question_id {mentry['bird_question_id']} not found in source JSON",
                errors,
            )
            continue
        # cross-check difficulty mapping against the BIRD source row
        expected_diff = BIRD_TO_PROJECT_DIFFICULTY.get(bird["difficulty"])
        if expected_diff != diff:
            fail(
                tid,
                f"difficulty mismatch: BIRD={bird['difficulty']!r} -> {expected_diff!r}, task.json={diff!r}",
                errors,
            )
        try:
            header, rows = sql_result_rows(slim_db, bird["SQL"])
        except Exception as e:
            fail(tid, f"gold SQL re-execute failed: {e}", errors)
            continue

        expected_rows = [header, *rows] if header else rows
        if written != expected_rows:
            # surface first divergence
            sample_w = written[:3]
            sample_e = expected_rows[:3]
            fail(
                tid,
                f"gold.csv != re-executed result "
                f"(written rows={len(written)}, expected={len(expected_rows)}; "
                f"first written={sample_w!r}, first expected={sample_e!r})",
                errors,
            )

        if len(rows) != mentry["row_count"]:
            warnings.append(
                f"[{tid}] manifest row_count={mentry['row_count']} but re-run yielded {len(rows)}"
            )

    print(f"verified {len(input_dirs)} task dirs")
    print(f"  difficulty distribution: {dict(counts)}")
    print(f"  db distribution:         {dict(sorted(db_counts.items()))}")
    missing_dbs = EXPECTED_DBS - set(db_counts)
    if missing_dbs:
        warnings.append(f"databases not represented: {sorted(missing_dbs)}")
    print(f"  warnings: {len(warnings)}")
    for w in warnings:
        print("    !", w)
    print(f"  errors: {len(errors)}")
    for e in errors:
        print("    X", e)

    if errors:
        print("\nVERIFICATION FAILED")
        return 1
    print("\nVERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
