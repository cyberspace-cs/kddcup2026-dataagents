"""Convert BIRD mini-dev (SQLite split) into DABench heterogeneous dataset format.

Produces 324 tasks with difficulty distribution:
  Easy: 68, Medium: 136, Hard: 115, Extreme: 5

Data modalities: CSV, JSON ({"table": ..., "records": [...]}), SQLite DB.

Usage:
    uv run python scripts/convert_bird_to_dabench.py
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIRD_DB_DIR = PROJECT_ROOT / "data" / "bird_databases" / "dev_databases"
COMP_INPUT_DIR = PROJECT_ROOT / "data" / "public" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "bird_dev"

TARGET = {"easy": 68, "medium": 136, "hard": 115, "extreme": 5}  # total 324

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class BirdEntry:
    question_id: int
    db_id: str
    question: str
    evidence: str
    sql: str
    difficulty: str  # simple / moderate / challenging
    sql_complexity: int = 0  # computed later


def sql_complexity_score(sql: str) -> int:
    """Rough heuristic for SQL complexity (higher = harder)."""
    sql_upper = sql.upper()
    score = 0
    score += sql_upper.count("JOIN") * 3
    score += sql_upper.count("SUBQUER") * 5  # won't match, use nesting
    score += sql_upper.count("SELECT") * 2  # nested selects
    score += sql_upper.count("GROUP BY") * 2
    score += sql_upper.count("HAVING") * 3
    score += sql_upper.count("CASE WHEN") * 2
    score += sql_upper.count("UNION") * 4
    score += sql_upper.count("INTERSECT") * 4
    score += sql_upper.count("EXCEPT") * 4
    score += sql_upper.count("EXISTS") * 3
    score += sql_upper.count("NOT IN") * 2
    score += len(sql) // 80  # length bonus
    return score


def extract_tables_from_sql(sql: str, db_tables: list[str]) -> list[str]:
    """Best-effort extraction of table names referenced in a SQL query."""
    found = []
    for t in db_tables:
        patterns = [
            rf"\b{re.escape(t)}\b",
            rf"`{re.escape(t)}`",
            rf'"{re.escape(t)}"',
        ]
        for pat in patterns:
            if re.search(pat, sql, re.IGNORECASE):
                if t not in found:
                    found.append(t)
                break
    return found if found else db_tables[:2]


def get_db_tables(db_path: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'"
        ).fetchall()
    ]
    conn.close()
    return tables


def quote_sqlite_identifier(name: str) -> str:
    """Quote a SQLite identifier with double-quote escaping."""
    return '"' + name.replace('"', '""') + '"'


def table_to_json(db_path: str, table_name: str) -> dict:
    """Export a table as {"table": name, "records": [...]}."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"SELECT * FROM {quote_sqlite_identifier(table_name)}").fetchall()
    records = [dict(r) for r in rows]
    conn.close()
    return {"table": table_name, "records": records}


def table_to_csv_rows(db_path: str, table_name: str) -> tuple[list[str], list[list]]:
    """Export a table as (headers, rows)."""
    conn = sqlite3.connect(db_path)
    cur = conn.execute(f"SELECT * FROM {quote_sqlite_identifier(table_name)}")
    headers = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    conn.close()
    return headers, rows


def copy_tables_to_new_db(src_db: str, dst_db: str, table_names: list[str]) -> None:
    """Copy specific tables from src to a new SQLite database."""
    src = sqlite3.connect(src_db)
    dst = sqlite3.connect(dst_db)
    for t in table_names:
        ddl = src.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (t,)
        ).fetchone()
        if ddl and ddl[0]:
            dst.execute(ddl[0])
            quoted = quote_sqlite_identifier(t)
            rows = src.execute(f"SELECT * FROM {quoted}").fetchall()
            if rows:
                placeholders = ",".join(["?"] * len(rows[0]))
                dst.executemany(f"INSERT INTO {quoted} VALUES ({placeholders})", rows)
    dst.commit()
    src.close()
    dst.close()


def build_knowledge_md(db_id: str, evidence: str, referenced_tables: list[str]) -> str:
    """Build knowledge.md from BIRD evidence + database_description CSVs."""
    desc_dir = BIRD_DB_DIR / db_id / "database_description"

    sections = []
    sections.append(f"# Knowledge Guide for {db_id.replace('_', ' ').title()} Database\n")
    sections.append("## 1. Introduction")
    sections.append(
        f'This guide provides semantic context for the "{db_id}" database, '
        "covering entity definitions, field semantics, and analytical conventions.\n"
    )

    # Core entities from database_description CSVs
    sections.append("## 2. Core Entities & Fields\n")
    if desc_dir.exists():
        for csv_file in sorted(desc_dir.iterdir()):
            if csv_file.suffix != ".csv":
                continue
            table_name = csv_file.stem
            sections.append(f"### {table_name}")
            try:
                with open(csv_file, encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        col = row.get("original_column_name", "").strip()
                        desc = row.get("column_description", "").strip()
                        fmt = row.get("data_format", "").strip()
                        val_desc = row.get("value_description", "").strip()
                        if not col:
                            continue
                        line = f"- **{col}** ({fmt}): {desc}"
                        if val_desc:
                            line += f" — {val_desc}"
                        sections.append(line)
            except Exception:
                pass
            sections.append("")

    # Evidence as analytical hints
    if evidence and evidence.strip():
        sections.append("## 3. Analytical Hints\n")
        sections.append("The following domain knowledge may be relevant for analysis:\n")
        sections.append(f"- {evidence.strip()}\n")

    # Constraints
    sections.append("## 4. Constraints & Conventions\n")
    sections.append("- All dates are stored in 'YYYY-MM-DD' format unless otherwise noted.")
    sections.append("- NULL values indicate missing or unknown data.")
    sections.append("- Integer IDs are used as primary keys across tables.\n")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Format assignment: decide which tables go to CSV / JSON / DB per task
# ---------------------------------------------------------------------------

FORMAT_CHOICES = ["csv", "json", "db"]


def assign_formats(
    tables: list[str],
    difficulty: str,
    rng: random.Random,
) -> dict[str, str]:
    """Assign each referenced table a format. Returns {table_name: format}."""
    n = len(tables)
    if n == 0:
        return {}

    if n == 1:
        # single table: pick a format weighted by difficulty
        if difficulty == "easy":
            fmt = rng.choice(["csv", "json"])
        elif difficulty in ("medium", "hard"):
            fmt = rng.choice(["csv", "json", "db"])
        else:  # extreme
            fmt = rng.choice(["csv", "json", "db"])
        return {tables[0]: fmt}

    # Multiple tables: ensure at least 2 different formats for heterogeneity
    pool = ["csv", "json"] if difficulty == "easy" else ["csv", "json", "db"]

    assignment = {}
    formats_used = set()

    for i, t in enumerate(tables):
        if i == 0:
            fmt = rng.choice(pool)
        elif len(formats_used) == 1 and i == len(tables) - 1:
            # last table and only 1 format used so far: force a different one
            remaining = [f for f in pool if f not in formats_used]
            fmt = rng.choice(remaining) if remaining else rng.choice(pool)
        else:
            fmt = rng.choice(pool)
        assignment[t] = fmt
        formats_used.add(fmt)

    return assignment


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def load_competition_questions() -> set[str]:
    """Load existing competition task questions for dedup."""
    questions = set()
    if not COMP_INPUT_DIR.exists():
        return questions
    for task_dir in COMP_INPUT_DIR.iterdir():
        tj = task_dir / "task.json"
        if tj.exists():
            with open(tj) as f:
                questions.add(json.load(f)["question"].strip())
    return questions


def select_and_assign_difficulties(
    entries: list[BirdEntry], rng: random.Random
) -> list[tuple[BirdEntry, str]]:
    """Select 324 entries and assign DABench difficulty levels.

    Strategy:
    - simple → easy (sample 68 from 133)
    - moderate → medium (sample 136) + some promoted to hard (fill gap)
    - challenging → hard (all 92) + extreme (top 5 by complexity)
    """
    by_diff = defaultdict(list)
    for e in entries:
        by_diff[e.difficulty].append(e)

    for lst in by_diff.values():
        rng.shuffle(lst)

    # Sort challenging by complexity for extreme selection
    for e in by_diff["challenging"]:
        e.sql_complexity = sql_complexity_score(e.sql)
    by_diff["challenging"].sort(key=lambda e: e.sql_complexity, reverse=True)

    result: list[tuple[BirdEntry, str]] = []

    # Extreme: top 5 challenging
    extreme_pool = by_diff["challenging"][:5]
    for e in extreme_pool:
        result.append((e, "extreme"))
    remaining_challenging = by_diff["challenging"][5:]

    # Hard: rest of challenging (87) + need 28 more from moderate
    hard_from_challenging = remaining_challenging[: TARGET["hard"]]
    for e in hard_from_challenging:
        result.append((e, "hard"))

    hard_deficit = TARGET["hard"] - len(hard_from_challenging)

    # Promote hardest moderate entries to hard
    for e in by_diff["moderate"]:
        e.sql_complexity = sql_complexity_score(e.sql)
    by_diff["moderate"].sort(key=lambda e: e.sql_complexity, reverse=True)

    hard_from_moderate = by_diff["moderate"][:hard_deficit]
    for e in hard_from_moderate:
        result.append((e, "hard"))
    remaining_moderate = by_diff["moderate"][hard_deficit:]

    # Medium: sample 136 from remaining moderate
    medium_pool = remaining_moderate[: TARGET["medium"]]
    for e in medium_pool:
        result.append((e, "medium"))

    # Easy: sample 68 from simple
    easy_pool = by_diff["simple"][: TARGET["easy"]]
    for e in easy_pool:
        result.append((e, "easy"))

    # Verify counts
    assigned_counts = Counter(d for _, d in result)
    for diff, target in TARGET.items():
        actual = assigned_counts.get(diff, 0)
        if actual != target:
            print(f"WARNING: {diff} has {actual}, expected {target}")

    return result


def execute_gold_sql(db_path: str, sql: str) -> tuple[list[str], list[list]]:
    """Execute gold SQL and return (headers, rows)."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(sql)
        headers = [desc[0] for desc in cur.description]
        rows = [list(r) for r in cur.fetchall()]
        return headers, rows
    finally:
        conn.close()


def write_task(
    task_id: str,
    entry: BirdEntry,
    difficulty: str,
    rng: random.Random,
) -> bool:
    """Write a single task directory. Returns True on success."""
    db_path = str(BIRD_DB_DIR / entry.db_id / f"{entry.db_id}.sqlite")
    if not os.path.exists(db_path):
        print(f"SKIP {task_id}: database {entry.db_id} not found")
        return False

    # Execute gold SQL
    try:
        gold_headers, gold_rows = execute_gold_sql(db_path, entry.sql)
    except Exception as exc:
        print(f"SKIP {task_id}: SQL execution failed: {exc}")
        return False

    if not gold_rows and not gold_headers:
        print(f"SKIP {task_id}: empty result")
        return False

    # Determine referenced tables
    all_tables = get_db_tables(db_path)
    ref_tables = extract_tables_from_sql(entry.sql, all_tables)

    # Assign formats
    fmt_assignment = assign_formats(ref_tables, difficulty, rng)

    # Create directories
    task_input = OUTPUT_DIR / "input" / task_id
    task_output = OUTPUT_DIR / "output" / task_id
    context_dir = task_input / "context"

    for d in [context_dir / "csv", context_dir / "json", context_dir / "db"]:
        d.mkdir(parents=True, exist_ok=True)

    # Write task.json
    with open(task_input / "task.json", "w") as f:
        json.dump(
            {"task_id": task_id, "difficulty": difficulty, "question": entry.question},
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Write context files per format assignment
    db_tables_for_sqlite = []
    for table_name, fmt in fmt_assignment.items():
        if fmt == "csv":
            headers, rows = table_to_csv_rows(db_path, table_name)
            csv_path = context_dir / "csv" / f"{table_name}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
        elif fmt == "json":
            data = table_to_json(db_path, table_name)
            json_path = context_dir / "json" / f"{table_name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str)
        elif fmt == "db":
            db_tables_for_sqlite.append(table_name)

    # Write SQLite DB if any tables assigned to db format
    if db_tables_for_sqlite:
        dst_db = context_dir / "db" / f"{entry.db_id}.db"
        copy_tables_to_new_db(db_path, str(dst_db), db_tables_for_sqlite)

    # Clean up empty format directories
    for sub in ["csv", "json", "db"]:
        d = context_dir / sub
        if d.exists() and not list(d.iterdir()):
            d.rmdir()

    # Write knowledge.md
    knowledge = build_knowledge_md(entry.db_id, entry.evidence, ref_tables)
    with open(context_dir / "knowledge.md", "w", encoding="utf-8") as f:
        f.write(knowledge)

    # Write gold.csv
    task_output.mkdir(parents=True, exist_ok=True)
    with open(task_output / "gold.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(gold_headers)
        writer.writerows(gold_rows)

    return True


def main():
    rng = random.Random(SEED)

    # Load BIRD mini-dev SQLite split
    print("Loading BIRD mini-dev dataset...")
    ds = load_dataset("birdsql/bird_mini_dev")
    raw = list(ds["mini_dev_sqlite"])

    entries = [
        BirdEntry(
            question_id=r["question_id"],
            db_id=r["db_id"],
            question=r["question"],
            evidence=r["evidence"],
            sql=r["SQL"],
            difficulty=r["difficulty"],
        )
        for r in raw
    ]

    # Dedup against competition
    comp_q = load_competition_questions()
    entries = [e for e in entries if e.question.strip() not in comp_q]
    print(f"After dedup: {len(entries)} entries available")

    # Select and assign difficulties
    selected = select_and_assign_difficulties(entries, rng)
    print(f"Selected: {len(selected)} tasks")
    counts = Counter(d for _, d in selected)
    for d in ["easy", "medium", "hard", "extreme"]:
        print(f"  {d}: {counts.get(d, 0)}")

    # Clean output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    # Write tasks
    success = 0
    fail = 0
    for i, (entry, diff) in enumerate(selected, 1):
        task_id = f"task_{i}"
        if write_task(task_id, entry, diff, rng):
            success += 1
        else:
            fail += 1

    print(f"\nDone: {success} tasks written, {fail} failed")
    print(f"Output: {OUTPUT_DIR}")

    # Summary
    diffs = defaultdict(int)
    for task_dir in sorted((OUTPUT_DIR / "input").iterdir()):
        tj = task_dir / "task.json"
        if tj.exists():
            with open(tj) as f:
                d = json.load(f)["difficulty"]
                diffs[d] += 1
    print("\nFinal distribution:")
    for d in ["easy", "medium", "hard", "extreme"]:
        print(f"  {d}: {diffs.get(d, 0)}")


if __name__ == "__main__":
    main()
