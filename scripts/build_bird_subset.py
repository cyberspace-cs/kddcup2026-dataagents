"""Build a slim test subset from BIRD mini-dev (sqlite split).

Output mirrors `data/public/{input,output}/task_<n>/` so the existing
`DABenchPublicDataset` loader can read it unchanged. Each generated task contains:

    data/bird_dev_subset/input/task_<n>/task.json
    data/bird_dev_subset/input/task_<n>/context/db/<db_id>.sqlite     (slim copy)
    data/bird_dev_subset/input/task_<n>/context/knowledge.md          (evidence + schema)
    data/bird_dev_subset/output/task_<n>/gold.csv                     (executed SQL result)

Difficulty mapping (BIRD -> project): moderate -> medium, challenging -> hard.

Usage:
    uv run python scripts/build_bird_subset.py \
        --json .scratch/bird/mini_dev_sqlite.json \
        --dev-db-root .scratch/bird/dev_databases \
        --out-root data/bird_dev_subset \
        [--medium 30] [--hard 15] [--seed 20260522]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class BirdRecord:
    question_id: int
    db_id: str
    question: str
    evidence: str
    sql: str
    difficulty: str


def load_records(json_path: Path) -> list[BirdRecord]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return [
        BirdRecord(
            question_id=int(r["question_id"]),
            db_id=str(r["db_id"]),
            question=str(r["question"]),
            evidence=str(r.get("evidence") or ""),
            sql=str(r["SQL"]),
            difficulty=str(r["difficulty"]),
        )
        for r in payload
    ]


# --------------------------------------------------------------------- sampling

def sql_signature(sql: str) -> tuple[bool, ...]:
    """Coarse fingerprint over SQL features, used to spread picks within a db."""
    s = sql.upper()
    has_subquery = s.count("SELECT") >= 2
    return (
        "JOIN" in s,
        "GROUP BY" in s,
        "ORDER BY" in s,
        "LIMIT" in s,
        "WHERE" in s,
        ("CASE" in s) or ("IIF(" in s),
        has_subquery,
        any(f"{fn}(" in s for fn in ("AVG", "SUM", "COUNT", "MAX", "MIN")),
        "DISTINCT" in s,
        "HAVING" in s,
        "LIKE" in s,
        "BETWEEN" in s,
    )


def stratified_sample(
    records: list[BirdRecord], n: int, difficulty: str, seed: int
) -> list[BirdRecord]:
    """Round-robin across db_id, preferring unseen SQL signatures inside each db."""
    pool = [r for r in records if r.difficulty == difficulty]
    by_db: dict[str, list[BirdRecord]] = defaultdict(list)
    for r in pool:
        by_db[r.db_id].append(r)

    rng = random.Random(seed)
    dbs = sorted(by_db.keys())
    rng.shuffle(dbs)
    shuffled: dict[str, list[BirdRecord]] = {
        d: rng.sample(by_db[d], k=len(by_db[d])) for d in dbs
    }
    used_sigs: dict[str, set[tuple[bool, ...]]] = defaultdict(set)
    cursors = dict.fromkeys(dbs, 0)
    chosen: list[BirdRecord] = []

    while len(chosen) < n:
        made_progress = False
        for d in dbs:
            if len(chosen) >= n:
                break
            pick_idx = -1
            for j in range(cursors[d], len(shuffled[d])):
                sig = sql_signature(shuffled[d][j].sql)
                if sig not in used_sigs[d]:
                    pick_idx = j
                    break
            if pick_idx < 0 and cursors[d] < len(shuffled[d]):
                pick_idx = cursors[d]
            if pick_idx >= 0:
                r = shuffled[d][pick_idx]
                chosen.append(r)
                used_sigs[d].add(sql_signature(r.sql))
                cursors[d] = pick_idx + 1
                made_progress = True
        if not made_progress:
            break
    return chosen


# ------------------------------------------------------------------- DB slimming

_IDENT = r"[`\"\[]?([A-Za-z_][\w]*)[`\"\]]?"
_TABLE_REF_RE = re.compile(
    rf"(?i)\b(?:FROM|JOIN)\s+{_IDENT}(?:\s*\.\s*{_IDENT})?"
)


def extract_tables(sql: str) -> list[str]:
    """Best-effort extract table identifiers from FROM/JOIN clauses."""
    cleaned = re.sub(r"--.*?$", "", sql, flags=re.M)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.S)
    found: list[str] = []
    for m in _TABLE_REF_RE.finditer(cleaned):
        first, second = m.group(1), m.group(2)
        name = second if second else first
        if name:
            found.append(name)
    # de-dup preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in found:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def slim_sqlite(
    src_db: Path, dst_db: Path, gold_sql: str
) -> tuple[list[str], list[tuple[str, str]]]:
    """Copy only the tables referenced by `gold_sql` into a fresh SQLite file.

    Returns:
        (table_names_copied, ddl_pairs)  where ddl_pairs is [(table, CREATE statement), ...]
    """
    if dst_db.exists():
        dst_db.unlink()
    dst_db.parent.mkdir(parents=True, exist_ok=True)

    src = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True)
    src.text_factory = bytes  # avoid decode errors on legacy BIRD blobs
    try:
        rows = src.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        existing = {r[0].decode("utf-8", errors="replace"): r[0] for r in rows}
        existing_lc = {k.lower(): k for k in existing}

        wanted = extract_tables(gold_sql)
        resolved: list[str] = []
        for w in wanted:
            real = existing_lc.get(w.lower())
            if real and real not in resolved:
                resolved.append(real)
        if not resolved:
            raise ValueError(
                f"No referenced tables resolved against {src_db.name}; "
                f"wanted={wanted!r}, available={sorted(existing_lc)!r}"
            )

        dst = sqlite3.connect(dst_db)
        ddl_pairs: list[tuple[str, str]] = []
        try:
            for t in resolved:
                ddl_row = src.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (t,),
                ).fetchone()
                if not ddl_row or not ddl_row[0]:
                    continue
                ddl_text = ddl_row[0].decode("utf-8", errors="replace")
                dst.execute(ddl_text)
                ddl_pairs.append((t, ddl_text))

                src_rows = src.execute(f'SELECT * FROM "{t}"').fetchall()
                if not src_rows:
                    continue
                ncols = len(src_rows[0])
                placeholders = ",".join("?" * ncols)
                # decode bytes->str for non-blob cells while preserving NULLs
                decoded_rows = [
                    tuple(
                        cell.decode("utf-8", errors="replace")
                        if isinstance(cell, bytes)
                        else cell
                        for cell in row
                    )
                    for row in src_rows
                ]
                dst.executemany(
                    f'INSERT INTO "{t}" VALUES ({placeholders})', decoded_rows
                )

            for t in resolved:
                idx_rows = src.execute(
                    "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL",
                    (t,),
                ).fetchall()
                for (idx_sql_bytes,) in idx_rows:
                    if not idx_sql_bytes:
                        continue
                    try:
                        dst.execute(
                            idx_sql_bytes.decode("utf-8", errors="replace")
                        )
                    except sqlite3.OperationalError:
                        pass
            dst.commit()
        finally:
            dst.close()
        return resolved, ddl_pairs
    finally:
        src.close()


# ----------------------------------------------------------- gold answer & docs

def run_gold(db_path: Path, sql: str) -> tuple[list[str], list[tuple]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.execute(sql)
        header = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
    finally:
        conn.close()
    return header, rows


def write_gold_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        for row in rows:
            w.writerow(row)


def write_knowledge(
    path: Path,
    db_id: str,
    evidence: str,
    ddl_pairs: list[tuple[str, str]],
) -> None:
    """Lightweight knowledge.md: schema dump + the BIRD evidence as a hint."""
    lines: list[str] = []
    pretty_db = db_id.replace("_", " ").title()
    lines.append(f"# Knowledge Guide for {pretty_db} Database\n")
    lines.append("## 1. Introduction")
    lines.append(
        f"This guide accompanies the `{db_id}` database derived from "
        "BIRD mini-dev. It lists the relevant table schemas and the "
        "expert evidence provided with the original question.\n"
    )
    lines.append("## 2. Tables in this Task")
    for tname, ddl in ddl_pairs:
        lines.append(f"### `{tname}`")
        lines.append("```sql")
        lines.append(ddl.strip())
        lines.append("```\n")
    lines.append("## 3. Domain Evidence")
    if evidence.strip():
        lines.append(evidence.strip())
    else:
        lines.append("_No additional evidence was provided for this task._")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------------ main

DIFFICULTY_MAP = {"moderate": "medium", "challenging": "hard"}


def build_subset(
    json_path: Path,
    dev_db_root: Path,
    out_root: Path,
    n_medium: int,
    n_hard: int,
    seed: int,
) -> None:
    records = load_records(json_path)
    print(f"loaded {len(records)} BIRD mini-dev records")

    medium = stratified_sample(records, n_medium, "moderate", seed)
    hard = stratified_sample(records, n_hard, "challenging", seed + 1)

    # Interleave medium first, then hard, so task_1..task_30 == medium, task_31..task_45 == hard
    plan = [(r, DIFFICULTY_MAP[r.difficulty]) for r in medium] + [
        (r, DIFFICULTY_MAP[r.difficulty]) for r in hard
    ]

    in_root = out_root / "input"
    out_dir = out_root / "output"
    if in_root.exists():
        shutil.rmtree(in_root)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    in_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    failures: list[dict] = []

    for idx, (rec, mapped_difficulty) in enumerate(plan, start=1):
        task_id = f"task_{idx}"
        task_in_dir = in_root / task_id
        task_ctx_dir = task_in_dir / "context"
        task_db_dir = task_ctx_dir / "db"

        src_db = dev_db_root / rec.db_id / f"{rec.db_id}.sqlite"
        if not src_db.exists():
            failures.append(
                {
                    "task_id": task_id,
                    "db_id": rec.db_id,
                    "question_id": rec.question_id,
                    "reason": f"source database not found: {src_db}",
                }
            )
            continue

        slim_db = task_db_dir / f"{rec.db_id}.sqlite"
        try:
            tables, ddl_pairs = slim_sqlite(src_db, slim_db, rec.sql)
            header, rows = run_gold(slim_db, rec.sql)
        except Exception as e:  # noqa: BLE001
            failures.append(
                {
                    "task_id": task_id,
                    "db_id": rec.db_id,
                    "question_id": rec.question_id,
                    "reason": f"{type(e).__name__}: {e}",
                }
            )
            shutil.rmtree(task_in_dir, ignore_errors=True)
            continue

        task_json = {
            "task_id": task_id,
            "difficulty": mapped_difficulty,
            "question": rec.question,
        }
        (task_in_dir / "task.json").write_text(
            json.dumps(task_json, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_knowledge(
            task_ctx_dir / "knowledge.md",
            rec.db_id,
            rec.evidence,
            ddl_pairs,
        )
        write_gold_csv(out_dir / task_id / "gold.csv", header, rows)

        manifest.append(
            {
                "task_id": task_id,
                "difficulty": mapped_difficulty,
                "bird_question_id": rec.question_id,
                "db_id": rec.db_id,
                "tables": tables,
                "row_count": len(rows),
                "bird_difficulty": rec.difficulty,
            }
        )
        print(
            f"  built {task_id} ({mapped_difficulty}, db={rec.db_id}, "
            f"tables={tables}, rows={len(rows)})"
        )

    (out_root / "manifest.json").write_text(
        json.dumps(
            {
                "source": "birdsql/bird_mini_dev (sqlite split)",
                "seed": seed,
                "n_medium": n_medium,
                "n_hard": n_hard,
                "tasks": manifest,
                "failures": failures,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"\ndone: {len(manifest)} tasks built, {len(failures)} skipped. "
        f"Manifest at {out_root / 'manifest.json'}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--json",
        type=Path,
        default=PROJECT_ROOT / ".scratch/bird/mini_dev_sqlite.json",
        help="path to mini_dev_sqlite JSON",
    )
    p.add_argument(
        "--dev-db-root",
        type=Path,
        default=PROJECT_ROOT / ".scratch/bird/dev_databases",
        help="root that contains per-db subdirs with <db_id>/<db_id>.sqlite",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "data/bird_dev_subset",
        help="where to write {input,output}/task_<n>",
    )
    p.add_argument("--medium", type=int, default=30, help="number of medium tasks")
    p.add_argument("--hard", type=int, default=15, help="number of hard tasks")
    p.add_argument("--seed", type=int, default=20260522, help="RNG seed")
    args = p.parse_args(argv)

    if not args.json.exists():
        print(f"missing JSON: {args.json}", file=sys.stderr)
        return 2
    if not args.dev_db_root.exists():
        print(f"missing dev_databases root: {args.dev_db_root}", file=sys.stderr)
        return 2

    build_subset(
        json_path=args.json,
        dev_db_root=args.dev_db_root,
        out_root=args.out_root,
        n_medium=args.medium,
        n_hard=args.hard,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
