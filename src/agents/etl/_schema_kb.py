"""Schema knowledge base: lookup known table schemas from real SQLite tables."""

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

_DB_PATH = Path(__file__).with_name("schema_kb.db")
_conn: sqlite3.Connection | None = None

_TYPE_MAP = {"INTEGER": "integer", "REAL": "number", "TEXT": "string"}


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
    return _conn


@dataclass(frozen=True, slots=True)
class KnownSchema:
    table_name: str
    columns: list[str] = field(default_factory=lambda: list[str]())
    dtypes: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    sample_rows: list[dict[str, object]] = field(default_factory=lambda: list[dict[str, object]]())


def lookup(table_key: str) -> KnownSchema | None:
    """Look up a known schema by table name (case-insensitive)."""
    if not _DB_PATH.exists():
        return None
    conn = _get_conn()

    # Find actual table name (sqlite_master is case-preserving)
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND LOWER(name)=?",
        (table_key.lower(),),
    ).fetchone()
    if not row:
        return None
    real_name = row[0]

    info: list[tuple[int, str, str, int, object, int]] = conn.execute(
        f"PRAGMA table_info([{real_name}])"
    ).fetchall()
    columns: list[str] = [r[1] for r in info]
    dtypes: dict[str, str] = {r[1]: _TYPE_MAP.get(r[2], "string") for r in info}

    cursor = conn.execute(f"SELECT * FROM [{real_name}] LIMIT 5")
    desc = cursor.description or []
    col_names: list[str] = [str(d[0]) for d in desc]
    sample_rows: list[dict[str, object]] = [
        dict(zip(col_names, row, strict=False)) for row in cursor.fetchall()
    ]

    return KnownSchema(
        table_name=real_name,
        columns=columns,
        dtypes=dtypes,
        sample_rows=sample_rows,
    )


def all_table_names() -> list[str]:
    """Return all known table names."""
    if not _DB_PATH.exists():
        return []
    conn = _get_conn()
    return [
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    ]
