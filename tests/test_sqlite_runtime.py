"""SQLite 工具运行时行为测试。

重点覆盖 `_maybe_raise_non_sqlite_path_error` 把 sqlite 的
`file is not a database` 翻译成带路由提示的 ValueError——以便 driver
（`react.py:374`）把它打包成下一轮的 `observation.error`，模型一步纠偏。

回归 run 20260430-013 task_16 类故障：模型对 `csv/trans.csv` 调
execute_context_sql，原始观察只有"file is not a database"，下一步往往
继续在 sqlite 工具上挣扎。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from data_agent_baseline.tools.sqlite import (
    execute_read_only_sql,
    inspect_sqlite_schema,
)


@pytest.mark.parametrize(
    ("ext", "redirect_keyword"),
    [
        (".csv", "read_csv"),
        (".json", "read_json"),
        (".md", "read_doc"),
        (".txt", "read_doc"),
    ],
)
def test_inspect_sqlite_schema_translates_non_db_path(
    tmp_path: Path, ext: str, redirect_keyword: str
) -> None:
    """非 sqlite 路径调 inspect_sqlite_schema → ValueError 含路径名/扩展名/路由提示。"""
    target = tmp_path / f"trans{ext}"
    target.write_text("dummy content\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        inspect_sqlite_schema(target)

    msg = str(exc_info.value)
    assert target.name in msg
    assert ext in msg
    assert "not a sqlite database" in msg
    assert redirect_keyword in msg
    # 仍要透出"应放弃 sqlite 工具"的元信号
    assert "inspect_sqlite_schema" in msg or "execute_context_sql" in msg


@pytest.mark.parametrize(
    ("ext", "redirect_keyword"),
    [
        (".csv", "read_csv"),
        (".json", "read_json"),
        (".md", "read_doc"),
        (".txt", "read_doc"),
    ],
)
def test_execute_read_only_sql_translates_non_db_path(
    tmp_path: Path, ext: str, redirect_keyword: str
) -> None:
    """非 sqlite 路径调 execute_read_only_sql → ValueError 含路径名/扩展名/路由提示。"""
    target = tmp_path / f"trans{ext}"
    target.write_text("dummy content\n", encoding="utf-8")

    # `SELECT 1` is a constant query and never reads the file, so sqlite never
    # validates the file header → no "file is not a database" → translation
    # path doesn't fire. Use a query that touches the file to mirror what real
    # model-issued SQL would do.
    with pytest.raises(ValueError) as exc_info:
        execute_read_only_sql(target, "SELECT * FROM sqlite_master")

    msg = str(exc_info.value)
    assert target.name in msg
    assert ext in msg
    assert "not a sqlite database" in msg
    assert redirect_keyword in msg


def test_unknown_extension_falls_back_to_execute_python_hint(tmp_path: Path) -> None:
    """未识别扩展名（如 .parquet）兜底建议 execute_python——不要把模型推回 sqlite 工具。"""
    target = tmp_path / "data.parquet"
    target.write_text("not a real parquet, just bytes for the test", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        inspect_sqlite_schema(target)

    msg = str(exc_info.value)
    assert ".parquet" in msg
    assert "execute_python" in msg


def test_no_extension_falls_back_to_execute_python_hint(tmp_path: Path) -> None:
    """无扩展名也要给具体路由（execute_python），不能空回"file is not a database"。"""
    target = tmp_path / "rawdata"
    target.write_text("plain bytes\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        inspect_sqlite_schema(target)

    msg = str(exc_info.value)
    assert "extension: none" in msg
    assert "execute_python" in msg


def test_real_sqlite_db_still_works(tmp_path: Path) -> None:
    """正向回归：真 sqlite 文件不被翻译路径误伤。"""
    db_path = tmp_path / "real.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO t VALUES (1, 'a'), (2, 'b')")
    conn.commit()
    conn.close()

    schema = inspect_sqlite_schema(db_path)
    assert schema["tables"] and schema["tables"][0]["name"] == "t"

    result = execute_read_only_sql(db_path, "SELECT id, name FROM t ORDER BY id")
    assert result["columns"] == ["id", "name"]
    assert result["rows"] == [[1, "a"], [2, "b"]]


def test_other_database_errors_pass_through(tmp_path: Path) -> None:
    """非 `file is not a database` 的 DatabaseError 继续向上抛——
    确保翻译逻辑只针对路径类型错误，不吞别的诊断信息。"""
    db_path = tmp_path / "real.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    conn.close()

    # 查一张不存在的表 → OperationalError("no such table"), 不是路径类型错
    with pytest.raises(sqlite3.OperationalError) as exc_info:
        execute_read_only_sql(db_path, "SELECT * FROM nonexistent_table")
    assert "no such table" in str(exc_info.value).lower()
