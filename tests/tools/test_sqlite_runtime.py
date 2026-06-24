"""SQLite 工具运行时行为测试。

重点覆盖 `_maybe_raise_non_sqlite_path_error` 把 sqlite 的
`file is not a database` 翻译成带路由提示的 ValueError——以便 driver
把它打包成下一轮的 `observation.error`，模型一步纠偏。

回归 run 20260430-013 task_16 类故障：模型对 `csv/trans.csv` 调
execute_context_sql，原始观察只有"file is not a database"，下一步往往
继续在 sqlite 工具上挣扎。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from agents.tools.execute_sql import execute_read_only_sql
from agents.tools.inspect_sqlite import inspect_sqlite_database


@pytest.mark.parametrize(
    "invoke",
    [
        pytest.param(inspect_sqlite_database, id="inspect_sqlite_schema"),
        # `SELECT 1` is a constant query and never reads the file, so sqlite never
        # validates the file header → no "file is not a database" → translation
        # path doesn't fire. Use a query that touches the file to mirror what real
        # model-issued SQL would do.
        pytest.param(
            lambda path: execute_read_only_sql(path, "SELECT * FROM sqlite_master"),
            id="execute_read_only_sql",
        ),
    ],
)
@pytest.mark.parametrize(
    ("ext", "redirect_keyword"),
    [
        (".csv", "preview_file"),
        (".json", "preview_file"),
        (".md", "preview_file"),
        (".txt", "preview_file"),
    ],
)
def test_sqlite_tools_translate_non_db_path(
    tmp_path: Path, invoke: Callable[[Path], object], ext: str, redirect_keyword: str
) -> None:
    """非 sqlite 路径 → ValueError 含路径名/扩展名/路由提示。"""
    target = tmp_path / f"trans{ext}"
    target.write_text("dummy content\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        invoke(target)

    msg = str(exc_info.value)
    assert target.name in msg
    assert ext in msg
    assert "not a sqlite database" in msg
    assert redirect_keyword in msg
    # 仍要透出"应放弃 sqlite 工具"的元信号
    assert "execute_context_sql" in msg


def test_unknown_extension_falls_back_to_execute_python_hint(tmp_path: Path) -> None:
    """未识别扩展名（如 .parquet）兜底建议 execute_python——不要把模型推回 sqlite 工具。"""
    target = tmp_path / "data.parquet"
    target.write_text("not a real parquet, just bytes for the test", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        inspect_sqlite_database(target)

    msg = str(exc_info.value)
    assert ".parquet" in msg
    assert "execute_python" in msg


def test_no_extension_falls_back_to_execute_python_hint(tmp_path: Path) -> None:
    """无扩展名也要给具体路由（execute_python），不能空回"file is not a database"。"""
    target = tmp_path / "rawdata"
    target.write_text("plain bytes\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        inspect_sqlite_database(target)

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

    schema = inspect_sqlite_database(db_path)
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


def test_empty_result_returns_zero_row_count(tmp_path: Path) -> None:
    """SQL query returning 0 rows has row_count=0 in the result dict."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE items (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO items VALUES (1, 'a'), (2, 'b')")
    conn.commit()
    conn.close()

    result = execute_read_only_sql(db_path, "SELECT * FROM items WHERE id = 999")
    assert result["row_count"] == 0
    assert result["rows"] == []
