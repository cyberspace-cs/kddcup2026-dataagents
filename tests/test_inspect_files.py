"""inspect_files 工具的单元测试。

覆盖点：
- mixed context（csv / json / md / sqlite / xlsx 占位 / 未知扩展）下的统一输出形态
- CSV dtype 推断（int/float/str）
- sqlite tables → columns 抽取
- supported=false 占位（xlsx + 未知扩展）
- file_count / unsupported_formats / truncated 字段语义
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.filesystem import inspect_context_files


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_x"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_x", difficulty="easy", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_sqlite(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("CREATE TABLE visits (id INTEGER, patient_id INTEGER, date TEXT)")
        conn.execute("INSERT INTO patients (id, name) VALUES (1, 'a')")
        conn.commit()


@pytest.fixture
def task_with_mixed_context(tmp_path: Path) -> PublicTask:
    task = _make_task(tmp_path)
    context = task.context_dir
    # CSV with int/float/str columns — dtype 推断三类样本
    _write_csv(
        context / "people.csv",
        header=["age", "score", "name"],
        rows=[
            ["12", "3.14", "alice"],
            ["34", "2.71", "bob"],
            ["56", "1.41", "carol"],
        ],
    )
    # JSON object with keys
    (context / "config.json").write_text(json.dumps({"alpha": 1, "beta": "x"}))
    # markdown doc
    (context / "knowledge.md").write_text("# Topic\nline 2\nline 3\nline 4")
    # sqlite database
    _write_sqlite(context / "main.sqlite")
    # xlsx 占位 — 已识别但本工具未原生支持
    (context / "data.xlsx").write_bytes(b"\x50\x4b\x03\x04")
    # 未知扩展 — 也走 supported=false 分支
    (context / "blob.bin").write_bytes(b"\x00\x01\x02")
    return task


def test_inspect_files_returns_one_entry_per_file(task_with_mixed_context: PublicTask) -> None:
    result = inspect_context_files(task_with_mixed_context)
    assert result["file_count"] == 6
    assert result["truncated"] is False
    paths = {entry["path"] for entry in result["files"]}
    assert paths == {
        "people.csv",
        "config.json",
        "knowledge.md",
        "main.sqlite",
        "data.xlsx",
        "blob.bin",
    }


def test_inspect_files_csv_dtype_inference(task_with_mixed_context: PublicTask) -> None:
    result = inspect_context_files(task_with_mixed_context)
    csv_entry = next(entry for entry in result["files"] if entry["path"] == "people.csv")
    assert csv_entry["format"] == "csv"
    assert csv_entry["supported"] is True
    schema = csv_entry["schema"]
    assert schema["columns"] == ["age", "score", "name"]
    assert schema["dtypes"] == ["int", "float", "str"]
    assert schema["row_count"] == 3
    assert schema["dtypes_confidence"] == "approx"


def test_inspect_files_sqlite_tables(task_with_mixed_context: PublicTask) -> None:
    result = inspect_context_files(task_with_mixed_context)
    sqlite_entry = next(entry for entry in result["files"] if entry["path"] == "main.sqlite")
    assert sqlite_entry["format"] == "sqlite"
    assert sqlite_entry["supported"] is True
    tables = sqlite_entry["schema"]["tables"]
    assert set(tables.keys()) == {"patients", "visits"}
    assert tables["patients"] == ["id", "name"]
    assert tables["visits"] == ["id", "patient_id", "date"]


def test_inspect_files_xlsx_marked_unsupported(task_with_mixed_context: PublicTask) -> None:
    result = inspect_context_files(task_with_mixed_context)
    xlsx_entry = next(entry for entry in result["files"] if entry["path"] == "data.xlsx")
    assert xlsx_entry["supported"] is False
    assert xlsx_entry["format"] == "xlsx"
    assert "execute_python" in xlsx_entry["reason"]
    assert "xlsx" in result["unsupported_formats"]


def test_inspect_files_unknown_extension_marked_unsupported(
    task_with_mixed_context: PublicTask,
) -> None:
    result = inspect_context_files(task_with_mixed_context)
    bin_entry = next(entry for entry in result["files"] if entry["path"] == "blob.bin")
    assert bin_entry["supported"] is False
    assert bin_entry["format"] == "bin"
    assert isinstance(bin_entry["reason"], str) and bin_entry["reason"]


def test_inspect_files_json_object_summary(task_with_mixed_context: PublicTask) -> None:
    result = inspect_context_files(task_with_mixed_context)
    json_entry = next(entry for entry in result["files"] if entry["path"] == "config.json")
    assert json_entry["format"] == "json"
    schema = json_entry["schema"]
    assert schema["kind"] == "object"
    assert set(schema["keys"]) == {"alpha", "beta"}
    assert schema["key_count"] == 2


def test_inspect_files_markdown_head(task_with_mixed_context: PublicTask) -> None:
    result = inspect_context_files(task_with_mixed_context)
    md_entry = next(entry for entry in result["files"] if entry["path"] == "knowledge.md")
    assert md_entry["format"] == "markdown"
    schema = md_entry["schema"]
    assert schema["line_count"] == 4
    assert schema["head"] == ["# Topic", "line 2", "line 3"]


def test_inspect_files_truncates_when_above_limit(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    # 创建 105 个空 .txt 占位文件，触发上限截断
    for index in range(105):
        (task.context_dir / f"file_{index:03d}.txt").write_text("")
    result = inspect_context_files(task)
    assert result["truncated"] is True
    assert result["file_count"] == 100


def test_inspect_files_empty_context(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    result = inspect_context_files(task)
    assert result["file_count"] == 0
    assert result["files"] == []
    assert result["unsupported_formats"] == []
    assert result["truncated"] is False
