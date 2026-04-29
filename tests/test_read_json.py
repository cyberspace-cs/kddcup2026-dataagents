"""`read_json_preview` 的单元测试：锁住 inspect-style 契约。

设计意图回顾（见 `filesystem.read_json_preview` docstring）：
- read_json 是 inspect-style 预览，输出按 top-level `kind` 分流。
- object → keys + key_count + 浅层 value_preview（标量原值；集合给 cardinality 不展开）。
- array → length + 前 20 项 head（单项 ≤ 2KB 走 shape stub；累计 ≤ 6KB）。
- scalar → 原值（字符串截断到 200 字符）。
- 文件 > 5MB → kind=unknown + skipped=True，提示走 execute_python。
- 不再有 `max_chars` 字段；`extra="forbid"` 在 ToolRegistry.execute 边界处直接拒绝。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools import filesystem
from data_agent_baseline.tools.filesystem import read_json_preview
from data_agent_baseline.tools.registry import create_default_tool_registry


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_json"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_json", difficulty="easy", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False))


def test_object_returns_keys_and_value_preview(tmp_path: Path) -> None:
    """object：top-level 4 keys；value_preview 给标量原值 + 集合 cardinality + array first_item。"""
    task = _make_task(tmp_path)
    _write_json(
        task.context_dir / "config.json",
        {"a": 1, "b": "x", "c": [1, 2, 3], "d": {"x": 1}},
    )

    out = read_json_preview(task, "config.json")

    assert out["kind"] == "object"
    assert out["key_count"] == 4
    assert out["keys"] == ["a", "b", "c", "d"]
    assert out["keys_truncated"] is False
    assert out["truncated"] is False
    assert out["value_preview"] == {
        "a": 1,
        "b": "x",
        # array 在 value_preview 里额外暴露 first_item（这里第一个元素是标量 1）
        "c": {"kind": "array", "length": 3, "first_item": 1},
        # object 仍保持 depth=1（避免 ∞ 嵌套），不暴露内部
        "d": {"kind": "object", "key_count": 1},
    }


def test_array_returns_head_and_length(tmp_path: Path) -> None:
    """array：length 永远反映全量；head 截到 20 项；truncated 提示后续要切 execute_python。"""
    task = _make_task(tmp_path)
    rows = [{"id": i} for i in range(30)]
    _write_json(task.context_dir / "rows.json", rows)

    out = read_json_preview(task, "rows.json")

    assert out["kind"] == "array"
    assert out["length"] == 30
    assert len(out["head"]) == 20
    assert out["head"] == rows[:20]
    assert out["truncated"] is True
    # 全部小项目 < 2KB 时不应触发任何 head_truncation flag
    assert out["head_truncation"] is None


def test_scalar_json_int(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    (task.context_dir / "int.json").write_text("42")

    out = read_json_preview(task, "int.json")

    assert out == {"path": "int.json", "kind": "scalar", "value": 42}


def test_scalar_json_string(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    _write_json(task.context_dir / "str.json", "hello")

    out = read_json_preview(task, "str.json")

    assert out == {"path": "str.json", "kind": "scalar", "value": "hello"}


def test_scalar_json_bool(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    (task.context_dir / "bool.json").write_text("true")

    out = read_json_preview(task, "bool.json")

    assert out == {"path": "bool.json", "kind": "scalar", "value": True}


def test_scalar_json_null(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    (task.context_dir / "null.json").write_text("null")

    out = read_json_preview(task, "null.json")

    assert out == {"path": "null.json", "kind": "scalar", "value": None}


def test_empty_array(tmp_path: Path) -> None:
    """空数组：length=0、head=[]、truncated=False；下游不应再请求 execute_python。"""
    task = _make_task(tmp_path)
    _write_json(task.context_dir / "empty.json", [])

    out = read_json_preview(task, "empty.json")

    assert out["kind"] == "array"
    assert out["length"] == 0
    assert out["head"] == []
    assert out["truncated"] is False
    assert out["head_truncation"] is None


def test_empty_object(tmp_path: Path) -> None:
    """空对象：key_count=0、keys=[]、value_preview={}、truncated=False。"""
    task = _make_task(tmp_path)
    _write_json(task.context_dir / "empty.json", {})

    out = read_json_preview(task, "empty.json")

    assert out["kind"] == "object"
    assert out["key_count"] == 0
    assert out["keys"] == []
    assert out["keys_truncated"] is False
    assert out["value_preview"] == {}
    assert out["truncated"] is False


def test_oversized_file_short_circuits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """文件 > 5MB：跳过 loads，返回 skipped=True + reason 引导走 execute_python。

    monkeypatch 把 size limit 调到 100 字节，避免实际写 5MB 文件拖慢测试。
    """
    monkeypatch.setattr(filesystem, "_READ_JSON_SIZE_LIMIT", 100)
    task = _make_task(tmp_path)
    payload: dict[str, str] = {"k": "x" * 200}  # 序列化后 > 100 字节
    _write_json(task.context_dir / "big.json", payload)
    actual_size = (task.context_dir / "big.json").stat().st_size
    assert actual_size > 100

    out = read_json_preview(task, "big.json")

    assert out["kind"] == "unknown"
    assert out["skipped"] is True
    assert out["size"] == actual_size
    assert "execute_python" in out["reason"]


def test_object_with_many_keys_caps_at_50(tmp_path: Path) -> None:
    """object 有 120 个 key：keys 截到 50；value_preview 截到 20。"""
    task = _make_task(tmp_path)
    payload = {f"k{i}": i for i in range(120)}
    _write_json(task.context_dir / "many.json", payload)

    out = read_json_preview(task, "many.json")

    assert out["kind"] == "object"
    assert out["key_count"] == 120
    assert len(out["keys"]) == 50
    assert out["keys_truncated"] is True
    assert len(out["value_preview"]) == 20
    assert out["truncated"] is True
    # 显式校验 value_preview 是按 keys 顺序构建的（k0..k19）
    assert list(out["value_preview"].keys()) == [f"k{i}" for i in range(20)]


def test_array_with_oversized_first_item_uses_shape_stub(tmp_path: Path) -> None:
    """array 含 > 2KB 单项：替换为 shape stub 而不是把原值塞进去。"""
    task = _make_task(tmp_path)
    items = [{"big": "x" * 5000}, {"small": 1}]
    _write_json(task.context_dir / "mixed.json", items)

    out = read_json_preview(task, "mixed.json")

    assert out["kind"] == "array"
    assert out["length"] == 2
    assert out["head"][0] == {"kind": "object", "key_count": 1, "keys": ["big"]}
    assert out["head"][1] == {"small": 1}
    assert out["head_truncation"] == "item_cap"
    assert out["truncated"] is True


def test_array_byte_cap_stops_early(tmp_path: Path) -> None:
    """array 单项 < 2KB 但累计 > 6KB：累计上限触发，head_truncation == byte_cap。

    每项 ~500 字节（远低于 2KB 单项上限），20 项约 10KB，必然在 head_cap 之前撞上
    6KB 累计上限——因此 head_truncation 必为 byte_cap，而非 item_cap。
    """
    task = _make_task(tmp_path)
    items = [{"id": i, "padding": "y" * 500} for i in range(50)]
    _write_json(task.context_dir / "wide.json", items)

    out = read_json_preview(task, "wide.json")

    assert out["kind"] == "array"
    assert out["length"] == 50
    # head 不会满 20 项（byte cap 在中途切断）
    assert len(out["head"]) < 20
    assert out["head_truncation"] == "byte_cap"
    assert out["truncated"] is True


def test_value_preview_does_not_recurse_into_objects(tmp_path: Path) -> None:
    """value_preview 中嵌套 object 严格 depth=1：只给 kind+key_count，避免 ∞ 嵌套。"""
    task = _make_task(tmp_path)
    _write_json(task.context_dir / "nested.json", {"a": {"b": {"c": {"d": 1}}}})

    out = read_json_preview(task, "nested.json")

    assert out["value_preview"] == {"a": {"kind": "object", "key_count": 1}}


def test_value_preview_array_exposes_first_item_for_wrapper_object(tmp_path: Path) -> None:
    """`{table, records: [...]}` 包装对象（dataset 通用形态）需要看到一条 record 的样子。

    回归 run 20260429T181710702941Z 的 Patient.json 案例：旧（结构化）实现只给
    `length: 1238` 而不暴露第一条 record 的字段，强制模型每次都先 read_json + 再
    execute_python 才能拿到 schema by example。
    """
    task = _make_task(tmp_path)
    _write_json(
        task.context_dir / "Patient.json",
        {
            "table": "Patient",
            "records": [
                {"id": 30609, "sex": "F", "birth_date": "1962-09-21"},
                {"id": 57266, "sex": "M", "birth_date": "1971-03-04"},
            ],
        },
    )

    out = read_json_preview(task, "Patient.json")

    assert out["value_preview"] == {
        "table": "Patient",
        "records": {
            "kind": "array",
            "length": 2,
            "first_item": {"id": 30609, "sex": "F", "birth_date": "1962-09-21"},
        },
    }


def test_value_preview_array_first_item_falls_back_to_shape_stub_when_oversized(
    tmp_path: Path,
) -> None:
    """value_preview 中 array 的第一个元素 > 2KB 时，first_item 走 shape stub 而非原值。"""
    task = _make_task(tmp_path)
    huge_record = {"big_field": "x" * 5000, "id": 1}
    _write_json(
        task.context_dir / "big.json",
        {"table": "Big", "records": [huge_record]},
    )

    out = read_json_preview(task, "big.json")

    records_summary = out["value_preview"]["records"]
    assert records_summary["kind"] == "array"
    assert records_summary["length"] == 1
    # 不是原值，是 shape stub
    assert records_summary["first_item"] == {
        "kind": "object",
        "key_count": 2,
        "keys": ["big_field", "id"],
    }


def test_value_preview_array_omits_first_item_when_empty(tmp_path: Path) -> None:
    """value_preview 中空数组：只有 kind+length，没有 first_item 字段。"""
    task = _make_task(tmp_path)
    _write_json(task.context_dir / "empty_records.json", {"table": "T", "records": []})

    out = read_json_preview(task, "empty_records.json")

    assert out["value_preview"]["records"] == {"kind": "array", "length": 0}


def test_max_chars_field_rejected(tmp_path: Path) -> None:
    """`max_chars` 已从 ReadJsonInput 移除：传入应在校验边界被拒，不进 handler。"""
    task = _make_task(tmp_path)
    _write_json(task.context_dir / "x.json", {"a": 1})
    registry = create_default_tool_registry()

    with pytest.raises(ValueError, match=r"read_json: extra field 'max_chars' is not permitted"):
        registry.execute(task, "read_json", {"path": "x.json", "max_chars": 4000})


def test_path_escaping_raises(tmp_path: Path) -> None:
    """安全边界：拒绝走出 context dir。"""
    task = _make_task(tmp_path)
    _write_json(task.context_dir / "ok.json", {"a": 1})

    with pytest.raises(ValueError, match="escapes context dir"):
        read_json_preview(task, "../escape.json")


def test_missing_file_raises(tmp_path: Path) -> None:
    task = _make_task(tmp_path)

    with pytest.raises(FileNotFoundError):
        read_json_preview(task, "nope.json")


def test_invalid_json_raises(tmp_path: Path) -> None:
    """非法 JSON：让 JSONDecodeError 直接传播，由模型在下一轮决定是否换工具。"""
    task = _make_task(tmp_path)
    (task.context_dir / "bad.json").write_text("{not valid}")

    with pytest.raises(json.JSONDecodeError):
        read_json_preview(task, "bad.json")


def test_unicode_preserved_in_value_preview(tmp_path: Path) -> None:
    """中文等非 ASCII 字符不应被 \\uXXXX 转义。"""
    task = _make_task(tmp_path)
    _write_json(task.context_dir / "zh.json", {"name": "李雷"})

    out = read_json_preview(task, "zh.json")

    assert out["value_preview"]["name"] == "李雷"


def test_long_string_value_truncated(tmp_path: Path) -> None:
    """value_preview 中的字符串值超 200 字符：截断 + value_truncated=True 标记。"""
    task = _make_task(tmp_path)
    _write_json(task.context_dir / "long.json", {"desc": "x" * 500})

    out = read_json_preview(task, "long.json")

    desc_preview = out["value_preview"]["desc"]
    assert isinstance(desc_preview, dict)
    assert desc_preview["value_truncated"] is True
    assert len(desc_preview["value"]) == 200
