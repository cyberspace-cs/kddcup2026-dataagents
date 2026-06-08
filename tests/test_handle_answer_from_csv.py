"""`handle_answer` 的 from_csv artifact 路径单元测试。

锁住设计契约：
- 列级 dtype 推断与 inline 路径在 prediction.csv 字面值上对齐（int 不被悄悄变 float）
- 失败一律以 `ValueError` 上抛（dispatcher 翻译成 `ToolErrorEvent`），让 ReAct 循环 retry
- ack content 给 trace 留 `from_csv.dtypes` + `head_preview` 便于离线对账

XOR 校验、空 columns 报错由 `tests/test_dispatcher.py` 端到端校验；本文件只测 handler 内部行为。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.benchmark.schema import (
    AnswerTable,
    PublicTask,
    TaskAssets,
    TaskRecord,
)
from agents.tools import handlers as _handlers
from agents.tools.handlers import handle_answer
from agents.tools.inputs import AnswerInput


@pytest.fixture
def task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    # Production now anchors _answer/ under `/tmp/dabench/<task_id>/_answer/` so
    # the §3.4 read-only `/input` mount doesn't break execute_python. Repointing
    # the scratch root at tmp_path keeps existing assertions (`task.task_dir /
    # "_answer"`) byte-equivalent — `_answer_dir_for(task)` collapses to
    # `tmp_path / "task_1" / "_answer"` which is exactly `task.task_dir / "_answer"`.
    monkeypatch.setattr(_handlers, "_ANSWER_SCRATCH_ROOT", tmp_path)
    return PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ----- #1 happy path：mixed dtypes -----


def test_from_csv_happy_path_returns_typed_answer(task: PublicTask, tmp_path: Path) -> None:
    """3 行 (int / str / float)：cell 类型按列推断而非全部落地为 str。"""
    csv_path = _write(
        tmp_path / "_answer" / "answer.csv",
        "id,name,ratio\n1,alpha,0.5\n2,beta,1.25\n3,gamma,2.0\n",
    )
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    result = handle_answer(task, args)

    assert result.ok is True
    assert result.is_terminal is True
    assert isinstance(result.answer, AnswerTable)
    assert result.answer.columns == ["id", "name", "ratio"]
    assert result.answer.rows == [
        [1, "alpha", 0.5],
        [2, "beta", 1.25],
        [3, "gamma", 2.0],
    ]
    # 类型必须严格保留（不是字符串等价）
    assert isinstance(result.answer.rows[0][0], int)
    assert isinstance(result.answer.rows[0][2], float)


# ----- #2 int IDs preservation -----


def test_from_csv_preserves_integer_ids(task: PublicTask, tmp_path: Path) -> None:
    """整数主键 ID 必须以 int 落地（与 inline 路径一致），不被 float 抢占。

    回归参照 `test_dispatch_answer_preserves_integer_cells`：
    avoid `163109.0` 写进 prediction.csv。
    """
    csv_path = _write(
        tmp_path / "_answer" / "ids.csv",
        "ID\n163109\n2803470\n",
    )
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    result = handle_answer(task, args)

    assert result.answer is not None
    cell0 = result.answer.rows[0][0]
    cell1 = result.answer.rows[1][0]
    assert cell0 == 163109 and isinstance(cell0, int)
    assert cell1 == 2803470 and isinstance(cell1, int)
    # bool 是 int 子类，单独防御
    assert not isinstance(cell0, bool)


# ----- #3 float preservation -----


def test_from_csv_preserves_float_values(task: PublicTask, tmp_path: Path) -> None:
    csv_path = _write(
        tmp_path / "_answer" / "ratios.csv",
        "ratio\n1.5\n0.25\n",
    )
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    result = handle_answer(task, args)

    assert result.answer is not None
    assert result.answer.rows == [[1.5], [0.25]]
    assert isinstance(result.answer.rows[0][0], float)
    assert isinstance(result.answer.rows[1][0], float)


# ----- #4 empty cell -> None -----


def test_from_csv_empty_cell_becomes_none(task: PublicTask, tmp_path: Path) -> None:
    """空字符串单元格落地为 None（而不是 `""`），保持与 inline `null` 语义对齐。"""
    csv_path = _write(
        tmp_path / "_answer" / "mix.csv",
        "id,sex,diag\n,F,SLE\n2,M,RA\n",
    )
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    result = handle_answer(task, args)

    assert result.answer is not None
    # 第一列因为含空值 -> _classify_dtype 看 non_empty=[2] 返回 'int'，
    # cell "" 被强转为 None；非空 cell 被强转为 int
    assert result.answer.rows[0] == [None, "F", "SLE"]
    assert result.answer.rows[1] == [2, "M", "RA"]


# ----- #5 bool column -----


def test_from_csv_bool_column(task: PublicTask, tmp_path: Path) -> None:
    csv_path = _write(
        tmp_path / "_answer" / "flags.csv",
        "flag\nTrue\nFalse\nTrue\n",
    )
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    result = handle_answer(task, args)

    assert result.answer is not None
    assert result.answer.rows == [[True], [False], [True]]
    for row in result.answer.rows:
        assert isinstance(row[0], bool)


# ----- #6 file not found -----


def test_from_csv_missing_file_raises(task: PublicTask, tmp_path: Path) -> None:
    missing = tmp_path / "_answer" / "ghost.csv"
    args = AnswerInput.model_validate({"from_csv": str(missing)})

    with pytest.raises(ValueError, match=r"from_csv file not found"):
        handle_answer(task, args)


# ----- #7 relative path -----


def test_from_csv_relative_path_raises(task: PublicTask) -> None:
    args = AnswerInput.model_validate({"from_csv": "_answer/answer.csv"})

    with pytest.raises(ValueError, match=r"from_csv must be an absolute path"):
        handle_answer(task, args)


# ----- #8 zero-byte file -----


def test_from_csv_zero_byte_raises(task: PublicTask, tmp_path: Path) -> None:
    csv_path = _write(tmp_path / "_answer" / "empty.csv", "")
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    with pytest.raises(ValueError, match=r"from_csv is empty"):
        handle_answer(task, args)


# ----- #9 header only, no data -----


def test_from_csv_header_only_returns_empty_answer(task: PublicTask, tmp_path: Path) -> None:
    """仅 header 无数据行：合法 → AnswerTable(header, [])。

    "空 answer 是合法的"——dispatcher 没有 row count 下限，符合 inline 路径
    `answer({"columns": ["x"], "rows": []})` 同等语义。
    """
    csv_path = _write(tmp_path / "_answer" / "head.csv", "id,name\n")
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    result = handle_answer(task, args)

    assert result.ok is True
    assert result.is_terminal is True
    assert result.answer is not None
    assert result.answer.columns == ["id", "name"]
    assert result.answer.rows == []


# ----- #10 row width mismatch -----


def test_from_csv_row_width_mismatch_raises(task: PublicTask, tmp_path: Path) -> None:
    csv_path = _write(
        tmp_path / "_answer" / "ragged.csv",
        "id,name,extra\n1,alpha\n",
    )
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    with pytest.raises(ValueError, match=r"from_csv row 0 has 2 cells but header has 3"):
        handle_answer(task, args)


# ----- #11 size cap -----


def test_from_csv_exceeds_size_limit_raises(
    task: PublicTask, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """5MB 上限：不真造 5MB 文件，把阈值临时调小到 64 字节再触发。"""
    monkeypatch.setattr(
        "agents.tools.handlers._FROM_CSV_SIZE_LIMIT_BYTES",
        64,
    )
    csv_path = _write(
        tmp_path / "_answer" / "big.csv",
        "id\n" + "\n".join(str(n) for n in range(200)) + "\n",
    )
    assert csv_path.stat().st_size > 64  # sanity
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    with pytest.raises(ValueError, match=r"from_csv exceeds 5MB"):
        handle_answer(task, args)


# ----- #14 UTF-8 corruption -----


def test_from_csv_invalid_utf8_raises(task: PublicTask, tmp_path: Path) -> None:
    """UTF-8 失败：错误消息要可让模型自己决定要不要换 encoding。"""
    csv_path = tmp_path / "_answer" / "broken.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # \xff\xfe 是 UTF-16 BOM；以 UTF-8 解码必然失败
    csv_path.write_bytes(b"\xff\xfeID\n1\n")
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    with pytest.raises(ValueError, match=r"from_csv is not valid UTF-8"):
        handle_answer(task, args)


# ----- #16 ack content shape -----


def test_from_csv_ack_content_includes_dtypes_and_preview(task: PublicTask, tmp_path: Path) -> None:
    """ack `content.from_csv.{dtypes, head_preview}`：trace 离线对账依赖该结构。"""
    csv_path = _write(
        tmp_path / "_answer" / "preview.csv",
        "id,name,ratio\n" + "\n".join(f"{n},name_{n},{n / 4}" for n in range(1, 11)) + "\n",
    )
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    result = handle_answer(task, args)

    assert result.content["status"] == "submitted"
    assert result.content["column_count"] == 3
    assert result.content["row_count"] == 10

    from_csv_meta = result.content["from_csv"]
    assert from_csv_meta["path"] == str(csv_path)
    assert from_csv_meta["dtypes"] == ["int", "str", "float"]
    # head_preview 永远 ≤ 5 行
    assert len(from_csv_meta["head_preview"]) == 5
    assert from_csv_meta["head_preview"][0] == [1, "name_1", 0.25]


# ----- 补充：non-existent absolute path with explicit hint -----


def test_from_csv_missing_file_hint_messages_are_actionable(
    task: PublicTask, tmp_path: Path
) -> None:
    """error message 包含具体路径：模型才能 retry execute_python 重写文件。"""
    args = AnswerInput.model_validate({"from_csv": str(tmp_path / "nope.csv")})

    with pytest.raises(ValueError) as exc_info:
        handle_answer(task, args)

    assert str(tmp_path / "nope.csv") in str(exc_info.value)


# ----- 补充：成功提交后 artifact 自动清理（避免污染输入树） -----


def test_from_csv_unlinks_artifact_after_successful_submission(
    task: PublicTask, tmp_path: Path
) -> None:
    """模型提交成功后，from_csv 指向的 CSV 与整个 `_answer/` 目录都必须被删除。

    输入树污染回归：曾出现 `data/public/input/task_*/_answer/answer.csv` 残留，
    造成离线分析时把它误读成"项目自带参考答案"；草稿/日志同理。失败路径不在本
    测试覆盖范围，那里保留文件让 ReAct 循环 retry 时 to_csv 直接覆盖更稳。
    """
    answer_dir = task.task_dir / "_answer"
    csv_path = _write(answer_dir / "answer.csv", "id,name\n1,alpha\n2,beta\n")
    # 草稿文件：模型可能把中间表/日志也丢在 DABENCH_ANSWER_DIR 下
    draft_path = _write(answer_dir / "draft.csv", "tmp\n1\n")
    args = AnswerInput.model_validate({"from_csv": str(csv_path)})

    result = handle_answer(task, args)

    assert result.ok is True
    assert not csv_path.exists()
    assert not draft_path.exists()
    assert not answer_dir.exists()


def test_from_csv_external_path_still_clears_answer_dir(task: PublicTask, tmp_path: Path) -> None:
    """from_csv 指向 `_answer/` 之外（如 `/tmp/...`）也要清掉 `_answer/` 目录。

    场景：模型把 artifact 写到非约定路径（绕开 DABENCH_ANSWER_DIR），但更早的
    execute_python 已经在 `_answer/` 留下文件；rmtree 兜底带走，避免下次离线
    分析读到错位的 stale answer。
    """
    answer_dir = task.task_dir / "_answer"
    stale = _write(answer_dir / "answer.csv", "id\n999\n")
    external = _write(tmp_path / "elsewhere" / "real.csv", "id\n1\n2\n")
    args = AnswerInput.model_validate({"from_csv": str(external)})

    result = handle_answer(task, args)

    assert result.ok is True
    assert not external.exists()
    assert not stale.exists()
    assert not answer_dir.exists()


def test_inline_submission_with_existing_artifact_raises(task: PublicTask, tmp_path: Path) -> None:
    """残留 `_answer/answer.csv` + inline 提交：handler 拒绝并指回 from_csv。

    回归参照 run 20260501-010/task_8：execute_python 已经把 140 行写到
    `_answer/answer.csv`，但 `answer` tool_call 的 JSON 参数里只 inline 了 head(5)
    的 5 行（attention drift 把更早一轮 head() 预览当成了完整表）。inline
    硬闸（10 行 / 50 cells）抓不住此类小体量截断；提交时间点检测到 artifact
    残留就直接拒绝，让 ReAct 循环把错误回灌给模型自行改用 from_csv。

    artifact 文件必须保留在原位——下一轮模型才能用同一路径走 from_csv 重交。
    """
    artifact = task.task_dir / "_answer" / "answer.csv"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("id\n999\n", encoding="utf-8")

    args = AnswerInput.model_validate({"columns": ["id"], "rows": [[1], [2]]})

    with pytest.raises(ValueError, match=r"answer artifact already exists") as exc_info:
        handle_answer(task, args)

    # 错误消息要把绝对路径与 from_csv 修复方向都贴上，模型下一轮才能直接照搬
    msg = str(exc_info.value)
    assert str(artifact) in msg
    assert "from_csv" in msg
    # 关键：保留 artifact 与 `_answer/` 目录，模型 retry 时还能指它
    assert artifact.exists()
    assert artifact.parent.exists()


def test_inline_submission_with_zero_byte_artifact_passes(task: PublicTask, tmp_path: Path) -> None:
    """zero-byte 残留不视作 artifact：inline 提交照常成功并清理零字节文件。

    `_FROM_CSV_SIZE_LIMIT_BYTES` 上限不是这里的拦截理由；空文件更像 mkdir/touch
    误产物，不应触发 from_csv 改写要求。
    """
    artifact = task.task_dir / "_answer" / "answer.csv"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"")
    assert artifact.exists()

    args = AnswerInput.model_validate({"columns": ["id"], "rows": [[1], [2]]})
    result = handle_answer(task, args)

    assert result.ok is True
    assert result.answer is not None
    assert result.answer.rows == [[1], [2]]
    assert not artifact.exists()
    assert not artifact.parent.exists()
