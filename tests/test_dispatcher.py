"""工具分发层 `dispatcher.dispatch_tool_call` 测试。

验证 D3 关键不变式：终止权威来自注册表元数据 + handler 成功；handler 抛异常一律
原样上抛，由调用方记 `tool_error` 事件；非终止工具永远 `should_terminate=False`。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_agent_baseline.agents.dispatcher import dispatch_tool_call
from data_agent_baseline.agents.model import ModelToolCall
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.registry import create_default_tool_registry


@pytest.fixture
def task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (context_dir / "hello.txt").write_text("hello\n")
    return PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _call(name: str, **arguments: object) -> ModelToolCall:
    return ModelToolCall(id=f"call_{name}", name=name, arguments=dict(arguments))


def test_dispatch_non_terminal_success_does_not_terminate(task: PublicTask) -> None:
    registry = create_default_tool_registry()
    call = _call("list_context", max_depth=2)

    result, should_terminate = dispatch_tool_call(task, call, registry)

    assert result.ok is True
    assert should_terminate is False


def test_dispatch_terminal_success_terminates(task: PublicTask) -> None:
    registry = create_default_tool_registry()
    call = _call("answer", columns=["greeting"], rows=[["hi"]])

    result, should_terminate = dispatch_tool_call(task, call, registry)

    assert result.ok is True
    assert should_terminate is True
    assert result.answer is not None
    assert result.answer.columns == ["greeting"]


def test_dispatch_invalid_answer_raises_and_does_not_terminate(task: PublicTask) -> None:
    """answer payload 校验失败 → handler 抛 ValueError，dispatch 原样上抛（不终止）。

    这个用例锁住 D3：terminal flag 是元数据，但只在 handler 成功返回后才生效。
    """
    registry = create_default_tool_registry()
    # rows 长度与 columns 不匹配
    call = _call("answer", columns=["a", "b"], rows=[["only one"]])

    with pytest.raises(ValueError, match="match the number of columns"):
        dispatch_tool_call(task, call, registry)


def test_dispatch_unknown_tool_raises_key_error(task: PublicTask) -> None:
    registry = create_default_tool_registry()
    call = _call("not_a_real_tool")

    with pytest.raises(KeyError, match="not_a_real_tool"):
        dispatch_tool_call(task, call, registry)


def test_dispatch_path_escape_raises_value_error(task: PublicTask) -> None:
    """sandboxing：`resolve_context_path` 阻止 `../` 越界访问。"""
    registry = create_default_tool_registry()
    call = _call("inspect_sqlite_schema", path="../../etc/passwd")

    with pytest.raises(ValueError, match="escapes context dir"):
        dispatch_tool_call(task, call, registry)


def test_dispatch_missing_file_raises_file_not_found(task: PublicTask) -> None:
    """合法路径但文件不存在 → FileNotFoundError 上抛。"""
    registry = create_default_tool_registry()
    call = _call("read_csv", path="missing.csv")

    with pytest.raises(FileNotFoundError, match="Missing context asset"):
        dispatch_tool_call(task, call, registry)


def test_dispatch_python_code_failure_returns_ok_false(task: PublicTask) -> None:
    """`execute_python` handler 不抛异常，但用户代码失败时 `result.ok=False` 且不终止。"""
    registry = create_default_tool_registry()
    call = _call("execute_python", code="raise RuntimeError('boom')")

    result, should_terminate = dispatch_tool_call(task, call, registry)

    assert result.ok is False
    assert should_terminate is False
