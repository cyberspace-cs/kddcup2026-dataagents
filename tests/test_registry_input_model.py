"""ToolDefinition.input_model：registry 局部校验（不经全局 _INPUT_MODELS）。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.tools.handlers import ToolExecutionResult
from agents.tools.registry import ToolDefinition, ToolRegistry


class _EchoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


def _make_task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_1", difficulty="easy", question="q"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _echo_tool(*, input_model: type[BaseModel] | None) -> ToolDefinition:
    def handler(_task: PublicTask, args: object) -> ToolExecutionResult:
        return ToolExecutionResult(ok=True, content={"args_type": type(args).__name__})

    return ToolDefinition(
        name="echo",
        description="echo",
        json_schema=None,
        handler=handler,
        input_model=input_model,
    )


def test_input_model_validates_and_passes_model_instance(tmp_path: Path) -> None:
    registry = ToolRegistry(definitions={"echo": _echo_tool(input_model=_EchoInput)})
    result = registry.execute(_make_task(tmp_path), "echo", {"text": "hi"})
    assert result.ok is True
    assert result.content["args_type"] == "_EchoInput"


def test_input_model_invalid_args_raise_single_line_value_error(tmp_path: Path) -> None:
    registry = ToolRegistry(definitions={"echo": _echo_tool(input_model=_EchoInput)})
    with pytest.raises(ValueError, match="echo: field 'text' is required"):
        registry.execute(_make_task(tmp_path), "echo", {})


def test_no_input_model_still_passes_raw_dict(tmp_path: Path) -> None:
    registry = ToolRegistry(definitions={"echo": _echo_tool(input_model=None)})
    result = registry.execute(_make_task(tmp_path), "echo", {"anything": 1})
    assert result.content["args_type"] == "dict"
