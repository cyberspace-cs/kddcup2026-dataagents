"""文本协议降级路径回归测试。

显式指定 `protocol="text"`，使用 `ScriptedModelAdapter` 注入 ```json fence 响应，
验证：
- ReAct 循环能正常结束（走 parse_model_step 路径）
- StepRecord.turn_index 与 step_index 相等（文本协议每轮恰好一个工具调用）
- StepRecord.tool_call_id 为 None
- StepRecord.raw_tool_calls 为 None
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_agent_baseline.agents.model import ScriptedModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.registry import create_default_tool_registry


@pytest.fixture
def task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    # 放一个简单文件便于 list_context 有内容可看
    (context_dir / "hello.txt").write_text("hello\n")
    return PublicTask(
        record=TaskRecord(
            task_id="task_1",
            difficulty="easy",
            question="What files are in context?",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _fenced(payload: dict) -> str:
    return "```json\n" + json.dumps(payload) + "\n```"


def test_text_protocol_runs_single_action_per_turn(task: PublicTask) -> None:
    adapter = ScriptedModelAdapter(
        [
            _fenced(
                {
                    "thought": "list the context",
                    "action": "list_context",
                    "action_input": {"max_depth": 2},
                }
            ),
            _fenced(
                {
                    "thought": "answer with the result",
                    "action": "answer",
                    "action_input": {"columns": ["file"], "rows": [["hello.txt"]]},
                }
            ),
        ]
    )
    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=4),
        protocol="text",
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.answer is not None
    assert result.answer.columns == ["file"]
    assert len(result.steps) == 2

    for step in result.steps:
        # text 协议：每个 StepRecord 单独占一轮，step_index == turn_index
        assert step.step_index == step.turn_index
        # text 协议不携带原生 tool_call 信息
        assert step.tool_call_id is None
        assert step.raw_tool_calls is None


def test_text_protocol_records_parse_errors(task: PublicTask) -> None:
    # 故意给一条解析不了的响应，然后再给合法的 answer
    adapter = ScriptedModelAdapter(
        [
            "not a fenced json at all",
            _fenced(
                {
                    "thought": "retry",
                    "action": "answer",
                    "action_input": {"columns": ["c"], "rows": [["v"]]},
                }
            ),
        ]
    )
    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=4),
        protocol="text",
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert len(result.steps) == 2
    error_step, answer_step = result.steps
    assert error_step.action == "__error__"
    assert error_step.ok is False
    assert answer_step.action == "answer"
    # 回归：文本降级路径依旧能让模型自我修正
