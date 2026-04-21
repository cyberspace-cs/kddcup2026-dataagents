"""原生 function calling ReAct 循环的测试。

覆盖点：
- ScriptedNativeModelAdapter + 默认工具 registry：多步 ReAct 直至 answer
- 单轮并行 tool_calls：同 turn_index、step_index 连续递增
- answer 短路：并行 tool_calls 中若首个就是 answer，后续 tool_calls 不执行
- native 模式消息构建：assistant 带 tool_calls、tool 消息带 tool_call_id
- 默认协议：`ReActAgent()` 不传 protocol 时走 native 路径
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data_agent_baseline.agents.model import (
    ModelResponse,
    ModelToolCall,
    ScriptedNativeModelAdapter,
)
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.agents.runtime import AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.registry import create_default_tool_registry


@pytest.fixture
def task(tmp_path: Path) -> PublicTask:
    task_dir = tmp_path / "task_1"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(
            task_id="task_1",
            difficulty="easy",
            question="What files exist?",
        ),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _build_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: dict,
) -> tuple[ModelToolCall, dict]:
    """返回（ModelToolCall, raw_tool_call dict）对，保持二者一致。"""
    raw = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": __import__("json").dumps(arguments)},
    }
    parsed = ModelToolCall(id=call_id, name=name, arguments=arguments)
    return parsed, raw


def test_native_multi_step_until_answer(task: PublicTask) -> None:
    # 先 list_context，再 answer 结束
    list_call, list_raw = _build_tool_call(
        call_id="call_1", name="list_context", arguments={"max_depth": 2}
    )
    answer_call, answer_raw = _build_tool_call(
        call_id="call_2",
        name="answer",
        arguments={"columns": ["result"], "rows": [["done"]]},
    )
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="thinking about files",
                tool_calls=[list_call],
                raw_response="thinking about files",
                raw_tool_calls=[list_raw],
            ),
            ModelResponse(
                content="submitting",
                tool_calls=[answer_call],
                raw_response="submitting",
                raw_tool_calls=[answer_raw],
            ),
        ]
    )
    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=4),
        protocol="native",
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.answer is not None
    assert result.answer.columns == ["result"]
    assert len(result.steps) == 2
    first, second = result.steps
    assert first.action == "list_context"
    assert first.tool_call_id == "call_1"
    assert first.turn_index == 1
    assert first.step_index == 1
    assert second.action == "answer"
    assert second.tool_call_id == "call_2"
    assert second.turn_index == 2
    assert second.step_index == 2


def test_native_parallel_tool_calls_share_turn_index(task: PublicTask) -> None:
    # 同一轮返回 2 个非终止 tool_calls + 下一轮 answer
    list_call, list_raw = _build_tool_call(
        call_id="c_a", name="list_context", arguments={"max_depth": 1}
    )
    # 造第二个 list_context（parallel 分支：测试循环能正确聚合）
    list_call_2, list_raw_2 = _build_tool_call(
        call_id="c_b", name="list_context", arguments={"max_depth": 2}
    )
    answer_call, answer_raw = _build_tool_call(
        call_id="c_done",
        name="answer",
        arguments={"columns": ["ok"], "rows": [["yes"]]},
    )
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="parallel explore",
                tool_calls=[list_call, list_call_2],
                raw_response="parallel explore",
                raw_tool_calls=[list_raw, list_raw_2],
            ),
            ModelResponse(
                content="done",
                tool_calls=[answer_call],
                raw_response="done",
                raw_tool_calls=[answer_raw],
            ),
        ]
    )
    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=4),
        protocol="native",
    )

    result = agent.run(task)

    assert result.succeeded is True
    # 3 条 StepRecord：2 个并行 list + 1 个 answer
    assert len(result.steps) == 3
    step_a, step_b, step_answer = result.steps

    # 并行两条共享 turn_index，step_index 连续
    assert step_a.turn_index == step_b.turn_index == 1
    assert step_a.step_index == 1
    assert step_b.step_index == 2
    # raw_tool_calls 只在首条上填写，节省空间 + 便于 message 回放
    assert step_a.raw_tool_calls == [list_raw, list_raw_2]
    assert step_b.raw_tool_calls is None

    # answer 是下一个 turn
    assert step_answer.turn_index == 2
    assert step_answer.step_index == 3


def test_native_answer_short_circuits_same_turn(task: PublicTask) -> None:
    # 并行：第一个就是 answer，第二个不应执行
    answer_call, answer_raw = _build_tool_call(
        call_id="first_answer",
        name="answer",
        arguments={"columns": ["c"], "rows": [["v"]]},
    )
    list_call, list_raw = _build_tool_call(
        call_id="should_not_run", name="list_context", arguments={"max_depth": 1}
    )
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="submitting early",
                tool_calls=[answer_call, list_call],
                raw_response="submitting early",
                raw_tool_calls=[answer_raw, list_raw],
            )
        ]
    )
    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=4),
        protocol="native",
    )

    result = agent.run(task)

    assert result.succeeded is True
    # 只有 answer 被记录；后续 list_context 被短路
    assert len(result.steps) == 1
    assert result.steps[0].action == "answer"
    assert result.steps[0].tool_call_id == "first_answer"


def test_native_build_messages_uses_tool_role(task: PublicTask) -> None:
    # 通过手动构造 state 检查消息构建：不跑完整循环
    registry = create_default_tool_registry()
    adapter = ScriptedNativeModelAdapter([])  # 不会被调用到
    agent = ReActAgent(
        model=adapter,
        tools=registry,
        config=ReActAgentConfig(max_steps=2),
        protocol="native",
    )

    # 模拟一个已完成步骤的 state：assistant 曾并行调用 2 个 list_context
    raw_tool_calls = [
        {
            "id": "tc1",
            "type": "function",
            "function": {"name": "list_context", "arguments": "{}"},
        },
        {
            "id": "tc2",
            "type": "function",
            "function": {"name": "list_context", "arguments": "{}"},
        },
    ]
    state = AgentRuntimeState()
    state.steps.append(
        StepRecord(
            step_index=1,
            turn_index=1,
            thought="exploring",
            action="list_context",
            action_input={},
            raw_response="exploring",
            observation={"ok": True, "tool": "list_context", "content": {}},
            ok=True,
            tool_call_id="tc1",
            raw_tool_calls=raw_tool_calls,
        )
    )
    state.steps.append(
        StepRecord(
            step_index=2,
            turn_index=1,
            thought="exploring",
            action="list_context",
            action_input={},
            raw_response="exploring",
            observation={"ok": True, "tool": "list_context", "content": {}},
            ok=True,
            tool_call_id="tc2",
            raw_tool_calls=None,
        )
    )

    messages = agent._build_messages(task, state)

    # 期望顺序：system, user(task), assistant(tool_calls=...), tool(tc1), tool(tc2)
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert messages[2].role == "assistant"
    assert messages[2].tool_calls == raw_tool_calls
    assert messages[3].role == "tool"
    assert messages[3].tool_call_id == "tc1"
    assert messages[4].role == "tool"
    assert messages[4].tool_call_id == "tc2"


def test_native_is_default_protocol(task: PublicTask) -> None:
    # 不显式传 protocol → 默认 native
    answer_call, answer_raw = _build_tool_call(
        call_id="a", name="answer", arguments={"columns": ["x"], "rows": [["y"]]}
    )
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="",
                tool_calls=[answer_call],
                raw_response="",
                raw_tool_calls=[answer_raw],
            )
        ]
    )
    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
    )

    assert agent.protocol == "native"
    result = agent.run(task)
    assert result.succeeded is True
    assert result.steps[0].tool_call_id == "a"


def test_native_empty_tool_calls_records_error(task: PublicTask) -> None:
    # 模拟 native 模式下模型只回纯文本（无 tool_calls）：应记一条 __error__ 并继续
    answer_call, answer_raw = _build_tool_call(
        call_id="final", name="answer", arguments={"columns": ["c"], "rows": [["v"]]}
    )
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="I am confused",
                tool_calls=[],
                raw_response="I am confused",
                raw_tool_calls=[],
            ),
            ModelResponse(
                content="",
                tool_calls=[answer_call],
                raw_response="",
                raw_tool_calls=[answer_raw],
            ),
        ]
    )
    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=4),
        protocol="native",
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert len(result.steps) == 2
    error_step, answer_step = result.steps
    assert error_step.action == "__error__"
    assert error_step.ok is False
    assert "no tool_calls" in error_step.observation["error"].lower()
    assert answer_step.action == "answer"
