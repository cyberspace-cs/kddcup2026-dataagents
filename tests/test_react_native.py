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
    TokenUsage,
)
from data_agent_baseline.agents.react import (
    ReActAgent,
    ReActAgentConfig,
    _extract_reasoning_tool_call_draft,
)
from data_agent_baseline.agents.runtime import AgentRuntimeState, StepRecord
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.tools.registry import create_default_tool_registry
from tests.helpers.scripted_adapters import ScriptedNativeModelAdapter


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


def _default_tool_names() -> set[str]:
    return set(create_default_tool_registry().definitions)


def test_reasoning_tool_call_draft_parser_handles_multiline_python() -> None:
    reasoning = """
I should run Python.
<tool_call>
<function=execute_python>
<parameter=code>
import os
print(sorted(os.listdir(".")))
</parameter>
</function>
</tool_call>
"""

    draft = _extract_reasoning_tool_call_draft(reasoning, _default_tool_names())

    assert draft == {
        "function": "execute_python",
        "arguments": {"code": 'import os\nprint(sorted(os.listdir(".")))'},
        "source": "reasoning_content",
        "executed": False,
    }


def test_reasoning_tool_call_draft_parser_handles_sql_parameters() -> None:
    reasoning = """
<tool_call>
<function=execute_context_sql>
<parameter=path>"db/main.sqlite"</parameter>
<parameter=sql>
SELECT city, COUNT(*) AS n
FROM people
GROUP BY city
</parameter>
<parameter=limit>5</parameter>
</function>
</tool_call>
"""

    draft = _extract_reasoning_tool_call_draft(reasoning, _default_tool_names())

    assert draft == {
        "function": "execute_context_sql",
        "arguments": {
            "path": "db/main.sqlite",
            "sql": "SELECT city, COUNT(*) AS n\nFROM people\nGROUP BY city",
            "limit": 5,
        },
        "source": "reasoning_content",
        "executed": False,
    }


def test_reasoning_tool_call_draft_parser_handles_answer_json_arguments() -> None:
    reasoning = """
<tool_call>
<function=answer>
<parameter=columns>["city", "count"]</parameter>
<parameter=rows>[["Paris", 2], ["Tokyo", 3]]</parameter>
</function>
</tool_call>
"""

    draft = _extract_reasoning_tool_call_draft(reasoning, _default_tool_names())

    assert draft == {
        "function": "answer",
        "arguments": {
            "columns": ["city", "count"],
            "rows": [["Paris", 2], ["Tokyo", 3]],
        },
        "source": "reasoning_content",
        "executed": False,
    }


@pytest.mark.parametrize(
    "reasoning",
    [
        "<tool_call><function=execute_python><parameter=code>x</parameter></function>",
        "<tool_call><function=unknown_tool></function></tool_call>",
        "<tool_call><parameter=code>x</parameter></tool_call>",
        "no tool-call draft here",
        (
            "<tool_call><function=unknown_tool></function></tool_call>"
            "<tool_call><function=inspect_files></function></tool_call>"
        ),
    ],
)
def test_reasoning_tool_call_draft_parser_rejects_invalid_drafts(reasoning: str) -> None:
    assert _extract_reasoning_tool_call_draft(reasoning, _default_tool_names()) is None


def test_native_multi_step_until_answer(task: PublicTask) -> None:
    # 先 inspect_files，再 answer 结束
    list_call, list_raw = _build_tool_call(call_id="call_1", name="inspect_files", arguments={})
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
    assert first.action == "inspect_files"
    assert first.tool_call_id == "call_1"
    assert first.turn_index == 1
    assert first.step_index == 1
    assert second.action == "answer"
    assert second.tool_call_id == "call_2"
    assert second.turn_index == 2
    assert second.step_index == 2


def test_native_parallel_tool_calls_share_turn_index(task: PublicTask) -> None:
    # 同一轮返回 2 个非终止 tool_calls + 下一轮 answer
    list_call, list_raw = _build_tool_call(call_id="c_a", name="inspect_files", arguments={})
    # 造第二个 inspect_files（parallel 分支：测试循环能正确聚合）
    list_call_2, list_raw_2 = _build_tool_call(call_id="c_b", name="inspect_files", arguments={})
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
        call_id="should_not_run", name="inspect_files", arguments={}
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
    # 只有 answer 被记录；后续 inspect_files 被短路
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

    # 模拟一个已完成步骤的 state：assistant 曾并行调用 2 个 inspect_files
    raw_tool_calls = [
        {
            "id": "tc1",
            "type": "function",
            "function": {"name": "inspect_files", "arguments": "{}"},
        },
        {
            "id": "tc2",
            "type": "function",
            "function": {"name": "inspect_files", "arguments": "{}"},
        },
    ]
    state = AgentRuntimeState()
    state.steps.append(
        StepRecord(
            step_index=1,
            turn_index=1,
            thought="exploring",
            action="inspect_files",
            action_input={},
            raw_response="exploring",
            observation={"ok": True, "tool": "inspect_files", "content": {}},
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
            action="inspect_files",
            action_input={},
            raw_response="exploring",
            observation={"ok": True, "tool": "inspect_files", "content": {}},
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
    error = error_step.observation["error"].lower()
    assert "did not produce an executable tool call" in error
    assert "reasoning/thinking text was ignored" in error
    assert "syntactically complete" in error
    assert "already decided which tool to call last turn" in error
    assert "structured tool call" in error
    assert "call answer directly" in error
    assert "must call a tool" in error
    assert "drafted_tool_call" not in error_step.observation
    assert answer_step.action == "answer"


def test_native_reasoning_tool_call_draft_is_observed_but_not_executed(
    task: PublicTask,
) -> None:
    structured_call, structured_raw = _build_tool_call(
        call_id="real_run",
        name="execute_python",
        arguments={
            "code": (
                "from pathlib import Path\n"
                "Path('structured_executed.txt').write_text('yes')\n"
                "print('structured ran')"
            )
        },
    )
    answer_call, answer_raw = _build_tool_call(
        call_id="final", name="answer", arguments={"columns": ["c"], "rows": [["v"]]}
    )
    reasoning_draft = """
I know the Python I want to run.
<tool_call>
<function=execute_python>
<parameter=code>
from pathlib import Path
Path('draft_executed.txt').write_text('bad')
</parameter>
</function>
</tool_call>
"""
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="",
                tool_calls=[],
                raw_response="",
                raw_tool_calls=[],
                reasoning_content=reasoning_draft,
            ),
            ModelResponse(
                content="",
                tool_calls=[structured_call],
                raw_response="",
                raw_tool_calls=[structured_raw],
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
    assert [step.action for step in result.steps] == ["__error__", "execute_python", "answer"]
    error_step = result.steps[0]
    assert error_step.reasoning == reasoning_draft
    assert error_step.observation["error"] == (
        "Your previous turn did not produce an executable tool call."
    )
    assert error_step.observation["drafted_tool_call"] == {
        "function": "execute_python",
        "arguments": {
            "code": ("from pathlib import Path\nPath('draft_executed.txt').write_text('bad')")
        },
        "source": "reasoning_content",
        "executed": False,
    }
    instruction = error_step.observation["instruction"].lower()
    assert "reasoning drafts are not executable" in instruction
    assert "real structured tool_call" in instruction
    assert not (task.context_dir / "draft_executed.txt").exists()
    assert (task.context_dir / "structured_executed.txt").read_text() == "yes"


def test_native_reasoning_tool_call_drafts_use_empty_retry_budget(
    task: PublicTask,
) -> None:
    reasoning_draft = """
<tool_call>
<function=inspect_files>
</function>
</tool_call>
"""
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="",
                tool_calls=[],
                raw_response="",
                raw_tool_calls=[],
                reasoning_content=reasoning_draft,
            ),
            ModelResponse(
                content="",
                tool_calls=[],
                raw_response="",
                raw_tool_calls=[],
                reasoning_content=reasoning_draft,
            ),
        ]
    )
    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
        # N=2：第一次空轮记录但不终止；第二次（>=N）触发 max_empty_tool_call_retries 失败
        config=ReActAgentConfig(max_steps=4, max_empty_tool_call_retries=2),
        protocol="native",
    )

    result = agent.run(task)

    assert result.succeeded is False
    assert len(result.steps) == 2
    assert all("drafted_tool_call" in step.observation for step in result.steps)
    assert result.failure_reason is not None
    assert "max_empty_tool_call_retries" in result.failure_reason


def test_native_empty_tool_calls_do_not_consume_max_steps(task: PublicTask) -> None:
    answer_call, answer_raw = _build_tool_call(
        call_id="final", name="answer", arguments={"columns": ["c"], "rows": [["v"]]}
    )
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="",
                tool_calls=[],
                raw_response="",
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
        # max_steps=1 验证 "空 tool_calls 不消耗 max_steps"——第二轮的 answer 仍然能跑；
        # max_empty=2 给第一次空轮留 budget（>= 触发，所以 N=1 会立刻终止）
        config=ReActAgentConfig(max_steps=1, max_empty_tool_call_retries=2),
        protocol="native",
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert len(result.steps) == 2
    error_step, answer_step = result.steps
    assert error_step.action == "__error__"
    assert answer_step.action == "answer"
    assert answer_step.turn_index == 2


def test_native_empty_tool_calls_have_independent_retry_limit(task: PublicTask) -> None:
    # max_empty_tool_call_retries=N 表示 "第 N 次空轮就是失败轮"——guard 用 >=，
    # 所以 N=1 时第一次空 tool_calls 就立即终止；max_steps=4 仍远大于实际步数，
    # 用以验证 empty-tool 的预算独立于 max_steps（失败原因里不应出现 max_steps）。
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(content="", tool_calls=[], raw_response="", raw_tool_calls=[]),
        ]
    )
    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=4, max_empty_tool_call_retries=1),
        protocol="native",
    )

    result = agent.run(task)

    assert result.succeeded is False
    assert len(result.steps) == 1
    assert result.steps[0].action == "__error__"
    assert result.failure_reason is not None
    assert "max_empty_tool_call_retries" in result.failure_reason
    assert "max_steps" not in result.failure_reason


def test_native_total_usage_aggregates_per_turn_usage(task: PublicTask) -> None:
    # 两轮调用，每轮塞入不同 usage；中间一轮返回并行 tool_calls 但只有 lead 承载 usage
    list_call_a, list_raw_a = _build_tool_call(call_id="p_a", name="inspect_files", arguments={})
    list_call_b, list_raw_b = _build_tool_call(call_id="p_b", name="inspect_files", arguments={})
    answer_call, answer_raw = _build_tool_call(
        call_id="done",
        name="answer",
        arguments={"columns": ["x"], "rows": [["y"]]},
    )
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="parallel",
                tool_calls=[list_call_a, list_call_b],
                raw_response="parallel",
                raw_tool_calls=[list_raw_a, list_raw_b],
                reasoning_content="parallel reasoning",
                usage=TokenUsage(
                    prompt_tokens=1000,
                    completion_tokens=200,
                    total_tokens=1200,
                    cached_tokens=400,
                    reasoning_tokens=80,
                ),
            ),
            ModelResponse(
                content="answer",
                tool_calls=[answer_call],
                raw_response="answer",
                raw_tool_calls=[answer_raw],
                reasoning_content="final reasoning",
                usage=TokenUsage(
                    prompt_tokens=1500,
                    completion_tokens=50,
                    total_tokens=1550,
                    cached_tokens=900,
                    reasoning_tokens=10,
                ),
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

    # 3 条 steps：2 个并行 list + 1 个 answer
    assert len(result.steps) == 3
    step_a, step_b, step_answer = result.steps

    # lead（step_a）承载第一轮的 usage 与 reasoning；step_b 是并行的第二条，留默认
    assert step_a.usage.prompt_tokens == 1000
    assert step_a.reasoning == "parallel reasoning"
    assert step_b.usage == TokenUsage()  # 全 0，避免双计
    assert step_b.reasoning == ""

    # answer 自成一轮，usage 单独计
    assert step_answer.usage.prompt_tokens == 1500
    assert step_answer.reasoning == "final reasoning"

    # 顶层 total_usage：第一轮 + answer 轮之和
    payload = result.to_dict()
    assert payload["total_usage"] == {
        "prompt_tokens": 2500,
        "completion_tokens": 250,
        "total_tokens": 2750,
        "cached_tokens": 1300,
        "reasoning_tokens": 90,
    }


def test_budget_warning_injected_at_75_percent(task: PublicTask) -> None:
    """Budget warning injected once when effective_turns reaches 75% of max_steps."""
    # max_steps=4 → 75% threshold at effective_turns=3
    # Need 3 non-answer turns + 1 answer turn = 4 responses total
    calls_and_raws = [
        _build_tool_call(call_id=f"c_{i}", name="inspect_files", arguments={}) for i in range(3)
    ]
    answer_call, answer_raw = _build_tool_call(
        call_id="final", name="answer", arguments={"columns": ["x"], "rows": [["y"]]}
    )
    responses = [
        ModelResponse(
            content=f"step {i}",
            tool_calls=[calls_and_raws[i][0]],
            raw_response=f"step {i}",
            raw_tool_calls=[calls_and_raws[i][1]],
        )
        for i in range(3)
    ]
    responses.append(
        ModelResponse(
            content="done",
            tool_calls=[answer_call],
            raw_response="done",
            raw_tool_calls=[answer_raw],
        )
    )

    captured_messages: list[list] = []

    class CapturingAdapter(ScriptedNativeModelAdapter):
        def complete(self, messages):
            captured_messages.append(messages)
            return super().complete(messages)

    adapter = CapturingAdapter(responses)
    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=4),
        protocol="native",
    )

    result = agent.run(task)
    assert result.succeeded is True

    # The 4th call (effective_turns=3 at that point, 3/4=0.75) should have budget warning
    fourth_call_messages = captured_messages[3]
    budget_msgs = [m for m in fourth_call_messages if "BUDGET WARNING" in (m.content or "")]
    assert len(budget_msgs) == 1
    assert "call answer NOW" in budget_msgs[0].content


def test_preact_planning_message_on_first_turn(task: PublicTask) -> None:
    """enable_preact=True injects planning instruction on first turn only."""
    answer_call, answer_raw = _build_tool_call(
        call_id="a", name="answer", arguments={"columns": ["x"], "rows": [["y"]]}
    )

    captured_messages: list[list] = []

    class CapturingAdapter(ScriptedNativeModelAdapter):
        def complete(self, messages):
            captured_messages.append(messages)
            return super().complete(messages)

    adapter = CapturingAdapter(
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
        config=ReActAgentConfig(max_steps=4, enable_preact=True),
        protocol="native",
    )

    result = agent.run(task)
    assert result.succeeded is True

    # First (and only) call should have the preact planning message
    first_messages = captured_messages[0]
    planning_msgs = [
        m for m in first_messages if "Before acting, outline a brief plan" in (m.content or "")
    ]
    assert len(planning_msgs) == 1
    assert "inspect_files" in planning_msgs[0].content


def test_preact_disabled_no_planning_message(task: PublicTask) -> None:
    """enable_preact=False (default) does not inject planning instruction."""
    answer_call, answer_raw = _build_tool_call(
        call_id="a", name="answer", arguments={"columns": ["x"], "rows": [["y"]]}
    )

    captured_messages: list[list] = []

    class CapturingAdapter(ScriptedNativeModelAdapter):
        def complete(self, messages):
            captured_messages.append(messages)
            return super().complete(messages)

    adapter = CapturingAdapter(
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
        config=ReActAgentConfig(max_steps=4, enable_preact=False),
        protocol="native",
    )

    result = agent.run(task)
    assert result.succeeded is True

    first_messages = captured_messages[0]
    planning_msgs = [
        m for m in first_messages if "Before acting, outline a brief plan" in (m.content or "")
    ]
    assert len(planning_msgs) == 0
