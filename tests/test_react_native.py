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

from agents.agent import ReActAgent, ReActAgentConfig
from agents.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from agents.llm.openai import ToolCallParseError
from agents.llm.protocol import extract_reasoning_tool_call_draft
from agents.llm.types import (
    ModelMessage,
    ModelResponse,
    ModelToolCall,
    TokenUsage,
)
from agents.prompts import build_task_prompt
from agents.runtime.state import AgentRuntimeState, StepRecord
from agents.tools import handlers as _handlers
from agents.tools.registry import create_default_tool_registry
from agents.verification.answer import AnswerVerifier
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


class _ScriptedVerifierAdapter:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.messages: list[list[ModelMessage]] = []

    def complete(self, messages: list[ModelMessage]) -> ModelResponse:
        self.messages.append(messages)
        if not self.responses:
            raise RuntimeError("No verifier responses remaining.")
        text = self.responses.pop(0)
        return ModelResponse(content=text, raw_response=text)


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

    draft = extract_reasoning_tool_call_draft(reasoning, _default_tool_names())

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

    draft = extract_reasoning_tool_call_draft(reasoning, _default_tool_names())

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

    draft = extract_reasoning_tool_call_draft(reasoning, _default_tool_names())

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
    assert extract_reasoning_tool_call_draft(reasoning, _default_tool_names()) is None


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


def test_native_answer_verifier_rejects_once_then_allows_resubmit(task: PublicTask) -> None:
    first_answer_call, first_answer_raw = _build_tool_call(
        call_id="first_answer",
        name="answer",
        arguments={"columns": ["result", "extra"], "rows": [["v", "leak"]]},
    )
    second_answer_call, second_answer_raw = _build_tool_call(
        call_id="second_answer",
        name="answer",
        arguments={"columns": ["result", "extra"], "rows": [["v", "leak"]]},
    )
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="submit first",
                tool_calls=[first_answer_call],
                raw_response="submit first",
                raw_tool_calls=[first_answer_raw],
            ),
            ModelResponse(
                content="submit again",
                tool_calls=[second_answer_call],
                raw_response="submit again",
                raw_tool_calls=[second_answer_raw],
            ),
        ]
    )
    verifier_adapter = _ScriptedVerifierAdapter(
        [('{"verdict": "reject", "check": "extra", "reason": "extra column should be removed"}')]
    )
    verifier = AnswerVerifier(model=verifier_adapter, max_rejections=1)
    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=4),
        protocol="native",
        answer_verifier=verifier,
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert len(verifier_adapter.messages) == 1
    verifier_user_message = verifier_adapter.messages[0][1].content
    assert task.question in verifier_user_message
    assert 'COLUMNS: ["result", "extra"]' in verifier_user_message
    assert "TOTAL ROWS: 1" in verifier_user_message
    assert len(result.steps) == 2
    rejected_step, submitted_step = result.steps
    assert rejected_step.action == "answer"
    assert "answer_rejected" in rejected_step.observation["content"]
    assert submitted_step.action == "answer"
    assert result.answer is not None
    assert result.answer.columns == ["result", "extra"]


def test_native_answer_verifier_rejection_stops_same_turn_tool_calls(
    task: PublicTask,
) -> None:
    inspect_call, inspect_raw = _build_tool_call(
        call_id="inspect_first",
        name="inspect_files",
        arguments={},
    )
    rejected_call, rejected_raw = _build_tool_call(
        call_id="rejected_answer",
        name="answer",
        arguments={"columns": ["result", "extra"], "rows": [["bad", "leak"]]},
    )
    same_turn_call, same_turn_raw = _build_tool_call(
        call_id="same_turn_answer",
        name="answer",
        arguments={"columns": ["result"], "rows": [["same_turn_should_not_submit"]]},
    )
    next_turn_call, next_turn_raw = _build_tool_call(
        call_id="next_turn_answer",
        name="answer",
        arguments={"columns": ["result"], "rows": [["fresh_turn_submit"]]},
    )
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="submit twice",
                tool_calls=[inspect_call, rejected_call, same_turn_call],
                raw_response="submit twice",
                raw_tool_calls=[inspect_raw, rejected_raw, same_turn_raw],
            ),
            ModelResponse(
                content="submit after feedback",
                tool_calls=[next_turn_call],
                raw_response="submit after feedback",
                raw_tool_calls=[next_turn_raw],
            ),
        ]
    )
    verifier_adapter = _ScriptedVerifierAdapter(
        [('{"verdict": "reject", "check": "extra", "reason": "extra column should be removed"}')]
    )
    verifier = AnswerVerifier(model=verifier_adapter, max_rejections=1)
    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=4),
        protocol="native",
        answer_verifier=verifier,
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.answer is not None
    assert result.answer.rows == [["fresh_turn_submit"]]
    assert [step.tool_call_id for step in result.steps] == [
        "inspect_first",
        "rejected_answer",
        "next_turn_answer",
    ]
    assert result.steps[0].raw_tool_calls == [inspect_raw, rejected_raw]


def test_native_answer_verifier_fallback_uses_latest_valid_rejection(
    task: PublicTask,
) -> None:
    first_call, first_raw = _build_tool_call(
        call_id="first_answer",
        name="answer",
        arguments={"columns": ["result"], "rows": [["first"]]},
    )
    second_call, second_raw = _build_tool_call(
        call_id="second_answer",
        name="answer",
        arguments={"columns": ["result"], "rows": [["second"]]},
    )
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="submit first",
                tool_calls=[first_call],
                raw_response="submit first",
                raw_tool_calls=[first_raw],
            ),
            ModelResponse(
                content="submit second",
                tool_calls=[second_call],
                raw_response="submit second",
                raw_tool_calls=[second_raw],
            ),
        ]
    )
    verifier_adapter = _ScriptedVerifierAdapter(
        [
            '{"verdict": "reject", "check": "row_count", "reason": "try again"}',
            '{"verdict": "reject", "check": "row_count", "reason": "still suspicious"}',
        ]
    )
    verifier = AnswerVerifier(model=verifier_adapter, max_rejections=2)
    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=4),
        protocol="native",
        answer_verifier=verifier,
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.answer is not None
    assert result.answer.rows == [["second"]]


def test_native_answer_verifier_fallback_ignores_invalid_rejection(
    task: PublicTask,
) -> None:
    bad_call, bad_raw = _build_tool_call(
        call_id="bad_answer",
        name="answer",
        arguments={"columns": ["result"], "rows": [["too", "wide"]]},
    )
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="submit malformed",
                tool_calls=[bad_call],
                raw_response="submit malformed",
                raw_tool_calls=[bad_raw],
            ),
        ]
    )
    verifier_adapter = _ScriptedVerifierAdapter(
        ['{"verdict": "reject", "check": "missing", "reason": "malformed"}']
    )
    verifier = AnswerVerifier(model=verifier_adapter, max_rejections=1)
    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=3),
        protocol="native",
        answer_verifier=verifier,
    )

    result = agent.run(task)

    assert result.succeeded is False
    assert result.answer is None
    assert result.failure_reason == "Agent did not submit an answer within max_steps."


def test_native_answer_verifier_fallback_from_csv_cleans_artifacts(
    task: PublicTask,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_handlers, "_ANSWER_SCRATCH_ROOT", tmp_path)
    answer_dir = tmp_path / task.task_id / "_answer"
    csv_path = answer_dir / "answer.csv"
    draft_path = answer_dir / "draft.csv"
    answer_dir.mkdir(parents=True)
    csv_path.write_text("result\nkept\n", encoding="utf-8")
    draft_path.write_text("tmp\nignored\n", encoding="utf-8")

    answer_call, answer_raw = _build_tool_call(
        call_id="csv_answer",
        name="answer",
        arguments={"from_csv": str(csv_path)},
    )
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="submit csv",
                tool_calls=[answer_call],
                raw_response="submit csv",
                raw_tool_calls=[answer_raw],
            ),
        ]
    )
    verifier_adapter = _ScriptedVerifierAdapter(
        ['{"verdict": "reject", "check": "row_count", "reason": "try again"}']
    )
    verifier = AnswerVerifier(model=verifier_adapter, max_rejections=1)
    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=2),
        protocol="native",
        answer_verifier=verifier,
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert result.answer is not None
    assert result.answer.rows == [["kept"]]
    assert not csv_path.exists()
    assert not draft_path.exists()
    assert not answer_dir.exists()


def test_native_answer_verifier_invalid_from_csv_type_falls_through_to_tool_error(
    task: PublicTask,
) -> None:
    bad_call, bad_raw = _build_tool_call(
        call_id="bad_answer",
        name="answer",
        arguments={"from_csv": []},
    )
    good_call, good_raw = _build_tool_call(
        call_id="good_answer",
        name="answer",
        arguments={"columns": ["result"], "rows": [["ok"]]},
    )
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="bad submit",
                tool_calls=[bad_call],
                raw_response="bad submit",
                raw_tool_calls=[bad_raw],
            ),
            ModelResponse(
                content="good submit",
                tool_calls=[good_call],
                raw_response="good submit",
                raw_tool_calls=[good_raw],
            ),
        ]
    )
    verifier_adapter = _ScriptedVerifierAdapter([])
    verifier = AnswerVerifier(model=verifier_adapter, max_rejections=1)
    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=4),
        protocol="native",
        answer_verifier=verifier,
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert len(verifier_adapter.messages) == 1
    assert 'COLUMNS: ["result"]' in verifier_adapter.messages[0][1].content
    assert result.steps[0].action == "__error__"
    assert "from_csv" in result.steps[0].observation["error"]
    assert result.answer is not None
    assert result.answer.rows == [["ok"]]


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

    messages = agent._build_messages(state, build_task_prompt(task))

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


def test_native_reasoning_tool_call_recovered_and_executed(
    task: PublicTask,
) -> None:
    """reasoning 中的 tool call 草稿被恢复为可执行调用并直接运行。"""
    answer_call, answer_raw = _build_tool_call(
        call_id="final", name="answer", arguments={"columns": ["c"], "rows": [["v"]]}
    )
    reasoning_draft = """
I know the Python I want to run.
<tool_call>
<function=execute_python>
<parameter=code>
from pathlib import Path
Path('recovered_executed.txt').write_text('yes')
print('recovered ran')
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
    assert [step.action for step in result.steps] == ["execute_python", "answer"]
    assert result.steps[0].tool_call_id == "recovered:1"
    assert (task.context_dir / "recovered_executed.txt").read_text() == "yes"


def test_native_content_tool_call_recovered_and_executed(
    task: PublicTask,
) -> None:
    """content 中的 tool call 标签被恢复为可执行调用并直接运行。"""
    answer_call, answer_raw = _build_tool_call(
        call_id="final", name="answer", arguments={"columns": ["c"], "rows": [["v"]]}
    )
    content_with_tool_call = (
        "<tool_call><function=execute_python>"
        "<parameter=code>print('from content')</parameter>"
        "</function></tool_call>"
    )
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content=content_with_tool_call,
                tool_calls=[],
                raw_response=content_with_tool_call,
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
    assert [step.action for step in result.steps] == ["execute_python", "answer"]
    assert result.steps[0].tool_call_id == "recovered:1"


def test_native_recovered_call_synthesizes_raw_tool_calls(
    task: PublicTask,
) -> None:
    """recovered tool call 合成 raw_tool_calls 以保证 native 消息回放正确。"""
    answer_call, answer_raw = _build_tool_call(
        call_id="final", name="answer", arguments={"columns": ["c"], "rows": [["v"]]}
    )
    reasoning = "<tool_call><function=inspect_files></function></tool_call>"
    adapter = ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="",
                tool_calls=[],
                raw_response="",
                raw_tool_calls=[],
                reasoning_content=reasoning,
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
    recovered_step = result.steps[0]
    assert recovered_step.action == "inspect_files"
    assert recovered_step.raw_tool_calls is not None
    assert len(recovered_step.raw_tool_calls) == 1
    assert recovered_step.raw_tool_calls[0]["id"] == "recovered:1"
    assert recovered_step.raw_tool_calls[0]["function"]["name"] == "inspect_files"


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


def test_native_tool_call_parse_error_preserves_partial_trace_and_recovers(
    task: PublicTask,
) -> None:
    """Native adapter JSON parse failures keep partial response metadata in trace."""
    bad_raw = {
        "id": "bad_answer",
        "type": "function",
        "function": {
            "name": "answer",
            "arguments": '{"columns": ["x"], "rows": [["unterminated"]',
        },
    }
    answer_call, answer_raw = _build_tool_call(
        call_id="good_answer",
        name="answer",
        arguments={"columns": ["x"], "rows": [["fixed"]]},
    )

    class ParseThenAnswerAdapter:
        def __init__(self) -> None:
            self.calls = 0
            self.messages: list[list[ModelMessage]] = []

        def complete(self, messages: list[ModelMessage]) -> ModelResponse:
            self.messages.append(messages)
            self.calls += 1
            if self.calls == 1:
                raise ToolCallParseError(
                    "Failed to parse tool_call arguments for 'answer': bad json",
                    partial_response=ModelResponse(
                        content="",
                        raw_response="",
                        raw_tool_calls=[bad_raw],
                        reasoning_content="bad reasoning",
                        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                        latency_ms=123,
                    ),
                )
            return ModelResponse(
                content="submit fixed answer",
                tool_calls=[answer_call],
                raw_response="submit fixed answer",
                raw_tool_calls=[answer_raw],
            )

    adapter = ParseThenAnswerAdapter()
    agent = ReActAgent(
        model=adapter,
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=3),
        protocol="native",
    )

    result = agent.run(task)

    assert result.succeeded is True
    assert len(result.steps) == 2
    parse_step, answer_step = result.steps
    assert parse_step.action == "__error__"
    assert parse_step.reasoning == "bad reasoning"
    assert parse_step.usage.total_tokens == 15
    assert parse_step.response_message is not None
    assert parse_step.response_message["raw_tool_calls"] == [bad_raw]
    assert "from_csv" in adapter.messages[1][-1].content
    assert answer_step.action == "answer"


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
    assert "call `answer` NOW" in budget_msgs[0].content


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
