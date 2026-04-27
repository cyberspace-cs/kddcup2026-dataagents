"""ReAct 循环把 prompt_messages / response_message 落到 StepRecord 的回归测试。

覆盖：
- 每条成功 step 的 `prompt_messages` 至少包含 system + user 两条消息
- 每条成功 step 的 `response_message` 含 content / tool_calls / usage
- 顶层 `system_prompt` 由 runner_artifacts 抽出，且与 step 内 system 消息一致
- Step 1 的 prompt 不含 assistant 回放；Step 2 应该有上一轮的 assistant + tool 消息
"""

from __future__ import annotations

import json
from pathlib import Path

from data_agent_baseline.agents.model import ModelResponse, ModelToolCall, TokenUsage
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig
from data_agent_baseline.run.runner import run_single_task
from data_agent_baseline.run.runner_artifacts import extract_system_prompt
from data_agent_baseline.tools.registry import create_default_tool_registry
from tests.helpers.scripted_adapters import ScriptedNativeModelAdapter


def _make_task(tmp_path: Path) -> PublicTask:
    """构造一个最小可用的任务（仅一个空的 context 目录）。"""
    task_dir = tmp_path / "task_capture"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    return PublicTask(
        record=TaskRecord(task_id="task_capture", difficulty="easy", question="q?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _build_two_turn_adapter() -> ScriptedNativeModelAdapter:
    """两轮脚本：list_context → answer。供"两步任务"路径使用。"""
    list_call = ModelToolCall(id="c1", name="list_context", arguments={"max_depth": 1})
    list_raw = {
        "id": "c1",
        "type": "function",
        "function": {"name": "list_context", "arguments": json.dumps({"max_depth": 1})},
    }
    answer_call = ModelToolCall(
        id="c2",
        name="answer",
        arguments={"columns": ["v"], "rows": [["ok"]]},
    )
    answer_raw = {
        "id": "c2",
        "type": "function",
        "function": {
            "name": "answer",
            "arguments": json.dumps({"columns": ["v"], "rows": [["ok"]]}),
        },
    }
    return ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="step1",
                tool_calls=[list_call],
                raw_response="step1",
                raw_tool_calls=[list_raw],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="step2",
                tool_calls=[answer_call],
                raw_response="step2",
                raw_tool_calls=[answer_raw],
                usage=TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
            ),
        ]
    )


def test_step_records_carry_prompt_and_response(tmp_path: Path) -> None:
    """端到端：跑两步 ReAct，断言每步都带 prompt_messages + response_message。"""
    task = _make_task(tmp_path)
    agent = ReActAgent(
        model=_build_two_turn_adapter(),
        tools=create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=4),
        protocol="native",
    )
    result = agent.run(task)

    assert result.succeeded is True
    assert len(result.steps) == 2

    # ---- Step 1: prompt 仅含 system + user；尚无 assistant 回放 ----
    step1 = result.steps[0]
    assert step1.prompt_messages is not None
    roles_step1 = [m["role"] for m in step1.prompt_messages]
    assert roles_step1 == ["system", "user"]
    # response_message 应包含 content / tool_calls / usage
    assert step1.response_message is not None
    assert step1.response_message["content"] == "step1"
    assert len(step1.response_message["tool_calls"]) == 1
    assert step1.response_message["tool_calls"][0]["name"] == "list_context"
    assert step1.response_message["usage"]["prompt_tokens"] == 10

    # ---- Step 2: prompt 应含 system + user + assistant(step1) + tool(step1 obs) ----
    step2 = result.steps[1]
    assert step2.prompt_messages is not None
    roles_step2 = [m["role"] for m in step2.prompt_messages]
    assert roles_step2 == ["system", "user", "assistant", "tool"]
    assert step2.response_message is not None
    assert step2.response_message["tool_calls"][0]["name"] == "answer"


def test_extract_system_prompt_helper() -> None:
    """`extract_system_prompt` 直接对 dict 列表工作，应能从 step[0] 中拣到 system 内容。"""
    steps = [
        {
            "prompt_messages": [
                {"role": "system", "content": "SYSTEM-CONTENT"},
                {"role": "user", "content": "user msg"},
            ],
        },
    ]
    assert extract_system_prompt(steps) == "SYSTEM-CONTENT"

    # 缺失 prompt_messages 时返回 None
    assert extract_system_prompt([{"action": "x"}]) is None
    # 空步列表
    assert extract_system_prompt([]) is None
    # 第一条 prompt 没 system 角色
    assert (
        extract_system_prompt(
            [{"prompt_messages": [{"role": "user", "content": "u"}]}],
        )
        is None
    )


def test_parallel_tool_calls_only_lead_step_carries_prompt_and_response(
    tmp_path: Path,
) -> None:
    """lead-only 不变式：单 turn 并行 N 个 tool_calls 时，只有首条 StepRecord
    携带 `prompt_messages` / `response_message`，其余应为 None。

    动机：完整 messages + response 落到每条 StepRecord 会让 trace.json 体积线性膨胀
    （N 倍），且回放时只需读首条即可还原。该约定与 `raw_tool_calls` / `usage` /
    `reasoning` 保持一致——见 `react.py` 中 `is_lead` 判定。
    """
    task = _make_task(tmp_path)

    list_call_a = ModelToolCall(id="par_a", name="list_context", arguments={"max_depth": 1})
    list_call_b = ModelToolCall(id="par_b", name="list_context", arguments={"max_depth": 2})
    list_raw_a = {
        "id": "par_a",
        "type": "function",
        "function": {"name": "list_context", "arguments": json.dumps({"max_depth": 1})},
    }
    list_raw_b = {
        "id": "par_b",
        "type": "function",
        "function": {"name": "list_context", "arguments": json.dumps({"max_depth": 2})},
    }
    answer_call = ModelToolCall(
        id="done",
        name="answer",
        arguments={"columns": ["v"], "rows": [["ok"]]},
    )
    answer_raw = {
        "id": "done",
        "type": "function",
        "function": {
            "name": "answer",
            "arguments": json.dumps({"columns": ["v"], "rows": [["ok"]]}),
        },
    }
    adapter = ScriptedNativeModelAdapter(
        [
            # 第一轮：并行 2 个 list_context（产生 2 条 StepRecord，共享 turn_index）
            ModelResponse(
                content="parallel explore",
                tool_calls=[list_call_a, list_call_b],
                raw_response="parallel explore",
                raw_tool_calls=[list_raw_a, list_raw_b],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
            # 第二轮：answer 终止
            ModelResponse(
                content="finalize",
                tool_calls=[answer_call],
                raw_response="finalize",
                raw_tool_calls=[answer_raw],
                usage=TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
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
    assert len(result.steps) == 3  # 2 (parallel) + 1 (answer)

    lead_parallel = result.steps[0]
    follower_parallel = result.steps[1]
    answer_step = result.steps[2]

    # ---- lead 必须承载 prompt_messages + response_message ----
    assert lead_parallel.turn_index == follower_parallel.turn_index, (
        "并行 tool_calls 应共享 turn_index，否则 lead-only 切片逻辑会失效"
    )
    assert lead_parallel.prompt_messages is not None
    assert lead_parallel.response_message is not None
    # raw_tool_calls / usage / reasoning 同步检查（与 prompt/response 同约定）
    assert lead_parallel.raw_tool_calls is not None
    assert lead_parallel.usage.prompt_tokens == 10

    # ---- follower 必须为 None（即避免双计 + 节省体积）----
    assert follower_parallel.prompt_messages is None
    assert follower_parallel.response_message is None
    assert follower_parallel.raw_tool_calls is None
    assert follower_parallel.usage == TokenUsage()  # 默认全 0

    # ---- 下一轮（answer）作为新 turn 的 lead，重新承载 ----
    assert answer_step.turn_index != lead_parallel.turn_index
    assert answer_step.prompt_messages is not None
    assert answer_step.response_message is not None


def test_trace_json_has_top_level_system_prompt(tmp_path: Path) -> None:
    """run_single_task 落盘后，trace.json 顶层 system_prompt 字段非空且与 step 内一致。"""
    dataset_root = tmp_path / "input"
    task_dir = dataset_root / "task_capture"
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": "task_capture",
                "difficulty": "easy",
                "question": "List the files.",
            }
        )
    )

    run_output_dir = tmp_path / "runs" / "capture"
    run_output_dir.mkdir(parents=True)
    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(api_key="test", max_steps=4),
        run=RunConfig(
            output_dir=run_output_dir.parent,
            run_id="capture",
            task_timeout_seconds=0,
        ),
    )
    artifact = run_single_task(
        task_id="task_capture",
        config=config,
        run_output_dir=run_output_dir,
        model=_build_two_turn_adapter(),
        tools=create_default_tool_registry(),
    )
    payload = json.loads(artifact.trace_path.read_text())

    assert payload["system_prompt"] is not None
    assert len(payload["system_prompt"]) > 0
    # 顶层 system_prompt 必须等于 steps[0].prompt_messages[0].content
    first_step_system = payload["steps"][0]["prompt_messages"][0]["content"]
    assert payload["system_prompt"] == first_step_system
