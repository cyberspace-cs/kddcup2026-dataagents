"""端到端 byte-stability 回归：固定 ScriptedNativeModelAdapter 脚本驱动 agent，
比对 trace.json / summary.json / prediction.csv 的字节内容与冻结的 golden 文件一致。

Golden 文件保存在 `tests/fixtures/golden/` 下；为了让 fixture 与机器无关：
- trace.json 内的 `<CONTEXT_DIR>` 占位符会在比对时替换为实际临时上下文目录
- trace.json 内的 `e2e_elapsed_seconds` 字段（壁钟时间）在比对前归零
- summary.json 内的 `<RUN_DIR>` 占位符会替换为实际运行输出目录
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from data_agent_baseline.agents.model import (
    ModelResponse,
    ModelToolCall,
    ScriptedNativeModelAdapter,
    TokenUsage,
)
from data_agent_baseline.config import AgentConfig, AppConfig, DatasetConfig, RunConfig
from data_agent_baseline.run.runner import run_single_task
from data_agent_baseline.tools.registry import create_default_tool_registry

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"


def _scripted_adapter() -> ScriptedNativeModelAdapter:
    """两轮固定脚本：list_context → answer。"""
    list_call = ModelToolCall(id="call_1", name="list_context", arguments={"max_depth": 1})
    list_raw = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "list_context", "arguments": json.dumps({"max_depth": 1})},
    }
    answer_call = ModelToolCall(
        id="call_2",
        name="answer",
        arguments={"columns": ["filename"], "rows": [["note.md"]]},
    )
    answer_raw = {
        "id": "call_2",
        "type": "function",
        "function": {
            "name": "answer",
            "arguments": json.dumps({"columns": ["filename"], "rows": [["note.md"]]}),
        },
    }
    return ScriptedNativeModelAdapter(
        [
            ModelResponse(
                content="Let me list the context first.",
                tool_calls=[list_call],
                raw_response="Let me list the context first.",
                raw_tool_calls=[list_raw],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                latency_ms=0,
            ),
            ModelResponse(
                content="The single file is note.md.",
                tool_calls=[answer_call],
                raw_response="The single file is note.md.",
                raw_tool_calls=[answer_raw],
                usage=TokenUsage(prompt_tokens=20, completion_tokens=8, total_tokens=28),
                latency_ms=0,
            ),
        ]
    )


def test_trace_and_prediction_byte_stable(tmp_path: Path) -> None:
    """运行 agent 后比对 trace.json / prediction.csv 与 golden 字节一致。"""
    dataset_root = tmp_path / "input"
    dataset_root.mkdir()
    shutil.copytree(GOLDEN_DIR / "task_fixture", dataset_root / "task_fixture")
    context_dir_str = str(dataset_root / "task_fixture" / "context")

    run_output_dir = tmp_path / "runs" / "golden-run"
    run_output_dir.mkdir(parents=True)

    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(api_key="test", max_steps=4),
        run=RunConfig(
            output_dir=run_output_dir.parent,
            run_id="golden-run",
            task_timeout_seconds=0,
        ),
    )

    artifact = run_single_task(
        task_id="task_fixture",
        config=config,
        run_output_dir=run_output_dir,
        model=_scripted_adapter(),
        tools=create_default_tool_registry(),
    )

    trace_text = artifact.trace_path.read_text()
    trace_text = trace_text.replace(context_dir_str, "<CONTEXT_DIR>")
    trace_data = json.loads(trace_text)
    trace_data["e2e_elapsed_seconds"] = 0.0
    actual_trace_bytes = (json.dumps(trace_data, ensure_ascii=False, indent=2) + "\n").encode()

    expected_trace_bytes = (GOLDEN_DIR / "trace.json").read_bytes()
    assert actual_trace_bytes == expected_trace_bytes

    assert artifact.prediction_csv_path is not None
    assert artifact.prediction_csv_path.read_bytes() == (GOLDEN_DIR / "prediction.csv").read_bytes()


def test_summary_byte_stable(tmp_path: Path) -> None:
    """run_benchmark 风格 summary.json 与 golden 字节一致（替换 <RUN_DIR> 占位符后）。"""
    dataset_root = tmp_path / "input"
    dataset_root.mkdir()
    shutil.copytree(GOLDEN_DIR / "task_fixture", dataset_root / "task_fixture")

    run_output_dir = tmp_path / "runs" / "golden-run"
    run_output_dir.mkdir(parents=True)

    config = AppConfig(
        dataset=DatasetConfig(root_path=dataset_root),
        agent=AgentConfig(api_key="test", max_steps=4),
        run=RunConfig(
            output_dir=run_output_dir.parent,
            run_id="golden-run",
            task_timeout_seconds=0,
        ),
    )

    artifact = run_single_task(
        task_id="task_fixture",
        config=config,
        run_output_dir=run_output_dir,
        model=_scripted_adapter(),
        tools=create_default_tool_registry(),
    )

    # Synthesize the run_benchmark-style summary structure from a single artifact.
    actual_summary = {
        "run_id": "golden-run",
        "task_count": 1,
        "succeeded_task_count": 1 if artifact.succeeded else 0,
        "max_workers": 1,
        "tasks": [artifact.to_dict()],
    }
    actual_summary_text = json.dumps(actual_summary, ensure_ascii=False, indent=2) + "\n"
    actual_summary_text = actual_summary_text.replace(str(run_output_dir), "<RUN_DIR>")

    expected_summary_text = (GOLDEN_DIR / "summary.json").read_text()
    assert actual_summary_text == expected_summary_text
