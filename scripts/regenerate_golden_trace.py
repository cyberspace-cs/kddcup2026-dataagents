"""一次性脚本：重新生成 tests/fixtures/golden/{trace.json, summary.json}。

用途：当 prompt 或 trace schema 改动导致 byte-stability 测试失败时，确认逻辑变化
是预期的，再跑此脚本同步 fixture。

逻辑严格复用 tests/test_golden_trace.py 的两个测试函数（脚本化 adapter + 占位符
替换），保证生成结果与 byte-stability 断言期待的字节一致。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

# 仓库根加入 sys.path，让 `tests.*` 可被 import（脚本不依赖 PYTHONPATH）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.helpers.scripted_adapters import ScriptedNativeModelAdapter
from tests.test_golden_trace import GOLDEN_DIR, _scripted_adapter

from data_agent_baseline.config import (
    AgentConfig,
    AppConfig,
    DatasetConfig,
    RunConfig,
)
from data_agent_baseline.run.runner import run_single_task
from data_agent_baseline.tools.registry import create_default_tool_registry


def _regenerate(
    scripted: ScriptedNativeModelAdapter, dataset_root: Path, run_output_dir: Path
) -> tuple[bytes, str, bytes]:
    artifact = run_single_task(
        task_id="task_fixture",
        config=AppConfig(
            dataset=DatasetConfig(root_path=dataset_root),
            agent=AgentConfig(api_key="test", max_steps=4),
            run=RunConfig(
                output_dir=run_output_dir.parent,
                run_id="golden-run",
                task_timeout_seconds=0,
            ),
        ),
        run_output_dir=run_output_dir,
        model=scripted,
        tools=create_default_tool_registry(),
    )

    context_dir_str = str(dataset_root / "task_fixture" / "context")
    trace_text = artifact.trace_path.read_text().replace(context_dir_str, "<CONTEXT_DIR>")
    trace_data = json.loads(trace_text)
    trace_data["e2e_elapsed_seconds"] = 0.0
    trace_bytes = (json.dumps(trace_data, ensure_ascii=False, indent=2) + "\n").encode()

    summary = {
        "run_id": "golden-run",
        "task_count": 1,
        "succeeded_task_count": 1 if artifact.succeeded else 0,
        "max_workers": 1,
        "tasks": [artifact.to_dict()],
    }
    summary_text = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").replace(
        str(run_output_dir), "<RUN_DIR>"
    )

    assert artifact.prediction_csv_path is not None
    prediction_bytes = artifact.prediction_csv_path.read_bytes()

    return trace_bytes, summary_text, prediction_bytes


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dataset_root = tmp_path / "input"
        dataset_root.mkdir()
        shutil.copytree(GOLDEN_DIR / "task_fixture", dataset_root / "task_fixture")
        run_output_dir = tmp_path / "runs" / "golden-run"
        run_output_dir.mkdir(parents=True)

        trace_bytes, summary_text, prediction_bytes = _regenerate(
            _scripted_adapter(), dataset_root, run_output_dir
        )

    (GOLDEN_DIR / "trace.json").write_bytes(trace_bytes)
    (GOLDEN_DIR / "summary.json").write_text(summary_text)
    (GOLDEN_DIR / "prediction.csv").write_bytes(prediction_bytes)
    print(f"updated: {GOLDEN_DIR / 'trace.json'} ({len(trace_bytes)} bytes)")
    print(f"updated: {GOLDEN_DIR / 'summary.json'} ({len(summary_text)} chars)")
    print(f"updated: {GOLDEN_DIR / 'prediction.csv'} ({len(prediction_bytes)} bytes)")


if __name__ == "__main__":
    main()
