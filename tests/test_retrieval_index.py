from __future__ import annotations

import csv
import json
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask, TaskAssets, TaskRecord
from data_agent_baseline.config import (
    AgentConfig,
    AppConfig,
    DatasetConfig,
    RetrievalIndexConfig,
    RunConfig,
)
from data_agent_baseline.retrieval.index import (
    build_or_load_task_index,
    lookup_field,
    search_context,
)
from data_agent_baseline.run.runner_subprocess import run_single_task_core
from data_agent_baseline.tools.filesystem import inspect_context_files
from data_agent_baseline.tools.registry import create_default_tool_registry
from tests.helpers.scripted_adapters import ScriptedModelAdapter


def _make_task(tmp_path: Path, task_id: str | None = None) -> PublicTask:
    if task_id is None:
        task_id = "task_" + tmp_path.name.replace("-", "_")
    task_dir = tmp_path / "input" / task_id
    context_dir = task_dir / "context"
    context_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "difficulty": "hard",
                "question": "What is Angela Sanders's major?",
            }
        )
    )
    return PublicTask(
        record=TaskRecord(task_id=task_id, difficulty="hard", question="?"),
        assets=TaskAssets(task_dir=task_dir, context_dir=context_dir),
    )


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _populate_context(task: PublicTask) -> None:
    doc_dir = task.context_dir / "doc"
    doc_dir.mkdir()
    (task.context_dir / "knowledge.md").write_text(
        "\n".join(
            [
                "# Knowledge",
                "",
                "FG: fibrinogen normal range, unit mg/dL.",
                "major: student declared major.",
            ]
        )
    )
    (doc_dir / "students.md").write_text(
        "\n".join(
            [
                "# Student Records",
                "",
                "## Angela Sanders",
                "Angela Sanders is enrolled in the Student_Club roster.",
                "Her declared major is Biology.",
                "",
                "## Marco Chen",
                "Marco Chen studies Mathematics.",
            ]
        )
    )
    _write_csv(
        task.context_dir / "students.csv",
        ["student_id", "full_name", "major"],
        [["1", "Angela Sanders", "Biology"], ["2", "Marco Chen", "Mathematics"]],
    )


def test_retrieval_index_builds_searches_and_caches(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    _populate_context(task)
    cfg = RetrievalIndexConfig(enabled=True, cache_dir=tmp_path / "indexes")

    first = build_or_load_task_index(task, cfg)
    assert first["built"] is True
    assert first["cache_hit"] is False
    assert first["chunk_count"] >= 3
    assert first["field_count"] >= 3

    search = search_context(task, query="Angela Sanders major")
    assert search["result_count"] >= 1
    assert search["results"][0]["path"] == "doc/students.md"
    assert "Angela" in search["results"][0]["preview"]

    fields = lookup_field(task, term="FG fibrinogen")
    assert fields["result_count"] >= 1
    assert fields["results"][0]["path"] == "knowledge.md"

    second = build_or_load_task_index(task, cfg)
    assert second["cache_hit"] is True


def test_search_context_honors_path_filter(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    _populate_context(task)
    cfg = RetrievalIndexConfig(enabled=True, cache_dir=tmp_path / "indexes")
    build_or_load_task_index(task, cfg)

    miss = search_context(task, query="Angela", path="knowledge.md")
    assert miss["result_count"] == 0

    hit = search_context(task, query="Angela", path="doc/students.md")
    assert hit["result_count"] >= 1


def test_inspect_files_reports_retrieval_index_metadata(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    _populate_context(task)
    cfg = RetrievalIndexConfig(enabled=True, cache_dir=tmp_path / "indexes")
    build_or_load_task_index(task, cfg)

    result = inspect_context_files(task)
    assert result["retrieval_index"]["built"] is True
    assert result["retrieval_index"]["chunk_count"] >= 1
    assert "search_context" in result["retrieval_index"]["hint"]


def test_retrieval_tools_return_recoverable_disabled_error(tmp_path: Path) -> None:
    task = _make_task(tmp_path)
    _populate_context(task)
    registry = create_default_tool_registry()

    result = registry.execute(task, "search_context", {"query": "Angela"})
    assert result.ok is False
    assert result.content["enabled"] is False


def test_runner_attaches_retrieval_index_payload(tmp_path: Path) -> None:
    task = _make_task(tmp_path, task_id="task_1")
    _populate_context(task)
    cfg = AppConfig(
        dataset=DatasetConfig(root_path=tmp_path / "input"),
        agent=AgentConfig(tool_protocol="text", max_steps=1),
        run=RunConfig(
            retrieval_index=RetrievalIndexConfig(
                enabled=True,
                cache_dir=tmp_path / "indexes",
            ),
            task_timeout_seconds=0,
        ),
    )
    model = ScriptedModelAdapter(
        [
            '```json\n{"thought":"done","action":"answer","action_input":'
            '{"columns":["major"],"rows":[["Biology"]]}}\n```'
        ]
    )
    result = run_single_task_core(
        task_id="task_1",
        config=cfg,
        model=model,
        tools=create_default_tool_registry(),
    )

    assert result["succeeded"] is True
    assert result["retrieval_index"]["built"] is True
