"""Submission entrypoint per §3.6: iterate /input task_<id>, write /output/<task_id>/prediction.csv.

Wires through pure-version's `run.app.build_application` factory rather than
hand-assembling model + tools + agent — that's how the on-laptop CLI runs
too, and it's the only path that picks the right adapter (NativeToolsOpenAIAdapter
when `tool_protocol="native"`, plain OpenAIModelAdapter when "text"). main's
old submission.py instantiated `OpenAIModelAdapter` against the default-native
ReActAgent, which silently breaks on the first tool call.

§3.7 trade-off: only minimal status lines go to /logs/main.log. ReAct traces
contain raw rows from /input CSVs (read_csv / inspect_files outputs); the
official rules treat that as leaked test-set data — best case loss of debug
support, worst case disqualification. For full traces, run the rehearsal
locally via scripts/eval_local.sh; the artifacts/eval_logs from that path
are unrestricted because they live on your machine.

Exit codes: 0 ok; 78 (EX_CONFIG) missing env / mounts; 73 (EX_CANTCREAT)
/output not writable.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.config import AppConfig, load_app_config
from data_agent_baseline.run.app import AgentApp, build_application

REQUIRED_ENV = ("MODEL_API_URL", "MODEL_API_KEY", "MODEL_NAME")
DEFAULT_INPUT = Path("/input")
DEFAULT_OUTPUT = Path("/output")
DEFAULT_LOGS = Path("/logs")
DEFAULT_CONFIG = Path("/app/configs/react_baseline.example.yaml")
DEFAULT_GLOBAL_BUDGET_SECONDS = 42_000.0  # ~11.7 h, fits a 50-task batch with headroom


def _log(logs_dir: Path, line: str) -> None:
    """Append a status line to /logs/main.log and stderr.

    Per §3.7 we only emit metadata here — never tool outputs or raw rows.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    with (logs_dir / "main.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    sys.stderr.write(line + "\n")


def _write_prediction_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    """Write prediction.csv at the exact byte shape `runner_artifacts._write_csv` produces.

    Inlined (instead of importing the private symbol) to keep the entrypoint
    independent of `run/` internals; if the runner format ever shifts, sync
    here.  `newline=""` is the csv-module-mandated form for cross-platform
    line endings.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def _build_config(input_root: Path) -> AppConfig:
    """Load the baked-in YAML and apply §3.4 / §3.5 overrides.

    Reading from `configs/react_baseline.example.yaml` (baked into the image
    at /app/configs/) preserves on-laptop tuning — `tool_protocol`,
    `enable_thinking`, `max_tokens`, `max_empty_tool_call_retries`, prompt
    knobs — that the local CLI uses. Override `CONFIG_PATH` to point at a
    different config inside the image.

    Forced overrides:
    - `dataset.root_path` → `/input` (§3.4 mount)
    - `agent.{model, api_base, api_key}` → §3.5 injected env vars
    - `agent.{api_keys, rate_limit}` → cleared. The judge gives exactly one
      key and the eval network owns QPS shaping; pure-version's pool-mode
      requires `len(api_keys) > 0`, so leaving a stale list would mis-route.
    - `agent.max_steps` ← `MAX_STEPS` env when set (smoke-test escape hatch)
    """
    config_path = Path(os.environ.get("CONFIG_PATH", DEFAULT_CONFIG))
    base = load_app_config(config_path) if config_path.is_file() else AppConfig()

    agent_overrides: dict[str, Any] = {
        "model": os.environ["MODEL_NAME"],
        "api_base": os.environ["MODEL_API_URL"],
        "api_key": os.environ["MODEL_API_KEY"],
        "api_keys": (),
        "rate_limit": None,
    }
    if max_steps := os.environ.get("MAX_STEPS"):
        agent_overrides["max_steps"] = int(max_steps)

    return replace(
        base,
        dataset=replace(base.dataset, root_path=input_root),
        agent=replace(base.agent, **agent_overrides),
    )


def _run_one_task(app: AgentApp, task: PublicTask, output_root: Path, logs_root: Path) -> bool:
    """Run a single task; return True if a prediction.csv landed on disk."""
    task_started = time.monotonic()
    task_output_dir = output_root / task.task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)

    answer = None
    failure_reason: str | None = None
    try:
        agent = app.new_agent()
        result = agent.run(task)
        answer = result.answer
        failure_reason = result.failure_reason
    except Exception as exc:
        # Tracebacks to stderr only — they may quote question strings, but
        # not raw row payloads. /logs/main.log records the redacted summary.
        traceback.print_exc(file=sys.stderr)
        failure_reason = f"{type(exc).__name__}: {exc}"

    wrote = False
    if answer is not None and answer.columns:
        _write_prediction_csv(
            task_output_dir / "prediction.csv",
            list(answer.columns),
            [list(row) for row in answer.rows],
        )
        wrote = True

    elapsed = time.monotonic() - task_started
    status = "ok" if wrote else f"no_answer ({failure_reason or 'unknown'})"
    _log(logs_root, f"task={task.task_id} status={status} elapsed={elapsed:.1f}s")
    return wrote


def main() -> int:
    input_root = Path(os.environ.get("INPUT_ROOT", DEFAULT_INPUT))
    output_root = Path(os.environ.get("OUTPUT_ROOT", DEFAULT_OUTPUT))
    logs_root = Path(os.environ.get("LOGS_ROOT", DEFAULT_LOGS))

    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        sys.stderr.write(f"FATAL: missing env vars: {missing}\n")
        return 78

    for path in (input_root, output_root, logs_root):
        if not path.is_dir():
            sys.stderr.write(f"FATAL: {path} not mounted as directory\n")
            return 78

    try:
        probe = output_root / ".writable_probe"
        probe.touch()
        probe.unlink()
    except OSError as exc:
        sys.stderr.write(f"FATAL: {output_root} not writable: {exc}\n")
        return 73

    global_budget = float(os.environ.get("GLOBAL_WALL_SECONDS", DEFAULT_GLOBAL_BUDGET_SECONDS))

    config = _build_config(input_root)
    app = build_application(config)
    tasks = app.dataset.iter_tasks()
    _log(
        logs_root,
        f"start input={input_root} output={output_root} budget={global_budget:.0f}s tasks={len(tasks)}",
    )

    started = time.monotonic()
    completed = 0
    written = 0
    for task in tasks:
        elapsed = time.monotonic() - started
        if elapsed >= global_budget:
            _log(logs_root, f"budget exhausted at task={task.task_id} elapsed={elapsed:.0f}s")
            break
        if _run_one_task(app, task, output_root, logs_root):
            written += 1
        completed += 1

    total_elapsed = time.monotonic() - started
    _log(
        logs_root,
        f"finished completed={completed}/{len(tasks)} predictions={written} elapsed={total_elapsed:.1f}s",
    )
    (logs_root / "summary.json").write_text(
        json.dumps(
            {
                "task_total": len(tasks),
                "task_completed": completed,
                "predictions_written": written,
                "elapsed_seconds": round(total_elapsed, 1),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
