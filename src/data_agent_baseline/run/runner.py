"""任务运行调度与产物落盘。

三层抽象：
- `_run_single_task_core`:         在当前进程里跑一个任务，返回 dict 结果
- `_run_single_task_with_timeout`: 套一层子进程 + join(timeout)，给超时/崩溃兜底
- `run_single_task / run_benchmark`: 对外 API，封装落盘与并行

输出目录结构：
    <output_dir>/<run_id>/
    ├── summary.json                  # 整次 run 的汇总
    └── <task_id>/
        ├── trace.json                # ReAct 全量 steps + answer + elapsed
        └── prediction.csv            # 答案表格（仅 Agent 提交了 answer 时）

注意：`summary.json` 里的 `succeeded_task_count` 仅统计 Agent 是否调用了 answer，
**不代表答案正确**。真正的评分需要另外比对 `prediction.csv` 与 gold.csv。
"""

from __future__ import annotations

import csv
import json
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_baseline.agents.model import OpenAIModelAdapter
from data_agent_baseline.agents.react import ReActAgent, ReActAgentConfig
from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.config import AppConfig
from data_agent_baseline.tools.registry import ToolRegistry, create_default_tool_registry


@dataclass(frozen=True, slots=True)
class TaskRunArtifacts:
    """单任务运行结束后，关于其产物的句柄集合。

    面向上层（CLI 展示、summary.json 序列化）：包含所有外部关心的字段，
    但不含 trace/answer 本体——那些已写进磁盘。
    """

    task_id: str
    task_output_dir: Path
    prediction_csv_path: Path | None  # 未提交 answer 时为 None
    trace_path: Path
    succeeded: bool  # Agent 是否正常调用 answer 工具
    failure_reason: str | None  # 失败时的可读原因

    def to_dict(self) -> dict[str, Any]:
        """序列化进 summary.json 的任务条目。

        Path → str：JSON 不原生支持 Path 类型。
        """
        return {
            "task_id": self.task_id,
            "task_output_dir": str(self.task_output_dir),
            "prediction_csv_path": str(self.prediction_csv_path)
            if self.prediction_csv_path
            else None,
            "trace_path": str(self.trace_path),
            "succeeded": self.succeeded,
            "failure_reason": self.failure_reason,
        }


def create_run_id() -> str:
    """生成默认 run_id：UTC 时间戳如 `20260420T123045Z`。

    用 UTC 保证跨时区机器不冲突；秒级精度对人工运行已足够。
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_run_id(run_id: str | None = None) -> str:
    """规整 run_id：None 时自动生成；传了字符串则必须是合法的单级目录名。

    拒绝包含 `/` `\\` 或等于 `.`/`..` 的输入，防止 Agent 误写到父目录或根目录。
    """
    if run_id is None:
        return create_run_id()

    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty.")
    # 单级目录名约束：禁 `.`/`..`/含路径分隔符
    if normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("run_id must be a single directory name, not a path.")
    return normalized


def create_run_output_dir(output_root: Path, *, run_id: str | None = None) -> tuple[str, Path]:
    """在 `output_root` 下创建 `<run_id>/` 子目录。

    `exist_ok=False` 是故意的：防止复用同名 run_id 覆盖历史产物。
    """
    effective_run_id = resolve_run_id(run_id)
    run_output_dir = output_root / effective_run_id
    run_output_dir.mkdir(parents=True, exist_ok=False)
    return effective_run_id, run_output_dir


def build_model_adapter(config: AppConfig):
    """根据 `config.agent` 构造 OpenAI 适配器。

    独立成函数主要是为了测试：可以 monkeypatch 掉这里注入 `ScriptedModelAdapter`。
    """
    return OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=config.agent.temperature,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """写 JSON：UTF-8 + 2 缩进 + 行尾换行，保证 git diff 可读。

    ensure_ascii=False 保留中文原样，避免 `\\uXXXX` 污染 trace.json。
    """
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    """写 prediction.csv（逐行 writer）。

    `newline=""` 按 csv 模块要求传，确保换行符跨平台一致（避免 Windows 写出 `\\r\\r\\n`）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)


def _failure_run_result_payload(task_id: str, failure_reason: str) -> dict[str, Any]:
    """构造 "任务彻底失败" 的占位结果，结构与 AgentRunResult.to_dict 同形。

    用于超时 / 子进程崩溃等情形，保证上层落盘/统计代码无分支。
    """
    return {
        "task_id": task_id,
        "answer": None,
        "steps": [],
        "failure_reason": failure_reason,
        "succeeded": False,
    }


def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    """纯业务逻辑：加载任务 → 构造 Agent → 跑循环 → 返回 dict。

    model/tools 若外部传入则复用（并行/测试场景），否则现建。
    """
    # 每次都现 new 一个 dataset 对象：它是 frozen dataclass，几乎零成本
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    agent = ReActAgent(
        model=model or build_model_adapter(config),
        tools=tools or create_default_tool_registry(),
        config=ReActAgentConfig(max_steps=config.agent.max_steps),
    )
    run_result = agent.run(task)
    return run_result.to_dict()


def _run_single_task_in_subprocess(
    task_id: str, config: AppConfig, queue: multiprocessing.Queue[Any]
) -> None:
    """子进程入口：跑核心逻辑并把结果/异常扔回主进程 Queue。

    注意 BaseException 而非 Exception——要兜住 KeyboardInterrupt / SystemExit，
    否则子进程退出码非 0 但 Queue 为空，主进程很难诊断。
    """
    try:
        queue.put(
            {
                "ok": True,
                "run_result": _run_single_task_core(task_id=task_id, config=config),
            }
        )
    except BaseException as exc:
        queue.put(
            {
                "ok": False,
                "error": str(exc),
            }
        )


def _run_single_task_with_timeout(*, task_id: str, config: AppConfig) -> dict[str, Any]:
    """跨进程执行带硬超时的单任务。

    三类退出路径：
    1. 超时还活着 → terminate + 兜底 kill，返回"超时"失败结果
    2. 退出但 Queue 为空 → 进程死于 OS kill / segfault，返回"无结果"失败
    3. Queue 有数据 → 按 ok/error 决定是 pass-through 还是再包一层失败

    `timeout_seconds <= 0` 时绕过子进程机制，直接在当前进程跑——
    用于显式不需要超时的单测场景。
    """
    timeout_seconds = config.run.task_timeout_seconds
    if timeout_seconds <= 0:
        return _run_single_task_core(task_id=task_id, config=config)

    queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_single_task_in_subprocess,
        args=(task_id, config, queue),
    )
    process.start()
    process.join(timeout_seconds)

    # Case 1: 超时 —— 先 SIGTERM 给清理机会，1 秒仍不退就 SIGKILL
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()
        return _failure_run_result_payload(
            task_id, f"Task timed out after {timeout_seconds} seconds."
        )

    # Case 2: 进程已退出但没 put 任何结果
    if queue.empty():
        exit_code = process.exitcode
        if exit_code not in (None, 0):
            # 非 0 退出码一般意味着 native crash 或被 OS 干掉
            return _failure_run_result_payload(
                task_id,
                f"Task exited unexpectedly with exit code {exit_code}.",
            )
        # 退出码 0 但 Queue 空：逻辑异常（理论上不会发生，留个明确错误信息）
        return _failure_run_result_payload(task_id, "Task exited without returning a result.")

    # Case 3: 正常拿到 Queue 数据
    result = queue.get()
    if result.get("ok"):
        # dict(...) 拷贝一次，脱离 Queue 的底层 pickle 壳
        return dict(result["run_result"])
    return _failure_run_result_payload(
        task_id, f"Task failed with uncaught error: {result['error']}"
    )


def _write_task_outputs(
    task_id: str, run_output_dir: Path, run_result: dict[str, Any]
) -> TaskRunArtifacts:
    """把单任务结果落盘为 trace.json（必出）+ prediction.csv（仅当有 answer）。

    trace.json 保留完整 ReAct 轨迹用于离线分析；prediction.csv 是评测面向的唯一产物。
    """
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / "trace.json"
    _write_json(trace_path, run_result)

    # prediction.csv 只有在 Agent 成功调用 answer 后才写
    prediction_csv_path: Path | None = None
    answer = run_result.get("answer")
    if isinstance(answer, dict):
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(
            prediction_csv_path,
            list(answer.get("columns", [])),
            [list(row) for row in answer.get("rows", [])],
        )

    return TaskRunArtifacts(
        task_id=task_id,
        task_output_dir=task_output_dir,
        prediction_csv_path=prediction_csv_path,
        trace_path=trace_path,
        succeeded=bool(run_result.get("succeeded")),
        failure_reason=run_result.get("failure_reason"),
    )


def run_single_task(
    *,
    task_id: str,
    config: AppConfig,
    run_output_dir: Path,
    model=None,
    tools: ToolRegistry | None = None,
) -> TaskRunArtifacts:
    """对外：跑单个任务并落盘产物。

    model/tools 的注入决定了隔离强度：
    - 两者都为 None（默认）：走 `_run_single_task_with_timeout`（子进程 + 硬超时）
    - 任一不为 None：走 `_run_single_task_core`（当前进程，复用共享资源；
      用于 workers=1 或测试场景）
    """
    started_at = perf_counter()
    if model is None and tools is None:
        run_result = _run_single_task_with_timeout(task_id=task_id, config=config)
    else:
        run_result = _run_single_task_core(task_id=task_id, config=config, model=model, tools=tools)
    # end-to-end 耗时给 trace.json 记一笔，便于后续做性能分析
    run_result["e2e_elapsed_seconds"] = round(perf_counter() - started_at, 3)
    return _write_task_outputs(task_id, run_output_dir, run_result)


def run_benchmark(
    *,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    limit: int | None = None,
    progress_callback: Callable[[TaskRunArtifacts], None] | None = None,
) -> tuple[Path, list[TaskRunArtifacts]]:
    """批量跑一批任务并生成 summary.json。

    `model` / `tools` 不为 None 时：
    - 强制 workers=1（资源被共享，无法安全并发）
    - 主要用于 ScriptedModelAdapter + mock tools 的端到端测试

    并发模型：
    - 当 workers == 1：串行 for，复用同一份 model/tools
    - 当 workers > 1：ThreadPoolExecutor + as_completed，每个 worker 通过子进程跑任务
      （线程池 + 子进程叠加是因为 OpenAI 调用主要是 IO 等待，线程足够释放并发；
       子进程用来硬隔离超时/崩溃）

    `progress_callback` 在每个任务完成时触发，主要供 CLI 渲染进度条。
    """
    # 先建目录再做其他事：run_id 冲突要立刻暴露
    effective_run_id, run_output_dir = create_run_output_dir(
        config.run.output_dir, run_id=config.run.run_id
    )

    dataset = DABenchPublicDataset(config.dataset.root_path)
    tasks = dataset.iter_tasks()
    # limit 起 "抽样调试" 作用；切前 N 个（任务已按编号排序）
    if limit is not None:
        tasks = tasks[:limit]

    effective_workers = config.run.max_workers
    if effective_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    # 共享 model/tools 时不能并行（OpenAI client 线程安全但工具注册表状态不可并发修改）
    if model is not None or tools is not None:
        effective_workers = 1

    task_ids = [task.task_id for task in tasks]

    task_artifacts: list[TaskRunArtifacts]
    if effective_workers == 1:
        # 串行路径：共享 model/tools 减少重复初始化开销
        shared_model = model or build_model_adapter(config)
        shared_tools = tools or create_default_tool_registry()
        task_artifacts = []
        for task_id in task_ids:
            artifact = run_single_task(
                task_id=task_id,
                config=config,
                run_output_dir=run_output_dir,
                model=shared_model,
                tools=shared_tools,
            )
            task_artifacts.append(artifact)
            if progress_callback is not None:
                progress_callback(artifact)
    else:
        # 并行路径：每个线程自己调 run_single_task，内部会 fork 子进程做超时隔离
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_index = {
                executor.submit(
                    run_single_task,
                    task_id=task_id,
                    config=config,
                    run_output_dir=run_output_dir,
                ): index
                for index, task_id in enumerate(task_ids)
            }
            # 按任务原始顺序占位，最后按顺序收集；保证 summary.json 的 tasks 数组稳定
            indexed_artifacts: list[TaskRunArtifacts | None] = [None] * len(task_ids)
            for future in as_completed(future_to_index):
                artifact = future.result()
                indexed_artifacts[future_to_index[future]] = artifact
                # 回调在 as_completed 的主线程里串行调用，UI 侧不用自己加锁
                if progress_callback is not None:
                    progress_callback(artifact)
            # 过滤 None（理论上所有格子都被填满；留一手防御 future.result() 意外）
            task_artifacts = [artifact for artifact in indexed_artifacts if artifact is not None]

    # 整次 run 的汇总文件：run_id、任务数、成功数、并发度、各任务摘要
    summary_path = run_output_dir / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": effective_run_id,
            "task_count": len(task_artifacts),
            "succeeded_task_count": sum(1 for artifact in task_artifacts if artifact.succeeded),
            "max_workers": effective_workers,
            "tasks": [artifact.to_dict() for artifact in task_artifacts],
        },
    )
    return run_output_dir, task_artifacts
