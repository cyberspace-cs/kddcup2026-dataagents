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
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from data_agent_baseline.agents.model import NativeToolsOpenAIAdapter, OpenAIModelAdapter
from data_agent_baseline.agents.rate_limit import RateLimitedAdapter
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


def build_model_adapter(config: AppConfig, tools: ToolRegistry | None = None):
    """根据 `config.agent` 构造模型适配器。

    按 `tool_protocol` 分发：
    - "native": 需要传入 `tools` 以把工具 schema 下推到 OpenAI `tools=[...]` 字段
    - "text":   传统文本协议，不需要 tools 参数

    当 `config.agent.rate_limit` 非 None 且 `api_key` 非空时，外面再套一层
    `RateLimitedAdapter`（PR-2）。线上 vLLM 路径留空 rate_limit 即跳过。

    独立成函数主要是为了测试：可以 monkeypatch 掉这里注入 `ScriptedModelAdapter`。
    """
    inner: NativeToolsOpenAIAdapter | OpenAIModelAdapter
    if config.agent.tool_protocol == "native":
        if tools is None:
            # 给一个防御性错误：native 协议必须提供 tools
            raise ValueError("NativeToolsOpenAIAdapter requires a ToolRegistry; pass `tools=...`.")
        inner = NativeToolsOpenAIAdapter(
            model=config.agent.model,
            api_base=config.agent.api_base,
            api_key=config.agent.api_key,
            temperature=config.agent.temperature,
            tools=tools,
            allow_parallel_tool_calls=config.agent.allow_parallel_tool_calls,
            max_tokens=config.agent.max_tokens,
            enable_thinking=config.agent.enable_thinking,
        )
    else:
        inner = OpenAIModelAdapter(
            model=config.agent.model,
            api_base=config.agent.api_base,
            api_key=config.agent.api_key,
            temperature=config.agent.temperature,
            max_tokens=config.agent.max_tokens,
            enable_thinking=config.agent.enable_thinking,
        )

    # PR-2：有 rate_limit 就包一层限速壳；没有就返回裸 adapter（线上路径不变）
    if config.agent.rate_limit is not None and config.agent.api_key:
        return RateLimitedAdapter(
            inner=inner,
            api_key=config.agent.api_key,
            rate_limit=config.agent.rate_limit,
        )
    return inner


def _resolve_ratelimit_state_dir(config: AppConfig, effective_run_id: str) -> Path:
    """推导 `state_dir` 默认值：`<output_dir.parent>/ratelimit/<run_id>/`。

    默认 `output_dir = artifacts/runs`，所以默认状态目录是 `artifacts/ratelimit/<run_id>/`，
    与 `artifacts/runs/<run_id>/` 做兄弟目录——run 粒度隔离，避免跨 run 残留污染令牌桶。
    """
    return config.run.output_dir.parent / "ratelimit" / effective_run_id


def _pin_key_on_config(config: AppConfig, *, task_index: int, effective_run_id: str) -> AppConfig:
    """返回 "单 Key 视图" 的子配置：按 task_index 轮询 api_keys 并写回 agent.api_key，
    同时把 rate_limit.state_dir 从 None 解成具体路径、把 run.run_id 从 None 解成真实值。

    用 `dataclasses.replace` 逐层生成新的 frozen dataclass；原 config 不变，可被多个
    子进程按不同 task_index 独立派生。

    若 `api_keys` 为空（未启用池），原样返回 config——对线上单 Key 路径无副作用。
    """
    if not config.agent.api_keys:
        return config
    picked = config.agent.api_keys[task_index % len(config.agent.api_keys)]
    rl = config.agent.rate_limit
    if rl is not None and rl.state_dir is None:
        rl = replace(rl, state_dir=_resolve_ratelimit_state_dir(config, effective_run_id))
    agent = replace(config.agent, api_key=picked, rate_limit=rl)
    # run_id 也落实：子进程 trace 里能看到真实 run_id，便于调试
    run = replace(config.run, run_id=effective_run_id)
    return replace(config, agent=agent, run=run)


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


def _agent_state_to_payload(task_id: str, state: Any) -> dict[str, Any]:
    """把运行中的 `AgentRuntimeState` 快照成 trace.json 兼容的 dict。

    用于增量落盘：结构与 `AgentRunResult.to_dict()` 对齐，但因为尚未终止：
    - `succeeded` 永远写 False（真正成功时，父进程会用 queue 返回的完整 payload 覆盖）
    - `answer` 在 answer 工具命中那一步之后由 callback 自然带上

    兼容 typing：state 参数走 Any 是为了避免在 runner 层引入对 AgentRuntimeState 的硬依赖
    （保留 runner 与 agent 内部类型的解耦）。
    """
    answer = state.answer.to_dict() if state.answer is not None else None
    return {
        "task_id": task_id,
        "answer": answer,
        "steps": [step.to_dict() for step in state.steps],
        "failure_reason": state.failure_reason,
        "succeeded": False,
    }


def _make_partial_trace_writer(task_id: str, trace_path: Path):
    """构造"把当前 state 写成 trace.json"的 StepCallback。

    每次 state.steps 变化都同步落盘：代价是几 KB 写 I/O，换来子进程被硬 kill 时
    主进程可以读到最后一次 append 后的部分 trace，不至于丢失全部上下文。

    写失败（磁盘满/权限）只静默记一下——不能让观测逻辑弄挂 Agent。
    """

    def _write(state: Any) -> None:
        try:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            payload = _agent_state_to_payload(task_id, state)
            trace_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
        except Exception:
            # 观测失败不能影响任务本身
            pass

    return _write


def _load_partial_trace(task_id: str, trace_path: Path) -> dict[str, Any] | None:
    """尝试从磁盘读出子进程写过的部分 trace；失败时返回 None。

    用于子进程超时/crash 后，从落盘快照里恢复 "死前最后一步" 的 steps 列表。
    """
    if not trace_path.exists():
        return None
    try:
        payload = json.loads(trace_path.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    # 必须是同一个任务的 trace，避免跨任务误取
    if payload.get("task_id") != task_id:
        return None
    return payload


def _normalize_answer_payload(value: Any) -> dict[str, Any] | None:
    """把 trace 中的 answer-like 对象校验并规整成可写 CSV 的 payload。"""
    if not isinstance(value, dict):
        return None

    columns = value.get("columns")
    rows = value.get("rows")
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(item, str) for item in columns)
    ):
        return None
    if not isinstance(rows, list):
        return None

    normalized_rows: list[list[Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(columns):
            return None
        normalized_rows.append(list(row))
    return {"columns": list(columns), "rows": normalized_rows}


def _recover_answer_from_steps(steps: list[Any]) -> dict[str, Any] | None:
    """从 partial trace 的最后一个成功 answer step 中恢复答案表。"""
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        if step.get("action") != "answer" or step.get("ok") is not True:
            continue
        recovered = _normalize_answer_payload(step.get("action_input"))
        if recovered is not None:
            return recovered
    return None


def _run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model=None,
    tools: ToolRegistry | None = None,
    step_callback=None,
) -> dict[str, Any]:
    """纯业务逻辑：加载任务 → 构造 Agent → 跑循环 → 返回 dict。

    model/tools 若外部传入则复用（并行/测试场景），否则现建。
    `step_callback` 允许 runner 在子进程里注入增量 trace 落盘逻辑。
    """
    # 每次都现 new 一个 dataset 对象：它是 frozen dataclass，几乎零成本
    public_dataset = DABenchPublicDataset(config.dataset.root_path)
    task = public_dataset.get_task(task_id)

    effective_tools = tools or create_default_tool_registry()
    effective_model = model or build_model_adapter(config, tools=effective_tools)

    agent = ReActAgent(
        model=effective_model,
        tools=effective_tools,
        config=ReActAgentConfig(max_steps=config.agent.max_steps),
        protocol=config.agent.tool_protocol,  # type: ignore[arg-type]
        step_callback=step_callback,
    )
    run_result = agent.run(task)
    return run_result.to_dict()


def _run_single_task_in_subprocess(
    task_id: str,
    config: AppConfig,
    queue: multiprocessing.Queue[Any],
    partial_trace_path: Path | None = None,
) -> None:
    """子进程入口：跑核心逻辑并把结果/异常扔回主进程 Queue。

    若 `partial_trace_path` 非 None，则每步结束都把当前状态同步写到该路径下。
    父进程在超时/crash 后可以从那里读出 "死前最后一步"。

    注意 BaseException 而非 Exception——要兜住 KeyboardInterrupt / SystemExit，
    否则子进程退出码非 0 但 Queue 为空，主进程很难诊断。
    """
    step_callback = (
        _make_partial_trace_writer(task_id, partial_trace_path)
        if partial_trace_path is not None
        else None
    )
    try:
        queue.put(
            {
                "ok": True,
                "run_result": _run_single_task_core(
                    task_id=task_id,
                    config=config,
                    step_callback=step_callback,
                ),
            }
        )
    except BaseException as exc:
        queue.put(
            {
                "ok": False,
                "error": str(exc),
            }
        )


def _run_single_task_with_timeout(
    *,
    task_id: str,
    config: AppConfig,
    partial_trace_path: Path | None = None,
) -> dict[str, Any]:
    """跨进程执行带硬超时的单任务。

    三类退出路径：
    1. 超时还活着 → terminate + 兜底 kill；如果子进程已经写过部分 trace，则回收那些 steps
    2. 退出但 Queue 为空 → 进程死于 OS kill / segfault；同样先尝试回收 partial trace
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
        args=(task_id, config, queue, partial_trace_path),
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
        return _timeout_payload(task_id, partial_trace_path, timeout_seconds)

    # Case 2: 进程已退出但没 put 任何结果
    if queue.empty():
        exit_code = process.exitcode
        if exit_code not in (None, 0):
            # 非 0 退出码一般意味着 native crash 或被 OS 干掉；尝试拿 partial trace 续命
            return _crash_payload(
                task_id,
                partial_trace_path,
                f"Task exited unexpectedly with exit code {exit_code}.",
            )
        # 退出码 0 但 Queue 空：逻辑异常（理论上不会发生，留个明确错误信息）
        return _crash_payload(
            task_id, partial_trace_path, "Task exited without returning a result."
        )

    # Case 3: 正常拿到 Queue 数据
    result = queue.get()
    if result.get("ok"):
        # dict(...) 拷贝一次，脱离 Queue 的底层 pickle 壳
        return dict(result["run_result"])
    return _crash_payload(
        task_id,
        partial_trace_path,
        f"Task failed with uncaught error: {result['error']}",
    )


def _timeout_payload(
    task_id: str, partial_trace_path: Path | None, timeout_seconds: int
) -> dict[str, Any]:
    """构造超时失败 payload，优先带上 partial trace 里的 steps。"""
    failure_reason = f"Task timed out after {timeout_seconds} seconds."
    return _merge_partial_with_failure(task_id, partial_trace_path, failure_reason)


def _crash_payload(
    task_id: str, partial_trace_path: Path | None, failure_reason: str
) -> dict[str, Any]:
    """构造子进程崩溃失败 payload，优先带上 partial trace 里的 steps。"""
    return _merge_partial_with_failure(task_id, partial_trace_path, failure_reason)


def _merge_partial_with_failure(
    task_id: str, partial_trace_path: Path | None, failure_reason: str
) -> dict[str, Any]:
    """合并规则：有 partial trace 就保留其 steps/answer；失败字段由 runner 覆写。

    拿到的 partial 是子进程写的快照；我们在父侧再加 failure_reason、强制 succeeded=False，
    若顶层 answer 缺失但 steps 里已有成功 answer 调用，则从 action_input 恢复可打分答案。
    """
    if partial_trace_path is not None:
        partial = _load_partial_trace(task_id, partial_trace_path)
        if partial is not None and isinstance(partial.get("steps"), list):
            merged = dict(partial)
            steps = partial["steps"]
            merged["task_id"] = task_id
            merged["failure_reason"] = failure_reason
            answer = _normalize_answer_payload(partial.get("answer"))
            if answer is None:
                answer = _recover_answer_from_steps(steps)
            merged["answer"] = answer
            # 失败路径下 answer 仅表示有可打分产物，运行状态仍由 failure_reason 决定
            merged["succeeded"] = False
            return merged
    return _failure_run_result_payload(task_id, failure_reason)


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

    # 只要 trace 中有结构合法的 answer 就写 prediction.csv；timeout/crash 任务也可能
    # 通过 partial trace 恢复出 answer，但 summary 里的 succeeded 仍保持 False。
    prediction_csv_path: Path | None = None
    answer = _normalize_answer_payload(run_result.get("answer"))
    if answer is not None:
        prediction_csv_path = task_output_dir / "prediction.csv"
        _write_csv(
            prediction_csv_path,
            answer["columns"],
            answer["rows"],
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

    子进程路径下，`partial_trace_path` 在任务开始前就确定了，子进程每步增量写，
    父进程在 kill/crash 后可从磁盘回收"死前"状态——避免 600s 超时的任务 steps=0。

    PR-2：CLI `dabench run-task <id>` 不经 run_benchmark，没人替它按 task_index 固定 Key。
    这里若检测到 "配了 api_keys 但 rate_limit.state_dir 仍是 None"（= 未被上游 pin 过），
    就按 `api_keys[0]` 决定性选 Key。上游 benchmark 路径已经做过 pin 的 config 不受影响。
    """
    if (
        config.agent.api_keys
        and config.agent.rate_limit is not None
        and config.agent.rate_limit.state_dir is None
    ):
        effective_run_id = resolve_run_id(config.run.run_id)
        config = _pin_key_on_config(config, task_index=0, effective_run_id=effective_run_id)

    started_at = perf_counter()
    if model is None and tools is None:
        # 子进程路径：预先确定 trace.json 位置，供子进程增量写 + 父进程超时时读
        partial_trace_path = run_output_dir / task_id / "trace.json"
        run_result = _run_single_task_with_timeout(
            task_id=task_id,
            config=config,
            partial_trace_path=partial_trace_path,
        )
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

    # PR-2：配了 api_keys 就走 "按 task_index 轮询 Key" 的路径；否则保持 PR-1 单 Key 行为
    use_key_pool = bool(config.agent.api_keys) and model is None

    task_artifacts: list[TaskRunArtifacts]
    if effective_workers == 1:
        # 串行路径：共享 model/tools 减少重复初始化开销
        shared_tools = tools or create_default_tool_registry()
        # 启用 Key 池时，model 需要按 task 轮询 Key 重建——放弃共享 model；
        # 但共享 tools 仍然成立（tool registry 是无状态的）
        shared_model = (
            None if use_key_pool else (model or build_model_adapter(config, tools=shared_tools))
        )
        task_artifacts = []
        for index, task_id in enumerate(task_ids):
            iter_config = (
                _pin_key_on_config(config, task_index=index, effective_run_id=effective_run_id)
                if use_key_pool
                else config
            )
            artifact = run_single_task(
                task_id=task_id,
                config=iter_config,
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
                    # PR-2：按 task_index 固定 Key；池未启用时 _pin_key_on_config 原样返回
                    config=_pin_key_on_config(
                        config, task_index=index, effective_run_id=effective_run_id
                    ),
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
