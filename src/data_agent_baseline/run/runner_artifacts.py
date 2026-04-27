"""单任务运行的产物结构 + I/O 与 payload 帮手。

本模块只承担"形成磁盘产物"和"在状态字典间转换"两件事，不感知子进程编排。被
`runner.py` 与 `runner_subprocess.py` 共同使用：

- `TaskRunArtifacts` / run-id 工厂 / `pin_key_on_config`：编排前需要的轻量帮手
- 部分轨迹 (`make_partial_trace_writer` / `load_partial_trace`)：子进程内增量写、
  父进程在 timeout/crash 时回收
- `timeout_payload` / `crash_payload` / `merge_partial_with_failure`：失败兜底
- `write_task_outputs`：核心结果落盘成 trace.json + prediction.csv

跨模块共享的符号统一不带前缀下划线（pyright strict 友好）；仅在本文件内部使用的
工具仍保留 `_xxx` 命名以表"模块私有"。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from data_agent_baseline.config import AppConfig


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
    """生成默认 run_id：UTC 微秒级时间戳如 `20260420T123045123456Z`。

    UTC 保证跨时区不冲突；微秒精度避免连发 run 命中同一秒造成目录冲突。
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


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


def resolve_ratelimit_state_dir(config: AppConfig, effective_run_id: str) -> Path:
    """推导 `state_dir` 默认值：`<output_dir.parent>/ratelimit/<run_id>/`。

    默认 `output_dir = artifacts/runs`，所以默认状态目录是 `artifacts/ratelimit/<run_id>/`，
    与 `artifacts/runs/<run_id>/` 做兄弟目录——run 粒度隔离，避免跨 run 残留污染令牌桶。
    """
    return config.run.output_dir.parent / "ratelimit" / effective_run_id


def pin_key_on_config(config: AppConfig, *, task_index: int, effective_run_id: str) -> AppConfig:
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
        rl = replace(rl, state_dir=resolve_ratelimit_state_dir(config, effective_run_id))
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


def extract_system_prompt(steps: list[Any]) -> str | None:
    """从 trace.json 的 steps 中拣出 system prompt（仅看第一步的 prompt_messages）。

    设计动机：完整 prompt_messages 已落在每条 step 上（见 `react.py`），但 system
    prompt 在所有 turn 中都重复，每条 step 复制一份会让 trace.json 体积重复膨胀；
    在顶层抽出一份，UI 渲染头卡时直接读顶层即可。

    输入容错：
    - `steps` 不是 list、为空、第一步不是 dict、`prompt_messages` 缺失或不是 list：返回 None
    - prompt_messages 中无 system 角色或 system 角色的 content 不是字符串：返回 None

    公开为模块级 API（`extract_system_prompt`）：UI 层与测试都直接 import；
    见 `data_agent_baseline.ui.streamlit_app` 与 `tests/test_recorder_prompt_capture.py`。
    """
    if not steps:
        return None
    first = steps[0]
    if not isinstance(first, dict):
        return None
    prompt_messages = cast(Any, first).get("prompt_messages")
    if not isinstance(prompt_messages, list):
        return None
    messages = cast(list[Any], prompt_messages)
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        msg_dict = cast(dict[str, Any], msg)
        if msg_dict.get("role") != "system":
            continue
        content = msg_dict.get("content")
        if isinstance(content, str):
            return content
    return None


def _agent_state_to_payload(task_id: str, state: Any) -> dict[str, Any]:
    """把运行中的 `AgentRuntimeState` 快照成 trace.json 兼容的 dict。

    用于增量落盘：结构与 `AgentRunResult.to_dict()` 对齐，但因为尚未终止：
    - `answer` 在 answer 工具命中那一步之后由 callback 自然带上
    - `succeeded` 在 answer 已命中且无 failure_reason 时即可为 True；这样即使
      子进程卡在结果回传阶段，父进程也能从 partial trace 判定任务已完成

    兼容 typing：state 参数走 Any 是为了避免在 runner 层引入对 AgentRuntimeState 的硬依赖
    （保留 runner 与 agent 内部类型的解耦）。
    """
    answer = state.answer.to_dict() if state.answer is not None else None
    steps = [step.to_dict() for step in state.steps]
    return {
        "task_id": task_id,
        "answer": answer,
        "steps": steps,
        "failure_reason": state.failure_reason,
        "succeeded": answer is not None and state.failure_reason is None,
        "system_prompt": extract_system_prompt(steps),
    }


def make_partial_trace_writer(task_id: str, trace_path: Path):
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


def load_partial_trace(task_id: str, trace_path: Path) -> dict[str, Any] | None:
    """尝试从磁盘读出子进程写过的部分 trace；失败时返回 None。

    用于子进程超时/crash 后，从落盘快照里恢复 "死前最后一步" 的 steps 列表。
    """
    if not trace_path.exists():
        return None
    try:
        payload: Any = json.loads(trace_path.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    payload_dict = cast(dict[str, Any], payload)
    # 必须是同一个任务的 trace，避免跨任务误取
    if payload_dict.get("task_id") != task_id:
        return None
    return payload_dict


def _normalize_answer_payload(value: Any) -> dict[str, Any] | None:
    """把 trace 中的 answer-like 对象校验并规整成可写 CSV 的 payload。"""
    if not isinstance(value, dict):
        return None
    value_dict = cast(dict[str, Any], value)

    columns = value_dict.get("columns")
    rows = value_dict.get("rows")
    if (
        not isinstance(columns, list)
        or not columns
        or not all(isinstance(item, str) for item in cast(list[Any], columns))
    ):
        return None
    if not isinstance(rows, list):
        return None

    columns_list = cast(list[str], columns)
    rows_list = cast(list[Any], rows)
    normalized_rows: list[list[Any]] = []
    for row in rows_list:
        if not isinstance(row, list) or len(cast(list[Any], row)) != len(columns_list):
            return None
        normalized_rows.append(list(cast(list[Any], row)))
    return {"columns": list(columns_list), "rows": normalized_rows}


def _recover_answer_from_steps(steps: list[Any]) -> dict[str, Any] | None:
    """从 partial trace 的最后一个成功 answer step 中恢复答案表。"""
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        step_dict = cast(dict[str, Any], step)
        if step_dict.get("action") != "answer" or step_dict.get("ok") is not True:
            continue
        recovered = _normalize_answer_payload(step_dict.get("action_input"))
        if recovered is not None:
            return recovered
    return None


def timeout_payload(
    task_id: str, partial_trace_path: Path | None, timeout_seconds: int
) -> dict[str, Any]:
    """构造超时失败 payload，优先带上 partial trace 里的 steps。"""
    failure_reason = f"Task timed out after {timeout_seconds} seconds."
    return merge_partial_with_failure(task_id, partial_trace_path, failure_reason)


def crash_payload(
    task_id: str, partial_trace_path: Path | None, failure_reason: str
) -> dict[str, Any]:
    """构造子进程崩溃失败 payload，优先带上 partial trace 里的 steps。"""
    return merge_partial_with_failure(task_id, partial_trace_path, failure_reason)


def merge_partial_with_failure(
    task_id: str, partial_trace_path: Path | None, failure_reason: str
) -> dict[str, Any]:
    """合并规则：有 partial trace 就保留其 steps/answer。

    拿到的 partial 是子进程写的快照。若顶层 answer 缺失但 steps 里已有成功 answer
    调用，则从 action_input 恢复答案。只要恢复出合法 answer，就说明 terminal tool
    已经命中，父进程应按成功处理；这兜住了子进程已完成 answer 但卡在 Queue 结果回传
    阶段导致父进程超时的情况。
    """
    if partial_trace_path is not None:
        partial = load_partial_trace(task_id, partial_trace_path)
        if partial is not None and isinstance(partial.get("steps"), list):
            merged = dict(partial)
            steps = partial["steps"]
            merged["task_id"] = task_id
            answer = _normalize_answer_payload(partial.get("answer"))
            if answer is None:
                answer = _recover_answer_from_steps(steps)
            merged["answer"] = answer
            if answer is not None:
                merged["failure_reason"] = None
                merged["succeeded"] = True
            else:
                merged["failure_reason"] = failure_reason
                merged["succeeded"] = False
            return merged
    return _failure_run_result_payload(task_id, failure_reason)


def write_task_outputs(
    task_id: str, run_output_dir: Path, run_result: dict[str, Any]
) -> TaskRunArtifacts:
    """把单任务结果落盘为 trace.json（必出）+ prediction.csv（仅当有 answer）。

    trace.json 保留完整 ReAct 轨迹用于离线分析；prediction.csv 是评测面向的唯一产物。

    顶层 `system_prompt` 字段在落盘前从 steps[0].prompt_messages 中抽出（若不存在
    则保留 None）：UI 头卡渲染时直接读顶层而不必扫遍每个 step；step 内
    prompt_messages 仍保留完整内容用于离线回放。
    """
    task_output_dir = run_output_dir / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = task_output_dir / "trace.json"

    # subprocess 路径下 run_result 由 AgentRunResult.to_dict() 产生，没有 system_prompt 字段；
    # crash/timeout 路径下若 partial trace 已含 system_prompt 则保留之，缺失时再补
    if "system_prompt" not in run_result:
        steps = run_result.get("steps")
        if isinstance(steps, list):
            run_result["system_prompt"] = extract_system_prompt(cast(list[Any], steps))
        else:
            run_result["system_prompt"] = None

    _write_json(trace_path, run_result)

    # 只要 trace 中有结构合法的 answer 就写 prediction.csv；timeout/crash 路径如果
    # 能从 partial trace 恢复 answer，也会被归并为 succeeded=True。
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


def write_summary_json(summary_path: Path, payload: dict[str, Any]) -> None:
    """把整次 run 的汇总 dict 写成 summary.json，供 runner.run_benchmark 使用。"""
    _write_json(summary_path, payload)
