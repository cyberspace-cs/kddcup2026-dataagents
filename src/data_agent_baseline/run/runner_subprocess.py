"""单任务的子进程编排：核心循环 + 硬超时 + crash 兜底。

三层抽象：
- `run_single_task_core`:         在当前进程跑核心 ReAct 循环并返回 dict 结果
- `_run_single_task_in_subprocess`: 子进程入口，把结果/异常塞回 multiprocessing.Queue
- `run_single_task_with_timeout`: 父进程：fork + join(timeout) + crash 兜底

`run_single_task_core` 同时供 in-process 路径（测试注入 model/tools 时）和子进程路径
共用，避免两份逻辑分叉。
"""

from __future__ import annotations

import multiprocessing
from queue import Empty
from time import perf_counter
from typing import TYPE_CHECKING, Any

from data_agent_baseline.config import AppConfig
from data_agent_baseline.run.app import build_application
from data_agent_baseline.run.runner_artifacts import (
    crash_payload,
    timeout_payload,
)

if TYPE_CHECKING:
    from data_agent_baseline.agents.answer_verifier import AnswerVerifier
    from data_agent_baseline.agents.model import ModelAdapter
    from data_agent_baseline.tools.registry import ToolRegistry


def run_single_task_core(
    *,
    task_id: str,
    config: AppConfig,
    model: ModelAdapter | None = None,
    tools: ToolRegistry | None = None,
    answer_verifier: AnswerVerifier | None = None,
) -> dict[str, Any]:
    """纯业务逻辑：构造 AgentApp → 拿任务 → 跑循环 → 返回 dict。

    `model` / `tools` / `answer_verifier` 不为 None 时透传给 `build_application` 作为 override，
    让测试无需真的初始化 OpenAI 客户端；线上 None 时走默认 wiring。
    """
    _trace_ctx = None
    _processor = None
    if config.tracing is not None and config.tracing.enabled:
        from data_agent_baseline.tracing.context import TraceCtxManager
        from data_agent_baseline.tracing.processors import SimpleProcessor, SQLiteExporter
        from data_agent_baseline.tracing.setup import get_trace_provider
        from data_agent_baseline.tracing.store import SQLiteTraceStore

        _store = SQLiteTraceStore(config.tracing.db_path)
        _exporter = SQLiteExporter(_store)
        _processor = SimpleProcessor(_exporter)
        get_trace_provider().register_processor(_processor)

        _trace_ctx = TraceCtxManager(
            workflow_name="ReActAgent",
            run_id=config.run.run_id or "unknown",
            task_id=task_id,
        )
        _trace_ctx.__enter__()

    try:
        app = build_application(
            config,
            model=model,
            tools=tools,
            answer_verifier=answer_verifier,
        )

        if _trace_ctx is not None:
            from dataclasses import replace as _replace

            from data_agent_baseline.tracing.instrument import (
                TracedModelAdapter,
                TracedToolRegistry,
            )

            traced_verifier = app.answer_verifier
            if traced_verifier is not None:
                traced_verifier = _replace(
                    traced_verifier,
                    model=TracedModelAdapter(traced_verifier.model),
                )

            app = _replace(
                app,
                model_adapter=TracedModelAdapter(app.model_adapter),
                registry=TracedToolRegistry(app.registry),
                answer_verifier=traced_verifier,
            )

        task = app.dataset.get_task(task_id)
        try:
            from data_agent_baseline.etl.extractor import run_etl_for_task

            run_etl_for_task(task, app.model_adapter)
        except Exception:
            pass
        retrieval_index: dict[str, Any] | None = None
        try:
            from data_agent_baseline.retrieval.index import prepare_task_retrieval_index

            retrieval_index = prepare_task_retrieval_index(task, config.run.retrieval_index)
        except Exception as exc:
            if config.run.retrieval_index.enabled:
                retrieval_index = {
                    "enabled": True,
                    "built": False,
                    "cache_hit": False,
                    "task_id": task.task_id,
                    "error": str(exc),
                }

        _agent_span = None
        if _trace_ctx is not None:
            from data_agent_baseline.tracing.create import agent_span as _create_agent_span

            tool_names = list(app.registry.definitions.keys())
            _agent_span = _create_agent_span(
                name="ReActAgent",
                tools=tool_names,
                max_steps=app.agent_config.max_steps,
                protocol=app.protocol,
            )
            _agent_span.span_data.question = task.question
            _agent_span.start(mark_as_current=True)

        try:
            agent = app.new_agent()
            result = agent.run(task).to_dict()
        finally:
            if _agent_span is not None:
                _agent_span.finish(reset_current=True)

        if retrieval_index is not None:
            result["retrieval_index"] = retrieval_index
        return result
    finally:
        if _trace_ctx is not None:
            _trace_ctx.__exit__(None, None, None)
        if _processor is not None:
            from data_agent_baseline.tracing.setup import get_trace_provider as _gtp

            _gtp().unregister_processor(_processor)
            _processor.shutdown()


def _attach_retrieval_index_payload(
    payload: dict[str, Any], *, task_id: str, config: AppConfig
) -> dict[str, Any]:
    """Attach retrieval metadata to timeout/crash payloads when retrieval was enabled."""
    if not config.run.retrieval_index.enabled or "retrieval_index" in payload:
        return payload
    try:
        from data_agent_baseline.retrieval.index import load_task_index_metadata_by_id

        metadata = load_task_index_metadata_by_id(task_id)
    except Exception:
        metadata = None
    if isinstance(metadata, dict) and metadata.get("enabled"):
        payload["retrieval_index"] = metadata
    return payload


def _mark_running_trace_error(task_id: str, config: AppConfig, failure_reason: str) -> None:
    """Best-effort parent-process cleanup for traced children that cannot exit cleanly."""
    if config.tracing is None or not config.tracing.enabled:
        return
    try:
        from data_agent_baseline.tracing.store import SQLiteTraceStore

        store = SQLiteTraceStore(config.tracing.db_path)
        try:
            store.mark_running_trace_error(
                run_id=config.run.run_id or "unknown",
                task_id=task_id,
                failure_reason=failure_reason,
            )
        finally:
            store.close()
    except Exception:
        return


def _run_single_task_in_subprocess(
    task_id: str,
    config: AppConfig,
    queue: multiprocessing.Queue[Any],
) -> None:
    """子进程入口：跑核心逻辑并把结果/异常扔回主进程 Queue。

    注意 BaseException 而非 Exception——要兜住 KeyboardInterrupt / SystemExit，
    否则子进程退出码非 0 但 Queue 为空，主进程很难诊断。
    """
    try:
        queue.put(
            {
                "ok": True,
                "run_result": run_single_task_core(
                    task_id=task_id,
                    config=config,
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


def run_single_task_with_timeout(
    *,
    task_id: str,
    config: AppConfig,
) -> dict[str, Any]:
    """跨进程执行带硬超时的单任务。

    三类退出路径：
    1. 超时还活着 → terminate + 兜底 kill
    2. 退出但 Queue 为空 → 进程死于 OS kill / segfault
    3. Queue 有数据 → 按 ok/error 决定是 pass-through 还是再包一层失败

    `timeout_seconds <= 0` 时绕过子进程机制，直接在当前进程跑——
    用于显式不需要超时的单测场景。
    """
    timeout_seconds = config.run.task_timeout_seconds
    if timeout_seconds <= 0:
        return run_single_task_core(task_id=task_id, config=config)

    queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_single_task_in_subprocess,
        args=(task_id, config, queue),
    )
    process.start()

    # 不能先 join 再 get：子进程返回的 run_result 可能很大，multiprocessing.Queue 的
    # feeder 线程会在父进程不读 pipe 时卡住，导致"结果已经 put 进队列，
    # 但子进程迟迟不退出，最终被父进程按 timeout kill"。
    deadline = perf_counter() + timeout_seconds
    result: dict[str, Any] | None = None
    while True:
        remaining = deadline - perf_counter()
        if remaining <= 0:
            break
        try:
            result = queue.get(timeout=min(0.2, remaining))
            break
        except Empty:
            if not process.is_alive():
                break

    if result is None:
        try:
            result = queue.get_nowait()
        except Empty:
            result = None

    if result is not None:
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join()
        if result.get("ok"):
            # dict(...) 拷贝一次，脱离 Queue 的底层 pickle 壳
            return dict(result["run_result"])
        return _attach_retrieval_index_payload(
            crash_payload(
                task_id,
                f"Task failed with uncaught error: {result['error']}",
            ),
            task_id=task_id,
            config=config,
        )

    # Case 1: 超时 —— 先 SIGTERM 给清理机会，1 秒仍不退就 SIGKILL
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()
        payload = timeout_payload(task_id, timeout_seconds)
        _mark_running_trace_error(task_id, config, payload["failure_reason"])
        return _attach_retrieval_index_payload(
            payload,
            task_id=task_id,
            config=config,
        )

    # Case 2: 进程已退出但没 put 任何结果
    exit_code = process.exitcode
    if exit_code not in (None, 0):
        payload = crash_payload(
            task_id,
            f"Task exited unexpectedly with exit code {exit_code}.",
        )
        _mark_running_trace_error(task_id, config, payload["failure_reason"])
        return _attach_retrieval_index_payload(
            payload,
            task_id=task_id,
            config=config,
        )
    # 退出码 0 但 Queue 空：逻辑异常（理论上不会发生，留个明确错误信息）
    payload = crash_payload(task_id, "Task exited without returning a result.")
    _mark_running_trace_error(task_id, config, payload["failure_reason"])
    return _attach_retrieval_index_payload(
        payload,
        task_id=task_id,
        config=config,
    )
