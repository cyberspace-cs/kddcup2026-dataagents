"""Python 代码执行工具（沙箱化）。

设计要点：
- **子进程执行**：用 `multiprocessing.Process` 隔离，代码里的副作用（全局变量、
  os.chdir、sys 修改）不污染主进程
- **硬超时**：`process.join(timeout)`；超时先 terminate，兜底 kill，返回可读错误
- **流捕获**：用文件 fd + dup2 替换 stdout/stderr，能抓到 C 扩展（numpy/pandas）
  通过底层 fd 写出的内容，而不只是 Python 层的 print
- **工作目录**：切到 `task.context_dir`，让 Agent 可以直接 open('file.csv') 等

⚠️ 重要：本工具**不是真正的安全沙箱**——没有 seccomp/namespace 隔离。
    如果评测环境允许任意代码执行，那就等同于开放 Agent 进程完整权限。
    仅通过资源/时间隔离防护 "失控卡死" 级别的问题。
"""

from __future__ import annotations

import contextlib
import io
import multiprocessing
import os
import sys
import tempfile
import traceback
from pathlib import Path
from queue import Empty
from typing import Any

# 单股 stdout/stderr 的硬上限（字节）。64KB ≈ 16K tokens（按 cl100k 经验 ~4 字节/token），
# 远低于典型 context 预算（≥128K），既给出足够的调试信息又防止单次 observation 把
# 模型 prompt 撑过上限——失败案例：run 20260430-025 task_2 一次 `print(records)` 输出
# 5.5MB stdout 后续 step 全部 400 InvalidParameter 至 max_steps。
_PYTHON_EXEC_STREAM_CAP_BYTES = 64 * 1024
# 截断时保留首部字节数（看到 traceback / 第一行 print 等开头信号）
_PYTHON_EXEC_STREAM_HEAD_BYTES = 32 * 1024
# 截断时保留尾部字节数（看到最终结果 / 最后一条 error 等结尾信号）
_PYTHON_EXEC_STREAM_TAIL_BYTES = 32 * 1024


@contextlib.contextmanager
def _capture_process_streams(stdout_path: Path, stderr_path: Path):
    """上下文管理器：把当前进程的 stdout/stderr 重定向到文件。

    做两层替换，是为了同时捕获：
    1. Python 层：`sys.stdout = ...`（普通 `print` 会写这里）
    2. C 层：`os.dup2` 改 fd 1/2（底层 C 扩展的 write/printf）
    退出时两层都按逆序复原，保证本上下文不会破坏主进程日志输出。
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    # 先备份 fd 1/2，以便最后 dup2 回来
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)

    with stdout_path.open("w+b") as stdout_file, stderr_path.open("w+b") as stderr_file:
        try:
            # 先 flush 原有缓冲，避免旧数据落到新文件里
            if original_stdout is not None:
                original_stdout.flush()
            if original_stderr is not None:
                original_stderr.flush()

            # 把 fd 1/2 指向文件：C 层输出从此刻开始被捕获
            os.dup2(stdout_file.fileno(), 1)
            os.dup2(stderr_file.fileno(), 2)

            # 与原始流保持一致的编码；None 时退化为 utf-8
            stdout_encoding = getattr(original_stdout, "encoding", None) or "utf-8"
            stderr_encoding = getattr(original_stderr, "encoding", None) or "utf-8"

            # 再替换 Python 层的 sys.stdout/stderr 为对同一 fd 的 TextIOWrapper
            # line_buffering + write_through 让 print 不依赖程序退出自动 flush
            sys.stdout = io.TextIOWrapper(
                os.fdopen(os.dup(1), "wb"),
                encoding=stdout_encoding,
                errors="replace",
                line_buffering=True,
                write_through=True,
            )
            sys.stderr = io.TextIOWrapper(
                os.fdopen(os.dup(2), "wb"),
                encoding=stderr_encoding,
                errors="replace",
                line_buffering=True,
                write_through=True,
            )
            yield
        finally:
            # 复原前先 flush，防止 close 时数据丢失
            if sys.stdout is not None:
                sys.stdout.flush()
            if sys.stderr is not None:
                sys.stderr.flush()

            # 关闭我们新建的 TextIOWrapper，但别误关 original_stdout
            if sys.stdout is not None and sys.stdout is not original_stdout:
                sys.stdout.close()
            if sys.stderr is not None and sys.stderr is not original_stderr:
                sys.stderr.close()

            # 恢复 Python 层引用
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            # 把 fd 1/2 dup 回备份，再关掉备份句柄
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


def _read_captured_stream(path: Path) -> str:
    """读取被捕获的 stdout/stderr 文件内容；超过 _PYTHON_EXEC_STREAM_CAP_BYTES 走首尾保留截断。

    `errors="replace"` 防止非 UTF-8 字节（如某些 C 扩展的二进制日志）导致崩溃。
    截断策略：保留头 32KB + 尾 32KB + 中间一行 marker `[TRUNCATED: N bytes elided]`，
    UTF-8 多字节字符在 cut-point 被切断时由 `errors="replace"` 用 U+FFFD 兜底——可读性
    不打折，对模型推理无影响。

    动机：避免 agent 一次 `print(huge_object)` 让 observation 撑过模型 context（详见
    run 20260430-025 task_2 的级联失败：5.5MB stdout 进入对话历史后所有后续 step 都
    400 InvalidParameter）。stdout / stderr 各自独立调用本函数，互相不挤占 64KB 预算。

    用 `path.stat().st_size` 提前判定尺寸，避免把 GB 级文件读进内存再截断。
    子进程已 join 退出后才被调用，stdout/stderr 临时文件无并发写者，stat 与 open
    之间无尺寸跳变。
    """
    size = path.stat().st_size
    if size <= _PYTHON_EXEC_STREAM_CAP_BYTES:
        return path.read_text(encoding="utf-8", errors="replace")

    elided = size - (_PYTHON_EXEC_STREAM_HEAD_BYTES + _PYTHON_EXEC_STREAM_TAIL_BYTES)
    with path.open("rb") as fp:
        head_bytes = fp.read(_PYTHON_EXEC_STREAM_HEAD_BYTES)
        fp.seek(size - _PYTHON_EXEC_STREAM_TAIL_BYTES)
        tail_bytes = fp.read(_PYTHON_EXEC_STREAM_TAIL_BYTES)
    head = head_bytes.decode("utf-8", errors="replace")
    tail = tail_bytes.decode("utf-8", errors="replace")
    return f"{head}\n... [TRUNCATED: {elided} bytes elided] ...\n{tail}"


def _run_python_code(
    context_root: str,
    code: str,
    stdout_path: str,
    stderr_path: str,
    answer_dir: str,
    queue: multiprocessing.Queue[Any],
) -> None:
    """子进程入口：切换 CWD，设置 exec 命名空间，执行代码。

    预置两个便捷变量 `context_root` / `Path` 到 `exec` 命名空间，让模型
    不必自己 `import pathlib` 也能访问 context 目录。
    `DABENCH_ANSWER_DIR` 在子进程内显式 set，避免依赖父进程 env 在 ThreadPoolExecutor
    路径下被多个 task 串扰；模型按 prompt 指引把 answer artifact CSV 写到该目录后
    用 `answer({"from_csv": <abs_path>})` 提交。
    """
    # 显式 set 而非 setdefault：每次调用都用本任务的目录覆盖（thread-safe by spawn 隔离）
    os.environ["DABENCH_ANSWER_DIR"] = answer_dir
    namespace: dict[str, Any] = {
        # __builtins__ 必须显式给，否则 exec 出来的代码没有 print/open 等
        "__builtins__": __builtins__,
        # 让 __name__=="__main__" 的常见代码片段也能正常工作
        "__name__": "__main__",
        # context 根目录的字符串，方便 os.path.join
        "context_root": context_root,
        # 导出 Path 类，模型可 `Path(context_root) / "foo.csv"`
        "Path": Path,
    }
    resolved_stdout_path = Path(stdout_path)
    resolved_stderr_path = Path(stderr_path)

    try:
        # CWD 切到 context_dir：Agent 代码可以用相对路径直接 open 文件
        os.chdir(context_root)
        with _capture_process_streams(resolved_stdout_path, resolved_stderr_path):
            # globals/locals 合并为同一个 dict：允许代码中定义的函数互相引用
            exec(code, namespace, namespace)
        # 顺利执行完：仅回传 success 标志；stdout/stderr 由主进程读文件取
        queue.put({"success": True})
    except BaseException as exc:
        # 回传错误文本和 traceback，让主进程能在 observation 里暴露给模型
        queue.put(
            {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def execute_python_code(
    context_root: Path,
    code: str,
    *,
    answer_dir: Path,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """主入口：在子进程中跑代码，带超时和结果回收。

    返回结构（各 case 保持同形）：
    - {success: bool, output: str, stderr: str, [error, traceback]}
    这样上层 ReAct 循环始终能拿到 output/stderr 字段用于展示。

    `answer_dir` 必填：子进程会把它 set 到 `os.environ['DABENCH_ANSWER_DIR']`，
    给 artifact handoff 路径（answer({from_csv}))一个稳定的写盘位置；本函数会
    `mkdir(parents=True, exist_ok=True)` 确保目录存在再 spawn 子进程。
    """
    resolved_context_root = context_root.resolve()
    answer_dir.mkdir(parents=True, exist_ok=True)
    resolved_answer_dir = answer_dir.resolve()
    # 用临时目录存 stdout/stderr，进程结束后自动清理
    with tempfile.TemporaryDirectory() as temp_dir:
        stdout_path = Path(temp_dir) / "stdout.txt"
        stderr_path = Path(temp_dir) / "stderr.txt"
        # 预创建空文件：子进程 dup2 到不存在的文件会失败
        stdout_path.write_text("")
        stderr_path.write_text("")

        # 用 Queue 回传结构化结果（success/error/traceback）；stdout/stderr 走文件
        queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_run_python_code,
            args=(
                resolved_context_root.as_posix(),
                code,
                stdout_path.as_posix(),
                stderr_path.as_posix(),
                resolved_answer_dir.as_posix(),
                queue,
            ),
        )
        process.start()
        # 阻塞到超时或子进程自行退出
        process.join(timeout_seconds)

        # Case 1: 超时还没退出 → 先 terminate（SIGTERM），兜底会直接 kill
        if process.is_alive():
            process.terminate()
            process.join()
            return {
                "success": False,
                "output": _read_captured_stream(stdout_path),
                "stderr": _read_captured_stream(stderr_path),
                "error": f"Python execution timed out after {timeout_seconds} seconds.",
            }

        # 进程已退出。用 get_nowait 直接判定有无结果——multiprocessing.Queue.empty()
        # 受后台 feeder 线程影响，跨进程语义不可靠（官方文档明示 approximate）。
        try:
            result = queue.get_nowait()
        except Empty:
            # OS 层 kill / segfault：进程死了但啥也没塞进队列
            return {
                "success": False,
                "output": _read_captured_stream(stdout_path),
                "stderr": _read_captured_stream(stderr_path),
                "error": "Python execution exited without returning a result.",
            }
        # 无论成功/失败，都补充 stdout/stderr 文本，保持字段齐整
        result["output"] = _read_captured_stream(stdout_path)
        result["stderr"] = _read_captured_stream(stderr_path)
        return result
