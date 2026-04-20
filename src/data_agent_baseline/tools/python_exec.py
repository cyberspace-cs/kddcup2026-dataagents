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
from typing import Any


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
            if sys.stdout is not original_stdout:
                sys.stdout.close()
            if sys.stderr is not original_stderr:
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
    """读取被捕获的 stdout/stderr 文件内容。

    errors="replace" 防止非 UTF-8 字节（如某些 C 扩展的二进制日志）导致崩溃。
    """
    return path.read_text(encoding="utf-8", errors="replace")


def _run_python_code(
    context_root: str,
    code: str,
    stdout_path: str,
    stderr_path: str,
    queue: multiprocessing.Queue[Any],
) -> None:
    """子进程入口：切换 CWD，设置 exec 命名空间，执行代码。

    预置两个便捷变量 `context_root` / `Path` 到 `exec` 命名空间，让模型
    不必自己 `import pathlib` 也能访问 context 目录。
    """
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
    except BaseException as exc:  # noqa: BLE001 — 刻意捕获所有异常，包括 SystemExit
        # 回传错误文本和 traceback，让主进程能在 observation 里暴露给模型
        queue.put(
            {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def execute_python_code(context_root: Path, code: str, *, timeout_seconds: int = 30) -> dict[str, Any]:
    """主入口：在子进程中跑代码，带超时和结果回收。

    返回结构（各 case 保持同形）：
    - {success: bool, output: str, stderr: str, [error, traceback]}
    这样上层 ReAct 循环始终能拿到 output/stderr 字段用于展示。
    """
    resolved_context_root = context_root.resolve()
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

        # Case 2: 进程已退出，但 Queue 是空的 → 可能是 OS 层 kill 或 segfault
        if queue.empty():
            return {
                "success": False,
                "output": _read_captured_stream(stdout_path),
                "stderr": _read_captured_stream(stderr_path),
                "error": "Python execution exited without returning a result.",
            }

        # Case 3: 正常有返回（可能 success=True 或 False+traceback）
        result = queue.get()
        # 无论成功/失败，都补充 stdout/stderr 文本，保持字段齐整
        result["output"] = _read_captured_stream(stdout_path)
        result["stderr"] = _read_captured_stream(stderr_path)
        return result
