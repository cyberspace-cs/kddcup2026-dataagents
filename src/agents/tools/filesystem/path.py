"""路径解析与目录树遍历。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.benchmark.schema import PublicTask
from agents.config import ETL_SCRATCH_ROOT


def resolve_context_path(task: PublicTask, relative_path: str) -> Path:
    """把 Agent 传入的相对路径解析为绝对路径，并防止目录逃逸。

    安全模型：
    - 所有路径必须落在 `task.context_dir` 或 ETL scratch 目录之内
    - 传 `../` 或绝对路径会被 `candidate.resolve()` 规整后触发校验失败
    - 不存在的路径直接抛 FileNotFoundError，避免 Agent 拿到空结果误判
    """
    raw = Path(relative_path)
    if raw.is_absolute():
        candidate = raw.resolve()
        etl_root = (ETL_SCRATCH_ROOT / task.task_id).resolve()
        if etl_root in candidate.parents or candidate == etl_root:
            if not candidate.exists():
                raise FileNotFoundError(f"Missing ETL asset: {relative_path}")
            return candidate

    candidate = (task.context_dir / relative_path).resolve()
    context_root = task.context_dir.resolve()
    if context_root not in candidate.parents and candidate != context_root:
        raise ValueError(f"Path escapes context dir: {relative_path}")
    if not candidate.exists():
        raise FileNotFoundError(f"Missing context asset: {relative_path}")
    return candidate


def list_context_tree(task: PublicTask, *, max_depth: int = 4) -> dict[str, Any]:
    """按深度优先遍历 `context/` 下的文件/目录树。

    - `max_depth` 从 1 起算（1 表示只列直接子项）
    - 同目录下 **目录先于文件**（排序键用 `(is_file, name)`），便于模型先看结构再下钻
    - 返回相对路径（POSIX 风格），跨平台一致
    """
    entries: list[dict[str, Any]] = []

    def walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name)):
            rel_path = child.relative_to(task.context_dir).as_posix()
            entries.append(
                {
                    "path": rel_path,
                    "kind": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
            if child.is_dir():
                walk(child, depth + 1)

    walk(task.context_dir, 1)
    return {
        "root": str(task.context_dir),
        "entries": entries,
    }
