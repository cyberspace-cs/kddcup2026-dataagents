"""只读文件系统工具集。

给 Agent 暴露有限的目录/文件读取能力，所有访问都限定在
`task.context_dir` 内（`resolve_context_path` 负责越界检查）。
这些工具都是幂等且无副作用的——Agent 可以反复调用而不影响外部状态。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask


def resolve_context_path(task: PublicTask, relative_path: str) -> Path:
    """把 Agent 传入的相对路径解析为绝对路径，并防止目录逃逸。

    安全模型：
    - 所有路径必须落在 `task.context_dir` 之内
    - 传 `../` 或绝对路径会被 `candidate.resolve()` 规整后触发校验失败
    - 不存在的路径直接抛 FileNotFoundError，避免 Agent 拿到空结果误判
    """
    # resolve() 会同时处理软链接和 ".."，给一个规范化的绝对路径
    candidate = (task.context_dir / relative_path).resolve()
    context_root = task.context_dir.resolve()
    # 同时允许 candidate == root（Agent 想列出 context 本身）与 candidate 在 root 下
    if context_root not in candidate.parents and candidate != context_root:
        raise ValueError(f"Path escapes context dir: {relative_path}")
    if not candidate.exists():
        raise FileNotFoundError(f"Missing context asset: {relative_path}")
    return candidate


def list_context_tree(task: PublicTask, *, max_depth: int = 4) -> dict[str, object]:
    """按深度优先遍历 `context/` 下的文件/目录树。

    - `max_depth` 从 1 起算（1 表示只列直接子项）
    - 同目录下 **目录先于文件**（排序键用 `(is_file, name)`），便于模型先看结构再下钻
    - 返回相对路径（POSIX 风格），跨平台一致
    """
    entries: list[dict[str, object]] = []

    def walk(path: Path, depth: int) -> None:
        # 超出深度直接返回，避免巨型目录导致 observation 过大
        if depth > max_depth:
            return
        # is_file=False（即目录）排在前面；同层按名字字典序
        for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name)):
            rel_path = child.relative_to(task.context_dir).as_posix()
            entries.append(
                {
                    "path": rel_path,
                    "kind": "dir" if child.is_dir() else "file",
                    # 目录没有有意义的 size，用 None 占位
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
            # 递归下钻，深度 +1
            if child.is_dir():
                walk(child, depth + 1)

    walk(task.context_dir, 1)
    return {
        "root": str(task.context_dir),
        "entries": entries,
    }


def read_csv_preview(task: PublicTask, relative_path: str, *, max_rows: int = 20) -> dict[str, object]:
    """读取 CSV 的表头与前 N 行。

    用 `csv.reader` 逐行读：不依赖 pandas，避免超大 CSV 引入内存风险。
    同时返回 `row_count` 让 Agent 知道是否还有更多数据没看到。
    """
    path = resolve_context_path(task, relative_path)
    # newline="" 按 csv 模块要求传，以正确处理嵌入换行的单元格
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        # 一次读全部到内存：CSV 在 benchmark 数据集里通常不大，换取代码简洁
        rows = list(reader)

    # 空文件特判：避免 rows[0] IndexError
    if not rows:
        return {
            "path": relative_path,
            "columns": [],
            "rows": [],
            "row_count": 0,
        }

    header = rows[0]
    data_rows = rows[1:]
    return {
        "path": relative_path,
        "columns": header,
        "rows": data_rows[:max_rows],  # 仅前 N 行，防止 observation 膨胀
        "row_count": len(data_rows),   # 实际总行数：让模型评估是否需要更深查询
    }


def read_json_preview(task: PublicTask, relative_path: str, *, max_chars: int = 4000) -> dict[str, object]:
    """读取 JSON 并按字符数截断预览。

    先 loads → dumps：一来验证 JSON 合法性，二来统一缩进让模型更易阅读。
    ensure_ascii=False 保留中文原样字符，减少 token 消耗。
    """
    path = resolve_context_path(task, relative_path)
    payload = json.loads(path.read_text())
    preview = json.dumps(payload, ensure_ascii=False, indent=2)
    return {
        "path": relative_path,
        "preview": preview[:max_chars],
        # truncated=True 提示模型需要换工具（execute_python）细读
        "truncated": len(preview) > max_chars,
    }


def read_doc_preview(task: PublicTask, relative_path: str, *, max_chars: int = 4000) -> dict[str, object]:
    """读取任意文本类文档（md / txt）并按字符数截断。

    errors="replace" 把非法字节替换为 U+FFFD，避免个别文件乱码让整个调用崩溃。
    """
    path = resolve_context_path(task, relative_path)
    text = path.read_text(errors="replace")
    return {
        "path": relative_path,
        "preview": text[:max_chars],
        "truncated": len(text) > max_chars,
    }
