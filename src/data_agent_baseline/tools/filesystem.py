"""只读文件系统工具集。

给 Agent 暴露有限的目录/文件读取能力，所有访问都限定在
`task.context_dir` 内（`resolve_context_path` 负责越界检查）。
这些工具都是幂等且无副作用的——Agent 可以反复调用而不影响外部状态。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.schema import PublicTask

# inspect_files 一次性扫描的文件数上限：超过即截断，避免单次 observation 爆炸。
_INSPECT_FILES_MAX_FILES = 100
# 单文件 introspection 的字节上限（仅对 JSON 生效）：5MB 之上直接跳过 loads。
_INSPECT_FILES_JSON_SIZE_LIMIT = 5 * 1024 * 1024
# inspect_files dtype 推断扫描行数；多了也无谓——只为给模型一个粗略提示。
_INSPECT_FILES_DTYPE_SAMPLE = 5
# read_csv 预览返回的 head / tail 行数（固定不让模型自调，避免被当作"取数据"工具）：
# - head 给前缀样本（多数 CSV 把表头敏感的行放前）
# - tail 给后缀样本（catch sort-order surprise，比如 task_22 那条第 36 行的 dues）
# 全量过滤/聚合一律走 execute_python，read_csv 只做 schema + sample。
_READ_CSV_HEAD_ROWS = 20
_READ_CSV_TAIL_ROWS = 5
# 文档类（md/txt）保留的预览行数。
_INSPECT_FILES_DOC_HEAD_LINES = 3
# inspect_files 识别的 sqlite 扩展名集合。
_SQLITE_EXTS = frozenset({".sqlite", ".sqlite3", ".db"})
# 直接走 read_csv 流的 CSV 扩展。
_CSV_EXTS = frozenset({".csv"})
# JSON 系（含 jsonl 暂不支持，留给 execute_python）。
_JSON_EXTS = frozenset({".json"})
# 文档类纯文本扩展。
_DOC_EXTS = frozenset({".md", ".txt"})
# inspect_files 已识别但本工具未原生支持的格式 → 提示模型转 execute_python。
_UNSUPPORTED_HINT_BY_EXT: dict[str, str] = {
    ".xlsx": "Use execute_python with pandas/openpyxl to read .xlsx.",
    ".xls": "Use execute_python with pandas/xlrd to read legacy .xls.",
    ".parquet": "Use execute_python with pandas/pyarrow to read .parquet.",
    ".feather": "Use execute_python with pandas/pyarrow to read .feather.",
    ".jsonl": "Use execute_python or read_doc to stream JSON Lines.",
    ".ndjson": "Use execute_python or read_doc to stream NDJSON.",
    ".tsv": "Use execute_python with pandas to read .tsv.",
}


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


def list_context_tree(task: PublicTask, *, max_depth: int = 4) -> dict[str, Any]:
    """按深度优先遍历 `context/` 下的文件/目录树。

    - `max_depth` 从 1 起算（1 表示只列直接子项）
    - 同目录下 **目录先于文件**（排序键用 `(is_file, name)`），便于模型先看结构再下钻
    - 返回相对路径（POSIX 风格），跨平台一致
    """
    entries: list[dict[str, Any]] = []

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


def read_csv_preview(task: PublicTask, relative_path: str) -> dict[str, Any]:
    """返回 CSV 的 schema + head/tail 样本（**不是**取数据工具）。

    设计意图：
    - 此前 `read_csv` 默认返前 20 行，模型容易误以为"我已经看到全部数据"——
      task_22 的 36 行 income.csv 第 36 行 `2019-09-12` 那条 Dues 记录就因此被静默丢失。
    - 改造后语义收窄到 inspect_csv：返回 `columns + dtypes(approx) + row_count + head + tail`，
      永远只暴露样本；任何过滤/聚合/全扫必须走 execute_python。
    - 固定 head=20 / tail=5，**不**给模型 max_rows 旋钮——避免它通过加大窗口
      绕过样本语义。

    `tail` 严格定义为"head 之后剩余行的最后 N 行"，与 head 不重叠：
    - row_count <= head_rows 时 tail 为空（head 已覆盖全部，无后缀可看）
    - row_count > head_rows 时 tail = data_rows[max(head_rows, row_count - tail_rows):]

    实现仍用 `csv.reader` 逐行读：避免 pandas 给热路径塞依赖。
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
            "dtypes": [],
            "row_count": 0,
            "head": [],
            "tail": [],
        }

    header = rows[0]
    data_rows = rows[1:]
    row_count = len(data_rows)
    head_rows = data_rows[:_READ_CSV_HEAD_ROWS]
    # tail 起点 = max(head 边界, row_count - tail 容量)；
    # 当 row_count <= head 容量时该起点 == row_count，slice 自然得到空列表。
    tail_start = max(_READ_CSV_HEAD_ROWS, row_count - _READ_CSV_TAIL_ROWS)
    tail_rows = data_rows[tail_start:]

    # dtype 推断：用 head 样本（与 inspect_files 同套 _classify_dtype），
    # 标 confidence=approx 提示模型不要把 int/float/str 当作权威结论。
    column_count = len(header)
    dtypes: list[str] = []
    sample_for_dtype = head_rows[:_INSPECT_FILES_DTYPE_SAMPLE]
    for column_index in range(column_count):
        column_values = [
            row[column_index] if column_index < len(row) else "" for row in sample_for_dtype
        ]
        dtypes.append(_classify_dtype(column_values))

    return {
        "path": relative_path,
        "columns": header,
        "dtypes": dtypes,
        "row_count": row_count,
        "head": head_rows,
        "tail": tail_rows,
    }


def read_json_preview(
    task: PublicTask, relative_path: str, *, max_chars: int = 4000
) -> dict[str, Any]:
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


def _normalize_line_count(line_count: int) -> int:
    """把读取行数限制在稳定的 observation 范围内。"""
    return min(300, max(1, int(line_count)))


def read_doc_preview(
    task: PublicTask,
    relative_path: str,
    *,
    start_line: int = 1,
    line_count: int = 80,
    keyword: str | None = None,
) -> dict[str, Any]:
    """按行读取任意文本类文档（md / txt），可从指定行开始搜索关键词。

    errors="replace" 把非法字节替换为 U+FFFD，避免个别文件乱码让整个调用崩溃。
    返回的 `preview` 不加行号，避免污染文档正文语义；行范围通过结构化字段表达。
    """
    path = resolve_context_path(task, relative_path)
    lines = path.read_text(errors="replace").splitlines()
    total_lines = len(lines)

    if total_lines == 0:
        return {
            "path": relative_path,
            "total_lines": 0,
            "returned_range": None,
            "has_more": False,
            "keyword_match": None,
            "preview": "",
        }

    normalized_start_line = max(1, int(start_line))
    normalized_line_count = _normalize_line_count(line_count)

    start_index = normalized_start_line - 1
    keyword_match: dict[str, Any] | None = None

    if keyword is not None and keyword != "":
        lowered_keyword = keyword.lower()
        match_index = next(
            (
                index
                for index in range(start_index, total_lines)
                if lowered_keyword in lines[index].lower()
            ),
            None,
        )
        if match_index is None:
            keyword_match = {
                "keyword": keyword,
                "found": False,
                "line": None,
            }
        else:
            context_before = normalized_line_count // 3
            start_index = max(0, match_index - context_before)
            keyword_match = {
                "keyword": keyword,
                "found": True,
                "line": match_index + 1,
            }

    end_index = min(total_lines, start_index + normalized_line_count)
    selected_lines = lines[start_index:end_index]
    returned_range = None
    if selected_lines:
        returned_range = {
            "start_line": start_index + 1,
            "end_line": end_index,
        }

    return {
        "path": relative_path,
        "total_lines": total_lines,
        "returned_range": returned_range,
        "has_more": end_index < total_lines,
        "keyword_match": keyword_match,
        "preview": "\n".join(selected_lines),
    }


def _classify_dtype(values: list[str]) -> str:
    """根据列前 N 个非空样本推断粗 dtype（int/float/bool/str/empty）。

    刻意做"前 5 行"的 quick-and-dirty 推断：模型只需要一个起点判断即可，
    更精确的类型由 execute_python + pandas 完成。
    """
    non_empty = [v for v in values if v != ""]
    if not non_empty:
        return "empty"
    bool_set = {"true", "false"}
    if all(v.lower() in bool_set for v in non_empty):
        return "bool"
    all_int = True
    for v in non_empty:
        candidate = v.lstrip("-+")
        if not candidate.isdigit():
            all_int = False
            break
    if all_int:
        return "int"
    all_float = True
    for v in non_empty:
        try:
            float(v)
        except ValueError:
            all_float = False
            break
    if all_float:
        return "float"
    return "str"


def _summarize_csv(path: Path) -> dict[str, Any]:
    """流式读取 CSV：拿 header + 前 N 行做 dtype 推断 + 累加总行数。

    用 `csv.reader` 逐行迭代，避免一次性 list(reader) 撑爆内存——
    inspect_files 会被一次性扫描整个 context，单文件大小不可控。
    """
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            # 空文件：没有 header 也没有数据行
            return {
                "columns": [],
                "dtypes": [],
                "row_count": 0,
                "dtypes_confidence": "approx",
            }
        sample_rows: list[list[str]] = []
        row_count = 0
        for row in reader:
            row_count += 1
            if len(sample_rows) < _INSPECT_FILES_DTYPE_SAMPLE:
                sample_rows.append(row)
    # 按列重组样本：每列取前 N 个值，长度不齐时缺位用空串
    column_count = len(header)
    dtypes: list[str] = []
    for column_index in range(column_count):
        column_values = [
            row[column_index] if column_index < len(row) else "" for row in sample_rows
        ]
        dtypes.append(_classify_dtype(column_values))
    return {
        "columns": header,
        "dtypes": dtypes,
        "row_count": row_count,
        # dtype 基于前 5 行；返回字段提示模型不要把推断当真值
        "dtypes_confidence": "approx",
    }


def _summarize_json(path: Path) -> dict[str, Any]:
    """提取 JSON top-level 形态（object/array/scalar）+ 浅层 schema。

    巨型文件（> 5MB）直接跳过 loads，返回 truncated_at: "filesystem"——
    模型可以通过 execute_python 流式处理。
    """
    size = path.stat().st_size
    if size > _INSPECT_FILES_JSON_SIZE_LIMIT:
        return {
            "kind": "unknown",
            "truncated_at": "filesystem",
            "size": size,
        }
    payload = json.loads(path.read_text(errors="replace"))
    if isinstance(payload, dict):
        keys: list[str] = list(payload.keys())
        sample_kinds = {key: type(payload[key]).__name__ for key in keys[:10]}
        return {
            "kind": "object",
            "keys": keys[:20],
            "key_count": len(keys),
            "sample_value_kinds": sample_kinds,
        }
    if isinstance(payload, list):
        first_kind: str | None = None
        if payload:
            first_kind = type(payload[0]).__name__
        return {
            "kind": "array",
            "length": len(payload),
            "first_item_kind": first_kind,
        }
    # 标量（int/float/str/bool/None）
    return {
        "kind": "scalar",
        "value_kind": type(payload).__name__,
    }


def _summarize_doc(path: Path) -> dict[str, Any]:
    """文本文档：行数 + 前 3 行预览。"""
    lines = path.read_text(errors="replace").splitlines()
    return {
        "line_count": len(lines),
        "head": lines[:_INSPECT_FILES_DOC_HEAD_LINES],
    }


def _summarize_sqlite(path: Path) -> dict[str, Any]:
    """sqlite/db：返回 `{"tables": {table_name: [col, ...]}}` 形态。

    `inspect_sqlite_schema` 会回传完整 CREATE SQL，inspect_files 全量扫描时太
    费 token。这里单连接直接走 sqlite_master + PRAGMA table_info 抽列名，
    显式 close 连接避免 ResourceWarning。
    """
    import sqlite3

    tables: dict[str, list[str]] = {}
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
        for (name,) in table_rows:
            if not isinstance(name, str):
                continue
            # PRAGMA table_info 不能参数化表名；name 来自 sqlite_master 不存在
            # 外部注入面，但仍双引号包裹以适配带特殊字符的表名
            quoted = name.replace('"', '""')
            cursor = conn.execute(f'PRAGMA table_info("{quoted}")')
            columns = [row[1] for row in cursor.fetchall()]
            tables[name] = columns
    finally:
        conn.close()
    return {"tables": tables}


def _format_for_inspect(rel_path: str, abs_path: Path) -> dict[str, Any]:
    """单文件 introspection 的统一入口：按扩展名分流到具体 summarizer。"""
    suffix = abs_path.suffix.lower()
    size = abs_path.stat().st_size
    base: dict[str, Any] = {
        "path": rel_path,
        "size": size,
    }
    if suffix in _CSV_EXTS:
        base["format"] = "csv"
        base["supported"] = True
        base["schema"] = _summarize_csv(abs_path)
        return base
    if suffix in _JSON_EXTS:
        base["format"] = "json"
        base["supported"] = True
        base["schema"] = _summarize_json(abs_path)
        return base
    if suffix in _SQLITE_EXTS:
        # sqlite/sqlite3/db 共用同一个 introspection；format 字段统一报 "sqlite"
        base["format"] = "sqlite"
        base["supported"] = True
        base["schema"] = _summarize_sqlite(abs_path)
        return base
    if suffix in _DOC_EXTS:
        # md/txt 用同一个 summarizer；format 字段保留具体扩展，方便模型选 read_doc 时直观
        base["format"] = "markdown" if suffix == ".md" else "text"
        base["supported"] = True
        base["schema"] = _summarize_doc(abs_path)
        return base
    base["format"] = suffix.lstrip(".") or "unknown"
    base["supported"] = False
    base["reason"] = _UNSUPPORTED_HINT_BY_EXT.get(
        suffix,
        "Unsupported by inspect_files; use execute_python or another preview tool.",
    )
    return base


def inspect_context_files(task: PublicTask) -> dict[str, Any]:
    """一次性扫描 context_dir 下所有文件的轻量 schema。

    替代"先列目录再对每个文件挨个 read_*"的 N+1 模式：
    遍历整棵目录树，按扩展名分流到对应 introspection（csv/json/sqlite/md/txt），
    其它格式（xlsx/parquet/...）按 supported=false 占位返回，提示模型转 execute_python。

    保护上限 100 个文件：超出后 `truncated=True`，仍按字典序前 100 项返回。
    """
    context_root = task.context_dir.resolve()
    files_summary: list[dict[str, Any]] = []
    unsupported_formats: set[str] = set()

    # 收集所有文件（不含目录），按相对路径字典序排序，保证输出稳定
    candidates: list[tuple[str, Path]] = []
    for child in context_root.rglob("*"):
        if not child.is_file():
            continue
        rel = child.relative_to(context_root).as_posix()
        candidates.append((rel, child))
    candidates.sort(key=lambda item: item[0])

    truncated = len(candidates) > _INSPECT_FILES_MAX_FILES
    for rel_path, abs_path in candidates[:_INSPECT_FILES_MAX_FILES]:
        entry = _format_for_inspect(rel_path, abs_path)
        if not entry["supported"]:
            unsupported_formats.add(str(entry["format"]))
        files_summary.append(entry)

    return {
        "context_root": str(context_root),
        "files": files_summary,
        "unsupported_formats": sorted(unsupported_formats),
        "file_count": len(files_summary),
        "truncated": truncated,
    }
