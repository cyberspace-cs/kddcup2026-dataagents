"""只读文件系统工具集。

给 Agent 暴露有限的目录/文件读取能力，所有访问都限定在
`task.context_dir` 内（`resolve_context_path` 负责越界检查）。
这些工具都是幂等且无副作用的——Agent 可以反复调用而不影响外部状态。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

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
# read_json 预览的硬上限：
# - 文件大小 5MB（与 _summarize_json 对齐；超出直接走 execute_python 流读）
# - object：列前 50 个 keys + 前 20 个 keys 的浅层 value preview
# - 顶层 array：前 20 项 head，单项 ≤ 2KB（超大转 shape stub），累计 ≤ 6KB
# - value_preview 中的嵌套 array：前 3 项 head（用更紧的 cap），累计 ≤ 4KB；
#   独立常量是为了不让 value_preview 在最坏情况撑爆 observation——top-level
#   array head 6KB × 20 keys 会变 120KB，而 value_preview head 用 3-item/4KB
#   能把同一脏 case 限制在 80KB 量级，仍保留每个 array 的"几条样本"信号
# - 单字符串值上限 200 字符（截断后挂 value_truncated 标记）
_READ_JSON_SIZE_LIMIT = 5 * 1024 * 1024
_READ_JSON_OBJECT_KEYS_CAP = 50
_READ_JSON_OBJECT_VALUE_PREVIEW_CAP = 20
_READ_JSON_ARRAY_HEAD_CAP = 20
_READ_JSON_ARRAY_HEAD_BYTES = 6 * 1024
_READ_JSON_VALUE_PREVIEW_HEAD_CAP = 3
_READ_JSON_VALUE_PREVIEW_HEAD_BYTES = 4 * 1024
_READ_JSON_ITEM_BYTE_LIMIT = 2 * 1024
_READ_JSON_STRING_VALUE_CAP = 200
# read_doc keyword 模式上限（"全文返回所有匹配段落"——避免模型用 read_doc 串行翻页）：
# - 至多 20 个匹配段落
# - 单段最多 2KB（超出则尾部追加 "\n..." 截断标记）
# - 累计字节最多 8KB（与 csv head/tail 输出量级对齐）
_READ_DOC_KEYWORD_MATCH_CAP = 20
_READ_DOC_KEYWORD_PARAGRAPH_BYTES = 2 * 1024
_READ_DOC_KEYWORD_TOTAL_BYTES = 8 * 1024
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


def _truncate_string_value(text: str) -> tuple[str, bool]:
    """对字符串型值做 200 字符上限截断；返回 (截断后字符串, 是否被截).

    超长字符串值（descriptions / 长文本字段）会撑爆 observation；统一在此处把限。
    上层只需要把 `value_truncated: True` 挂到对应字段，提示模型换 execute_python 细读。
    """
    if len(text) <= _READ_JSON_STRING_VALUE_CAP:
        return text, False
    return text[:_READ_JSON_STRING_VALUE_CAP], True


def _summarize_value_for_preview(value: Any) -> Any:
    """object.value_preview 的"浅层"摘要：标量原值；集合给 kind+基数；array 额外给 head 样本。

    与 inspect_files._summarize_json 形成 strict superset：
    - 标量给真实值（read_json 的核心增量价值），string 截断到 200 字符
    - object 仅返回 cardinality（key_count），不递归展开内部——避免 depth=∞ 风险
    - array 返回与顶层 array 同形态的 `{kind, length, head, head_truncation, truncated}`，
      但用更紧的 cap（前 3 项 / 累计 4KB）；单 item > 2KB 仍走 shape stub

    array 暴露 head 样本是为了：
    (a) 恢复 wrapper-object 形态（dataset 通用 `{table, records: [...]}`）下"看几条记录长什么样"
        的能力——run 20260429T181710702941Z 显示 13 次 read_json 调用里 12 次落到这种 shape
    (b) 消除"顶层 array head[:20] vs 嵌套 array head[:1]"的 20× 取样不对称：3 条样本是异质性
        / 排序信号检测的标准小样本，远比单条更有判断力
    """
    if isinstance(value, dict):
        return {"kind": "object", "key_count": len(cast(dict[Any, Any], value))}
    if isinstance(value, list):
        value_list = cast(list[Any], value)
        length = len(value_list)
        head, head_truncation = _array_head_with_caps(
            value_list,
            item_cap=_READ_JSON_VALUE_PREVIEW_HEAD_CAP,
            aggregate_bytes=_READ_JSON_VALUE_PREVIEW_HEAD_BYTES,
        )
        return {
            "kind": "array",
            "length": length,
            "head": head,
            "head_truncation": head_truncation,
            "truncated": length > len(head) or head_truncation is not None,
        }
    if isinstance(value, str):
        truncated_text, was_truncated = _truncate_string_value(value)
        if was_truncated:
            return {"value": truncated_text, "value_truncated": True}
        return truncated_text
    # int / float / bool / None：原值返回
    return value


def _shape_stub_for_array_item(item: Any) -> dict[str, Any]:
    """单 item > 2KB 时的占位形态：尽量保留结构线索而不携带原始内容。"""
    if isinstance(item, dict):
        keys: list[str] = list(cast(dict[str, Any], item).keys())
        return {
            "kind": "object",
            "key_count": len(keys),
            "keys": keys[:10],
        }
    if isinstance(item, list):
        item_list = cast(list[Any], item)
        first_kind: str | None = None
        if item_list:
            first_kind = type(item_list[0]).__name__
        return {
            "kind": "array",
            "length": len(item_list),
            "first_item_kind": first_kind,
        }
    return {"kind": "scalar", "value_kind": type(item).__name__}


def _array_head_with_caps(
    items: list[Any],
    *,
    item_cap: int = _READ_JSON_ARRAY_HEAD_CAP,
    aggregate_bytes: int = _READ_JSON_ARRAY_HEAD_BYTES,
) -> tuple[list[Any], str | None]:
    """生成 array 的前 N 项预览，二级上限：单项 _READ_JSON_ITEM_BYTE_LIMIT + 累计 aggregate_bytes。

    顶层 `kind=array` 用默认（20 项 / 6KB）；value_preview 中嵌套 array 用收紧的
    （3 项 / 4KB）。共享同一段截断逻辑，避免两套实现漂移。

    返回 (head, head_truncation):
    - head_truncation == "byte_cap"   累计字节超 aggregate_bytes 提前停止
    - head_truncation == "item_cap"   至少有一项被替换为 shape stub
    - head_truncation is None         全量内嵌、无截断
    """
    head: list[Any] = []
    cumulative_bytes = 0
    item_cap_hit = False
    byte_cap_hit = False
    for item in items[:item_cap]:
        serialized = json.dumps(item, ensure_ascii=False)
        if len(serialized) > _READ_JSON_ITEM_BYTE_LIMIT:
            stub = _shape_stub_for_array_item(item)
            stub_bytes = len(json.dumps(stub, ensure_ascii=False))
            if cumulative_bytes + stub_bytes > aggregate_bytes and head:
                # stub 都塞不下：说明累计已经爆了，停止追加
                byte_cap_hit = True
                break
            head.append(stub)
            cumulative_bytes += stub_bytes
            item_cap_hit = True
            continue
        if cumulative_bytes + len(serialized) > aggregate_bytes and head:
            byte_cap_hit = True
            break
        head.append(item)
        cumulative_bytes += len(serialized)
    truncation: str | None = None
    if byte_cap_hit:
        truncation = "byte_cap"
    elif item_cap_hit:
        truncation = "item_cap"
    return head, truncation


def read_json_preview(task: PublicTask, relative_path: str) -> dict[str, Any]:
    """JSON 的 inspect-style 预览：返回结构化 `{kind, length|key_count, head|keys+value_preview}`。

    与 `inspect_files._summarize_json` 形成 strict superset：
    - object：列出前 50 个 keys + 前 20 个 keys 的浅层 value preview（标量原值；
      集合给 cardinality 而不展开）
    - array：前 20 项 head，单项 > 2KB 替换为 shape stub，累计 > 6KB 提前停止
    - scalar：直接返回原值，字符串超 200 字符截断
    - 文件 > 5MB：跳过 loads，返回 `kind="unknown" + skipped=True`，提示走 execute_python

    与旧版 `preview` 字符串前缀截断的关键差异：array 的 length 与 object 的 key_count
    永远不会被截断丢失——这两个数字是模型决定下一步（继续看还是切 execute_python）的核心。
    """
    path = resolve_context_path(task, relative_path)
    size = path.stat().st_size
    if size > _READ_JSON_SIZE_LIMIT:
        return {
            "path": relative_path,
            "kind": "unknown",
            "size": size,
            "skipped": True,
            "reason": "File exceeds 5MB; use execute_python with json.load to stream.",
        }
    payload = json.loads(path.read_text())

    if isinstance(payload, dict):
        payload_dict = cast(dict[str, Any], payload)
        keys: list[str] = list(payload_dict.keys())
        key_count = len(keys)
        listed_keys = keys[:_READ_JSON_OBJECT_KEYS_CAP]
        # 显式按 keys[:20] 顺序构建 value_preview，保证 dict 插入顺序确定（CPython
        # 3.7+ 默认即 insertion order，但写显式循环避免依赖隐式行为）。
        value_preview: dict[str, Any] = {}
        for key in keys[:_READ_JSON_OBJECT_VALUE_PREVIEW_CAP]:
            value_preview[key] = _summarize_value_for_preview(payload_dict[key])
        keys_truncated = key_count > _READ_JSON_OBJECT_KEYS_CAP
        return {
            "path": relative_path,
            "kind": "object",
            "key_count": key_count,
            "keys": listed_keys,
            "keys_truncated": keys_truncated,
            "value_preview": value_preview,
            # truncated 是"上游 observation 不完整"的统一信号，便于模型一眼识别要不要切 execute_python
            "truncated": keys_truncated or key_count > _READ_JSON_OBJECT_VALUE_PREVIEW_CAP,
        }

    if isinstance(payload, list):
        payload_list = cast(list[Any], payload)
        length = len(payload_list)
        head, head_truncation = _array_head_with_caps(payload_list)
        truncated = length > len(head) or head_truncation is not None
        return {
            "path": relative_path,
            "kind": "array",
            "length": length,
            "head": head,
            "head_truncation": head_truncation,
            "truncated": truncated,
        }

    # scalar（int / float / str / bool / None）：直接给原值，字符串过长截断
    if isinstance(payload, str):
        truncated_text, was_truncated = _truncate_string_value(payload)
        result: dict[str, Any] = {
            "path": relative_path,
            "kind": "scalar",
            "value": truncated_text,
        }
        if was_truncated:
            result["value_truncated"] = True
        return result
    return {
        "path": relative_path,
        "kind": "scalar",
        "value": payload,
    }


def _normalize_line_count(line_count: int) -> int:
    """把读取行数限制在稳定的 observation 范围内。"""
    return min(300, max(1, int(line_count)))


def _split_paragraphs(lines: list[str]) -> list[tuple[int, int, str]]:
    """把行列表切成段落（连续非空行为一段；空行为分隔），返回 (start, end, text) 列表。

    行号 1-based，闭区间。给 keyword 模式提供"段落级"返回单元——dataset 里的大
    `doc/*.md` 多为"一段一条记录"的 prose 数据库（superhero.md 1087 行 / 524 段；
    每段 ≈ 一个英雄/病人/赛事），段是天然的检索粒度。
    """
    paragraphs: list[tuple[int, int, str]] = []
    current: list[str] = []
    current_start = 0  # 0 表示"当前没有进行中的段"；进入新段时重置为 1-based 行号
    for one_based_idx, ln in enumerate(lines, start=1):
        if ln.strip() == "":
            if current:
                paragraphs.append((current_start, one_based_idx - 1, "\n".join(current)))
                current = []
                current_start = 0
        else:
            if not current:
                current_start = one_based_idx
            current.append(ln)
    if current:
        paragraphs.append((current_start, len(lines), "\n".join(current)))
    return paragraphs


def _collect_keyword_matches(
    paragraphs: list[tuple[int, int, str]], keyword: str
) -> tuple[list[dict[str, Any]], int, bool]:
    """从段落集合中收集所有包含 keyword 的段落（大小写不敏感子串匹配）。

    返回 (matches, match_count, matches_truncated):
    - matches:           已应用 cap 的返回子集
    - match_count:       全文匹配总数（即使被 cap 裁掉也如实计数，方便模型判断是否需要更精确的 keyword）
    - matches_truncated: 返回数 < 总数时为 True（数量上限或字节上限触发）

    上限策略（同 read_csv/read_json 思路："SAMPLE preview not full data access"）：
    - 至多 20 段；超出仅计数不返回内容
    - 单段超 2KB → 截到 2KB 并追加 "\n..." 标记 + paragraph_truncated=True
    - 累计超 8KB → 停止追加新段（已收集的段保留）
    """
    needle = keyword.lower()
    match_count = 0
    matches: list[dict[str, Any]] = []
    cumulative_bytes = 0
    capacity_exhausted = False
    for start, end, text in paragraphs:
        if needle not in text.lower():
            continue
        match_count += 1
        if capacity_exhausted:
            # 仍然计数，但不再 emit；让模型从 match_count vs len(matches) 看出还有多少未返回
            continue
        if len(matches) >= _READ_DOC_KEYWORD_MATCH_CAP:
            capacity_exhausted = True
            continue
        paragraph_truncated = False
        text_to_emit = text
        if len(text) > _READ_DOC_KEYWORD_PARAGRAPH_BYTES:
            text_to_emit = text[:_READ_DOC_KEYWORD_PARAGRAPH_BYTES] + "\n..."
            paragraph_truncated = True
        if cumulative_bytes + len(text_to_emit) > _READ_DOC_KEYWORD_TOTAL_BYTES and matches:
            capacity_exhausted = True
            continue
        matches.append(
            {
                "start_line": start,
                "end_line": end,
                "text": text_to_emit,
                "paragraph_truncated": paragraph_truncated,
            }
        )
        cumulative_bytes += len(text_to_emit)
    matches_truncated = len(matches) < match_count
    return matches, match_count, matches_truncated


def read_doc_preview(
    task: PublicTask,
    relative_path: str,
    *,
    start_line: int = 1,
    line_count: int = 80,
    keyword: str | None = None,
) -> dict[str, Any]:
    """文本文档预览。两种模式互斥：

    1. **paging 模式**（无 keyword）：返回 [start_line, start_line+line_count) 行窗口；
       通过 `has_more` + 下一次 `start_line` 续读。
    2. **keyword 模式**（提供 keyword）：返回**全文**所有包含该关键词的段落（空行分隔），
       上限 20 段 / 单段 2KB / 累计 8KB；`start_line` / `line_count` 在此模式下被忽略。

    keyword 模式是为破解模型"用 80 行窗口串行翻 1087 行 prose 数据库"的失败模式
    （task_46 trace：7 次 read_doc + 11 次 execute_python 仍超步耗尽）设计——
    一次 keyword 调用替代整段串行扫描。需要全文 regex / 跨段抽取 / 结构化解析时，
    走 execute_python（read_doc 不返回完整文档内容）。

    errors="replace" 把非法字节替换为 U+FFFD，避免个别文件乱码让整个调用崩溃。
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

    if keyword:
        paragraphs = _split_paragraphs(lines)
        matches, match_count, matches_truncated = _collect_keyword_matches(paragraphs, keyword)
        return {
            "path": relative_path,
            "total_lines": total_lines,
            # keyword 模式下 paging 字段无意义：清空避免模型把 returned_range 当作"窗口已读"
            "returned_range": None,
            "has_more": False,
            "keyword_match": {
                "keyword": keyword,
                "found": match_count > 0,
                "match_count": match_count,
                "matches": matches,
                "matches_truncated": matches_truncated,
            },
            # preview 是 paging 模式专属——keyword 模式的内容全部装在 keyword_match.matches[*].text 里
            "preview": None,
        }

    normalized_start_line = max(1, int(start_line))
    normalized_line_count = _normalize_line_count(line_count)

    start_index = normalized_start_line - 1
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
        "keyword_match": None,
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
