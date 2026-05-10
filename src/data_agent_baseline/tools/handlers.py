"""八件套工具的 handler 实现 + 统一返回结构。

每个 `handle_xxx` 接收一个已被对应 Pydantic 模型校验过的 args 实例（参见
`tools/inputs.py` 的 `XxxInput`）；类型转换 / 必填校验 / 边界检查全部在
`ToolRegistry.execute` 边界处完成，handler 内部可信任 args。

`ToolExecutionResult` 与 `ToolHandler` 类型别名在此处定义；`tools/registry.py` 顶端
重导出它们，保持 `ToolExecutionResult` 在 `tools.registry` 命名空间下的历史导入路径不变。
"""

from __future__ import annotations

import csv
import pathlib
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.filesystem import (
    _classify_dtype,  # pyright: ignore[reportPrivateUsage]
    inspect_context_files,
    read_csv_preview,
    read_doc_preview,
    read_json_preview,
    resolve_context_path,
)
from data_agent_baseline.tools.inputs import (
    EXECUTE_PYTHON_TIMEOUT_SECONDS,
    AnswerInput,
    ExecuteContextSqlInput,
    ExecutePythonInput,
    InspectFilesInput,
    InspectSqliteSchemaInput,
    ReadCsvInput,
    ReadDocInput,
    ReadJsonInput,
)
from data_agent_baseline.tools.python_exec import execute_python_code
from data_agent_baseline.tools.sqlite import execute_read_only_sql, inspect_sqlite_schema


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """工具执行的统一返回结构。

    - ok:          True 表示工具正常完成；即使业务层"失败"（如 SQL 查空）也是 True
    - content:     回传给模型看的结果 JSON
    - is_terminal: 是否应让 ReAct 循环立即终止（仅 answer 置 True）
    - answer:      终止工具附带的答案表（非终止工具为 None）
    """

    ok: bool
    content: dict[str, Any]
    is_terminal: bool = False
    answer: AnswerTable | None = None


# 统一的 handler 签名：第二个参数对内置工具是已验证的 BaseModel 实例（详见 _INPUT_MODELS），
# 对未在 _INPUT_MODELS 注册的用户自定义工具则透传原始 dict。
ToolHandler = Callable[[PublicTask, Any], ToolExecutionResult]


def handle_inspect_files(task: PublicTask, _args: InspectFilesInput) -> ToolExecutionResult:
    """`inspect_files`: 一次性回传 context 下所有文件的轻量 schema。

    入参模型刻意为空（参见 `InspectFilesInput`），handler 直接调度核心 walker。
    """
    return ToolExecutionResult(ok=True, content=inspect_context_files(task))


def handle_read_csv(task: PublicTask, args: ReadCsvInput) -> ToolExecutionResult:
    """`read_csv`: 返回 CSV 的 schema + head(20) + tail(5)。

    样本固定不让模型调，全量过滤/聚合走 execute_python——参见 `read_csv_preview` docstring。
    """
    return ToolExecutionResult(ok=True, content=read_csv_preview(task, args.path))


def handle_read_json(task: PublicTask, args: ReadJsonInput) -> ToolExecutionResult:
    """`read_json`: 返回 JSON 的结构化预览（kind + length/keys + 浅层 value preview）。"""
    return ToolExecutionResult(ok=True, content=read_json_preview(task, args.path))


def handle_read_doc(task: PublicTask, args: ReadDocInput) -> ToolExecutionResult:
    """`read_doc`: 按行分页读取任意文本（md/txt），可选关键词检索。"""
    # 沿用历史"空字符串与 None 同义"行为，避免下层把空 keyword 当作精确空匹配
    keyword = None if not args.keyword else args.keyword
    return ToolExecutionResult(
        ok=True,
        content=read_doc_preview(
            task,
            args.path,
            start_line=args.start_line,
            line_count=args.line_count,
            keyword=keyword,
        ),
    )


def handle_inspect_sqlite_schema(
    task: PublicTask, args: InspectSqliteSchemaInput
) -> ToolExecutionResult:
    """`inspect_sqlite_schema`: 查看 sqlite 数据库的全部表结构。"""
    # resolve_context_path 同时做越界校验 + 存在性检查
    path = resolve_context_path(task, args.path)
    return ToolExecutionResult(ok=True, content=inspect_sqlite_schema(path))


def handle_execute_context_sql(
    task: PublicTask, args: ExecuteContextSqlInput
) -> ToolExecutionResult:
    """`execute_context_sql`: 对 context/ 下的 sqlite 跑只读 SQL，默认 200 行上限。"""
    path = resolve_context_path(task, args.path)
    content = execute_read_only_sql(path, args.sql, limit=args.limit)
    if content.get("row_count", -1) == 0:
        content["warning"] = (
            "Query returned 0 rows. Verify filter values with "
            "SELECT DISTINCT or check for case/format mismatches."
        )
    return ToolExecutionResult(ok=True, content=content)


_ANSWER_SCRATCH_ROOT = pathlib.Path("/tmp/dabench")


def _answer_dir_for(task: PublicTask) -> pathlib.Path:
    """每个任务的 `_answer/` artifact 目录（execute_python 写、handle_answer 读）。

    锚定到 `/tmp/dabench/<task_id>/_answer/` 而非 `task.task_dir/_answer/`：§3.4
    把判题容器的 `/input` 挂为 read-only，沿用 task_dir 会让 mkdir/写入抛
    `[Errno 30] Read-only file system`，agent 在 execute_python 失败上死循环
    直到 max_steps。docker 内外统一指向 `/tmp/dabench`，本地 CLI 行为对齐
    判题环境，少一处隐式分叉。
    """
    return _ANSWER_SCRATCH_ROOT / task.task_id / "_answer"


def handle_execute_python(task: PublicTask, args: ExecutePythonInput) -> ToolExecutionResult:
    """`execute_python`: 在子进程中跑任意 Python 代码，30s 超时。

    注意这里根据 `content.success` 决定 `ok` 的真假——和其他工具不同，
    这个工具本身可能因用户代码抛错而失败，模型需要一个明确的 ok 信号。

    `answer_dir`：execute_python 子进程把它 set 到 `os.environ['DABENCH_ANSWER_DIR']`，
    让 artifact handoff（`answer({from_csv})`)有一个稳定写盘位置；目录由
    `execute_python_code` 保证 mkdir。具体路径解析见 `_answer_dir_for`。
    """
    content = execute_python_code(
        context_root=task.context_dir,
        code=args.code,
        answer_dir=_answer_dir_for(task),
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


# 5MB 上限呼应 read_json：超出意味着模型该抽样或换 inline 而不是把整张表喂回。
_FROM_CSV_SIZE_LIMIT_BYTES = 5 * 1024 * 1024


def _coerce_row(row: list[str], dtypes: list[str]) -> list[Any]:
    """按列 dtype 把 CSV 字符串 cell 强转回 int / float / bool / None。

    与 `_classify_dtype` 的判定一一对应：'empty' 列保留为 str（推断不出更窄类型时
    宁可保守，避免把 ID-like 字段误转成 int 丢精度）。空字符串始终视为 None，
    回填成 prediction.csv 时与 inline 路径的 `null` cell 行为一致。
    """
    out: list[Any] = []
    for cell, dtype in zip(row, dtypes, strict=True):
        if cell == "":
            out.append(None)
        elif dtype == "int":
            out.append(int(cell))
        elif dtype == "float":
            out.append(float(cell))
        elif dtype == "bool":
            out.append(cell.lower() == "true")
        else:
            out.append(cell)
    return out


def _parse_answer_csv(path: pathlib.Path) -> tuple[list[str], list[list[Any]], list[str]]:
    """读取并类型化一份 answer artifact CSV。

    设计意图：模型若用 `pandas.to_csv` 写 artifact，所有 cell 都会被序列化为字符串。
    本函数复用 `_classify_dtype` 做列级类型推断（int 优先于 float，再依次 bool/str），
    再用 `_coerce_row` 把字符串 cell 强转回原类型，让 prediction.csv 与 inline 路径
    生成的字面值一致——尤其是整数主键 ID（避免落地成 `163109.0`）与浮点全精度
    （避免被 float→str→float 截断）。

    所有失败一律 raise ValueError（messages 以 `answer:` 不再前置——caller 端会贴）；
    dispatcher 会把异常翻译成 `ToolErrorEvent` 让 ReAct 循环 retry。
    """
    if not path.exists():
        raise ValueError(f"from_csv file not found: {path}")
    if path.stat().st_size > _FROM_CSV_SIZE_LIMIT_BYTES:
        raise ValueError("from_csv exceeds 5MB; split the table or use inline columns/rows")
    try:
        # newline="" 按 csv 模块要求传，正确处理嵌入换行的单元格
        with path.open("r", encoding="utf-8", newline="") as handle:
            raw = list(csv.reader(handle))
    except UnicodeDecodeError as exc:
        raise ValueError(f"from_csv is not valid UTF-8: {exc}") from exc
    except csv.Error as exc:
        raise ValueError(f"failed to parse from_csv: {exc}") from exc
    if not raw:
        raise ValueError("from_csv is empty; expected header row + data rows")
    header, *data = raw
    if not header:
        raise ValueError("from_csv header row is empty")
    width = len(header)
    for index, row in enumerate(data):
        if len(row) != width:
            raise ValueError(f"from_csv row {index} has {len(row)} cells but header has {width}")
    # 列级 dtype 推断：与 inline 路径在 prediction.csv 字面值上对齐
    dtypes: list[str] = []
    for col_idx in range(width):
        column_values = [row[col_idx] for row in data]
        dtypes.append(_classify_dtype(column_values))
    typed = [_coerce_row(row, dtypes) for row in data]
    return header, typed, dtypes


def _prune_null_columns(
    columns: list[str], rows: list[list[Any]]
) -> tuple[list[str], list[list[Any]], list[str]]:
    """剔除所有值均为 null/空串/None 的列，返回 (pruned_columns, pruned_rows, removed)。

    grader 对多余列的惩罚是确定性的；如果一列在所有行上都是空值，
    它不可能是 gold 列，剔除只会减少 penalty，不可能伤害 recall。
    """
    if not rows or not columns:
        return columns, rows, []

    width = len(columns)
    keep_indices: list[int] = []
    removed: list[str] = []

    for col_idx in range(width):
        all_null = all(
            row[col_idx] is None or (isinstance(row[col_idx], str) and row[col_idx].strip() == "")
            for row in rows
            if col_idx < len(row)
        )
        if all_null and len(rows) > 0:
            removed.append(columns[col_idx])
        else:
            keep_indices.append(col_idx)

    if not removed:
        return columns, rows, []

    pruned_columns = [columns[i] for i in keep_indices]
    pruned_rows = [[row[i] for i in keep_indices] for row in rows]
    return pruned_columns, pruned_rows, removed


def handle_answer(task: PublicTask, args: AnswerInput) -> ToolExecutionResult:
    """`answer`: 终止工具，提交最终答案表。

    两种入参分支由 `AnswerInput._check_payload` 互斥强制：
    - inline (`columns` + `rows`)：所有结构性校验在 Pydantic 边界完成，handler 直接拷贝
    - artifact (`from_csv`)：handler 读取磁盘 CSV → 列级 dtype 推断 → cell 强转 →
      构 AnswerTable；任何 IO/格式/越界错误以 `ValueError` 上抛，dispatcher 翻译成
      `ToolErrorEvent`，ReAct 循环续跑让模型 retry

    提交成功路径上会清理整个 `_answer/` 目录：模型在 `execute_python` 里写出的
    artifact 此时已被 `_parse_answer_csv` 全量读入内存（或 inline 路径压根没用到），
    保留磁盘文件只会污染输入树（曾被误读为参考答案）。`DABENCH_ANSWER_DIR` 还可能
    存放草稿/备份/日志等非 answer.csv 文件，单独 unlink 主文件不够，整个目录 rmtree。

    Inline + 残留 artifact 互斥：模型若已经把表写到 `_answer/answer.csv`，
    必须用 `from_csv` 指向它；继续 inline 一律拒绝。回归参照见
    `run 20260501-010/task_8`——`to_csv` 写出 140 行后，模型在 `answer` tool_call
    里只 inline 了 `head(20)` 的 20 行，inline 硬闸（30 行 / 200 cells）抓不住。
    raise ValueError 让 dispatcher 翻译为 `ToolErrorEvent`，下一轮模型自己改 `from_csv`。
    """
    artifact_path = _answer_dir_for(task) / "answer.csv"
    if args.from_csv is not None:
        path = pathlib.Path(args.from_csv).expanduser()
        if not path.is_absolute():
            # 拒绝相对路径：execute_python 的 CWD 是 task.context_dir，模型若传相对路径
            # 会落到 context 之内污染源数据；强制绝对路径 + DABENCH_ANSWER_DIR 提示让
            # artifact 唯一栖身处明确
            raise ValueError(
                f"from_csv must be an absolute path, got '{args.from_csv}' "
                "(hint: use os.environ['DABENCH_ANSWER_DIR'] to build the path)"
            )
        columns, rows, dtypes = _parse_answer_csv(path)
        columns, rows, pruned_cols = _prune_null_columns(columns, rows)
        answer = AnswerTable(columns=columns, rows=rows)
        path.unlink(missing_ok=True)
        shutil.rmtree(artifact_path.parent, ignore_errors=True)
        content: dict[str, Any] = {
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(rows),
            "from_csv": {
                "path": str(path),
                "dtypes": dtypes,
                "head_preview": rows[:5],
            },
        }
        if pruned_cols:
            content["pruned_null_columns"] = pruned_cols
        return ToolExecutionResult(
            ok=True,
            content=content,
            is_terminal=True,
            answer=answer,
        )
    # 残留 artifact + inline 提交：拒绝并指回 from_csv（详见 docstring 中
    # run 20260501-010/task_8 案例）。size > 0 防御 zero-byte 残留误伤。
    if artifact_path.exists() and artifact_path.stat().st_size > 0:
        raise ValueError(
            f"answer artifact already exists at {artifact_path} but inline "
            "columns/rows were sent; the artifact is the canonical answer "
            "and inline cannot override it. Resubmit as "
            f'answer({{"from_csv": "{artifact_path}"}}) with no columns/rows.'
        )
    # inline 路径：list(...) 拷贝让最终落盘的 AnswerTable 不与外部对象共享底层引用
    columns = list(args.columns)
    rows = [list(row) for row in args.rows]
    columns, rows, pruned_cols = _prune_null_columns(columns, rows)
    answer = AnswerTable(columns=columns, rows=rows)
    shutil.rmtree(artifact_path.parent, ignore_errors=True)
    content: dict[str, Any] = {
        "status": "submitted",
        "column_count": len(columns),
        "row_count": len(rows),
    }
    if pruned_cols:
        content["pruned_null_columns"] = pruned_cols
    return ToolExecutionResult(
        ok=True,
        content=content,
        is_terminal=True,
        answer=answer,
    )
