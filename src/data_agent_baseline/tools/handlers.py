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
    return ToolExecutionResult(
        ok=True, content=execute_read_only_sql(path, args.sql, limit=args.limit)
    )


def handle_execute_python(task: PublicTask, args: ExecutePythonInput) -> ToolExecutionResult:
    """`execute_python`: 在子进程中跑任意 Python 代码，30s 超时。

    注意这里根据 `content.success` 决定 `ok` 的真假——和其他工具不同，
    这个工具本身可能因用户代码抛错而失败，模型需要一个明确的 ok 信号。

    `answer_dir = <task_dir>/_answer`：execute_python 子进程把它 set 到
    `os.environ['DABENCH_ANSWER_DIR']`，让 artifact handoff（`answer({from_csv})`)
    有一个稳定写盘位置；目录由 `execute_python_code` 保证 mkdir。
    """
    content = execute_python_code(
        context_root=task.context_dir,
        code=args.code,
        answer_dir=task.task_dir / "_answer",
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


def handle_answer(_task: PublicTask, args: AnswerInput) -> ToolExecutionResult:
    """`answer`: 终止工具，提交最终答案表。

    两种入参分支由 `AnswerInput._check_payload` 互斥强制：
    - inline (`columns` + `rows`)：所有结构性校验在 Pydantic 边界完成，handler 直接拷贝
    - artifact (`from_csv`)：handler 读取磁盘 CSV → 列级 dtype 推断 → cell 强转 →
      构 AnswerTable；任何 IO/格式/越界错误以 `ValueError` 上抛，dispatcher 翻译成
      `ToolErrorEvent`，ReAct 循环续跑让模型 retry
    """
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
        answer = AnswerTable(columns=columns, rows=rows)
        return ToolExecutionResult(
            ok=True,
            content={
                "status": "submitted",
                "column_count": len(columns),
                "row_count": len(rows),
                "from_csv": {
                    "path": str(path),
                    "dtypes": dtypes,
                    # 仅前 5 行做 trace 审计：模型回显已经写过的内容毫无意义，
                    # 但留个抽样便于离线对账（prediction.csv vs gold）
                    "head_preview": rows[:5],
                },
            },
            is_terminal=True,
            answer=answer,
        )
    # inline 路径：list(...) 拷贝让最终落盘的 AnswerTable 不与外部对象共享底层引用
    columns = list(args.columns)
    rows = [list(row) for row in args.rows]
    answer = AnswerTable(columns=columns, rows=rows)
    return ToolExecutionResult(
        ok=True,
        content={
            # 给模型一个简洁的提交确认信号，不把完整表回显（避免浪费 token）
            "status": "submitted",
            "column_count": len(columns),
            "row_count": len(rows),
        },
        is_terminal=True,  # 触发 ReAct 循环退出
        answer=answer,  # 答案对象交给循环，最终写入 prediction.csv
    )
