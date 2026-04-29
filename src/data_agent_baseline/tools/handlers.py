"""八件套工具的 handler 实现 + 统一返回结构。

每个 `handle_xxx` 接收一个已被对应 Pydantic 模型校验过的 args 实例（参见
`tools/inputs.py` 的 `XxxInput`）；类型转换 / 必填校验 / 边界检查全部在
`ToolRegistry.execute` 边界处完成，handler 内部可信任 args。

`ToolExecutionResult` 与 `ToolHandler` 类型别名在此处定义；`tools/registry.py` 顶端
重导出它们，保持 `ToolExecutionResult` 在 `tools.registry` 命名空间下的历史导入路径不变。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable, PublicTask
from data_agent_baseline.tools.filesystem import (
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
    """`read_json`: 读取 JSON 并按字符数截断，默认 4000 字符。"""
    return ToolExecutionResult(
        ok=True, content=read_json_preview(task, args.path, max_chars=args.max_chars)
    )


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
    """
    content = execute_python_code(
        context_root=task.context_dir,
        code=args.code,
        timeout_seconds=EXECUTE_PYTHON_TIMEOUT_SECONDS,
    )
    return ToolExecutionResult(ok=bool(content.get("success")), content=content)


def handle_answer(_task: PublicTask, args: AnswerInput) -> ToolExecutionResult:
    """`answer`: 终止工具，提交最终答案表。

    所有结构性校验（columns 非空、每行列数 == len(columns)、单元格类型）由
    `AnswerInput` 在 ToolRegistry.execute 边界处完成；handler 收到的 `args`
    可信，直接拷贝成 AnswerTable 落盘。
    """
    # list(...) 拷贝：让最终落盘的 AnswerTable 不与外部对象共享底层引用
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
