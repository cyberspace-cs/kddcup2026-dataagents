"""SQLite 只读访问工具。

通过 URI `mode=ro` 强制以只读方式连接，杜绝 Agent 误写入数据集。
支持两种操作：查看表结构、执行只读 SQL 查询。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def _connect_read_only(path: Path) -> sqlite3.Connection:
    """以只读 URI 模式连接 sqlite 文件。

    `file:...?mode=ro` 语法要求 `uri=True`。任何试图写入的语句都会
    抛 OperationalError，为 Agent 提供结构级别的安全保证。
    """
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def inspect_sqlite_schema(path: Path) -> dict[str, object]:
    """列出所有用户表及其 CREATE 语句。

    过滤 `sqlite_%` 系统表；按名字排序让结果稳定可对比。
    """
    with _connect_read_only(path) as conn:
        rows = conn.execute(
            """
            SELECT name, sql
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
        tables: list[dict[str, object]] = []
        for name, create_sql in rows:
            # 保留原始 CREATE SQL：列类型和约束对 Agent 推导 schema 很有帮助
            tables.append(
                {
                    "name": name,
                    "create_sql": create_sql,
                }
            )
    return {
        "path": str(path),
        "tables": tables,
    }


def execute_read_only_sql(path: Path, sql: str, *, limit: int = 200) -> dict[str, object]:
    """执行 SELECT/WITH/PRAGMA 语句并按行数截断。

    两道防线：
    1. **SQL 前缀白名单**：只放行读语句（定性拦住 DML/DDL）
    2. **连接只读**：即使绕过前缀检查，数据库层也会拒写

    - `limit + 1` 抓取策略：多抓一行用于判断 `truncated`，返回前再截回 limit
    """
    # 去掉首部空白（含换行）后小写比较，允许用户风格多样
    normalized_sql = sql.lstrip().lower()
    # 显式白名单：严格起见不接受 EXPLAIN（怕引发副作用也防止歧义）
    if not normalized_sql.startswith(("select", "with", "pragma")):
        raise ValueError("Only read-only SQL statements are allowed.")

    with _connect_read_only(path) as conn:
        cursor = conn.execute(sql)
        # description 在 PRAGMA 某些子命令下可能为 None，故 or [] 兜底
        column_names = [item[0] for item in cursor.description or []]
        # 多取 1 行：若真取到 limit+1 行即表示被截断，模型可决定是否换更细查询
        rows = cursor.fetchmany(limit + 1)

    truncated = len(rows) > limit
    limited_rows = rows[:limit]
    return {
        "path": str(path),
        "columns": column_names,
        # sqlite3 返回 tuple，统一成 list 让 JSON 序列化语义更直观
        "rows": [list(row) for row in limited_rows],
        "row_count": len(limited_rows),
        "truncated": truncated,
    }
