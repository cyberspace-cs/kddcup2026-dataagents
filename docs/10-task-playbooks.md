# 10. 任务手册（Task Playbooks）

> 针对 easy/medium/hard/extreme 四档，给出 Agent 典型处理路径与 prompt 附加段。
> 这不是硬编码——运行时 Agent 可偏离；只是**默认推荐路径**，用于 prompt 设计与 debug 时对照。

## 10.1 easy — Python 代码生成与数据分析

**典型场景**：单 CSV 或单 JSON，问题形如"列出 X 最高的 3 个 Y"、"求 Z 的总和"

**数据源组合**：CSV/JSON + knowledge.md

**典型步骤（4-6 步）**：
```
Step 1: list_context                    # 确认文件树（若 profile 已完整可跳过）
Step 2: read_csv (主表, max_rows=10)    # 看 schema + 样本
Step 3: execute_python                  # pandas/polars 聚合
Step 4: answer                          # 提交
```

**ReAct 预算**：max_steps=16（实际常 4-6 步），wall=300s

**Prompt 附加**：
```
This task is `easy`. Expect direct computation on one or two tabular files.
Aim for ≤8 tool calls. Start with read_csv on the obvious main table.
```

**常见坑**：
- CSV 列名中英混杂、含空格 → 用 polars 或 `df.columns = df.columns.str.strip()`
- 数值列混了千分位逗号 → `pd.to_numeric(s.str.replace(",", ""))`
- 问题里说"top 3"但实际表里只有 2 条 → 返回所有，不要 fabricate

**Answer 构造要点**：
- 列名建议与问题语义一致（便于 debug，不参与评分）
- 行数严格按问题要求（"top 3" → 3 行不多不少）
- 数值保留 2 位小数（OutputWriter 会归一化，但 Agent 保精度最好）

---

## 10.2 medium — Text-to-SQL + 多源融合

**典型场景**：表格 + SQLite 库，问题需要 join 或跨源查询

**数据源组合**：CSV/JSON + db/*.sqlite + knowledge.md

**典型步骤（6-10 步）**：
```
Step 1: inspect_sqlite_schema           # 必做，看表结构
Step 2: read_csv (关联 CSV)             # 若问题涉及
Step 3: execute_context_sql             # 主查询
Step 4: execute_python                  # 合并 CSV + SQL 结果
Step 5: answer
```

**ReAct 预算**：max_steps=20，wall=500s

**Prompt 附加**：
```
This task is `medium`. Data spans multiple sources (CSV + SQLite typical).
MUST call inspect_sqlite_schema before execute_context_sql on unknown DBs.
When joining CSV with SQL results, prefer doing the SQL aggregation first,
then loading results into pandas for final joins — this keeps SQL simple.
```

**常见坑**：
- SQLite 表名含保留字或连字符 → 用双引号 `"table-name"`
- 时间字段存为 TEXT 格式不统一（`2023-01-15` vs `2023/1/15`）→ 用 `strftime` 标准化或先 SELECT 后用 pandas 处理
- 写 SQL 时 LIMIT 忘记 → 工具默认 limit=200，但问题要 all 时需显式大 LIMIT
- 跨表计数：`COUNT(DISTINCT x)` vs `COUNT(x)` 的区别

**SQL 风格指引（写进 prompt 的示例）**：
```sql
-- 好：用 CTE + 明确聚合
WITH region_totals AS (
    SELECT region, SUM(amount) AS total
    FROM orders
    WHERE date >= '2023-07-01' AND date < '2023-10-01'
    GROUP BY region
)
SELECT region, total FROM region_totals ORDER BY total DESC LIMIT 10;
```

---

## 10.3 hard — 非结构化文档推理

**典型场景**：表格 + SQLite + PDF/MD 文档，问题需要从文档读出定性信息再与表格数据结合

**数据源组合**：CSV/JSON + db/ + doc/*.pdf|*.md + knowledge.md

**典型步骤（8-14 步）**：
```
Step 1: inspect_pdf (主文档)            # 看页数、是否含表
Step 2: read_pdf (mode="text", pages=[目录页])  # 锁定相关章节
Step 3: read_pdf (mode="text", pages=[X-Y])    # 精读目标章节
Step 4: inspect_sqlite_schema
Step 5: execute_context_sql             # 获取量化数据
Step 6: search_knowledge                # 查术语含义
Step 7: execute_python                  # 融合计算
Step 8: answer
```

**ReAct 预算**：max_steps=24，wall=700s

**Prompt 附加**：
```
This task is `hard`. You need to extract qualitative facts from documents
(PDFs, markdown) and combine them with quantitative data from tables.

Strategy:
1. Scope the document first with inspect_pdf → read_pdf on index / TOC.
2. Read the specific section that answers a sub-question.
3. Cross-check numbers from doc vs table — they should usually agree;
   if not, the table is authoritative unless the question says otherwise.
4. Use search_knowledge to resolve ambiguous terms.
```

**常见坑**：
- PDF 表格被 pdfplumber 抽成乱序文本 → 切 `mode="tables"` 或用 pymupdf
- 扫描版 PDF 文本抽不出 → fallback 到 OCR（tool 内部会做，agent 不必管）
- 文档里给的百分比与表格算出的差 0.1-0.2 → 通常是四舍五入差异，以表格为准
- 多份文档交叉引用 → 用 search_knowledge 做全库 query 而非逐文件 read_doc

---

## 10.4 extreme — 长上下文 + 记忆管理

**典型场景**：大量文档 + 多表 + 图表，问题需要 5+ 步骤的推理链

**数据源组合**：所有上面的类型可能同时出现

**策略切换**：**Plan-and-Execute**（不走纯 ReAct）

**Plan 阶段**：
```
Planner LLM 调用 1 次 → 输出 DAG，例如：
{
  "plan_summary": "Find Q3 top-growth region; read target from dashboard chart; compute % gap",
  "nodes": [
    {"id": "n1", "goal": "List Q3 region revenue from sales.csv",
     "tool_hint": "execute_python", "deps": []},
    {"id": "n2", "goal": "From q3_dashboard.png extract Q3 target value",
     "tool_hint": "read_chart", "deps": []},
    {"id": "n3", "goal": "Compute (actual - target) / target for each region",
     "tool_hint": "execute_python", "deps": ["n1", "n2"]},
    {"id": "n4", "goal": "Produce final table sorted by gap desc",
     "tool_hint": "execute_python", "deps": ["n3"]}
  ],
  "final_node": "n4"
}
```

**Execute 阶段**：每节点开子 ReAct（max_steps=6），结果写入 blackboard：
```python
blackboard = {
    "n1": AnswerTable(columns=[...], rows=[...]),
    "n2": AnswerTable(columns=["q3_target"], rows=[[4_200_000]]),
    "n3": ...,
}
```

子节点 prompt 里把 `deps` 节点的 result 作为输入：
```
GOAL: Compute (actual - target) / target for each region

INPUTS FROM PRIOR NODES:
n1 = {"columns":["region","revenue"],"rows":[["north",3500000],["south",2800000],...]}
n2 = {"columns":["q3_target"],"rows":[[4200000]]}
```

**Reflection 阶段（仅失败时触发）**：
- 若 final_node 失败（子 ReAct 没产出 answer），dump 失败原因，重走 Plan 阶段一次（最多 1 次）
- 重规划时 prompt 加入上次失败摘要

**ReAct 预算**：max_steps=32（单节点上限 6），wall=900s

**Context Folding**（长上下文管理）：
- 当 `sum(step.observation.tokens) > 180K` 时，启动 folding：
  - 把最老的 50% steps 压缩成 `"[folded: <steps X-Y summary>]"`
  - 保留最近 3 步完整 observation
- 每节点独立 fold（不跨节点）

**常见坑**：
- Plan 阶段 LLM 输出不合法 JSON → 重试一次；再失败退化到 ReAct（不挂整个任务）
- 节点 goal 描述太宽泛 → 子 ReAct 步数爆炸；Planner prompt 里加"atomic goal"约束
- blackboard 某节点 answer 为空 → 下游节点收到 `null`，prompt 指引 "node X returned nothing; reconsider approach"

---

## 10.5 预算与超时矩阵（汇总）

| 难度 | max_steps | wall_seconds | max_tokens（累积） | Planner |
|---|---|---|---|---|
| easy | 16 | 300 | 50K | ReAct |
| medium | 20 | 500 | 80K | ReAct |
| hard | 24 | 700 | 120K | ReAct |
| extreme | 32 | 900 | 200K | Plan-Execute |

总超时 42000s = 11.67h。假设 20 个任务平均 600s，总 12000s，安全。

## 10.6 降级路径（全局预算告急时）

`TaskOrchestrator` 监控剩余 wall clock：

| 触发条件 | 动作 |
|---|---|
| 剩余 < 2h 且剩余任务 > 10 | 所有待处理任务强制 easy 配置（steps=8, wall=150s） |
| 剩余 < 30 min | 所有剩余任务跳过，写空 CSV |
| 剩余 < 5 min | 立刻 flush logs 退出 |

降级日志：
```json
{"event": "orchestrator_degrade", "reason": "deadline_pressure",
 "remaining_seconds": 1023, "remaining_tasks": 15, "new_policy": "easy_fallback"}
```

## 10.7 Answer 格式决策树

```
question 关键词：
├─ "list all X" / "show X" → columns=[X]，rows 全列出
├─ "top N" / "first N" / "largest N" → 严格 N 行
├─ "count" / "how many" → 1 行 1 列 [[N]]
├─ "average" / "sum" / "total" → 1 行 1 列 [[value]]
├─ "for each X, show Y" → columns=[X, Y]，按 X 分组
├─ "compare X and Y" → columns=[metric, X_value, Y_value] 或拆两行
└─ "when" / "which date" → columns=[date]，rows=[[ISO 格式]]
```

列名选择（虽不评分，但 debug 友好）：
- 优先用 question 里的词（"region", "quarter", "revenue"）
- 不要用中文（避免 CSV 编码问题；gold 也是英文）
- 统一 snake_case

## 10.8 Playbook 与 prompt 集成

`PromptBuilder.build_system(task)` 根据 `task.difficulty` 拼接：
```
[7.2 base system prompt]
+
[10.1/10.2/10.3/10.4 difficulty 附加段]
+
[7.10 data profile summary]
```

## 10.9 未来迭代（v2+）

- **few-shot 示例库**：每档难度维护 3-5 个成功 case 的 (question, plan, answer) 三元组，按语义检索后作为 in-context example。需拿到至少一次提交分数后做，避免早期过拟合 demo
- **自适应难度**：不仅看 `task.difficulty`，也看 `profile.sources` 类型数量决定 Planner
- **节点级并行**（Plan-Execute）：无 deps 关系的节点并行跑（需 tool 调用改造为 async）
