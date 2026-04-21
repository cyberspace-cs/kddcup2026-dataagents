# 4. 详细设计（Detailed Design）

> 本文件定义数据结构、接口契约、关键流程的可实现级细节。不包含业务代码——仅接口与状态机。

## 4.1 模块目录

```
src/dabench/
├── __init__.py
├── main.py                    # 容器 ENTRYPOINT；读 /input，写 /output
├── config.py                  # AppConfig dataclass + env 覆盖层
├── orchestrator.py            # TaskOrchestrator
│
├── agents/
│   ├── __init__.py
│   ├── react.py               # ReAct 循环
│   ├── plan_execute.py        # Plan-and-Execute（hard/extreme）
│   ├── prompt.py              # 所有 prompt 模板
│   ├── parser.py              # ResponseParser（JSON fence + <think> 剥离）
│   └── budget.py              # 步数/时间/token 预算管理
│
├── llm/
│   ├── __init__.py
│   ├── client.py              # LLMClient（OpenAI SDK 包装 + 重试）
│   └── env.py                 # 读 MODEL_API_URL/_KEY/_NAME
│
├── tools/
│   ├── __init__.py
│   ├── registry.py            # Tool 注册与调度
│   ├── base.py                # ToolSpec / ToolResult / AnswerTable
│   ├── filesystem.py          # list_context / read_csv / read_json / read_doc
│   ├── sqlite.py              # inspect_sqlite_schema / execute_context_sql
│   ├── python_exec.py         # execute_python（multiprocessing 沙箱）
│   ├── pdf.py                 # read_pdf / inspect_pdf
│   ├── image.py               # read_chart / inspect_image
│   ├── rag.py                 # build_index / search_docs
│   └── answer.py              # 终态 tool：answer
│
├── profiler/
│   ├── __init__.py
│   └── scan.py                # DataProfiler：context/ 扫描与摘要
│
├── io/
│   ├── __init__.py
│   ├── output_writer.py       # CSV 归一化写
│   └── logger.py              # /logs 结构化日志
│
└── utils/
    ├── __init__.py
    ├── retry.py               # 指数退避
    ├── timeout.py             # 跨平台超时
    └── paths.py               # 路径沙箱
```

## 4.2 核心数据结构

### 4.2.1 任务与上下文

```python
@dataclass(frozen=True, slots=True)
class TaskMeta:
    task_id: str           # "task_1"
    difficulty: Difficulty # Enum: EASY/MEDIUM/HARD/EXTREME
    question: str
    task_dir: Path         # /input/task_1
    context_dir: Path      # /input/task_1/context

@dataclass(frozen=True, slots=True)
class DataSourceSummary:
    path: Path
    kind: Literal["csv","json","sqlite","markdown","pdf","image","knowledge"]
    size_bytes: int
    # 结构化摘要（按 kind 不同填不同字段）
    columns: list[str] | None = None    # csv/sqlite
    row_count: int | None = None        # csv/sqlite
    sample_rows: list[dict] | None = None
    tables: list[str] | None = None     # sqlite: 表名
    top_keys: list[str] | None = None   # json
    page_count: int | None = None       # pdf
    preview: str | None = None          # md/txt

@dataclass(frozen=True, slots=True)
class TaskProfile:
    meta: TaskMeta
    sources: list[DataSourceSummary]
    knowledge_text: str | None          # knowledge.md 全文
    has_images: bool                    # 是否含 chart/image
```

### 4.2.2 Agent 状态

```python
class AgentState:
    """可变，每任务独立实例"""
    task: TaskMeta
    profile: TaskProfile
    messages: list[ChatMessage]         # 与 LLM 的对话历史
    steps: list[AgentStep]              # 每步记录（thought/action/obs）
    answer: AnswerTable | None = None
    failure_reason: str | None = None
    tokens_used: int = 0
    wall_time_seconds: float = 0.0

@dataclass(frozen=True, slots=True)
class AgentStep:
    step_index: int
    thought: str
    action: str
    action_input: dict
    raw_response: str
    observation: dict | None           # tool 执行结果
    ok: bool
    latency_ms: int
```

### 4.2.3 Tool 契约

```python
@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict                 # JSON Schema（不是 starter kit 的示例 dict）
    handler: Callable[[TaskMeta, dict], ToolResult]
    is_terminal: bool = False

@dataclass(frozen=True, slots=True)
class ToolResult:
    observation: dict                  # {"ok": bool, "data"/"error": ...}
    is_terminal: bool = False
    answer: AnswerTable | None = None  # 仅当 is_terminal=True 时

@dataclass(frozen=True, slots=True)
class AnswerTable:
    columns: list[str]                 # 列名（不参与评分）
    rows: list[list[Any]]              # 二维原始值
```

### 4.2.4 预算

```python
@dataclass
class TaskBudget:
    max_steps: int                     # 按难度 16/24/32
    wall_seconds: int                  # 300/600/900
    max_tokens: int                    # 上下文窗口预算（留 20% 给输出）
    start_ts: float

    def exceeded(self) -> TaskAbortReason | None: ...
```

## 4.3 LLMClient 接口

```python
class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 120.0, max_retries: int = 3): ...

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop: list[str] | None = None,
    ) -> CompletionResult:
        """
        调用 OpenAI Chat Completions API。
        - 重试：指数退避（1s, 2s, 4s），仅对 APIError / Timeout / 5xx
        - 返回 content 已剥离 <think>...</think>（若存在）
        - 消息中若含 image，自动走 content blocks 格式
        """

    @classmethod
    def from_env(cls) -> "LLMClient":
        """
        读取 MODEL_API_URL / MODEL_API_KEY / MODEL_NAME。
        缺失时：fallback 到 YAML；都没有则 RuntimeError
        """

@dataclass(frozen=True)
class CompletionResult:
    content: str                       # 已剥离 <think>
    reasoning: str | None              # <think> 内容（可选记录）
    usage: TokenUsage
    latency_ms: int
    raw: Any                           # 保留原始响应用于 debug
```

**重试策略细节**：
- 指数退避基数 1s，倍数 2，抖动 ±250ms
- 只对 `openai.APIError`、`openai.APITimeoutError`、`httpx.ReadTimeout` 重试
- 4xx（包括 429）：首次 429 等 5s 再试；其他 4xx 直接抛
- 超时：单次 120s（Qwen 输出长）；全局 15 分钟最大累计

## 4.4 ReAct 循环状态机

```
         ┌──────────────────────────────┐
         │ INIT                         │
         │  - 载入 profile              │
         │  - 构造 system + task prompt │
         │  - state.messages = [...]    │
         └──────────┬───────────────────┘
                    ▼
         ┌──────────────────────────────┐
  ┌─────▶│ THINK                        │
  │      │  - LLM.complete(messages)    │
  │      │  - parse → thought/act/input │
  │      └──────────┬───────────────────┘
  │                 │
  │                 ├── 解析失败 ──┐
  │                 │              ▼
  │                 │       ┌──────────────────┐
  │                 │       │ RETRY_PARSE      │
  │                 │       │ 重提示，计数+1   │
  │                 │       │ >2 → FAIL        │
  │                 │       └──────────┬───────┘
  │                 │                  │
  │                 ▼                  │
  │      ┌──────────────────────────────┐
  │      │ ACT                          │
  │      │  - Registry.dispatch(...)    │
  │      │  - 超时/异常 → obs.ok=false  │
  │      │  - is_terminal? → DONE       │
  │      └──────────┬───────────────────┘
  │                 │
  │                 ▼
  │      ┌──────────────────────────────┐
  │      │ OBSERVE                      │
  │      │  - obs 追加到 messages       │
  │      │  - step_index++              │
  │      │  - budget.exceeded? → FAIL   │
  │      └──────────┬───────────────────┘
  │                 │
  └─────────────────┘  （loop back to THINK）

DONE: OutputWriter.write(task, answer) → /output/task_<id>/prediction.csv
FAIL: OutputWriter.write_empty(task) + log failure_reason
```

**关键不变式**：
- `len(state.messages)` 随步数单调递增
- `state.answer is not None` ⇔ 循环通过 DONE 退出
- 无论 DONE/FAIL，**必写** `/output/task_<id>/prediction.csv`（失败时写空 header "answer"）
- trace.json 在每步后 flush（防进程被强杀后丢失）

## 4.5 Plan-and-Execute（hard/extreme）

适用触发：`difficulty in {HARD, EXTREME}` **或** context 文件数 > 10 或包含 ≥2 种数据源类型。

```
1. PLAN 阶段：
   输入：question + profile
   输出：Plan DAG
     {
       "nodes": [
         {"id": "n1", "goal": "从 sales.csv 找 Q3 top-3 region",
          "deps": [], "tool_hint": "execute_python"},
         {"id": "n2", "goal": "从 report.pdf 读 Q3 target",
          "deps": [], "tool_hint": "read_pdf"},
         {"id": "n3", "goal": "计算 n1 和 n2 的 diff%",
          "deps": ["n1","n2"], "tool_hint": "execute_python"}
       ],
       "final_node": "n3"
     }

2. EXECUTE 阶段：
   按拓扑序执行每个 node
   每个 node 内部跑一个小 ReAct（max_steps=6）
   node.result 写入共享 blackboard
   final_node 的结果转成 AnswerTable

3. REFLECT 阶段（仅 extreme）：
   若 Plan 失败，重新规划一次
```

**为什么不直接用 LangGraph/AutoGen**：额外依赖 + 容器体积 + 学习成本。我们自己实现 DAG 调度仅 ~100 LOC，可控。

## 4.6 Tool 清单与扩展

### 4.6.1 继承自 starter kit（微调）

| Tool | 输入 | 输出 | 改动 |
|---|---|---|---|
| `list_context` | `{max_depth?:int}` | 文件树 JSON | 默认 max_depth=3 |
| `read_csv` | `{path, max_rows?, columns?}` | 预览 + dtype | 加 `columns` 选列；用 polars 应对大文件 |
| `read_json` | `{path, max_chars?}` | 字符串预览 + top keys | — |
| `read_doc` | `{path, start_line?, line_count?, keyword?}` | 行分页预览 + 总行数/范围/has_more | 支持 keyword 首个命中窗口 |
| `inspect_sqlite_schema` | `{path}` | 表与列列表 | — |
| `execute_context_sql` | `{path, sql, limit?}` | 行列表 | — |
| `execute_python` | `{code, timeout?}` | stdout/stderr/return | timeout 可调 30-120s |
| `answer` | `{columns, rows}` | — | 终态 |

### 4.6.2 新增

| Tool | 输入 | 输出 | 实现要点 |
|---|---|---|---|
| `read_pdf` | `{path, pages?, mode?:"text"|"tables"|"all"}` | 文本/表格 JSON | pdfplumber 主；PyMuPDF fallback；扫描版走 OCR |
| `inspect_pdf` | `{path}` | 页数、首页摘要、是否含表 | 预扫元信息 |
| `read_chart` | `{path}` | 图表描述 JSON | **主路径**：传 Qwen vision；**兜底**：paddleocr + 启发式 |
| `inspect_image` | `{path}` | dimensions + 是否有 OCR 可读文本 | 用于 Agent 判断走哪条路 |
| `search_knowledge` | `{query, k?}` | top-k 片段 | 对 knowledge.md + doc/ 建 FAISS index；embedding 用 `bge-small` 或 `text2vec-base` CPU 版 |

### 4.6.3 JSON Schema 示例（answer）

```json
{
  "name": "answer",
  "description": "Submit the final answer table and terminate the task. After calling answer, no further actions are allowed.",
  "input_schema": {
    "type": "object",
    "properties": {
      "columns": {"type": "array", "items": {"type": "string"}, "minItems": 1},
      "rows": {"type": "array", "items": {"type": "array"}, "minItems": 1}
    },
    "required": ["columns", "rows"]
  }
}
```

## 4.7 PromptBuilder 模板骨架

### 4.7.1 System Prompt（核心段落）

```
You are an autonomous data analysis agent. You receive a dataset and a question.
Your job: decompose the question, invoke tools, and produce a CSV answer.

CONSTRAINTS (read carefully):
- You MUST end with an `answer` action that submits columns + rows.
- Column names are IGNORED during scoring; values are matched as unordered vectors.
- Numbers are rounded to 2 decimals; keep sufficient precision.
- Empty values MUST be empty strings, not "NULL" or "NaN".
- For datetime values: with timezone → UTC (Z suffix); without timezone → ISO as-is.

RESPONSE FORMAT (strict):
You MUST respond with exactly one fenced JSON block:
```json
{
  "thought": "brief reasoning",
  "action": "<tool_name>",
  "action_input": { ... }
}
```
No other text. No multiple blocks.

TOOLS AVAILABLE:
{tool_descriptions}

DATA CONTEXT (pre-scanned):
{data_profile_summary}
```

### 4.7.2 Task Prompt
```
Question: {task.question}
Difficulty: {task.difficulty}

Begin. Respond with your first JSON action.
```

### 4.7.3 Observation Prompt
```
Observation:
{json.dumps(observation, indent=2, ensure_ascii=False)[:4000]}
```

截断到 4000 chars，避免上下文爆炸。长结果用 summary，详情让 Agent 再查。

## 4.8 OutputWriter 归一化规则

```python
def normalize_cell(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, (int, float, Decimal, np.integer, np.floating)):
        return f"{float(v):.2f}".rstrip("0").rstrip(".") or "0"
        # 或：保留原始精度，不做 strip；让评测端的 round(x, 2) 去做（§6.4）
    if isinstance(v, (pd.Timestamp, datetime)):
        if v.tzinfo is not None:
            return v.astimezone(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return v.isoformat()
    s = str(v).strip().replace("\r\n", "\n")
    return s
```

写 CSV 用标准库 `csv.writer(quoting=csv.QUOTE_MINIMAL)`。

## 4.9 可观测性

### 4.9.1 日志结构

`/logs/main.log` — 文本日志，等级 INFO，格式 `[%(asctime)s] %(levelname)s %(name)s: %(message)s`

`/logs/task_<id>.jsonl` — 每行一个 JSON，追加式：
```json
{"ts": "2026-04-24T03:15:22Z", "event": "step",
 "task": "task_42", "step": 3,
 "thought": "...", "action": "read_csv", "action_input": {...},
 "observation_ok": true, "latency_ms": 4231, "tokens_in": 1024, "tokens_out": 87}
```

`/logs/summary.jsonl` — 所有任务结束后汇总
```json
{"task_id": "task_42", "succeeded": true, "steps": 7,
 "wall_seconds": 123.4, "tokens": 18234, "answer_rows": 12}
```

### 4.9.2 指标

- `tasks_total`, `tasks_succeeded`, `tasks_failed`, `tasks_empty`
- `step_latency_p50/p95/p99`
- `llm_retries_total`
- `tool_errors_by_name{read_csv: 3, execute_python: 1}`

## 4.10 错误处理与降级矩阵

| 场景 | 动作 | 影响 |
|---|---|---|
| LLM API 单次失败 | 指数退避重试 3 次 | 延迟 ~7s |
| LLM API 持续失败 >3 次 | 本任务 FAIL；写空 CSV；继续下个任务 | 本任务 0 分 |
| tool handler 抛异常 | 捕获 → obs.ok=false | Agent 下一步决策 |
| 单任务超时 | 子进程强杀，写空 CSV | 本任务 0 分 |
| OOM risk（polars 超 16GB） | 切 streaming / chunked 读 | ~2x 慢 |
| 全局剩余时间 < 2h 且剩余任务 >10 | 降级：max_steps=8, max_tokens=2048 | 精度降低，完成度提升 |
| 全局剩余时间 < 30min | 剩余任务全跳过写空 | 保已完成任务不丢 |

## 4.11 配置样例

### 4.11.1 `configs/eval.yaml`（评测用）

```yaml
agent:
  planner: adaptive           # easy/medium→react; hard/extreme→plan_execute
  max_steps_by_difficulty:
    easy: 16
    medium: 20
    hard: 24
    extreme: 32
  temperature: 0.0
  max_output_tokens: 4096
  # model 字段留空；运行时读 MODEL_NAME env

llm:
  # base_url/key 留空；运行时读 MODEL_API_URL/MODEL_API_KEY
  timeout_seconds: 120
  max_retries: 3

runtime:
  input_root: /input
  output_root: /output
  logs_root: /logs
  max_workers: 4
  task_wall_seconds_by_difficulty:
    easy: 300
    medium: 500
    hard: 700
    extreme: 900
  global_wall_seconds: 42000   # 11.67h，留 20min 安全余量

tools:
  execute_python:
    subprocess_timeout: 60
    mem_limit_mb: 8192
  read_csv:
    default_max_rows: 50
    large_file_threshold_mb: 100   # > 100MB 走 polars lazy
```

## 4.12 接口契约（与 starter kit 的对照）

为降低 fork 摩擦，保持下列接口与 starter kit 一致：
- `data_agent_baseline.benchmark.dataset.DABenchPublicDataset` → 改名 `dabench.profiler.TaskDataset`，API 相同
- `ToolSpec(name, description, input_schema, handler)` → 保持字段名
- `AnswerTable(columns, rows)` → 保持

不一致处：
- Config 加 env 覆盖
- Output path `/output/<task_id>/prediction.csv`（去掉 `<run_id>` 层）
- Trace 写 `/logs/task_<id>.jsonl`（覆盖 starter kit 的 `artifacts/runs/...` 路径）
