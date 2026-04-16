# 6. 接口契约（Interfaces）

> 本文件给每个模块的**公共 API** 以 Python 类型签名形式定义契约（非实现）。
> 所有 signature 经检视后视为"冻结"——实现时不得修改语义；如需修改，先修本文件再改代码。
> 私有函数（下划线开头）不在本文件管辖范围。

## 6.1 包与命名空间

```python
dabench                          # 顶层包
├── __init__.py                  # exports: __version__
├── main                         # python -m dabench.main
├── config                       # AppConfig, load_config
├── orchestrator                 # TaskOrchestrator
├── agents
│   ├── __init__.py              # re-export: Planner, AgentState
│   ├── react                    # ReActPlanner
│   ├── plan_execute             # PlanExecutePlanner
│   ├── prompt                   # build_* functions
│   ├── parser                   # ResponseParser
│   └── budget                   # TaskBudget
├── llm
│   ├── client                   # LLMClient, CompletionResult
│   └── env                      # load_llm_from_env
├── tools
│   ├── base                     # ToolSpec, ToolResult, AnswerTable
│   ├── registry                 # ToolRegistry
│   ├── filesystem / sqlite / python_exec / pdf / image / rag / answer
├── profiler
│   └── scan                     # DataProfiler
├── io
│   ├── output_writer            # OutputWriter
│   └── logger                   # StructuredLogger
└── utils
    ├── retry / timeout / paths
```

## 6.2 `dabench.config`

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Difficulty = Literal["easy", "medium", "hard", "extreme"]

@dataclass(frozen=True, slots=True)
class AgentConfig:
    planner: Literal["adaptive", "react", "plan_execute"] = "adaptive"
    max_steps_by_difficulty: dict[Difficulty, int] = field(default_factory=lambda: {
        "easy": 16, "medium": 20, "hard": 24, "extreme": 32,
    })
    temperature: float = 0.0
    max_output_tokens: int = 4096

@dataclass(frozen=True, slots=True)
class LLMConfig:
    base_url: str = ""        # env MODEL_API_URL 覆盖
    api_key: str = ""         # env MODEL_API_KEY 覆盖
    model: str = ""           # env MODEL_NAME 覆盖
    timeout_seconds: float = 120.0
    max_retries: int = 3
    backoff_base_seconds: float = 1.0
    backoff_multiplier: float = 2.0

@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    input_root: Path = Path("/input")
    output_root: Path = Path("/output")
    logs_root: Path = Path("/logs")
    max_workers: int = 4
    task_wall_seconds_by_difficulty: dict[Difficulty, int] = field(default_factory=lambda: {
        "easy": 300, "medium": 500, "hard": 700, "extreme": 900,
    })
    global_wall_seconds: int = 42000   # 11.67h

@dataclass(frozen=True, slots=True)
class ToolsConfig:
    execute_python_timeout_seconds: int = 60
    execute_python_mem_limit_mb: int = 8192
    read_csv_default_max_rows: int = 50
    read_csv_large_file_threshold_mb: int = 100
    read_doc_max_chars: int = 4000

@dataclass(frozen=True, slots=True)
class AppConfig:
    agent: AgentConfig
    llm: LLMConfig
    runtime: RuntimeConfig
    tools: ToolsConfig
    # 追加字段 → 默认值，避免老配置失效

def load_config(path: Path, *, env_overlay: bool = True) -> AppConfig:
    """
    加载 YAML → 合并默认值 → 若 env_overlay，用环境变量覆盖 LLMConfig 字段。
    
    解析顺序（后者覆盖前者）：
      1. AppConfig 全字段默认
      2. YAML 指定字段
      3. 环境变量 MODEL_API_URL/MODEL_API_KEY/MODEL_NAME（仅 LLMConfig）

    Raises
    ------
    FileNotFoundError: path 不存在
    ValueError: YAML 语法错误 / 字段类型不匹配
    """
```

**不变式**：`AppConfig` 是 `frozen=True`，运行期不可变更；加载后所有字段已确定。

## 6.3 `dabench.llm.client`

```python
from dataclasses import dataclass
from typing import Any

ChatRole = Literal["system", "user", "assistant", "tool"]

@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str | list[ContentBlock]   # list 用于多模态
    name: str | None = None              # 仅 role="tool"

@dataclass(frozen=True, slots=True)
class ContentBlock:
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: dict | None = None        # {"url": "file:///...", "detail": "auto"}

@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass(frozen=True, slots=True)
class CompletionResult:
    content: str                         # 已剥离 <think>...</think>
    reasoning: str | None                # <think> 内容（若有）
    usage: TokenUsage
    latency_ms: int
    raw_response: Any                    # 供 debug；不序列化进 trace 除非 debug mode

class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
    ) -> None: ...

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop: list[str] | None = None,
    ) -> CompletionResult:
        """
        Chat Completions 调用，OpenAI 兼容。
        
        - 自动重试：指数退避（base=1s, mul=2, jitter=±250ms），仅对 5xx / Timeout / APIError
        - 429 首次等 5s；其他 4xx 直接抛 RuntimeError
        - 消息含 image_url 时走 OpenAI multimodal content blocks
        - 返回 content 已剥离 <think> 块；原始思考保留在 reasoning

        Raises
        ------
        RuntimeError: 超过 max_retries 仍失败 / 4xx 硬失败 / 认证错误
        """

    @classmethod
    def from_env_or_config(
        cls,
        config: LLMConfig,
        *,
        require_env: bool = False,
    ) -> "LLMClient":
        """
        构造客户端。环境变量优先于 config。
        require_env=True 时，三个 env 必须存在，否则 RuntimeError
        （评测时用 require_env=True，开发时 False）。
        """
```

## 6.4 `dabench.tools.base`

```python
from typing import Callable, Any

@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str                           # 小写下划线
    description: str                    # 给 LLM 看，< 300 chars
    input_schema: dict                  # JSON Schema draft-07
    handler: Callable[["ToolContext", dict], "ToolResult"]
    is_terminal: bool = False

@dataclass(frozen=True, slots=True)
class AnswerTable:
    columns: list[str]
    rows: list[list[Any]]

    def __post_init__(self) -> None:
        # 校验：columns 非空；每行长度 == len(columns)
        ...

@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    observation: dict                   # 给 Agent 看；将 json.dumps 注入 messages
    error: str | None = None
    is_terminal: bool = False
    answer: AnswerTable | None = None   # 仅当 is_terminal=True
    latency_ms: int = 0

@dataclass(frozen=True, slots=True)
class ToolContext:
    """传给 handler 的上下文；每任务独立实例，只读。"""
    task: "TaskMeta"
    context_dir: Path                   # = task.context_dir
    profile: "TaskProfile"              # DataProfiler 的输出
    config: ToolsConfig
    logger: "StructuredLogger"
```

**契约约束**：
- `handler` 必须是纯函数式（给定 ctx + args → ToolResult；不改变外部状态除文件系统/日志外）
- `handler` 不抛异常；所有异常 → `ToolResult(ok=False, error=str(e))`（由 `ToolRegistry` 统一 catch）
- `answer` tool 是唯一 `is_terminal=True`

## 6.5 `dabench.tools.registry`

```python
class ToolRegistry:
    def __init__(self) -> None: ...

    def register(self, spec: ToolSpec) -> None:
        """注册 tool；重名 → ValueError"""

    def get(self, name: str) -> ToolSpec | None: ...

    def list_specs(self) -> list[ToolSpec]: ...

    def render_descriptions(self) -> str:
        """拼 prompt 用的 tool 列表文本（JSON Schema 格式化）"""

    def execute(
        self,
        ctx: ToolContext,
        name: str,
        action_input: dict,
    ) -> ToolResult:
        """
        调度入口。
        - 查 name；不存在 → ToolResult(ok=False, error="unknown tool: ...")
        - 校验 action_input 符合 spec.input_schema（jsonschema 库）；不合法 → ok=False
        - 调 handler，捕获任何异常 → ok=False, error=repr(e)
        - 记录 latency_ms
        """

# 预注册的全局 registry
DEFAULT_REGISTRY: ToolRegistry = ...  # 包含 filesystem/sqlite/python_exec/pdf/image/rag/answer
```

## 6.6 `dabench.agents`

```python
@dataclass(frozen=True, slots=True)
class TaskMeta:
    task_id: str
    difficulty: Difficulty
    question: str
    task_dir: Path
    context_dir: Path

@dataclass(frozen=True, slots=True)
class AgentStep:
    step_index: int
    thought: str
    action: str
    action_input: dict
    raw_response: str
    observation: dict | None
    ok: bool
    latency_ms: int
    tokens_in: int
    tokens_out: int

class AgentState:
    task: TaskMeta
    profile: "TaskProfile"
    messages: list[ChatMessage]
    steps: list[AgentStep]
    answer: AnswerTable | None
    failure_reason: str | None
    start_ts: float

    def append_step(self, step: AgentStep) -> None: ...
    def mark_answered(self, ans: AnswerTable) -> None: ...
    def mark_failed(self, reason: str) -> None: ...
    def to_dict(self) -> dict: ...   # 序列化到 trace.jsonl

class Planner(Protocol):
    """策略抽象。ReActPlanner / PlanExecutePlanner 实现此 Protocol。"""

    def run(
        self,
        task: TaskMeta,
        profile: "TaskProfile",
        *,
        budget: "TaskBudget",
        llm: LLMClient,
        tools: ToolRegistry,
        logger: "StructuredLogger",
    ) -> AgentState:
        """跑一次完整循环。返回终态 state（答案或失败）。"""

def build_planner(
    config: AgentConfig,
    difficulty: Difficulty,
) -> Planner:
    """按 config.planner + difficulty 选具体实现。"""
```

## 6.7 `dabench.agents.parser`

```python
@dataclass(frozen=True, slots=True)
class ParsedAction:
    thought: str
    action: str
    action_input: dict

class ResponseParser:
    @staticmethod
    def strip_thinking(raw: str) -> str:
        """移除 <think>...</think>（可能多个；贪婪匹配）。"""

    @staticmethod
    def extract_json_block(raw: str) -> str:
        """
        提取 ```json ... ``` 或 ``` ... ``` 的第一个 fence；
        无 fence 时尝试整段 json.loads。
        """

    @classmethod
    def parse(cls, raw: str) -> ParsedAction:
        """
        完整流程：strip_thinking → extract_json_block → loads → 校验 keys。
        
        Raises
        ------
        ParseError: 任一步骤失败
        """

class ParseError(RuntimeError):
    pass
```

## 6.8 `dabench.agents.budget`

```python
AbortReason = Literal[
    "max_steps_exceeded", "wall_seconds_exceeded",
    "tokens_exceeded", "global_budget_exceeded",
]

@dataclass
class TaskBudget:
    max_steps: int
    wall_seconds: int
    max_tokens_total: int
    start_ts: float
    # 外部监控器注入
    global_deadline_ts: float

    # 可变状态
    steps_used: int = 0
    tokens_used: int = 0

    def tick_step(self, tokens: int) -> None: ...

    def exceeded(self) -> AbortReason | None:
        """检查所有维度；None 表示仍可继续。"""
```

## 6.9 `dabench.profiler.scan`

```python
@dataclass(frozen=True, slots=True)
class DataSourceSummary:
    path: Path
    kind: Literal["csv", "json", "sqlite", "markdown", "pdf", "image", "knowledge"]
    size_bytes: int
    columns: list[str] | None = None
    dtypes: dict[str, str] | None = None
    row_count: int | None = None
    sample_rows: list[dict] | None = None
    tables: list[str] | None = None          # sqlite
    table_schemas: dict[str, list[str]] | None = None  # sqlite: table → columns
    top_keys: list[str] | None = None        # json
    page_count: int | None = None            # pdf
    preview: str | None = None               # md/txt 前 500 chars
    image_dims: tuple[int, int] | None = None

@dataclass(frozen=True, slots=True)
class TaskProfile:
    meta: TaskMeta
    sources: list[DataSourceSummary]
    knowledge_text: str | None               # knowledge.md 全文（可能长）
    has_images: bool
    summary_for_prompt: str                  # 压缩成 ≤1500 tokens 的摘要卡片

class DataProfiler:
    def __init__(self, config: ToolsConfig) -> None: ...

    def profile(self, task: TaskMeta) -> TaskProfile:
        """
        扫描 task.context_dir，生成 summary。
        - CSV：读首 N 行 + schema（polars.scan_csv → describe）
        - SQLite：pragma 列 schema + 行数估算
        - JSON：顶层 key + 前 2KB
        - MD/PDF：元信息 + 首段预览
        - knowledge.md：全文（不截断，但留意长度）
        
        性能约束：单任务 profile 总耗时 < 5s；>1GB 文件降级到元信息 only
        """
```

## 6.10 `dabench.io.output_writer`

```python
class OutputWriter:
    def __init__(self, output_root: Path) -> None: ...

    def write(self, task: TaskMeta, answer: AnswerTable) -> Path:
        """
        写到 output_root/task_<id>/prediction.csv
        - 先 mkdir parents=True, exist_ok=True
        - 每 cell 走 normalize_cell
        - csv.writer quoting=QUOTE_MINIMAL, lineterminator="\n"
        - UTF-8 encoding, no BOM
        返回写入的 Path。
        """

    def write_empty(self, task: TaskMeta, reason: str) -> Path:
        """任务失败时写占位 CSV（单列 "answer" 无数据）；log reason。"""

    @staticmethod
    def normalize_cell(v: Any) -> str:
        """
        规则 §6.5 的归一化：
        - None / NaN → ""
        - 数值 → str，保留足够精度（评测端会 round 到 2 位）
        - 时间 aware → UTC ISO 带 Z
        - 时间 naive → 原 ISO
        - 其他 → str(v).strip().replace("\r\n", "\n")
        """
```

## 6.11 `dabench.orchestrator`

```python
@dataclass(frozen=True, slots=True)
class TaskRunResult:
    task_id: str
    succeeded: bool
    prediction_path: Path | None
    failure_reason: str | None
    steps: int
    wall_seconds: float
    tokens_total: int

class TaskOrchestrator:
    def __init__(
        self,
        config: AppConfig,
        *,
        llm: LLMClient,
        tools: ToolRegistry,
        profiler: DataProfiler,
        writer: OutputWriter,
        logger: "StructuredLogger",
    ) -> None: ...

    def run_all(self) -> list[TaskRunResult]:
        """
        1. 扫 input_root/task_*，按 task_id 排序（数字自然序）
        2. ThreadPoolExecutor(max_workers=config.runtime.max_workers)
        3. 每任务：new subprocess → Planner.run → OutputWriter.write
        4. 全局 deadline 监控；近超时激活降级
        5. 汇总写 /logs/summary.jsonl
        """

    def run_one(self, task: TaskMeta) -> TaskRunResult:
        """供 CLI run-task 调用；不并行。"""
```

## 6.12 `dabench.io.logger`

```python
class StructuredLogger:
    def __init__(self, logs_root: Path) -> None: ...

    def main(self, level: str, msg: str, **kv: Any) -> None:
        """写 /logs/main.log（text）。"""

    def task(self, task_id: str, event: str, **kv: Any) -> None:
        """追加一行 JSON 到 /logs/task_<id>.jsonl。"""

    def summary(self, results: list[TaskRunResult]) -> None:
        """写 /logs/summary.jsonl（每任务一行）。"""

    def flush(self) -> None:
        """关键节点强制 fsync（避免进程被强杀丢日志）。"""
```

## 6.13 `dabench.main`

```python
def main() -> int:
    """
    ENTRYPOINT。
    1. load_config("/app/configs/eval.yaml", env_overlay=True)
    2. init logger / llm / tools / profiler / writer
    3. orchestrator.run_all()
    4. logger.flush()
    5. 返回 0 即使部分任务失败（只有系统级错误返回非零）
    """

if __name__ == "__main__":
    sys.exit(main())
```

**约定**：
- 进程 exit code 0 = 正常（即使有任务失败，但框架本身没崩）
- 非 0 = 系统级错误（配置缺失、LLM 完全连不上、OOM 等）——让组委会能通过 exit code 判别

## 6.14 错误类型体系

```python
# dabench.errors
class DabenchError(Exception):
    """基类"""

class ConfigError(DabenchError): pass
class LLMError(DabenchError): pass
class ToolError(DabenchError): pass
class ParseError(DabenchError): pass
class BudgetError(DabenchError): pass

# 分层原则：
# - 系统错误（ConfigError、LLMError init 阶段）→ main() 抛出，返回非零 exit
# - 任务错误（ToolError、ParseError、BudgetError）→ 捕获转 failure_reason，继续下一任务
```

## 6.15 冻结规则

- 所有 `@dataclass(frozen=True, slots=True)` 的字段**只能增加不能删除**；删除需要版本号 bump + 迁移说明
- `ToolSpec.input_schema` 的 JSON Schema 约束**可放宽不可收紧**（避免模型已训练的调用失败）
- 环境变量三件套 `MODEL_API_URL/MODEL_API_KEY/MODEL_NAME` 的读取优先级与 fallback 逻辑一旦确定**不得调整**（评测对接面）
