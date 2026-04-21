# 3. 系统架构（Architecture）

> **方法论**：C4 模型（Context → Container → Component → Code）。本文件止于 Component 层；Code 层（类图、详细接口）移交 `04-detailed-design.md`。

## 3.1 Context 图（外部参与者）

```
┌────────────────────┐         Submit .tar.gz + Drive link
│  参赛者 (User)     │ ────────────────────────────────────────┐
│  @ready2dev...     │                                         │
└────────────────────┘                                         ▼
          │ dev-loop                                 ┌──────────────────────┐
          │ (DeepSeek / Qwen-Max)                    │ 组委会评测系统       │
          ▼                                           │  (HKUST-GZ infra)   │
┌────────────────────┐                                │                      │
│  KDD Data Agent    │ ◄──── MODEL_API_URL / KEY ─────┤  vLLM Qwen3.5-35B   │
│  (本项目，容器内)  │         OpenAI-compat HTTP      │  A3B 推理服务        │
└────────────────────┘                                │                      │
          │                                           │                      │
          │ 写 prediction.csv                         │  Scorer              │
          ▼                                           │  Recall-λ*Penalty    │
      /output/ ───────────────────────────────────► │                      │
                                                      └──────────────────────┘
                                                                │
                                                                ▼
                                                      Public Leaderboard
```

**关键外部接口**：
- **输入**：`/input/task_<id>/task.json` + `context/` 文件树（组委会挂载，只读）
- **输出**：`/output/task_<id>/prediction.csv`（UTF-8，唯一被评分文件）
- **辅助输出**：`/logs/*.log`（组委会在异常时取走）
- **网络**：仅 `MODEL_API_URL`（内网），其余端口全封

## 3.2 Container 图（进程级架构）

评测时容器内只有 **一个进程树**（Python 主进程 + multiprocessing 子进程）。

```
Docker Container (16 vCPU / 64GB / no GPU / no internet)
│
├── ENTRYPOINT: python -m dabench.main
│                    │
│                    ▼
│   ┌───────────────────────────────────────────────────────────┐
│   │  主进程：TaskOrchestrator                                 │
│   │  - 扫 /input 所有 task_<id>                              │
│   │  - 调度（线程池 max_workers=N 并行；N 按 vCPU 与内存估） │
│   │  - 每任务派发到 Agent Runtime                              │
│   │  - 收集 prediction.csv 写 /output                          │
│   └───────────────────────────────────────────────────────────┘
│                    │
│                    ├── 线程 worker 1 ──┐
│                    ├── 线程 worker 2 ──┤
│                    └── 线程 worker N ──┤
│                                         │
│                                         ▼
│                       ┌─────────────────────────────────┐
│                       │  Agent Runtime (per task)       │
│                       │  - ReAct 循环（max_steps 动态） │
│                       │  - 调 LLM（OpenAI SDK）         │
│                       │  - 调 Tool                      │
│                       └─────────────────────────────────┘
│                                         │
│                                         │ fork
│                                         ▼
│                       ┌─────────────────────────────────┐
│                       │  Tool 执行子进程                │
│                       │  (multiprocessing.Process)      │
│                       │  - execute_python 沙箱          │
│                       │  - SQLite 只读                  │
│                       │  - PDF 解析                     │
│                       │  - Image/Chart 解析             │
│                       └─────────────────────────────────┘
│
└── 挂载：/input (ro) /output (rw) /logs (rw)
              │              │            │
              │              │            └─── logs/main.log
              │              │                 logs/task_<id>.jsonl (trace)
              │              └─── task_<id>/prediction.csv
              └─── 组委会提供
```

**为什么不用更细粒度的微服务/容器拆分**：
- 规则明确"每个 submission 只有一个 container，运行独立全程"（§5.5）
- 进程内 ThreadPoolExecutor 即可榨干 16 vCPU；I/O 为主，GIL 不阻塞
- 避免 IPC 开销

## 3.3 Component 图（Agent 内部）

```
┌─────────────────────────────────────────────────────────────────┐
│  Agent Runtime （per task）                                     │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │ TaskContext  │   │  Planner     │   │ PromptBuilder      │  │
│  │ (元数据缓存) │──▶│ (ReAct /      │──▶│ (system/task/obs) │  │
│  │              │   │  Plan-Exec)  │   │                    │  │
│  └──────────────┘   └──────────────┘   └────────────────────┘  │
│         │                  │                       │            │
│         ▼                  ▼                       ▼            │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │ DataProfiler │   │  LLMClient   │◄──┤ 消息数组           │  │
│  │ (启动扫描)   │   │  (OpenAI SDK)│   │                    │  │
│  └──────────────┘   └──────────────┘   └────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│                     ┌─────────────────┐                         │
│                     │ ResponseParser  │                         │
│                     │ (JSON / XML)    │                         │
│                     └─────────────────┘                         │
│                            │                                    │
│                            ▼                                    │
│                     ┌─────────────────┐                         │
│                     │ ToolDispatcher  │                         │
│                     │  + Registry      │                         │
│                     └─────────────────┘                         │
│                            │                                    │
│     ┌──────────┬──────────┼──────────┬──────────┬───────────┐  │
│     ▼          ▼          ▼          ▼          ▼           ▼  │
│  list_ctx  read_csv   read_doc   sql_query  exec_py    read_img│
│  inspect    read_json inspect_           +rag_search   inspect_│
│  _ctx                  _pdf                              chart │
│     │                                                           │
│     ▼                                                           │
│  ┌─────────────────────────────────┐                           │
│  │ answer (terminal tool)           │                           │
│  │  → AnswerTable {columns, rows}   │                           │
│  │  → is_terminal=True              │                           │
│  └─────────────────────────────────┘                           │
│                            │                                    │
│                            ▼                                    │
│                     ┌─────────────────┐                         │
│                     │ OutputWriter    │                         │
│                     │ (CSV normalize) │                         │
│                     └─────────────────┘                         │
│                            │                                    │
└────────────────────────────┼────────────────────────────────────┘
                             ▼
                 /output/task_<id>/prediction.csv
```

### 3.3.1 每个 Component 的职责

| Component | 职责 | 继承自 starter？ | 备注 |
|---|---|---|---|
| `TaskContext` | 任务元数据 + 缓存预扫描结果（文件清单、schema） | 否（新增） | 保 DataProfiler 的结果 |
| `DataProfiler` | 启动时扫描 context/ 下所有数据源，生成结构摘要卡片 | 否（新增） | **对 hard/extreme 至关重要**，避免 Agent 自己慢慢探索 |
| `Planner` | 根据难度选择 ReAct or Plan-and-Execute | 否（替换 starter 的 react.py） | easy/medium → ReAct；hard/extreme → Plan-Exec |
| `PromptBuilder` | 组装 system + task + observations | 继承 starter prompt.py 结构 | 按 Qwen 风格重写 |
| `LLMClient` | 封装 OpenAI SDK，补**重试/超时/env 覆盖** | 继承 starter model.py 但大改 | 加 backoff、image_url 支持 |
| `ResponseParser` | 提取模型输出里的 action/action_input | 继承 starter react.py 的 fence 解析 | 加 `<think>` 剥离容错 |
| `ToolDispatcher` + `Registry` | tool schema → 执行器映射 | 继承 starter registry.py | 扩展 PDF/chart 工具 |
| `OutputWriter` | 按规则 §2.4 / §6.5 归一化后写 CSV | 否（新增） | 数字保留 2 位；空值空串；时间 UTC Z |
| `TaskOrchestrator` | 遍历 /input，调度 Agent，收集输出，全局超时 | 继承 starter runner.py | 加 task-level budget、/logs 写入 |

## 3.4 关键数据流

### 3.4.1 单任务流（主路径）

```
1. TaskOrchestrator 读 /input/task_<id>/task.json
2. DataProfiler 扫 context/：
   - csv/    → 每文件：行数、列名、dtype、前 5 行样本
   - db/     → 每库：表名、每表 schema、行数
   - json/   → 前 2KB 预览 + 顶层 key
   - doc/    → 文件名、页数、首页摘要
   - knowledge.md → 全文
3. Planner 选策略：
   - easy/medium → ReAct，max_steps=16, budget=300s
   - hard        → ReAct，max_steps=24, budget=600s，开启"分支合并"提示
   - extreme     → Plan-and-Execute，先规划 DAG，每节点子 ReAct
4. 循环：
   a. PromptBuilder 组装消息（含 profile 摘要作为 "Data Context" 段）
   b. LLMClient.complete(messages) → 响应
   c. ResponseParser 提取 action / action_input
   d. ToolDispatcher 执行 → observation
   e. 判断 terminal（action == "answer"）
5. OutputWriter 接收 AnswerTable，归一化后写 /output/task_<id>/prediction.csv
6. TaskOrchestrator 追加 /logs/task_<id>.jsonl（trace）
```

### 3.4.2 异常路径

| 异常 | 处理 |
|---|---|
| LLM API 瞬时失败 | 指数退避，最多 3 次（避免在某次任务里因抖动失败） |
| Tool 执行异常 | 捕获为 observation（`{ok:false, error:str}`），回送 Agent 做下一步决策 |
| 单任务 budget 耗尽 | 强制终止子进程，写空 prediction.csv（空 CSV 比缺文件好：规则 §6.6 "missing file → 0"） |
| 解析 action 失败 | 回发 observation："Your previous response was not valid JSON. Respond again." 最多 2 次，否则放弃任务 |
| 全局 budget 预警（剩 <1h 而剩 >5 个任务） | 降级为 easy-only 策略（max_steps=8），保完成度 |

## 3.5 部署视图

### 3.5.1 开发时（本机）

```
E:\KDD比赛\
├── src/dabench/               ← Fork 自 starter kit，改名避免冲突
├── data/public/input/         ← Phase 1 demo 数据（百度/Drive 下）
├── data/public/output/        ← gold.csv（对拍用）
├── configs/
│   ├── dev.yaml               ← 指向 DeepSeek/Qwen-Max 开发 API
│   └── eval.yaml              ← 模拟评测（读 env）
├── scripts/
│   ├── eval_local.sh          ← 模拟组委会评测 docker run
│   └── submit.sh              ← 打 tar.gz + 生成邮件模板
├── docker/
│   └── Dockerfile
└── artifacts/                 ← 本地运行产物
```

### 3.5.2 评测时（容器内）

```
/
├── app/                       ← 镜像内 COPY 来的源码
│   └── dabench/
├── usr/local/bin/python3.11
└── （挂载点）
    ├── input/                 (ro，组委会)
    ├── output/                (rw，组委会预创建)
    └── logs/                  (rw，组委会取)
```

## 3.6 关键设计决策（ADR 简版）

### ADR-001: ReAct 文本 JSON 协议 vs OpenAI 原生 tool calling
- **选择**：**原生 function calling 为默认，文本 JSON 协议为降级**（原决定翻转）
- **背景**：starter kit 原本沿用文本协议，M4+ 冲分阶段切为原生以提升格式稳定性并解锁 parallel tool calls
- **理由（翻转决定）**：
  - 评测端 vLLM 已启用 `--enable-auto-tool-choice --tool-call-parser qwen3_coder`（见 `docs/01-requirements.md:73`），原生协议在比赛环境是一等公民
  - Qwen3.5-35B-A3B 后训练对 tool use 格式做了专门强化，原生协议的格式错误率系统性低于文本协议；每个 JSON fence 解析失败都会吃掉一个 `max_steps` 预算，这是硬损失
  - 原顾虑「`enable_thinking` + `tool_calls` 交互未知」已被 vLLM `--reasoning-parser qwen3` 把 `<think>` 块塞进 `message.reasoning_content`、不污染 `message.content` / `message.tool_calls` 的事实消解
  - 支持一轮多 `tool_calls`（parallel），有机会压缩步数预算
- **文本路径保留**为：
  - 端点不支持 `tools=` 字段时的兜底
  - 离线调试 / 非 Qwen 开发替身
  - A/B 对比时的基线
- **切换开关**：`AgentConfig.tool_protocol: "native" | "text"`（默认 `native`）

### ADR-002: Planner 分层（ReAct + Plan-Execute）
- **选择**：easy/medium 用 ReAct，hard/extreme 用 Plan-and-Execute
- **理由**：hard/extreme 任务推理 DAG 复杂，纯 ReAct 容易迷路 + 步数爆炸；Plan-Exec 先出总计划再执行，更稳
- **代价**：两套代码路径，测试面增加

### ADR-003: Docker 基础镜像 `python:3.11-slim` + uv
- **选择**：`python:3.11-slim` + `uv pip install --system`
- **理由**：
  - slim 约 45MB，加全部依赖 + duckdb/pyarrow/pandas 约 800MB-1.5GB
  - uv 安装速度与 reproducibility 都优
  - 镜像总大小预估 2-3 GB 压缩后 ≤ 1 GB，远在 10 GB 上限内
- **替代**：`nvidia/cuda` 被排除——规则明确无 GPU；`alpine` 被排除——pandas/pyarrow wheel 兼容性差

### ADR-004: 并行策略
- **选择**：任务间 `ThreadPoolExecutor(max_workers=4)`；任务内串行
- **理由**：
  - 任务间：I/O-bound（主要等 LLM），线程即可；4 路与 16 vCPU 不完全饱和，但 LLM API 本身有并发上限
  - 任务内并行：ReAct 步是串行依赖，无法并行；Plan-Exec 内部 DAG 节点理论上可并行，但增加实现复杂度，**延后到 v2**
- **风险**：若 LLM API 对单客户端有并发限制（如 TPM），并发路数过高会触发限流；需监控后调

### ADR-005: 输出归一化放在 OutputWriter 而非 Agent
- **选择**：Agent 输出 `AnswerTable {columns, rows}` 时不做归一化；写 CSV 时统一做（数字→2 位、时间→UTC Z、空→空串）
- **理由**：
  - 模型输出不稳定，不要让 Agent 承担格式纪律
  - 归一化逻辑集中，一处修可影响全体
  - 评分规则是"写入前归一化"（§6.5），我们做一遍等价的归一化可避免双倍误差

### ADR-006: 视觉任务主路径 vs 兜底
- **主路径**：直接把 image_url（base64 或容器内路径 file://）传给 Qwen3.5-35B-A3B API
- **兜底**：pdfplumber + PyMuPDF 提文本；图表走 OCR（Tesseract 或 PaddleOCR CPU 版）+ 表格抽取规则
- **理由**：主路径风险点是评测服务可能禁用视觉（见 R1）；兜底必须在 v1 就有，不能 v1 拿 0 分再补

## 3.7 架构质量属性（Non-Functional Attributes）

| 属性 | 目标 | 达成方法 |
|---|---|---|
| 可评测性 | 本地 `eval_local.sh` 完全等价于组委会调用 | Docker run 使用 `--network=none`、`--cpus=16`、`--memory=64g`；环境变量注入；只读 `/input` |
| 可观测 | 每任务 trace 可复现 | `/logs/task_<id>.jsonl` 每步 append，含 messages + tool_calls + observations |
| 确定性 | 同一 seed + 同一输入 → 同一输出 | Qwen 服务固定 seed；我方 temperature=0.0（虽官方推荐 1.0，trade-off 取确定性）；`random.seed(42)` |
| 健壮性 | 单任务失败不影响其他 | 每任务独立子进程；父进程捕获 `Process.join` 超时与非零返回 |
| 可维护性 | 新 tool 20 LOC 可接入 | 通过 `Registry.register(ToolSpec)` 装饰器 |
| 资源保护 | 不超 64GB/16vCPU | 按 max_workers 限并发；pandas 大文件用 polars/duckdb 流式读 |

## 3.8 开放问题（架构层）

1. **Plan-and-Execute 对 hard/extreme 是否真比 ReAct 强？** 需 demo 数据上做 A/B，v1 不搞，v2 对比后决定
2. **DataProfiler 的成本**：若每任务扫描 10+ 大 CSV 花 30s，12h 预算里不小；需加 lazy + 抽样
3. **并发路数 N**：未知评测端对我方 API Key 的 QPS 限制，先定 4，观察后调整
4. **CSV 归一化是否破坏原文**（如用户期望保留"1,200"字符串而非数字 1200）：默认启发式，需 demo 数据验证
