# 5. 里程碑、测试策略、风险登记、交付物

## 5.1 里程碑（Milestones）

**时间轴总览**（基准日期 2026-04-16 周四）：

```
4/16 周四 ──────────────── 4/24 周五 ─────────────────── 5/23 周六 ─── 5/27 ─── 6/30 ──── 8/9
                │                 │                          │            │        │        │
 M0 环境/注册   │  M1 Fork+通     │ M3 首次提交              │           │        │        │
 M0 (4/16-4/19) │   M2 Docker     │ M4 覆盖 medium           │ M5 H/X    │ Phase2 │ Phase2  │
                │   (4/20-4/23)   │ (4/25-4/30)              │ 冲榜      │ 开赛   │ 截止    │ 颁奖
                                    └─ Phase 1 (4/24-5/23) ──┘
```

### M0 — 环境就绪 + 注册（4/16 周四 → 4/19 周日）

**目标**：具备开发与注册所需的所有基础条件。

| 任务 | 负责 | 退出标准 |
|---|---|---|
| 注册比赛（队长） | 用户 | 收到 team_id 邮件（4/21-23 窗口） |
| 队员身份验证 | 用户 + 搭档 | 两人各自完成 Informed Consent Form |
| 下载 Phase 1 demo 数据 | 用户 | 百度网盘 `ymdc` 或 Google Drive 获取；数据在 `data/public/input/` 与 `data/public/output/` 下 |
| 装 Python 3.11 + uv | 用户 | `uv --version` 可执行 |
| 装 Docker Desktop（WSL2） | 用户 | `docker run hello-world` 成功；`.wslconfig` 调好 |
| 申请 **Qwen-Max** API key（DashScope） | 用户 | 可调通 `qwen-max-latest` 测试请求 |
| 加入官方 Discord / 微信群 | 用户 | 能发问（用它澄清 Q2-Q7） |
| Discord 问清 Q2-Q7 | 用户（发）/ 我（起草） | Q2、Q3 至少其一回复（最关键）；其余异步追 |

**退出标准（门禁）**：`uv`、`docker`、`Qwen-Max API`、`demo 数据` 四项 100% 可用。

### M1 — 官方 Starter Fork & 通跑（4/20 周一 → 4/22 周三）

**目标**：在本地用开发 API 跑通 starter kit 官方 baseline，确认环境没问题。

| 任务 | 退出标准 |
|---|---|
| `git clone` starter kit 到 `E:/KDD比赛/vendor/starter/` | 存在 |
| `uv sync` 安装依赖 | 无报错 |
| 填 `configs/react_baseline.example.yaml`（Qwen-Max）| — |
| `uv run dabench status --config ...` | 输出 task 数量 |
| `uv run dabench run-task task_1 --config ...` | 生成 prediction.csv + trace.json |
| `uv run dabench run-benchmark --config ... --limit 3` | 3 个任务全部跑完，至少 1 个 succeeded |
| 记录：每任务平均步数、tokens、耗时 | 写入 `docs/baseline_metrics.md` |

**退出标准**：能复现官方 baseline，有基线指标。

### M2 — Fork 定制 + Docker 化（4/20 并行开始，4/23 周四 截止）

**目标**：把 starter 改造成符合评测契约的镜像。这是**最关键的里程碑**。

| 任务 | 退出标准 |
|---|---|
| 新建 `src/dabench/` 模块骨架（按 §4.1） | 文件结构齐 |
| 移植 starter 的 tools/agents 核心代码 | 单元测试继承自 starter pass |
| 加 **env 覆盖层** `llm/env.py` | 单测：YAML 空 + env 存在 → 用 env |
| 改输出路径 `/output/<task_id>/prediction.csv`（去 run_id 层） | 集成测：本地 run → 文件路径正确 |
| 加 **main.py 容器 ENTRYPOINT** | `python -m dabench.main` 在本机跑通 demo |
| 加 `/logs/` 写入 | 文件存在、格式正确 |
| 放宽 `task.json` schema 到 subset 校验 | 单测：多字段不报错 |
| 编写 `docker/Dockerfile` | `docker build` 成功；镜像 < 3 GB |
| 编写 `scripts/eval_local.sh` 模拟评测 | `docker run --network=none --cpus=16 ... -v demo_input:/input:ro -v demo_output:/output` 能跑完 |
| `docker save` + gzip 检查大小 | ≤ 10 GB（实际应 < 2 GB） |
| OutputWriter 归一化单测 | 数字/日期/空值 case 全通过 |

**退出标准（硬指标）**：
- `bash scripts/eval_local.sh` 在本机跑通 Phase 1 demo 数据集
- 输出目录结构完全符合 `/output/task_<id>/prediction.csv`
- 镜像压缩后 ≤ 3 GB

### M3 — 首次提交 baseline（4/24 周五 开赛当天）

**目标**：占位上榜，确认提交链路通。

| 任务 | 退出标准 |
|---|---|
| 收到 team_id（最晚 4/23） | — |
| 镜像打 `team<NNNN>:v1` 标 | — |
| 导出 tar.gz + 检查文件名 | `team<NNNN>_v1.tar.gz` ≤ 10 GB |
| 上传 Google Drive 并开放链接 | 可匿名访问 |
| 发邮件 `[KDDCup2026 Data Agents] Submission - team<NNNN> - v1` | 发出且无退信 |
| 24h 内收到评测结果 | — |
| 分数非 0 | — |

**不达标应对**：若分数 0，从 `/logs` 排查；最常见是输出路径、env 未读、CSV 编码错误。

### M4 — 覆盖 medium（4/25 周六 → 4/30 周四）

**目标**：从 baseline 扩到对 medium 难度有效。

| 任务 | 退出标准 |
|---|---|
| 实现 `DataProfiler`（扫 context/ 生成摘要） | 摘要 < 2000 token 但信息完整 |
| 实现 `read_pdf` + `inspect_pdf` | 扫描版 PDF fallback 到 OCR |
| 加 Text-to-SQL 专项 prompt 提示 | demo 里的 db/ 任务对拍 gold，Recall > 0.5 |
| 实现 `search_knowledge`（向量检索 knowledge.md） | k=5 召回相关度合理 |
| 周中提交 v2 | 分数较 v1 提升 |

**退出标准**：demo 数据集上，medium 任务平均分 ≥ 0.5（Recall 口径）。

### M5 — hard & extreme 攻坚（5/1 周五 → 5/10 周六）

**目标**：Plan-and-Execute 上线；视觉路径验证。

| 任务 | 退出标准 |
|---|---|
| 实现 Plan-and-Execute（§4.5） | DAG 拓扑执行 + 回滚 |
| 实现 `read_chart`（Qwen vision 主 + OCR 兜底） | 两种路径单测通过 |
| 对 Q3（视觉是否启用）有明确答案 | Discord 回复 或 实证（我方提交 image_url 请求） |
| 加 reflection（extreme 任务失败时重规划一次） | 至少 1 个 extreme 任务因 reflection 救回 |
| 提交 v3-v5 | 每次都有回归测试保证不降分 |

**退出标准**：hard 任务平均分 ≥ 0.3；extreme 至少有 1 个非零。

### M6 — 冲榜与稳定（5/11 周日 → 5/23 周六，Phase 1 截止）

**目标**：最后 12 天每次提交都要有目标。

| 任务 | 退出标准 |
|---|---|
| 分析 leaderboard 与 demo 对拍差距 | 写 gap 报告 |
| Prompt 工程针对 Qwen 风格调优 | temp/top_p 对比实验 |
| 长上下文管理（context folding） | extreme 平均分上升 |
| 提交次数预算管理：剩余次数 = 30 - 已用 | 每次提交前写 `CHANGELOG.md` |
| 最后一次提交预留 5/22 | 避免 5/23 AoE 时区陷阱 |

**退出标准**：排名进入 cutoff（具体分数待组委会公布）。

### Phase 2 准备（5/24-5/27 过渡）

- 若进线 → 评估 Leaderboard vs Creative 子赛道：
  - Leaderboard：chart OCR 已成熟 → 继续冲
  - Creative：产品化能力强 → 转 Creative（但需从零做 UI）
- 5/27 前做决策

## 5.2 测试策略（Test Strategy）

### 5.2.1 测试金字塔

```
           ┌─────────────┐
           │   E2E (few) │   docker run 整镜像跑 demo
           ├─────────────┤
           │ Integration │   真 LLM（DeepSeek-V3）跑单个 task
           ├─────────────┤
           │   Unit      │   tool 函数、归一化、解析
           └─────────────┘
```

### 5.2.2 Unit Tests（目标 > 70% 覆盖）

| 测试对象 | 测试点 |
|---|---|
| `OutputWriter.normalize_cell` | 整数/float/Decimal/NaN/None/datetime(aware/naive)/str |
| `ResponseParser` | 合法 JSON fence；带 `<think>` 块；多 fence；非法字符；空响应 |
| `tools.filesystem.resolve_context_path` | 路径越界 → 拒绝；符号链接 → 拒绝 |
| `tools.python_exec` | 超时；stdout 捕获；异常传播；工作目录正确 |
| `tools.sqlite` | 只读禁写；危险 SQL 拒绝 |
| `llm.env.LLMClient.from_env` | 缺失变量 fallback；mock OpenAI |
| `agents.budget.TaskBudget.exceeded` | 步数、时间、token 各触发 |

### 5.2.3 Integration Tests

| 场景 | 预期 |
|---|---|
| Mock LLM 返回固定 JSON → ReAct 跑完 3 步 | 终态写 prediction.csv |
| 真 DeepSeek-V3 → demo task_1 | 分数非 0 |
| Tool 异常（sql 语法错误） | Agent 收到 obs.ok=false，继续 |
| 单任务超时 | 写空 CSV；日志有 "task_timeout" |

### 5.2.4 E2E Tests

| 场景 | 命令 | 预期 |
|---|---|---|
| 完整镜像跑 demo 全量 | `bash scripts/eval_local.sh` | 所有 task 有 prediction.csv |
| 镜像大小检查 | `bash scripts/check_size.sh` | `.tar.gz` ≤ 10 GB |
| 网络封锁验证 | `docker run --network=none ...` | 非 MODEL_API_URL 请求 fail；但 LLM 调用（开发 API）… |

**注意**：E2E 测时用 `--network=none` 会阻断开发 API 调用；需用 `--network=bridge` + 仅允许 `api.deepseek.com` 的路由表，或本地起 **mock vLLM** 暴露在 `host.docker.internal`。

### 5.2.5 Regression Tests（每次提交前强制）

`scripts/pre_submit_check.sh`：
1. `pytest tests/` 全通过
2. `bash scripts/eval_local.sh` demo 通跑
3. 对拍 `data/public/output/task_<id>/gold.csv` → 生成 `artifacts/latest_scores.json`
4. 与 `artifacts/submission_v<N-1>_scores.json` 对比，任何任务 0→1 回退需 ACK
5. 通过后才允许 `git tag v<N>` 与打镜像

### 5.2.6 本地评测脚本（核心）

```bash
#!/bin/bash
# scripts/eval_local.sh
set -e

docker build -t dabench:dev -f docker/Dockerfile .
docker run --rm \
  --network=none \
  --cpus=16 --memory=64g --memory-swap=64g \
  -e MODEL_API_URL="http://host.docker.internal:8000/v1" \
  -e MODEL_API_KEY="sk-local-test" \
  -e MODEL_NAME="qwen3.5-35b-a3b" \
  -v "$PWD/data/public/input:/input:ro" \
  -v "$PWD/artifacts/eval_output:/output:rw" \
  -v "$PWD/artifacts/eval_logs:/logs:rw" \
  dabench:dev

python scripts/score_predictions.py \
  --pred artifacts/eval_output \
  --gold data/public/output
```

`host.docker.internal:8000` 指向本机 LiteLLM/OpenAI-mock，它代理到 Qwen-Max API（对 Docker `--network=none` 不兼容——所以其实 eval_local 用 `--network=bridge` 并手动关其他外网流量；真正的"断网"测是 M6 压测用的专项测试）。

## 5.3 风险登记册（Risk Register）

> 沿用 §2.4 的 R1-R10，在此扩展应对细节与 triggers。

| ID | 风险 | 概率 | 影响 | 预防 | 缓解 | Trigger（触发时切到缓解） |
|---|---|---|---|---|---|---|
| R1 | 评测端 Qwen 禁用视觉 | 中 | 高 | Discord 问 Q3 | CPU OCR 兜底（Tesseract+pdfplumber），ADR-006 v1 就做 | v2 提交后若 hard/extreme 0 分，切主路径到 OCR |
| R2 | λ 具体值未知 | 中 | 低-中 | Discord 问 Q2（Q1 已裁决为连续公式） | 先用 λ=0.1 假设；leaderboard 反推 | 本地分 vs leaderboard 差 >0.1 时反推 |
| R3 | Qwen tool call 格式兼容 | 低 | 中 | 文本 JSON 协议绕过 | — | — |
| R4 | 镜像超 10 GB | 中 | 高 | `python:3.11-slim` base；分层构建 | 砍依赖（如去 polars，留 pandas） | `docker save` 输出 >8 GB 即预警 |
| R5 | 12h 总超时 | 中 | 高 | 每任务硬超时；即写即存 | 全局预算监控 → 降级策略 | 剩余 <2h 剩任务 >10，激活降级 |
| R6 | 注册卡审核 | 低 | 致命 | 4/19 前发起 | 紧急联系 `kddcup@hkust-gz.edu.cn` | 4/22 还没 team_id，发邮件 |
| R7 | 30 次提交被浪费 | 中 | 中 | pre_submit_check.sh | 保留最后 3 次给压轴 | 每周回顾已用次数 |
| R8 | 依赖版本冲突 | 低 | 中 | uv.lock 锁定 | 回滚 uv.lock | CI 构建失败 |
| R9 | 长文档 token 爆 | 中 | 中 | Profiler 抽样；pdf 分页 | Context folding | 上下文用量 >200K |
| R10 | task.json schema 校验失败 | 低 | 高 | subset 校验 | try/except + 写空 CSV | 某任务 KeyError |
| R11 | 本机 Docker Desktop WSL2 问题 | 中 | 中 | 提前验证 | 换用 Rancher Desktop 或云服务器 | 构建卡住 >10min |
| R12 | 评测 API 对我方 key QPS 限流 | 中 | 中 | max_workers=4 起步 | 降并发，加 retry | 429 错误出现 |
| R13 | Qwen 输出 `<think>` 污染 tool 解析 | 中 | 中 | parser 先剥离 think | 回提示"no reasoning in action" | 解析失败率 >5% |
| R14 | 并行任务 OOM | 中 | 高 | polars lazy；监控 RSS | 串行降级 | 容器 OOM |
| R15 | DNS/代理问题导致 docker pull 慢 | 低 | 低 | 预拉 base 镜像 | — | — |

## 5.4 交付物清单（Deliverables）

### 代码与文档
- `src/dabench/**` 完整源码
- `docker/Dockerfile` + `scripts/eval_local.sh`
- `tests/**` 单元/集成/E2E
- `docs/01-requirements.md` / `02-feasibility.md` / `03-architecture.md` / `04-detailed-design.md` / `05-milestones-testing-risk.md`（本文件）
- `docs/baseline_metrics.md`（M1 产出）
- `docs/submission_log.md`（每次提交记录：版本、改动、预期、实际得分）
- `docs/qna_organizer.md`（Discord 问答归档）

### 运行产物
- `artifacts/eval_output/**` 本地评测预测
- `artifacts/eval_logs/**` 运行日志
- 提交镜像 `team<NNNN>_v<N>.tar.gz`

### 提交物（每次）
- Google Drive 链接
- 邮件主题与正文（见规则 §3.3）
- `CHANGELOG.md` 更新

## 5.5 Phase 1 提交节奏建议

Phase 1 共 30 天 / 30 次。建议节奏：
```
Week 1 (4/24-4/30): v1-v3，每 2-3 天一次，铺框架
Week 2 (5/1-5/7):   v4-v7，每 1-2 天一次，加能力
Week 3 (5/8-5/14):  v8-v15，高频迭代打磨
Week 4 (5/15-5/22): v16-v25，精细调优
最后 3 天:          v26-v30，保守微调，防翻车
```
总共预计 ~25-28 次，留 2-5 次余量应对意外。

## 5.6 决策门禁（Go/No-Go）

每个里程碑结束，必须回答：

| 门禁 | 问题 | Go 标准 |
|---|---|---|
| M0→M1 | 环境齐全吗？注册完成了吗？ | 4 项就绪 + team_id 预期到手 |
| M1→M2 | starter kit 跑通了吗？ | 至少 1 task succeeded |
| M2→M3 | 本地 eval_local 通过吗？镜像合规吗？ | eval_local 通 + tar.gz ≤ 10 GB |
| M3→M4 | v1 非零分吗？ | 分数 > 0 |
| M4→M5 | medium 覆盖了吗？ | demo 上 medium ≥ 0.5 |
| M5→M6 | hard/extreme 有办法吗？ | hard ≥ 0.3 |
| M6→Phase2 | 进资格线了吗？ | 组委会邀请 |

**任何门禁不 Go**：立即开 post-mortem，识别根因，重排后续计划。
