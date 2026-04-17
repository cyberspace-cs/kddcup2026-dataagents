# KDD Cup 2026 Data Agents — 项目开发计划

> **项目**：KDD Cup 2026: Data Agents for Complex Data Analysis
> **仓库**：<https://github.com/Reverse-7/kddcup2026-dataagents>（private）
> **基准日期**：2026-04-16
> **Phase 1 开赛**：2026-04-24
> **方法**：标准 SDLC — 需求 → 可行性 → 架构 → 详细设计 → 里程碑/测试/风险
> **上一版讨论计划**：`C:\Users\16321\.claude\plans\immutable-whistling-storm.md`（输入源，已被本套文档完全取代）

## 文档清单

| # | 文档 | 内容 | 状态 |
|---|---|---|---|
| 01 | [需求分析](01-requirements.md) | 业务目标、功能需求、硬约束、输入/输出契约、待澄清项 | ✅ |
| 02 | [可行性分析](02-feasibility.md) | 技术/资源/时间可行性、Qwen3.5-35B-A3B 能力评估、starter kit gap 分析 | ✅ |
| 03 | [系统架构](03-architecture.md) | C4 模型、关键设计决策 ADR-001~006 | ✅ |
| 04 | [详细设计](04-detailed-design.md) | 模块目录、数据结构、ReAct 状态机、Plan-Execute、Tool 清单、prompt 模板 | ✅ |
| 05 | [里程碑·测试·风险·交付](05-milestones-testing-risk.md) | M0-M6 里程碑、测试金字塔、风险登记册 R1-R15、交付物清单 | ✅ |
| 06 | [接口契约](06-interfaces.md) | 所有模块公共 API 的 Python 类型签名 + 不变式 | ✅ |
| 07 | [Prompt 库](07-prompts.md) | system/task/observation/plan 完整 prompt + 每个 tool 的 description | ✅ |
| 08 | [本地评分器](08-local-scorer.md) | 按规则 §6 精确实现的列值向量匹配 scorer | ✅ |
| 09 | [部署方案](09-deployment.md) | Dockerfile 多阶段、eval_local.sh、pre_submit_check.sh、submit.sh | ✅ |
| 10 | [任务手册](10-task-playbooks.md) | easy/medium/hard/extreme 四档处理路径与 prompt 变体 | ✅ |
| 11 | [运维 Playbook](11-ops.md) | 两人 git 工作流、提交 checklist、回滚、实验跟踪、Phase 1 执行节奏 | ✅ |
| 12 | [两人作战计划](12-two-person-plan.md) | 逐日分工、并行度分析、沟通协议、onboarding checklist | ✅ |
| 13 | [Gantt 图](13-gantt.md) | Mermaid 甘特图：总览 / 华vs吴并行 / 里程碑+提交节奏 | ✅ |

## 核心事实卡（Cheat Sheet）

### 硬约束（违反即 0 分或 DQ）
- Docker 镜像 ≤ **10 GB**（tar.gz 压缩后）
- 运行环境：**16 vCPU / 64 GB RAM / 无 GPU / 无外网 / 12 小时总时长**
- 必须读环境变量 `MODEL_API_URL` / `MODEL_API_KEY` / `MODEL_NAME`
- 模型锁定 **Qwen3.5-35B-A3B**（MoE，3B 活跃，256K 上下文，**原生视觉**，vLLM `qwen3_coder` tool parser）
- 每天 ≤ 1 次提交；Phase 1 总计 ≤ **30 次**

### 输出路径
```
/output/task_<id>/prediction.csv  ← 唯一被评分文件
```

### 评分公式（以规则 §6.3 为准，用户裁决 2026-04-16）
```
Recall = Matched Columns / Gold Columns
Score  = Recall − λ · (Extra Columns / Predicted Columns)
下界 0；总分 = 所有任务算术平均；同分按提交时间升序
```
（λ 具体值仍未知，待 Discord 澄清——见 Q2）

### 时间窗
```
注册        : 3/22 – 4/23
Phase 1    : 4/24 – 5/23  （30 天，≤30 次提交）
资格审核   : 5/24 – 5/27
Phase 2    : 5/28 – 6/30
最终验证   : 7/1 – 8/8
KDD 颁奖   : 8/9
```

## 待澄清问题（Discord/邮件）

- ~~Q1~~ ✅ 已裁决：规则页 §6.3 连续公式为准
- **Q2** λ 的具体值？
- **Q3** 评测 vLLM 是否启用视觉（能否传 image_url）？
- **Q4** `doc/` 是否只含 PDF/MD，还是也可能有 docx/xlsx/pptx？
- **Q5** 单任务超时与 12h 总超时的关系？是否支持单任务超时继续下一个？
- **Q6** Qwen 服务是否启用 `enable_thinking=True`，输出是否含 `<think>` 块？
- **Q7** `MODEL_API_URL` 接口契约细节：是否严格 OpenAI `chat/completions`？是否支持 `vision` messages（`type: image_url`）？

## 关键决策已定（ADR + D）

### ADR（架构层）

- **ADR-001** ReAct 用文本 JSON 协议（不用 OpenAI 原生 tool calling）
- **ADR-002** Planner 分层：easy/medium 用 ReAct；hard/extreme 用 Plan-and-Execute
- **ADR-003** Docker base：`python:3.11-slim` + uv
- **ADR-004** 并发：任务间 ThreadPoolExecutor(max_workers=4)，任务内串行
- **ADR-005** 输出归一化在 OutputWriter 集中做
- **ADR-006** 视觉主路径走 Qwen vision，CPU OCR 作兜底（v1 就要有）

### 用户决策（2026-04-16 全部敲定）

| # | 决策点 | 结论 | 备注 |
|---|---|---|---|
| D1 | 参赛规模 | ✅ **两人组队** | 队员身份验证最晚 4/23；搭档待定 |
| D2 | 开发替身 LLM | ✅ **Qwen-Max（DashScope）** | 格式最接近评测端；DeepSeek-V3 做对比验证 |
| D3 | 官方 starter kit 处理 | ✅ **Fork + 定向重构** | 参 §4.1 / §6.1 模块结构 |
| D4 | 是否等 Q 回应再开工 | ✅ **并行推进** | M0 完 → M1 直接开干，Q2-Q7 Discord 异步问 |
| D5 | 视觉兜底 OCR 路径 | ✅ **v1 就做** | 不等 Q3 回应；双路径并行 |

### 本机规格（影响本地评测）

- CPU：24 核（估 12C24T）
- RAM：**16 GB**（评测目标 64 GB；本地只能 correctness 测；capacity 测 M5-M6 前租云 16C64G 做 dry run）
- OS：Windows 11 + Git Bash + Docker Desktop (WSL2)
- `.wslconfig` 推荐：`memory=12GB swap=8GB processors=20`

## 下一步（M0 执行清单）

### 用户（并行）
1. 注册流程：4/23 前提交（队长 + 搭档双人验证）
2. 下载 Phase 1 demo → `data/public/input/` 与 `data/public/output/`（百度网盘 `ymdc` / Google Drive）
3. 申请 DashScope（Qwen-Max）API key → 测试 `qwen-max-latest`
4. 装 Python 3.11 + uv + Docker Desktop (WSL2)；调 `.wslconfig`
5. 搭档确认邮箱 → 决定加 collaborator 时机

### 我（并行）
1. 起草 Q2-Q7 的 Discord 提问文案
2. M0 门禁通过后立刻 fork starter kit 到 `vendor/starter/`，进入 M1
3. 维护文档、同步仓库

## 项目目录规划（将在 M1-M2 建起）

```
E:\KDD比赛\
├── docs/                      ← 本套计划文档（已在）
├── src/dabench/               ← 主源码（M2 建）
├── vendor/starter/            ← starter kit fork（M1 建）
├── data/public/input/         ← Phase 1 demo（M0 下载）
├── data/public/output/        ← gold.csv（M0 下载）
├── configs/                   ← YAML 配置（M2 建）
│   ├── dev.yaml
│   └── eval.yaml
├── scripts/                   ← 脚本（M2 建）
│   ├── eval_local.sh
│   ├── pre_submit_check.sh
│   └── submit.sh
├── docker/Dockerfile          ← M2 建
├── tests/                     ← 测试（M2 建）
├── artifacts/                 ← 本地运行产物（gitignore）
└── CHANGELOG.md               ← 每次提交记录（M3 起）
```
