# KDD Cup 2026 Data Agents — 项目开发计划

> **项目**：KDD Cup 2026: Data Agents for Complex Data Analysis
> **基准日期**：2026-04-16
> **Phase 1 开赛**：2026-04-24（距今 8 天）
> **方法**：标准 SDLC — 需求 → 可行性 → 架构 → 详细设计 → 里程碑/测试/风险
> **上一版讨论计划**：`C:\Users\16321\.claude\plans\immutable-whistling-storm.md`（作为输入，已被本套文档取代）

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

## 关键决策已定（ADR）

- **ADR-001** ReAct 用文本 JSON 协议（不用 OpenAI 原生 tool calling）
- **ADR-002** Planner 分层：easy/medium 用 ReAct；hard/extreme 用 Plan-and-Execute
- **ADR-003** Docker base：`python:3.11-slim` + uv
- **ADR-004** 并发：任务间 ThreadPoolExecutor(max_workers=4)，任务内串行
- **ADR-005** 输出归一化在 OutputWriter 集中做
- **ADR-006** 视觉主路径走 Qwen vision，CPU OCR 作兜底（v1 就要有）

## 待用户决策（开工前）

下列决策尚未敲定，影响后续方向：

| # | 决策点 | 选项 | 我的建议 |
|---|---|---|---|
| D1 | 独自参赛还是拉队友？ | 1-5 人 | 独自（小团队决策快；30 次提交限制与规模无关） |
| D2 | 开发 LLM 主力选谁？ | Qwen-Max / DeepSeek-V3 / GPT-4.1-mini | **Qwen-Max**（格式最接近评测） |
| D3 | Fork starter kit 还是重写？ | Fork / 从零 / 混合 | **Fork + 定向重构**（见 §4.1 模块命名） |
| D4 | 是否立刻写代码？ | 立即 / 先等 Q1-Q7 回应 / M1 再写 | **M0 完成后立即 M1 并行问 Q** |
| D5 | 先做 OCR 兜底还是假设视觉可用？ | 两路径并行 / 先主路径 | **主路径先做，OCR 列入 M5 必做项**（若 Q3 确认视觉禁用，M2 就补） |

## 下一步（立刻可以做的事）

按优先级：

1. **用户**：今天或明天开始注册流程（避免审核卡住，4/23 截止）
2. **用户**：下载 Phase 1 demo 数据到 `E:/KDD比赛/data/public/input/`
3. **用户**：申请 DashScope（Qwen-Max）API key
4. **用户**：装 Docker Desktop + WSL2；调 `.wslconfig`（memory=12GB）
5. **我**：起草 Q2-Q7 的 Discord 提问文案
6. **我**：M1 开始时 fork starter kit 到 `vendor/starter/`
7. **我们**：M0 完成后进入 M1（fork 与定制开发）

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
