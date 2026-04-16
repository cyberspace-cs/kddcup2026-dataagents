# 1. 需求分析（Requirements）

> **来源**：`dataagent.top`（官网首页 + `/rules` 技术规范页）、`HKUSTDial/kddcup2026-data-agents-starter-kit` 仓库、`huggingface.co/Qwen/Qwen3.5-35B-A3B` config.json。
> **基准日期**：2026-04-16。所有条款均引自官方文档。

---

## 1.1 业务目标

| 层级 | 目标 | 可验证指标 |
|---|---|---|
| 最低 | 完成注册、Phase 1 至少提交一次有效 Docker 镜像，Leaderboard 非零得分 | 组委会发回评测结果邮件；leaderboard 有条目 |
| 中等 | Phase 1 总分进入资格线（cutoff 未公布，按登记规模公告） | 组委会 Phase 2 邀请 |
| 冲刺 | Phase 2 子赛道（Leaderboard 或 Creative）完赛，争奖金 + KDD Workshop 展示 | 颁奖邮件；KDD 2026 workshop 出席 |

冠军奖金：Leaderboard 赛道（CNY，税前；USD 以 1:6.91 折算为参考）。**Creative 子赛道奖金由委员会最终决定**。

## 1.2 功能需求（Agent 必须做什么）

**输入**（评测时，由组委会挂载）：
- 目录：`/input/task_<id>/`，只读
- 每个任务包含：
  - `task.json` — 字段：`task_id` / `difficulty`（easy|medium|hard|extreme）/ `question`
  - `context/` 目录，**子目录非固定**：可能是 `csv/`、`db/`（.sqlite/.db）、`json/`、`doc/`（.md/.pdf/其他）的任意组合；可选 `knowledge.md`（业务术语）

**输出**（Agent 必须写）：
- 路径：`/output/task_<id>/prediction.csv`
- 格式：UTF-8、首行列名（列名不参与评分）、UTF-8 CSV
- `/output` 目录由评测系统预创建；Agent 只写 `task_<id>/prediction.csv`，**不要创建父目录**
- 空值按空字符串处理；数值保留≥2位小数；ISO 时间带时区转 UTC（末尾 Z）

**能力需求**（按难度分级来自规则 2.3 表格）：

| 难度 | 数据源 | 需要的 Agent 能力 |
|---|---|---|
| easy | CSV/JSON + knowledge.md | Python 代码生成与数据分析 |
| medium | CSV/JSON + db/ + knowledge.md | Text-to-SQL + 多源数据融合 |
| hard | 同上 + doc/ 数据文档 | 非结构化文档推理 |
| extreme | 同上 | 长上下文工程 + 记忆管理 |

**非线性推理拓扑**（官网 DataAgent-Bench 章节）：
- Sequential Chain：步骤依赖
- Branching & Merging：并行子查询后合并
- Iterative Refinement：迭代自修正

## 1.3 非功能需求（硬约束，违反即 0 分或 DQ）

### 1.3.1 运行环境（规则 §4）
| 约束 | 值 | 违反后果 |
|---|---|---|
| CPU | 16 vCPU（x86-64） | — |
| 内存 | 64 GB | **OOM → 本次提交所有结果作废** |
| GPU | **无** | — |
| 总运行时长 | **12 小时** 跑完所有任务 | 超时强制终止 |
| 网络 | 外网完全阻断；仅允许 `MODEL_API_URL` | 越权访问 → 评测失败 |

### 1.3.2 提交物（规则 §3）
| 约束 | 值 |
|---|---|
| 打包形式 | **单个 Docker 镜像** tar.gz 压缩包 |
| 大小上限 | **≤ 10 GB**（压缩后） |
| 命名 | `team<NNNN>:vN` → 归档 `team<NNNN>_v<N>.tar.gz` |
| 提交渠道 | 上传 Google Drive → 邮件 `kddcup@hkust-gz.edu.cn` |
| 邮件主题 | `[KDDCup2026 Data Agents] Submission - team<NNNN> - v<N>` |
| `ENTRYPOINT` 或 `CMD` | 必须设置，`docker run` 直接启动 |
| root 权限 | 允许；但**禁止写 `/input`** |

### 1.3.3 模型（规则 §5，**核心约束**）
- 评测时强制走 `MODEL_API_URL` 指向的 **Qwen3.5-35B-A3B**（OpenAI 兼容 Chat Completions）
- 环境变量：`MODEL_API_URL`、`MODEL_API_KEY`、`MODEL_NAME`（不得硬编码）
- **禁止**在容器内运行任何非 Qwen3.5-35B-A3B 的 LLM 作为主推理引擎，包括 CPU 本地推理其他 LLM
- **允许**：embedding 模型、小辅助模型仅用于文档处理/检索，不作为主求解器
- vLLM 部署参数（参考）：`--max-model-len 262144`、`--enable-auto-tool-choice`、`--tool-call-parser qwen3_coder`、`--reasoning-parser qwen3`、`--tensor-parallel-size 8`、`--served-model-name qwen3.5-35b-a3b`

### 1.3.4 提交频率（规则 §7）
| 约束 | 值 |
|---|---|
| 每日上限 | 1 次/天/队 |
| Phase 1 总上限 | **30 次** |
| 串行 | 必须收到上一次评测结果后才能再提 |
| 评测延迟 | 组委会 24 小时内开始评测，实际完成视代码运行时间 |

### 1.3.5 评分（规则 §6，**直接决定得分天花板**）

**已确认（用户裁决 2026-04-16）：以规则页 §6.3 为准**。官网首页"二值 0/1"表述作废。

规则页 §6.3 原文：
> `Recall = Matched Columns / Gold Columns`
> `Score = Recall − λ · (Extra Columns / Predicted Columns)`
> "The lower bound of the final score is 0"

**代码层决策**：按连续公式优化（最大化 Recall，最小化冗余）。**λ 具体数值仍未知**，需 Discord 问组委会（Q2）。

**列匹配细节**（§6.2, 6.5）：
- 按列值"向量"匹配，列名忽略；支持重复列（签名次数必须一致）
- 数值：四舍五入到 2 位小数后比较
- 日期：带时区 → UTC(Z)；无时区 → 保持 ISO
- 字符串：去首尾空白与 `\r\n`，**大小写敏感**
- 姓名：姓/名拆两列 vs 合并全名，两种都算对

**总分**：所有任务分数的算术平均；并列时先提交者排名更高。

### 1.3.6 被禁行为（规则 §8）
越权调其他 LLM、运行非 Qwen3.5-35B-A3B 作为主解答模型、探测/攻击评测系统、修改 `/input`、多队共享镜像、通过非法手段获取标答、评测时人工介入 → 评测失败或 DQ。

## 1.4 输入/输出契约（Interface Contract）

```
评测容器挂载：
  /input/  (ro)        ← 组委会挂载
    task_<id>/
      task.json        {task_id, difficulty, question}
      context/
        csv/*.csv         （可选）
        db/*.sqlite|*.db  （可选）
        json/*.json       （可选）
        doc/*.md|*.pdf    （可选）
        knowledge.md      （可选）
  /output/ (rw)        ← Agent 写
    task_<id>/
      prediction.csv   ← **唯一被评分文件**
  /logs/   (rw)        ← Agent 写（用于 debug，超时/OOM 后组委会可取走）

环境变量（由 docker run -e 注入）：
  MODEL_API_URL  内网 vLLM 地址
  MODEL_API_KEY  推理认证 key
  MODEL_NAME     固定 "qwen3.5-35b-a3b"
```

## 1.5 明确排除（Out of Scope）

- 非 KDD Cup 目标：大计赛 A 题由弟弟团队承担，本项目**不直接参赛**，但架构选型应兼顾 70% 通用性以便顾问式复用。
- 不做 Phase 2 Creative 子赛道专属的交互 UI 直到 Phase 2 开赛并确认进线。
- 不训练/微调任何模型（严禁，违反 §8）。
- 不做 A/B 服务化部署——仅离线批处理镜像。

## 1.6 待澄清项（需 Discord/邮件问组委会）

| # | 问题 | 理由 | 状态 |
|---|---|---|---|
| ~~Q1~~ | ~~评分是二值还是连续？~~ | ~~直接影响优化目标~~ | ✅ 2026-04-16 裁决：以规则页 §6.3 连续公式为准 |
| Q2 | λ 的具体数值 | 决定冗余列惩罚强度 | 待问 |
| Q3 | 评测 vLLM 服务是否启用视觉（chart PNG 能否直接当 image_url 传）？官方 deployment 参数里未写 `--language-model-only` | 决定架构是否需 CPU OCR 回退 | 待问 |
| Q4 | `doc/` 是否会出现非 PDF/MD 的格式（docx/xlsx/pptx）？ | 决定需加载哪些解析器 | 待问 |
| Q5 | 单任务超时与总超时的关系：是否允许某任务超时后继续下一个？ | 决定调度策略 | 待问 |
| Q6 | Qwen 服务是否启用 `enable_thinking=True`？输出是否含 `<think>` 块？ | 决定 prompt 解析 | 待问 |
| Q7 | `MODEL_API_URL` 接口契约细节：是否严格 OpenAI `chat/completions`？是否支持 `vision` messages（`type: image_url`）？ | 决定多模态调用方式 | 待问 |
