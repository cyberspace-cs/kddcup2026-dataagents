# 12. 两人作战计划（Two-吴attle Plan）

> 本文件是 M0-M6 全周期的两人逐日分工计划。按模块分工。
> **华**（@Reverse-7）与 **吴**（搭档）——按模块分工，各管各的线。
> 角色定义灵活——搭档入队后可根据实际能力交换特定模块。

---

## 12.1 角色与职责划分

### 核心原则
1. **按模块分工**：各管各的线；谁先有 idea 谁做
2. **并行最大化**：同一时间段两人做不同模块，避免串行等待
3. **接口先行**：先定好模块接口（§06），各自实现，最后集成
4. **每日同步**：晚 21:00 微信 10 分钟——"做了啥 / 卡在哪 / 明天干啥"
5. **PR 互审**：任何合入 main 的代码对方需 1 分钟审（看 diff，不是逐行）
6. **提交权共享**：两人都有 push + tag + 发邮件权限，谁方便谁操作

### 职责矩阵

> 🟢 = 负责 | 🔵 = review / 兜底 | 🟢🔵 = 共同

| 模块 | 华 | 吴 | 说明 |
|---|---|---|---|
| **Agent 核心循环** (react.py / plan_execute.py) | 🟢🔵 | 🟢🔵 | **两人一起做**：结对/mob 编程 |
| **LLM Client** (client.py / env.py / 重试) | 🟢🔵 | 🟢🔵 | **两人一起做**：与 Agent 耦合 |
| **ResponseParser** (JSON fence + `<think>` 剥离) | 🟢🔵 | 🟢🔵 | **两人一起做** |
| **Tools — 结构化** (filesystem / sqlite / python_exec) | 🟢 | 🔵 | — |
| **Tools — 非结构化** (pdf / image / rag) | 🔵 | 🟢 | — |
| **DataProfiler** (scan.py) | 🟢 | 🔵 | — |
| **Prompt 工程** (所有 prompt 调优) | 🟢🔵 | 🟢🔵 | 两人独立实验 → 对比合并 |
| **OutputWriter + 本地 Scorer** | 🟢 | 🔵 | — |
| **Docker / Dockerfile / scripts** | 🟢 | 🔵 | — |
| **TaskOrchestrator / Runner** | 🟢 | 🔵 | — |
| **Context Folding + Reflection** | 🟢🔵 | 🟢🔵 | **两人一起做** |
| **测试** (unit / integration) | 🟢🔵 各写各的 | 🟢🔵 交叉审 | — |
| **数据探索 / demo 分析 / 对拍** | 🟢 | 🔵 | — |
| **文档 / 提交 / CHANGELOG** | 🟢🔵 | 🟢🔵 | 谁做的谁写 |

**简记**：
- **两人一起** = Agent 循环 + LLM Client + Parser + Reflection + Folding + Prompt 调优 + 测试
- **华** = Profiler + 结构化 Tools + OutputWriter + Scorer + Docker + 提交 + 调度 + 数据探索
- **吴** = 非结构化 Tools（PDF / Image / RAG）

---

## 12.2 逐日作战计划

### 🗓 M0 — 环境就绪 + 注册（4/17 周四 → 4/19 周日）

> M0 重点：**不写代码**，只做注册 + 环境 + 下载 + Discord Q

| 日期 | 华 | 吴 |
|---|---|---|
| **4/17 周四** | 注册 Google Form; 装 Docker Desktop + WSL2 调 .wslconfig; 申请 DashScope API key | 确认 GitHub 邮箱、clone 仓库; 装 Python 3.11 + uv; 装 Docker Desktop |
| **4/18 周五** | 测试 `docker run hello-world`; 测试 Qwen-Max API 调通; 下载 Phase 1 demo 数据（百度网盘 ymdc） | 同步下载 demo 数据; 加入 Discord / 微信群; 浏览 starter kit README |
| **4/19 周六** | 发 Discord 提问 Q2-Q7（我帮起草文案）; 验证 demo 数据目录结构正确; 队员身份验证（若 B 已注册） | 完成身份验证（Informed Consent Form）; 阅读 docs/00-12 全套文档熟悉架构; 列出自己的 skill gaps 告知 A |

**M0 门禁**：`uv`✅ + `docker`✅ + `Qwen-Max API`✅ + `demo 数据在位`✅ + `两人都 clone 了 repo`✅

---

### 🗓 M1 — Fork Starter Kit & 跑通（4/20 周日 → 4/22 周二）

| 日期 | 华 | 吴 |
|---|---|---|
| **4/20 周日** | `git clone` starter kit 到 `vendor/starter/`; `uv sync` 安装依赖; 填 config 指向 Qwen-Max API; 跑 `dabench status` + `dabench run-task task_1` | 同步 A 的 vendor/ 环境; 跑 `dabench inspect-task` 对 demo 所有任务做数据探索; 记录每任务的数据源类型、行数、文件大小 → 输出 `docs/demo_data_survey.md` |
| **4/21 周一** | 跑 `dabench run-benchmark --limit 5` 拿 baseline; 记录 metrics → `docs/baseline_metrics.md`（每任务步数/tokens/耗时） | 继续数据探索：手动看 gold.csv 的列结构，理解评分目标; 对比 prediction vs gold 手动算几个分数; 验证 `scripts/score_predictions.py` 框架（先写骨架） |
| **4/22 周二** | 跑全量 demo benchmark; 汇总 baseline_metrics; **两人 sync**：确认 starter kit 能力边界 + 下一步定制方向 | 完成 `docs/demo_data_survey.md`; 在 baseline_metrics 里标注哪些任务通过/失败/超时; 对失败任务做根因分类（SQL 错？PDF 没读？格式错？） |

**M1 门禁**：starter kit 能跑 ≥ 1 task succeeded ✅ + baseline_metrics 有了 ✅ + demo_data_survey 完成 ✅

---

### 🗓 M2 — Fork 定制 + Docker 化（4/20 并行 → 4/23 周三）

> M1 和 M2 时间重叠——A 做 M1 跑通时，B 可以并行做 M2 的数据探索和工具原型。

| 日期 | 华 | 吴 |
|---|---|---|
| **4/20-21** | （M1 跑通任务） | （M1 数据探索任务） |
| **4/22 周二** | 写 `tools/filesystem.py` 改进（polars 大文件）; 写 `tools/sqlite.py` 改进; 写 `io/output_writer.py` 归一化 | 建 `src/dabench/` 骨架 §4.1; 移植 starter 核心到 dabench; 写 `llm/env.py` 环境变量覆盖 + `llm/client.py` 重试 |
| **4/23 周三** | 写 `main.py` ENTRYPOINT; 写 `docker/Dockerfile`; 跑 `eval_local.sh`; **docker build + run 通过** | 写 `agents/react.py` 改进版（`<think>` 剥离 + 错误重试）; 写 `agents/parser.py`; 写 `tools/python_exec.py` 改进（timeout 60s） |

**4/23 晚两人 sync**：集成 A 的框架 + B 的 tools → 合入 main → docker build → eval_local.sh 通过

**M2 门禁**：`eval_local.sh` 通过 ✅ + 镜像 < 3 GB ✅ + `/output/task_<id>/prediction.csv` 路径正确 ✅

---

### 🗓 M3 — 首次提交 baseline（4/24 周四，Phase 1 开赛）

| 时段 | 华 | 吴 |
|---|---|---|
| **上午** | `pre_submit_check.sh v1` 全绿; tag v1; docker save + tar.gz | 并行写 `score_predictions.py` 完整版; 对拍 demo gold，记录 v1 本地分数 |
| **下午** | 重命名 archive → `team<NNNN>_v1.tar.gz`; 上传 Google Drive; 发邮件 | 更新 CHANGELOG.md + submission_log.md |
| **晚上** | 两人 sync：v1 已提交；review M4 任务分配 | 同上 |

**M3 门禁**：邮件已发 ✅ + Drive 链接可访问 ✅ + submission_log 有记录 ✅

---

### 🗓 M4 — 覆盖 medium（4/25 周五 → 4/30 周三）

**主线任务**：DataProfiler 上线 + PDF 工具 + Text-to-SQL prompt + 本地对拍系统成熟

| 日期 | 华 | 吴 |
|---|---|---|
| **4/25 周五** | 实现 `profiler/scan.py`：CSV/JSON/SQLite 扫描 | 实现 `tools/pdf.py` (pdfplumber + pymupdf) + `tools/rag.py`（embedding + FAISS） |
| **4/26 周六** | 完善 `score_predictions.py` 全量对拍; 写 regression 检测 | 将 Profiler + PDF/RAG 集成进 Agent 循环；改 prompt.py 加 `{data_profile_summary}` |
| **4/27 周日** | Text-to-SQL 专项 prompt A/B 调优 | 调试 medium 任务的 Agent 推理链路; 分析失败根因 |
| **4/28 周一** | 集成测试; 修 bug; 写 pdf/rag 单元测试 | 修 Agent 循环 bug; 优化 ReAct 步数效率 |
| **4/29 周二** | `pre_submit_check.sh v2`; 提交 v2 | 分析 v1 评测结果; 更新 CHANGELOG |
| **4/30 周三** | review v2 leaderboard; gap 分析 | 规划 M5：Plan-Execute 架构设计 |

**提交节奏**：v2（4/29 or 4/30）——medium 覆盖

**M4 门禁**：demo 上 medium 平均 Recall ≥ 0.5 ✅ + DataProfiler 上线 ✅ + v2 提交 ✅

---

### 🗓 M5 — hard & extreme 攻坚（5/1 周四 → 5/10 周六）

**主线任务**：Plan-and-Execute 上线 + 视觉路径 + reflection + context folding

| 时段 | 华 | 吴 |
|---|---|---|
| **5/1-5/2** | 实现 `tools/image.py` Qwen vision + tesseract 兜底; 实现 `inspect_pdf` / `inspect_image` | 实现 `agents/plan_execute.py`（Plan + DAG + blackboard）; prompt 模板 §7.6, §7.7 |
| **5/3-5/4** | Profiler 加 PDF 页数/图片维度; image tool 单元测试 | 集成 Plan-Exec 到 adaptive 路由; hard demo A/B 对比 |
| **5/5-5/6** | Plan-Execute 集成测试; 回归确认 | 实现 reflection（extreme 重规划）+ context folding（压缩旧 steps） |
| **5/7 周三** | 集成 + 回归测 + v3 提交 | 分析 v2 leaderboard; 按 difficulty 拆分 |
| **5/8-5/9** | Prompt 实验 E1-E3（scoring rules / tool desc 长度 / temp） | Prompt 实验 E4-E5（中英 / few-shot）; 优化 Agent 多轮推理 |
| **5/10 周六** | v4/v5 提交 | gap_analysis 更新; M6 冲榜清单 |

**提交节奏**：v3（5/7）、v4/v5（5/10）——hard 覆盖 + vision + plan-exec

**M5 门禁**：hard 平均 ≥ 0.3 ✅ + extreme 至少 1 个非零 ✅ + vision 路径有效 ✅

---

### 🗓 M6 — 冲榜与稳定（5/11 周日 → 5/22 周四）

**主线任务**：精细调优 + 资源管理 + 鲁棒性 + 压轴提交

| 阶段 | 华 | 吴 |
|---|---|---|
| **5/11-5/14 (Week 1)** | 鲁棒性加固：异常覆盖、OOM 防护、降级验证 | Prompt 精调：按 leaderboard 反馈逐难度优化; temperature/top_p/top_k 扫参 |
| **5/15-5/18 (Week 2)** | 性能优化：并行度调优、减 token 消耗; 云 16C64G dry run | 长上下文：context folding 调参; extreme 逐任务分析; few-shot 示例库 |
| **5/19-5/20** | 全量回归; 版本逐任务对比表 | **冷冻策略**：仅 prompt 微调 + bug fix |
| **5/21 周三** | v24-v25 提交（保守微调） | 最终回归; CHANGELOG 归档 |
| **5/22 周四** | **最后提交** v26-v28 | 监控邮件确认评测 |
| **5/23 周五** | ❌ **不提交** — AoE 时区风险 | 同上 |

**提交节奏**：v6-v28（约 15-22 次），按 §5.5 节奏

**M6 门禁**：排名进入 cutoff ✅（或无法控制则尽力而为）

---

## 12.3 并行度分析（关键路径）

```
                          华（工程保障）                    吴（推理引擎）
                          ──────────                        ──────────
M0 (4/17-4/19)           注册 + 环境 + API                环境 + clone + 通读文档
                               │                               │
                               ▼                               ▼
M1 (4/20-4/22)           跑 starter kit ──────────────── 数据探索 + demo survey
                               │                               │
                               ▼                               ▼
M2 (4/22-4/23)           结构化 Tools + Docker ──── Agent 骨架 + LLM Client + Parser
                               │            ╲                  │
                               │             ╲ 集成            │
                               ▼              ╲                ▼
M3 (4/24)                提交 v1 ◄────────────── v1 对拍验证
                               │                               │
                               ▼                               ▼
M4 (4/25-4/30)           Profiler + Scorer ──────── PDF/RAG + Agent 推理优化
                               │                               │
                               ▼                               ▼
M5 (5/1-5/10)            Vision Tools + 测试 ──── Plan-Exec + Reflection + Folding
                               │                               │
                               ▼                               ▼
M6 (5/11-5/22)           鲁棒性 + 提交 ───────── Prompt 精调 + 长上下文优化
```

**关键路径**（阻塞链）：
```
starter kit 跑通 → Agent 骨架(吴) + Docker(华) → 集成 → v1 → Profiler(华)+推理优化(吴) → Plan-Exec(吴) → 冲榜
```
华的工程保障与吴的推理引擎**始终并行**，最大化两人效率。

**最大风险点**：M2 集成（4/23 晚）——华的 Docker/Tools + 吴的 Agent/LLM 第一次对接。应对：4/22 先做一次 dry integration（吴在华的骨架上挂 1 个 tool 跑通）。

---

## 12.4 沟通协议

| 频率 | 形式 | 内容 | 时长 |
|---|---|---|---|
| **每日** | 微信消息（21:00） | 三句话：做了 / 卡了 / 明天做 | 10 min |
| **集成日**（周三/周四） | 腾讯会议/微信通话 | 合码 + 跑测试 + 讨论下一步 | 30-60 min |
| **提交日** | 实时微信 | A 操作提交流程；B 实时确认 checklist | 不限 |
| **周日复盘** | 结构化文档 + 通话 | 上周 OKR + 本周计划 + 得分趋势 | 30 min |
| **紧急** | 直接打电话 | Docker 构建崩了 / leaderboard 暴跌 / 评测 0 分 | — |

### 消息模板（每日 sync）

```
【4/25 同步】
A:
- ✅ DataProfiler CSV/JSON/SQLite 三类完成
- ❌ 卡在 polars schema inference 对带 BOM 的 CSV 报错
- 明天：修 BOM bug + 集成进 Agent 循环

B:
- ✅ read_pdf text mode done; tables mode 50%
- ❌ pymupdf 在某 PDF 上 segfault（已提 issue）
- 明天：绕过 segfault + rag.py embedding 选型
```

---

## 12.5 代码协作规范

### 12.5.1 谁写测试？

| 场景 | 写测试的人 |
|---|---|
| 新 tool | **写 tool 的人** 写单元测试 |
| Agent 循环 | A 写；B review |
| 集成测试 | **A** 写（要理解端到端流程） |
| Bug fix | **修 bug 的人** 必须加回归测试 |

### 12.5.2 PR 流程

```
1. 功能分支上开发 + 自测
2. 推到 origin
3. 开 PR → 对方 review（标注 LGTM 或 提 comment）
4. 合入 main
5. 本地 pre_submit_check.sh 确认无回归（每日至少 1 次）
```

紧急时可口头 approve + squash merge，事后补 review comment。

### 12.5.3 冲突解决

两人同时改一个文件？概率低（模块分工清晰），万一碰到：
- **谁先合谁留**——后合的人 rebase 解冲突
- **prompt.py** 是唯一高冲突区——用**分段负责**：A 写 system prompt，B 写 tool descriptions

---

## 12.6 吴入队第一天（Onboarding）

吴拿到 GitHub collaborator 权限后的 Day 1 清单：

```markdown
### 吴的 Onboarding Checklist

- [ ] clone 仓库 `git clone https://github.com/Reverse-7/kddcup2026-dataagents.git`
- [ ] 从头到尾读 `docs/00-index.md`（20 分钟）
- [ ] 精读自己负责的核心模块文档：
  - docs/06-interfaces.md §6.3（LLMClient）+ §6.6（AgentState / Planner）+ §6.7（ResponseParser）
  - docs/04-detailed-design.md §4.3（ReAct 状态机）+ §4.5（Plan-Execute）
  - docs/07-prompts.md §7.2-7.8（全部 prompt）
  - docs/10-task-playbooks.md（理解四档难度差异，这是吴做 adaptive routing 的依据）
- [ ] 装环境：Python 3.11 + uv + Docker Desktop
- [ ] 下载 Phase 1 demo 到 `data/public/`
- [ ] 跑一遍 starter kit：`uv run dabench run-task task_1 --config ...`
- [ ] 读 `src/data_agent_baseline/agents/react.py`（当前 baseline 循环逻辑）
- [ ] 手动看 3 个 gold.csv 理解输出格式
- [ ] 微信确认华：环境 OK，准备接任务
```

---

## 12.7 吴未到位的 Plan B

**最坏情况**：Phase 1 全程只有华一人。

| 影响 | 应对 |
|---|---|
| 核心 Agent 循环没人写 | 华自己写简版 ReAct（沿用 starter kit 原版，只改 env + output path）；**放弃 Plan-Execute** |
| 工作量翻倍 | **砍 scope**：v1 只做 easy/medium（基础 ReAct + SQL + CSV）；hard 推到 5 月中 |
| 无人 code review | 用 CI lint（ruff）替代人工审 |
| 无人做 PDF/vision/RAG | **延后到 M5**；v1-v3 不处理非结构化题 |

---

## 12.8 每周 OKR 模板

```markdown
## Week of 4/25 - 4/30

### Objectives
1. Medium 难度 demo 平均 Recall ≥ 0.5
2. 完成 DataProfiler + PDF tool
3. 提交 v2

### Key Results

| KR | Owner | Target | Actual | Status |
|---|---|---|---|---|
| Profiler scan CSV/JSON/SQLite | A | Done by 4/26 | | |
| read_pdf text+tables mode | B | Done by 4/27 | | |
| Medium Recall | A+B | ≥ 0.5 | | |
| v2 submitted | A | By 4/30 | | |

### Retrospective (周日填)
- What went well:
- What didn't:
- Action items for next week:
```

---

## 12.9 总结——两人一句话

> **一起做：Agent 循环 / LLM Client / Parser / Reflection / Folding / Prompt / 测试。
> 华：Profiler / 结构化 Tools / Scorer / Docker / 提交 / 调度 / 数据探索。
> 吴：非结构化 Tools（PDF / Image / RAG）。
> 通过 PR 互审和每日 sync 对齐。**
