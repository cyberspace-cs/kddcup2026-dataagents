# 12. 两人作战计划（Two-Person Battle Plan）

> 本文件是 M0-M6 全周期的两人逐日分工计划。
> **Person A**（@Reverse-7）= 队长/主架构 | **Person B**（搭档，待定）= 开发+数据
> 角色定义灵活——搭档入队后可根据实际能力交换特定模块。

---

## 12.1 角色与职责划分

### 核心原则
1. **并行最大化**：同一时间段两人应做不同模块，避免串行等待
2. **接口先行**：先定好模块接口（§06），各自实现，最后集成
3. **每日同步**：晚 21:00 微信 10 分钟——"做了啥 / 卡在哪 / 明天干啥"
4. **PR 互审**：任何合入 main 的代码对方需 1 分钟审（看 diff，不是逐行）

### 职责矩阵

| 模块 | Person A (队长) | Person B (搭档) | 理由 |
|---|---|---|---|
| **项目管理 / 文档 / 提交流程** | 🟢 主力 | 🔵 协助 | A 掌握全局 context |
| **Agent 核心循环** (react.py / plan_execute.py) | 🟢 主力 | 🔵 Review | 架构核心，需统一设计视角 |
| **LLM Client** (client.py / env.py) | 🟢 主力 | 🔵 协助 | 与 Agent 强耦合 |
| **Prompt 工程** (prompt.py / 调优实验) | 🟢🔵 共同 | 🟢🔵 共同 | 两人独立实验 → 合并最优 |
| **Tools — 结构化** (filesystem / sqlite / python_exec) | 🔵 Review | 🟢 主力 | 独立模块，接口清晰 |
| **Tools — 非结构化** (pdf / image / rag) | 🔵 协助 | 🟢 主力 | 需要调 OCR 库，工作量大 |
| **DataProfiler** (scan.py) | 🔵 协助 | 🟢 主力 | 需要理解各数据源细节 |
| **OutputWriter + 本地 Scorer** | 🟢 主力 | 🔵 测试 | 评分逻辑关键；A 对规则更熟 |
| **Docker / Dockerfile / scripts** | 🟢 主力 | 🔵 测试 | 部署面 A 把控 |
| **TaskOrchestrator / Runner** | 🟢 主力 | 🔵 协助 | 全局调度与 Agent 紧密 |
| **测试** (unit / integration) | 🟢🔵 各写各的模块 | 🟢🔵 交叉审 | — |
| **数据探索 / demo 分析 / 对拍** | 🔵 协助 | 🟢 主力 | B 出数据洞察，A 转化为 prompt |

**图例**：🟢 主力（写代码/做决策）| 🔵 协助（review / 写测试 / 兜底）

---

## 12.2 逐日作战计划

### 🗓 M0 — 环境就绪 + 注册（4/17 周四 → 4/19 周日）

> M0 重点：**不写代码**，只做注册 + 环境 + 下载 + Discord Q

| 日期 | Person A (队长) | Person B (搭档) |
|---|---|---|
| **4/17 周四** | 注册 Google Form（队长发起）; 装 Docker Desktop + WSL2 调 .wslconfig; 申请 DashScope API key | 确认 GitHub 邮箱、clone 仓库; 装 Python 3.11 + uv; 装 Docker Desktop |
| **4/18 周五** | 测试 `docker run hello-world`; 测试 Qwen-Max API 调通; 下载 Phase 1 demo 数据（百度网盘 ymdc） | 同步下载 demo 数据; 加入 Discord / 微信群; 浏览 starter kit README |
| **4/19 周六** | 发 Discord 提问 Q2-Q7（我帮起草文案）; 验证 demo 数据目录结构正确; 队员身份验证（若 B 已注册） | 完成身份验证（Informed Consent Form）; 阅读 docs/00-12 全套文档熟悉架构; 列出自己的 skill gaps 告知 A |

**M0 门禁**：`uv`✅ + `docker`✅ + `Qwen-Max API`✅ + `demo 数据在位`✅ + `两人都 clone 了 repo`✅

---

### 🗓 M1 — Fork Starter Kit & 跑通（4/20 周日 → 4/22 周二）

| 日期 | Person A (队长) | Person B (搭档) |
|---|---|---|
| **4/20 周日** | `git clone` starter kit 到 `vendor/starter/`; `uv sync` 安装依赖; 填 config 指向 Qwen-Max API; 跑 `dabench status` + `dabench run-task task_1` | 同步 A 的 vendor/ 环境; 跑 `dabench inspect-task` 对 demo 所有任务做数据探索; 记录每任务的数据源类型、行数、文件大小 → 输出 `docs/demo_data_survey.md` |
| **4/21 周一** | 跑 `dabench run-benchmark --limit 5` 拿 baseline; 记录 metrics → `docs/baseline_metrics.md`（每任务步数/tokens/耗时） | 继续数据探索：手动看 gold.csv 的列结构，理解评分目标; 对比 prediction vs gold 手动算几个分数; 验证 `scripts/score_predictions.py` 框架（先写骨架） |
| **4/22 周二** | 跑全量 demo benchmark; 汇总 baseline_metrics; **两人 sync**：确认 starter kit 能力边界 + 下一步定制方向 | 完成 `docs/demo_data_survey.md`; 在 baseline_metrics 里标注哪些任务通过/失败/超时; 对失败任务做根因分类（SQL 错？PDF 没读？格式错？） |

**M1 门禁**：starter kit 能跑 ≥ 1 task succeeded ✅ + baseline_metrics 有了 ✅ + demo_data_survey 完成 ✅

---

### 🗓 M2 — Fork 定制 + Docker 化（4/20 并行 → 4/23 周三）

> M1 和 M2 时间重叠——A 做 M1 跑通时，B 可以并行做 M2 的数据探索和工具原型。

| 日期 | Person A (队长) | Person B (搭档) |
|---|---|---|
| **4/20-21** | （M1 跑通任务） | （M1 数据探索任务） |
| **4/22 周二** | 建 `src/dabench/` 骨架（目录结构 §4.1）; 移植 starter kit 核心代码到 dabench; 加 `llm/env.py` 环境变量覆盖层 | 写 `tools/filesystem.py` 的改进版（加 columns 选列、polars 大文件）; 写 `tools/sqlite.py` 改进版（放宽 SQL 前缀检查） |
| **4/23 周三** | 写 `main.py` 容器 ENTRYPOINT; 写 `docker/Dockerfile`; 跑 `scripts/eval_local.sh` 首次; **测试：docker build + docker run 通过** | 写 `tools/python_exec.py` 改进版（提升 timeout 到 60s）; 写 `io/output_writer.py` 归一化逻辑; 写 `io/output_writer.py` 的单元测试 |

**4/23 晚两人 sync**：集成 A 的框架 + B 的 tools → 合入 main → docker build → eval_local.sh 通过

**M2 门禁**：`eval_local.sh` 通过 ✅ + 镜像 < 3 GB ✅ + `/output/task_<id>/prediction.csv` 路径正确 ✅

---

### 🗓 M3 — 首次提交 baseline（4/24 周四，Phase 1 开赛）

| 时段 | Person A (队长) | Person B (搭档) |
|---|---|---|
| **上午** | `pre_submit_check.sh v1` 全绿; tag v1; docker save + tar.gz | 并行写 `score_predictions.py` 完整版; 对拍 demo gold，记录 v1 本地分数 |
| **下午** | 重命名 archive → `team<NNNN>_v1.tar.gz`; 上传 Google Drive; 发邮件 | 更新 CHANGELOG.md + submission_log.md |
| **晚上** | 两人 sync：v1 已提交；review M4 任务分配 | 同上 |

**M3 门禁**：邮件已发 ✅ + Drive 链接可访问 ✅ + submission_log 有记录 ✅

---

### 🗓 M4 — 覆盖 medium（4/25 周五 → 4/30 周三）

**主线任务**：DataProfiler 上线 + PDF 工具 + Text-to-SQL prompt + 本地对拍系统成熟

| 日期 | Person A (队长) | Person B (搭档) |
|---|---|---|
| **4/25 周五** | 实现 `profiler/scan.py`：CSV/JSON/SQLite 扫描部分 | 实现 `tools/pdf.py` (pdfplumber + pymupdf fallback) |
| **4/26 周六** | 将 DataProfiler 集成进 Agent 循环（修 prompt.py 加 `{data_profile_summary}`） | 实现 `tools/rag.py`（bge-small embedding + FAISS 检索 knowledge.md） |
| **4/27 周日** | Text-to-SQL 专项 prompt 调优; 在 medium demo 任务上跑 A/B | 完善 `score_predictions.py`; 全量 demo 对拍; 写 regression 检测 |
| **4/28 周一** | 集成 B 的 tools; 整体测试; 修 bug | 写 pdf/rag 的单元测试; 探索失败的 medium 任务根因 |
| **4/29 周二** | `pre_submit_check.sh v2`; 提交 v2 | 分析 v1 评测结果（若已返回）; 更新 CHANGELOG + submission_log |
| **4/30 周三** | review v2 leaderboard; 规划 M5 冲刺方向 | 根据 v1/v2 分数做 gap 分析 → `docs/gap_analysis.md` |

**提交节奏**：v2（4/29 or 4/30）——medium 覆盖

**M4 门禁**：demo 上 medium 平均 Recall ≥ 0.5 ✅ + DataProfiler 上线 ✅ + v2 提交 ✅

---

### 🗓 M5 — hard & extreme 攻坚（5/1 周四 → 5/10 周六）

**主线任务**：Plan-and-Execute 上线 + 视觉路径 + reflection + context folding

| 时段 | Person A (队长) | Person B (搭档) |
|---|---|---|
| **5/1-5/2** | 实现 `agents/plan_execute.py`（Plan + DAG 拓扑执行 + blackboard）; 对应 prompt 模板（§7.6, §7.7） | 实现 `tools/image.py`：Qwen vision 主路径; 实现 `tools/image.py`：tesseract OCR 兜底路径 |
| **5/3-5/4** | 集成 Plan-Execute 到 `build_planner` adaptive 路由; hard demo 任务 A/B 对比（ReAct vs Plan-Exec） | 实现 `inspect_pdf` / `inspect_image` 工具; Profiler 加 PDF 页数/图片维度扫描 |
| **5/5-5/6** | 实现 reflection 层（extreme 失败重规划）; context folding 机制（压缩旧 steps） | 写 Plan-Execute 的集成测试; 写 image tool 的单元测试（mock LLM 视觉调用） |
| **5/7 周三** | 集成 + 回归测 + v3 提交 | 分析 v2 leaderboard 结果; 按 difficulty 做分数拆解 |
| **5/8-5/9** | 根据 leaderboard 反馈调优; prompt 实验 E1-E3（§7.12） | 实验 E4-E5（中文 vs 英文 prompt; few-shot） |
| **5/10 周六** | v4 or v5 提交（视实验结果） | 更新 gap_analysis; 准备 M6 冲榜清单 |

**提交节奏**：v3（5/7）、v4/v5（5/10）——hard 覆盖 + vision + plan-exec

**M5 门禁**：hard 平均 ≥ 0.3 ✅ + extreme 至少 1 个非零 ✅ + vision 路径有效 ✅

---

### 🗓 M6 — 冲榜与稳定（5/11 周日 → 5/22 周四）

**主线任务**：精细调优 + 资源管理 + 鲁棒性 + 压轴提交

| 阶段 | Person A (队长) | Person B (搭档) |
|---|---|---|
| **5/11-5/14 (Week 1)** | Prompt 精调：按 leaderboard 反馈逐难度优化; temperature / top_p / top_k 扫参 | 鲁棒性加固：异常处理覆盖; 大文件 OOM 防护（polars lazy）; 超时降级路径验证 |
| **5/15-5/18 (Week 2)** | 长上下文专项：context folding 调参; extreme 任务逐个分析; few-shot 示例库构建 | 性能优化：并行度调优（max_workers 实验）; 减少每步 token 消耗; 本地评测在**云 16C64G** 做一次 dry run |
| **5/19-5/20** | **冷冻策略**：不再大改架构; 仅 prompt 微调 + bug fix | 全量回归测; 每个版本 vs 上版逐任务对比表 |
| **5/21 周三** | v24-v25 提交（保守微调） | 最终回归确认; CHANGELOG 归档 |
| **5/22 周四** | **最后一次提交** v26-v28（若还有次数）; 提交截止前 6h 发 | 监控邮件确认评测成功 |
| **5/23 周五** | ❌ **不提交** — 避免 AoE 时区风险 | 同上 |

**提交节奏**：v6-v28（约 15-22 次），按 §5.5 节奏

**M6 门禁**：排名进入 cutoff ✅（或无法控制则尽力而为）

---

## 12.3 并行度分析（关键路径）

```
                          Person A                    Person B
                          ────────                    ────────
M0 (4/17-4/19)           注册 + 环境 + API           环境 + clone + 读文档
                               │                          │
                               ▼                          ▼
M1 (4/20-4/22)           跑 starter kit ─────────── 数据探索 + demo survey
                               │                          │
                               ▼                          ▼
M2 (4/22-4/23)           骨架 + env + Docker ─────── Tools 改进 + OutputWriter
                               │            ╲             │
                               │             ╲ 集成       │
                               ▼              ╲           ▼
M3 (4/24)                提交 v1 ◄─────────── 对拍 + scorer
                               │                          │
                               ▼                          ▼
M4 (4/25-4/30)           Profiler + 循环优化 ───── PDF + RAG tools + 测试
                               │                          │
                               ▼                          ▼
M5 (5/1-5/10)            Plan-Execute + reflect ─── Vision/OCR tools + 测试
                               │                          │
                               ▼                          ▼
M6 (5/11-5/22)           Prompt 精调 + 提交 ─────── 鲁棒性 + 性能 + 云 dry run
```

**关键路径**（阻塞链）：
```
starter kit 跑通 → 骨架 + env → Docker → v1 → Profiler → Plan-Exec → Prompt 调优 → v_final
```
Person B 的 tools/数据工作与关键路径**并行**，不阻塞 A。

**最大风险点**：M2 集成（4/23 晚）——A 的框架 + B 的 tools 第一次对接。应对：4/22 先做一次 dry integration（B 在 A 的骨架上挂 1 个 tool 跑通）。

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

## 12.6 B 入队第一天（Onboarding）

搭档拿到 GitHub collaborator 权限后的 Day 1 清单：

```markdown
### B 的 Onboarding Checklist

- [ ] clone 仓库 `git clone https://github.com/Reverse-7/kddcup2026-dataagents.git`
- [ ] 从头到尾读 `docs/00-index.md`（20 分钟）
- [ ] 精读自己负责的模块文档：
  - docs/06-interfaces.md §6.4（ToolSpec/ToolResult）
  - docs/06-interfaces.md §6.9（DataProfiler）
  - docs/07-prompts.md §7.9（Tool descriptions）
  - docs/10-task-playbooks.md（理解四档难度差异）
- [ ] 装环境：Python 3.11 + uv + Docker Desktop
- [ ] 下载 Phase 1 demo 到 `data/public/`
- [ ] 跑一遍 starter kit：`uv run dabench run-task task_1 --config ...`
- [ ] 手动看 3 个 gold.csv 理解输出格式
- [ ] 微信确认 A：环境 OK，准备接任务
```

---

## 12.7 搭档未到位的 Plan B

**最坏情况**：Phase 1 全程只有 A 一人。

| 影响 | 应对 |
|---|---|
| 工作量翻倍 | **砍 scope**：v1 只做 easy/medium（ReAct + SQL + CSV）；hard 推到 5 月中再碰 |
| 无人 code review | 用 CI lint（ruff）替代人工审；prompt 变更用 A/B 对拍替代 review |
| 无人做数据探索 | 先跑 Profiler 自动输出 demo survey；跳过手动深度分析 |
| 无人做 OCR/vision 兜底 | **延后到 M5**；v1-v3 不处理图表题（拿 0 分也行，先保 easy/medium） |

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

> **A 搭框架跑流程提交镜像，B 铺工具做数据保质量。两人通过 PR 和每日 sync 对齐。**
