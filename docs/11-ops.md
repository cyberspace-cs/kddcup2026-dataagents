# 11. 运维 Playbook（Ops）

> 两人团队的协作、提交、回滚、决策归档标准流程。

## 11.1 Git 工作流（两人规模）

**分支模型**：简化 trunk-based + 提交前 code review

```
main            ← 稳定分支；仅通过 PR 合入；每个 tag v<N> 对应一次提交
 │
 ├─ feat/<name> ← 功能分支，从 main 拉出
 ├─ fix/<name>  ← 修 bug
 └─ exp/<name>  ← 实验分支（prompt 调优、参数 A/B），不一定合入
```

**规则**：
- `main` 受保护：禁止直推；所有改动走 PR
- PR 需 **1 人 approve**（两人团队互审）
- PR 合并前必须本地 `scripts/pre_submit_check.sh` 等价检查通过（CI 若未配好，口头约定）
- commit message 用 Conventional Commits：`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`

**分工**（详见 `12-two-person-plan.md`）：
| 领域 | 华 | 吴 |
|---|---|---|
| Agent 核心 / Planner / LLM Client / Reflection / Folding | 🟢🔵 | 🟢🔵 |
| Tools 结构化 / Profiler / Scorer | 🟢 | 🔵 |
| Tools 非结构化 (PDF/Image/RAG) | 🔵 | 🟢 |
| Docker / scripts / 提交 | 🟢 | 🔵 |
| Prompt / 测试 / 文档 | 🟢🔵 | 🟢🔵 |

谁先有 idea 谁做；🟢 是出问题时先找谁。

## 11.2 CHANGELOG 维护

`CHANGELOG.md` 每次提交前更新，格式：

```markdown
# Changelog

## v3 — 2026-04-27 (pending submission)
### Changed
- 切换 Planner 到 adaptive 模式（extreme 走 Plan-Execute）
### Added
- read_chart tool（走 Qwen vision 主路径 + tesseract 兜底）
### Fixed
- csv 读取千分位 comma 导致 dtype=object 的问题
### Local demo score
- Total: 0.42 (vs v2: 0.31)
- By diff: easy=0.85, medium=0.55, hard=0.18, extreme=0.05

## v2 — 2026-04-25
...
```

## 11.3 提交日志

`docs/submission_log.md` 记录每次提交的**完整元数据**，用于事后复盘：

```markdown
| Version | Date (AoE) | Git tag | Image SHA | Archive size | Local score | Leaderboard score | Key change | Notes |
|---|---|---|---|---|---|---|---|---|
| v1 | 2026-04-24 | v1 | sha256:abc... | 820 MB | 0.18 | ? | baseline | 占位 |
| v2 | 2026-04-25 | v2 | sha256:def... | 850 MB | 0.31 | ? | 加 PDF 工具 | — |
```

## 11.4 每次提交的 checklist

一个版本从 "想提交" 到 "发出邮件" 要走的所有步骤：

```markdown
### v<N> 提交 checklist

- [ ] CHANGELOG.md 更新完（含本地得分 + 对比上版）
- [ ] git status 干净；所有改动已 commit
- [ ] 本地 `scripts/pre_submit_check.sh <N>` 全绿
  - [ ] ruff 通过
  - [ ] pytest 通过
  - [ ] docker build 成功
  - [ ] 镜像大小 < 9 GB
  - [ ] demo 评测通过
  - [ ] 本地评分不回退（或回退已 ACK）
  - [ ] tar.gz 大小 < 10 GB
- [ ] `git tag v<N> && git push origin v<N>`
- [ ] 重命名 archive：`mv dabench_v<N>.tar.gz team<NNNN>_v<N>.tar.gz`
- [ ] 上传 Google Drive，设"有链接任何人可看"
- [ ] 验证链接能匿名访问（用隐身窗口）
- [ ] 跑 `scripts/submit.sh team<NNNN> <N> '<gdrive_link>'`
- [ ] 邮件主题严格匹配 `[KDDCup2026 Data Agents] Submission - team<NNNN> - v<N>`
- [ ] 发邮件给 `kddcup@hkust-gz.edu.cn`
- [ ] 抄送自己（留痕）
- [ ] 记录到 `docs/submission_log.md`
- [ ] 等评测完成（≤24h），结果回邮后更新 submission_log.md
```

## 11.5 回滚流程

### 场景 A：本地 pre_submit_check 失败

```
1. 不 tag，不打镜像，不提交
2. 在 feat 分支修，不污染 main
3. git reset --soft HEAD~1 如需重写 commit
```

### 场景 B：提交后评测 0 分 / 明显回退

```
1. 立刻从 /logs 查错：
   docker run --rm -v "$PWD/artifacts/eval_logs:/logs:ro" alpine cat /logs/main.log
2. 定位根因（env 未读？输出路径错？依赖缺失？）
3. 决策：下一版是修这个还是继续既定路线
4. 若紧急修复，开 fix/ 分支快速迭代
5. submission_log.md 标注"v<N> scored 0, reverted strategy in v<N+1>"
```

**不要**：从 Drive 删已提交的 tar.gz（规则明确要求保留到评测完成通知）

### 场景 C：镜像构建成功但本地跑不通

```
1. 用 docker run 进 shell 手动调试：
   docker run --rm -it --entrypoint /bin/bash dabench:dev
2. 在容器里手动跑 python -m dabench.main
3. 对比本地 Python 与容器 Python 版本差异
```

## 11.6 实验跟踪（M5-M6 用）

`docs/experiments.md`：

```markdown
## E1: ReAct vs Plan-Execute on hard tasks

Date: 2026-05-03
Branch: exp/planexecute-hard
Hypothesis: hard tasks benefit from explicit planning

Setup:
- Baseline: ReAct all difficulties
- Treatment: ReAct easy/medium, Plan-Execute hard/extreme
- Dataset: demo Phase 1, 4 hard tasks
- Runs: 3 each, mean

Results:
| Task | Baseline | Treatment |
|------|----------|-----------|
| task_15 | 0.22 | 0.58 |
| task_18 | 0.00 | 0.33 |
| task_21 | 0.45 | 0.42 |
| task_25 | 0.12 | 0.51 |
| Mean | 0.20 | 0.46 |

Decision: merge exp branch; enable plan_execute for hard+extreme in v<N>
```

## 11.7 Discord / 邮件 问答归档

`docs/qna_organizer.md`：

```markdown
## 2026-04-17: 关于评分（Q2）λ 值

**Asked (Discord #general):**
"What is the specific value of λ in Score = Recall − λ·(Extra/Predicted)?"

**Organizer (...):** (reply pasted here)

**Our interpretation:**
...

**Action:**
- Update scripts/score_predictions.py default from 0.1 to <value>
- CHANGELOG: v3 noted
```

## 11.8 预算监控（Phase 1 提交次数）

```markdown
## 提交预算跟踪

| 日期 | 累计提交 | 剩余预算 | 备注 |
|---|---|---|---|
| 4/24 | 1/30 | 29 | v1 baseline |
| 4/25 | 2/30 | 28 | v2 加 PDF |
| ... | | | |
```

规则：
- 不在同一天提两次（日上限 1）
- 连续 3 次无提升 → 下次必须重大改动（prompt 重构 / 架构调整）
- 保留最后 **3 次**给 5/20-5/22 压轴稳定微调，不做冒险改动

## 11.9 夜间 / 周末作业

**鉴于 Phase 1 只有 30 天**，高强度期建议：

| 时段 | 做什么 |
|---|---|
| 工作日晚 20:00-24:00 | 开发 + 跑本地 demo |
| 工作日白天 | 思考 / 读论文 / 写 prompt 变体 |
| 周末 | 高强度开发 + 当日提交 |
| 提交日（任何日）  | 17:00 前跑完 pre_submit_check，18:00 前发邮件；留出回复窗口 |

**AoE 时区陷阱**：所有"每日 1 次"按 AoE（UTC-12），即中国时间**次日 20:00** 前都算同一天。晚于中国 20:00 的提交可能跨日。建议**中国时间 17:00 前完成提交邮件发送**以避开边界。

## 11.10 故障手册（FAQ 自建）

| 症状 | 常见原因 | 快速排查 |
|---|---|---|
| pre_submit_check 在 pytest 卡住 | fixture 卡 LLM 调用 | 检查是否有测试没 mock LLM |
| 镜像构建在 pip install 超时 | 网络到 PyPI 慢 | 启用国内镜像源（§9.4.2） |
| docker run 报 "no space left" | WSL2 虚拟磁盘满 | PowerShell: `wsl --shutdown`; 压缩 `ext4.vhdx` |
| 容器跑 10 分钟无输出 | Python buffering | `PYTHONUNBUFFERED=1`（已在 Dockerfile） |
| main.py 退出码 78 | MODEL_API_KEY 未注入 | `docker run -e MODEL_API_KEY=...` |
| prediction.csv 空 / 损坏 | OutputWriter 归一化 bug | 查 `/logs/task_X.jsonl` 看 answer 字段 |
| leaderboard 分数比本地低 | λ 不同 / normalization 不同 / 数据不同 | 记入 `qna_organizer.md`，看组委会回复 |

## 11.11 M6 最终周执行计划（5/17-5/23）

最后一周每天的节奏（示例）：

| 日期 | 目标 | 提交 v? |
|---|---|---|
| 5/17 周日 | 分析前一周所有 leaderboard 反馈，排序 TODO | v20 |
| 5/18 周一 | 针对 medium 薄弱任务 prompt 调优 | v21 |
| 5/19 周二 | 对 extreme 做 context folding 调参 | v22 |
| 5/20 周三 | **冷冻**：不做新改动，纯 prompt 微调 | v23 |
| 5/21 周四 | 再测一遍全量 demo，对比历次 | v24 |
| 5/22 周五 | 最终优化——保守改动 | v25 |
| 5/23 周六 | **截止日**；**不提交**，避免跨时区风险 | — |

## 11.12 项目会议节奏

两人规模，重"结果沟通"轻"仪式"：

| 节奏 | 形式 | 时长 |
|---|---|---|
| 每日 | 晚 21:00 微信同步 "今天做了啥/卡住啥" | 10 分钟 |
| 每周日 | 复盘：上周提交得分、本周目标 | 30 分钟 |
| 提交后 | 当天即时讨论评测结果 | 15 分钟 |
| Phase 1 结束 | 深度复盘 + Phase 2 子赛道决策 | 2 小时 |
