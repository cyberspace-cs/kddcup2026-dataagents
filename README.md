# KDD Cup 2026: Data Agents

Solution repo for **KDD Cup 2026: Data Agents for Complex Data Analysis**
(organized by Tsinghua University & HKUST-GZ; official KDD conference track).

> ⚠️ **Private repository.** Do not share code, docs, or discussion outside the team.
> Competition rules §8 prohibit sharing solutions between teams.

---

## Documentation

Full project plan under [`docs/`](docs/). Start at [`docs/00-index.md`](docs/00-index.md).

| # | Doc | Topic |
|---|---|---|
| 00 | [Index](docs/00-index.md) | 文档总览 + 事实卡 + ADR 索引 |
| 01 | [Requirements](docs/01-requirements.md) | 需求、硬约束、I/O 契约、待澄清问题 |
| 02 | [Feasibility](docs/02-feasibility.md) | Qwen3.5-35B-A3B 能力、starter kit gap、风险前置 |
| 03 | [Architecture](docs/03-architecture.md) | C4 架构 + 6 个 ADR |
| 04 | [Detailed Design](docs/04-detailed-design.md) | 模块、数据结构、ReAct 状态机、Plan-Execute |
| 05 | [Milestones / Testing / Risk](docs/05-milestones-testing-risk.md) | M0-M6、测试金字塔、风险登记册 |
| 06 | [Interfaces](docs/06-interfaces.md) | 所有模块的 Python 公共 API |
| 07 | [Prompts](docs/07-prompts.md) | System / Task / Tool prompt 库 |
| 08 | [Local Scorer](docs/08-local-scorer.md) | 按规则 §6 实现的对拍评分器 |
| 09 | [Deployment](docs/09-deployment.md) | Dockerfile、eval_local.sh、submit.sh |
| 10 | [Task Playbooks](docs/10-task-playbooks.md) | easy/medium/hard/extreme 四档处理 |
| 11 | [Ops](docs/11-ops.md) | 两人 git 工作流、提交 checklist、回滚 |

---

## Status

- [x] Planning & research（2026-04-16）— 12 docs, ~3700 lines
- [ ] **M0** — 环境就绪 + 注册（截止 4/23）
- [ ] **M1** — Fork starter kit 并跑通（4/20-4/22）
- [ ] **M2** — Docker 化 + 本地评测闭环（4/20-4/23）
- [ ] **M3** — 首次提交占位（4/24，Phase 1 开赛）
- [ ] **M4** — medium 难度覆盖（4/25-4/30）
- [ ] **M5** — hard / extreme 攻坚（5/1-5/10）
- [ ] **M6** — 冲榜（5/11-5/23，Phase 1 截止）

---

## Competition Reference

- Official site: <https://dataagent.top>
- Rules: <https://dataagent.top/rules>
- Starter kit: <https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit>
- Submission: `kddcup@hkust-gz.edu.cn`
- Phase 1 demo: Baidu Netdisk code `ymdc` / Google Drive link on official site

---

## Team

- [@Reverse-7](https://github.com/Reverse-7) — lead
- TBD — teammate（pending）

---

## Layout (planned, built through M1-M2)

```
.
├── docs/                   ← 计划文档（当前）
├── src/dabench/            ← 主源码（M2 建）
├── vendor/starter/         ← starter kit fork（M1 建，gitignored）
├── configs/                ← YAML 配置（M2 建）
├── scripts/                ← eval_local.sh / pre_submit_check.sh / submit.sh（M2 建）
├── docker/Dockerfile       ← M2 建
├── tests/                  ← 单元 / 集成 / E2E（M2 建）
├── data/                   ← Phase 1 demo（本地，gitignored）
├── artifacts/              ← 本地运行产物（gitignored）
├── CHANGELOG.md            ← 每次提交记录（M3 起）
├── pyproject.toml          ← M2 建
├── uv.lock                 ← M2 建
├── .gitignore
└── README.md               ← (this file)
```
