<div align="center">

# KDD Cup 2026: Data Agents — Team Repo

English | [中文](README.zh.md)

[![Official Website](https://img.shields.io/badge/Official%20Website-Visit%20dataagent.top-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white&labelColor=0f172a)](https://dataagent.top)
[![Demo Dataset](https://img.shields.io/badge/Demo%20Dataset-Download%20Phase%201-f59e0b?style=for-the-badge&logo=googledrive&logoColor=white&labelColor=0f172a)](https://drive.google.com/file/d/1lICQVM_LfyQ5DMEIZjssq6aaOPTWCtNd/view?usp=share_link)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?style=for-the-badge&logo=discord&logoColor=white&labelColor=0f172a)](https://discord.com/invite/7eFwJQN3Fx)

</div>

> ⚠️ **Private repository.** Do not share code, docs, or discussion outside the team.
> Competition rules §8 prohibit sharing solutions between teams.

Solution repo for **KDD Cup 2026: Data Agents for Complex Data Analysis**
(organized by Tsinghua University & HKUST-GZ; official KDD conference track).
This repo combines the team's planning docs with the official DataAgent-Bench starter kit.

---

## Team Documentation

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

## Status

- [x] Planning & research（2026-04-16）— 12 docs, ~3700 lines
- [ ] **M0** — 环境就绪 + 注册（截止 4/23）
- [ ] **M1** — Fork starter kit 并跑通（4/20-4/22）
- [ ] **M2** — Docker 化 + 本地评测闭环（4/20-4/23）
- [ ] **M3** — 首次提交占位（4/24，Phase 1 开赛）
- [ ] **M4** — medium 难度覆盖（4/25-4/30）
- [ ] **M5** — hard / extreme 攻坚（5/1-5/10）
- [ ] **M6** — 冲榜（5/11-5/23，Phase 1 截止）

## Team

- **华** [@Reverse-7](https://github.com/Reverse-7) — 工程保障（Profiler / 结构化 Tools / Scorer / Docker / 提交）
- **吴** TBD — 推理引擎（Agent / LLM Client / Plan-Execute / 非结构化 Tools）

---

## Starter Kit (Upstream)

> Official starter kit for the KDD Cup 2026 DataAgent-Bench challenge. The repository reads tasks from `data/public/input/` and writes predictions for downstream evaluation.

### Overview

| Item | Value |
| --- | --- |
| Dataset input | `data/public/input/` |
| Public demo ground truth | `data/public/output/task_<id>/gold.csv` |
| Hidden test data | `input/` only, no `output/` |
| Entry command | `uv run dabench <command> --config PATH` |
| Default run output | `artifacts/runs/` |

### Quick Start

1. Install `uv` by following the official guide:
   - https://docs.astral.sh/uv/getting-started/installation/
2. On macOS and Linux, the standalone installer is:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Install project dependencies:

   ```bash
   uv sync
   ```

4. Confirm the dataset root is visible:

   ```bash
   uv run dabench status --config configs/react_baseline.example.yaml
   ```

5. Run the baseline:

   ```bash
   uv run dabench run-benchmark --config configs/react_baseline.example.yaml
   ```

### Dataset

The public demo dataset lives under `data/public/input/`. Each task directory follows this structure:

```text
data/public/input/task_<id>/
├── task.json
└── context/
```

The corresponding public demo answers live separately under `data/public/output/task_<id>/gold.csv`.
Hidden test sets only include `input/`, so there is no `output/` directory there.

`task.json` contains:

- `task_id`
- `difficulty`
- `question`

The `context/` directory may contain one or more of:

- CSV files
- JSON files
- SQLite / DB files
- Text documents

### Configuration

An example config file lives at `configs/react_baseline.example.yaml`.

```yaml
dataset:
  root_path: data/public/input

agent:
  model: YOUR_MODEL_NAME
  api_base: YOUR_API_BASE_URL
  api_key: YOUR_API_KEY
  max_steps: 16
  temperature: 0.0

run:
  output_dir: artifacts/runs
  run_id:
  max_workers: 4
  task_timeout_seconds: 600
```

Config fields:

| Field | Meaning |
| --- | --- |
| `dataset.root_path` | Root directory of the public demo `input/` dataset. Relative paths are resolved from the project root. |
| `agent.model` | Model name. |
| `agent.api_base` | OpenAI-compatible API base URL. |
| `agent.api_key` | API key, read directly from the config file. |
| `agent.max_steps` | Maximum ReAct steps per task. |
| `agent.temperature` | Sampling temperature. |
| `run.output_dir` | Output directory for run artifacts. |
| `run.run_id` | Optional run directory name. Defaults to a UTC timestamp if omitted. Must be a single directory name; existing run directories are rejected. |
| `run.max_workers` | Parallel worker count for `run-benchmark`. |
| `run.task_timeout_seconds` | Maximum wall-clock time per task. Set to `0` or a negative value to disable the task-level timeout. |

### CLI

```bash
uv run dabench <command> --config PATH [options]
```

| Command | Purpose | Example |
| --- | --- | --- |
| `status` | Show project paths, config path, dataset root, and public task counts. | `uv run dabench status --config configs/react_baseline.example.yaml` |
| `inspect-task` | Show task metadata and list accessible files under `context/`. | `uv run dabench inspect-task task_1 --config configs/react_baseline.local.yaml` |
| `run-task` | Run the baseline on one task and write outputs. | `uv run dabench run-task task_1 --config configs/react_baseline.local.yaml` |
| `run-benchmark` | Run the baseline across the public dataset. | `uv run dabench run-benchmark --config configs/react_baseline.local.yaml` |

`run-benchmark` also supports `--limit N` to cap the number of tasks.

### Tools

The baseline exposes these tools to the model:

| Tool | Purpose | Inputs |
| --- | --- | --- |
| `list_context` | List files and directories under `context/`. | `max_depth` |
| `read_csv` | Read a CSV preview. | `path`, `max_rows` |
| `read_json` | Read a JSON preview. | `path`, `max_chars` |
| `read_doc` | Read a text document preview. | `path`, `max_chars` |
| `inspect_sqlite_schema` | Inspect tables in a SQLite / DB file. | `path` |
| `execute_context_sql` | Execute read-only SQL against a SQLite / DB file in `context/`. | `path`, `sql`, `limit` |
| `execute_python` | Execute arbitrary Python code inside the task `context/` directory. | `code` |
| `answer` | Submit the final answer table and terminate the task. | `columns`, `rows` |

All file paths passed to tools must be relative to the task `context/` directory.

### Outputs

Each successful task run may produce:

- `trace.json`
- `prediction.csv`

Per-task outputs are written to:

```text
artifacts/runs/<run_id>/<task_id>/
├── trace.json
└── prediction.csv
```

Benchmark runs also write:

```text
artifacts/runs/<run_id>/summary.json
```

### Main Modules

| Module | Responsibility |
| --- | --- |
| `src/data_agent_baseline/benchmark/dataset.py` | Public dataset loader |
| `src/data_agent_baseline/tools/filesystem.py` | `list_context`, `read_csv`, `read_json`, `read_doc` |
| `src/data_agent_baseline/tools/python_exec.py` | `execute_python` |
| `src/data_agent_baseline/tools/sqlite.py` | `inspect_sqlite_schema`, `execute_context_sql` |
| `src/data_agent_baseline/tools/registry.py` | Tool registration and terminal `answer` |
| `src/data_agent_baseline/agents/prompt.py` | System prompt, task prompt, observation prompt |
| `src/data_agent_baseline/agents/react.py` | ReAct runtime with JSON action protocol |
| `src/data_agent_baseline/run/runner.py` | Single-task and benchmark execution |

---

## Competition Reference

- Official site: <https://dataagent.top>
- Rules: <https://dataagent.top/rules>
- Upstream starter kit: <https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit>
- Submission: `kddcup@hkust-gz.edu.cn`
- Phase 1 demo: Baidu Netdisk code `ymdc` / Google Drive link on official site

## Contact

- Open issues (upstream): https://github.com/HKUSTDial/kddcup2026-data-agents-starter-kit/issues
- Official website: https://dataagent.top
- Discord: https://discord.com/invite/7eFwJQN3Fx
- WeChat official account: `数据智能与分析实验室 DIAL`

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://dataagent.top">
          <img
            src="https://api.qrserver.com/v1/create-qr-code/?size=144x144&data=https://dataagent.top&bgcolor=ffffff&color=111827&margin=8"
            alt="Official website QR code"
            width="144"
          />
        </a>
        <br />
        Official Website
      </td>
      <td align="center">
        <a href="https://discord.com/invite/7eFwJQN3Fx">
          <img
            src="https://api.qrserver.com/v1/create-qr-code/?size=144x144&data=https://discord.com/invite/7eFwJQN3Fx&bgcolor=ffffff&color=111827&margin=8"
            alt="Discord QR code"
            width="144"
          />
        </a>
        <br />
        Discord
      </td>
      <td align="center">
        <img
          src="https://dataagent.top/HKUSTGZ_DIAL.jpg"
          alt="WeChat official account QR code"
          width="144"
        />
        <br />
        WeChat Official Account
      </td>
    </tr>
  </table>
</div>
