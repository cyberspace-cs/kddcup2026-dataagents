# 2. 可行性分析（Feasibility）

## 2.1 技术可行性

### 2.1.1 模型能力匹配度（Qwen3.5-35B-A3B）

**已从 HuggingFace `config.json` 亲自核实**（文件留存：`C:/Users/16321/AppData/Local/Temp/qwen_research/qwen35-35b-a3b-config.json`）：

| 能力 | 数据 | 对赛题影响 |
|---|---|---|
| 架构 | MoE：256 experts，每 token 激活 8 个，~3B 活跃参数 / 35B 总参数 | 推理成本低 |
| 上下文 | `max_position_embeddings: 262144`（256K 原生） | **覆盖 extreme 难度的长文档任务** |
| 视觉 | 原生多模态：`vision_config` 27 层 ViT + `image_token_id: 248056` + `video_token_id: 248057` | **PDF/图表可直接走多模态通道**，无需 CPU OCR（**假设评测服务未禁用视觉**——见 Q3） |
| 注意力 | Hybrid：每 4 层一个 full-attention，其余 linear_attention | 长上下文效率高 |
| 工具调用 | `--tool-call-parser qwen3_coder` | **格式非 OpenAI JSON**，而是 XML 风格 `<tool_call><function=...><parameter=...>`；vLLM 会自动转换为 OpenAI 风格响应给客户端 |
| 思考模式 | 默认启用，输出含 `<think>...</think>`；vLLM 的 `--reasoning-parser qwen3` 会拆成 `reasoning_content` 字段 | 主流 OpenAI SDK 调用 `.choices[0].message.content` 得到的是**已剥离思考部分的最终答案** |
| 官方采样推荐（thinking general） | `temp=1.0, top_p=0.95, top_k=20, presence_penalty=1.5` | 与 starter kit 默认 `temperature=0.0` **不一致**——需重新调参 |

### 2.1.2 已知弱点（Qwen README 官方 benchmark）

- **Long-context retrieval (AA-LCR) 58.5** — 是 Qwen3.5 系列中该指标最弱的型号，弱于 27B dense（66.1）和 122B-A10B（66.9）
- **IFBench 70.2** — 指令遵循在约束条件下偏弱
- **README 自己推荐** search agent 在 256k 上下文时做 context folding（裁剪旧工具响应）

**工程影响**：extreme 难度的"长上下文+记忆管理"不能依赖模型原生长程检索，**必须**在 Agent 侧做工具响应裁剪 + 关键信息索引。

### 2.1.3 官方 Starter Kit 评估

**已从 GitHub 亲自核实源码**（本地副本：`C:/Users/16321/AppData/Local/Temp/starter_src/`）。关键发现：

| 项 | 现状 | 与评测契约的 gap |
|---|---|---|
| 包管理 | `uv` + `pyproject.toml` | 需加 `requirements.txt` 或用 uv 的 `uv export` |
| 依赖 | `openai>=2.28.0`, `typer`, `rich`, `pandas`, `polars`, `duckdb`, `pyarrow`, `pydantic`, `pyyaml` | 够用；需加 PDF 解析（pdfplumber/pymupdf）、embedding 模型 |
| ReAct 协议 | **文本 JSON 协议**（ass 输出 \`\`\`json ... \`\`\` fence，解析 `thought/action/action_input`） | 不用 OpenAI 原生 tool calling；vs `qwen3_coder` 格式是否有兼容问题需测（Qwen 响应端自动转 OpenAI 格式，但**客户端若用文本协议根本不走 tool_calls** → 无冲突） |
| 模型调用 | OpenAI SDK，**只从 YAML 读 api_base/api_key/model，不读 env** | 必须补 env 覆盖逻辑 |
| `execute_python` | `multiprocessing.Process` + `os.chdir(context_root)`，30s 超时 | 无网络沙箱/资源限制，但评测环境已经封网，可用 |
| `execute_context_sql` | SQLite `file:...?mode=ro`，只接受 `select/with/pragma` | 够用 |
| 输出路径 | `output_dir/<run_id>/<task_id>/prediction.csv` | **不直接兼容** `/output/<task_id>/prediction.csv`（需压平 run_id 或换写逻辑） |
| Dockerfile | **无** | 必须自建 |
| `/logs` | 无显式写入 | 必须加文件 handler |
| `task.json` schema | **严格等价校验** `{task_id, difficulty, question}`，额外字段会报错 | 如果评测集多一个 field 会 crash，需改为 subset 校验 |

**结论**：官方 baseline 架构清晰（~1100 LOC，15 文件），fork 后打补丁比重写省 3-5 天。

## 2.2 资源可行性

### 2.2.1 开发环境

| 资源 | 现状（2026-04-16 确认） | 需要补充 |
|---|---|---|
| 本机 OS | Windows 11 Home，Git Bash 可用 | — |
| 本机 CPU | **24 逻辑核（估 12C24T）** | — |
| 本机 RAM | **16 GB**（评测目标 64 GB，见下 §2.2.3） | — |
| Python | 未验证 | 安装 3.11（uv 会管理） |
| Docker | 未验证 | Docker Desktop (WSL2 后端) |
| LLM API | 未配置 | **Qwen-Max**（DashScope）API key（D2 确定） |
| 磁盘 | — | 镜像构建 + 测试 ≥ 30 GB 预留 |
| 网络 | — | 下载 Phase 1 demo（百度网盘 `ymdc` 或 Google Drive） |

### 2.2.3 本地评测 vs 评测端规格差距

| 维度 | 本机 | 评测端 | 影响 |
|---|---|---|---|
| vCPU | 24 | 16 | 不受限（本机多） |
| RAM | 16 GB | 64 GB | **本机只能做 correctness 测试**；大 CSV/PDF 可能本地 OOM 但评测端能过 |
| GPU | 无 | 无 | 对齐 |
| 网络 | 有外网 | 无外网 | 本地测用 `--network=bridge` + 真 API；真断网测是可选专项 |
| 可运行 Docker 容器 | `--cpus=12 --memory=10g`（给 WSL2 留余量） | `--cpus=16 --memory=64g` | 见下 |

**应对**：
- 代码层：polars/duckdb streaming 规避 OOM
- M5-M6 关键提交前租 **云服务器 16C64G**（阿里云/AWS ¥5-15/h）做 dry run 1-2 小时
- `.wslconfig` 建议 `memory=12GB swap=8GB processors=20`

### 2.2.2 开发替身 LLM 选型

| 候选 | 价格/百万 token | 与评测模型相似度 | 推荐度 |
|---|---|---|---|
| **Qwen-Max / Qwen3-235B** API（DashScope） | 中 | **高**（同家族，tool calling 格式一致） | ★★★★★ |
| DeepSeek-V3 | 低 | 中（MoE 类似，但 tool 格式不同） | ★★★ |
| GPT-4.1-mini（starter kit 默认） | 中 | 低（OpenAI 风格完全不同） | ★★ |

**决策**：开发用 **Qwen-Max** 主力（格式最接近），DeepSeek-V3 做对比验证。

## 2.3 时间可行性

**今天 2026-04-16，Phase 1 开赛 2026-04-24（距今 8 天）。**

| 任务 | 乐观 | 现实 | 保守 |
|---|---|---|---|
| 注册完成 + demo 下载 | 0.5 d | 1 d | 2 d |
| Fork starter kit + 补 env/Docker | 1 d | 2 d | 3 d |
| demo 全量跑通 easy 任务 | 1 d | 2 d | 4 d |
| Docker 本地评测闭环 | 1 d | 2 d | 3 d |
| **首次提交（baseline 占位上榜）** | 3.5 d | 7 d | 12 d |

**结论**：4/24 开赛当天提交 baseline **可达**但紧迫，必须并行推进注册 + 开发。若被卡在注册审核（每人独立身份验证），最晚 4/23 前必须提交注册申请。

## 2.4 关键风险前置识别

> 完整 15 条风险登记 + 触发条件见 `05-milestones-testing-risk.md §5.3`。本节只列前置高优。

| # | 风险 | 发生概率 | 影响 | 早期应对 |
|---|---|---|---|---|
| R1 | 评测端 Qwen 未启用视觉（Q3 未答） | 中 | 高（hard/extreme 图表任务归零） | Discord 问；**同时准备 CPU OCR 兜底**（ADR-006，v1 就做） |
| R2 | λ 具体值未知（Q2 未答） | 中 | 低-中 | 公式已确认连续（Q1 已裁决）；λ 先假设 0.1，leaderboard 出分后反推 |
| R3 | Qwen `qwen3_coder` XML 格式兼容性 | 低 | 中 | 文本 JSON 协议（ADR-001）彻底绕过原生 tool call |
| R4 | Docker 镜像超 10 GB | 中 | 高（被拒） | `python:3.11-slim` + 多阶段（ADR-003） |
| R5 | 12h 总超时被打爆 | 中 | 高 | 每任务硬超时；即写即存；降级策略（见 §10.6） |
| R6 | 注册卡审核 | 低 | 致命（无法提交） | 4/19 前发起 |
| R7 | 提交次数浪费（30 次极限） | 中 | 中 | `pre_submit_check.sh` + 回归对拍（§9.3, §11.4） |
| R8 | 依赖版本与 Python 3.11 不兼容 | 低 | 中 | uv.lock + Docker pin |
| R9 | PDF/图像 token 爆上下文 | 中 | 中 | Profiler 抽样 + context folding |
| R10 | `task.json` 多字段导致 schema 校验 fail | 低 | 高 | subset 校验（fork 时改 `schema.py`） |

## 2.5 可行性结论

**可行，但 8 天时间窗口极紧，必须严格按里程碑推进。**
- 架构可行：starter kit 已覆盖 80% 骨架；剩下的是 Docker 化 + 环境变量 + 输出路径
- 模型可行：Qwen3.5-35B-A3B 256K 原生上下文 + 多模态，覆盖 easy-hard 难度
- 唯一真正高风险点：**评测端视觉是否启用** — 决定 hard 档是否需要 CPU OCR 兜底路径
