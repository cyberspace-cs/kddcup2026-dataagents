# 13. Gantt 图（项目甘特图）

> GitHub 原生渲染 Mermaid。若本地不渲染，用 [mermaid.live](https://mermaid.live) 粘贴查看。

## 13.1 总览 Gantt（M0 → M6）

```mermaid
gantt
    title KDD Cup 2026 — Phase 1 项目甘特图
    dateFormat YYYY-MM-DD
    axisFormat %m/%d

    section M0 环境/注册
    注册比赛 (华+吴)                :done, m0_reg, 2026-04-17, 2026-04-18
    装环境 Python/uv/Docker (华)    :active, m0_env_h, 2026-04-17, 2d
    装环境 Python/uv/Docker (吴)    :m0_env_w, 2026-04-17, 2d
    下载 demo 数据 (华)             :m0_data, 2026-04-18, 1d
    申请 DashScope API key (华)     :m0_api, 2026-04-17, 2d
    Discord 问 Q2-Q7 (华)          :m0_discord, 2026-04-18, 2d
    吴通读文档 + clone repo         :m0_onboard, 2026-04-18, 2d

    section M1 跑通 Starter Kit
    uv sync + 跑 baseline (华+吴)  :m1_run, 2026-04-20, 2d
    数据探索 demo survey (华)       :m1_survey, 2026-04-20, 2d
    baseline metrics 记录 (华+吴)   :m1_metrics, 2026-04-21, 1d

    section M2 Fork 定制 + Docker
    Agent 骨架 + LLM Client (华+吴) :crit, m2_agent, 2026-04-22, 2d
    结构化 Tools 改进 (华)           :m2_tools_h, 2026-04-22, 2d
    非结构化 Tools 原型 (吴)         :m2_tools_w, 2026-04-22, 2d
    ResponseParser (华+吴)          :m2_parser, 2026-04-22, 1d
    OutputWriter 归一化 (华)         :m2_writer, 2026-04-22, 1d
    Dockerfile + eval_local (华)    :crit, m2_docker, 2026-04-23, 1d
    M2 集成测试 (华+吴)             :crit, m2_integ, 2026-04-23, 1d

    section M3 首次提交
    pre_submit_check v1 (华+吴)     :crit, m3_check, 2026-04-24, 1d
    打包 + 上传 + 邮件 (华)          :crit, m3_submit, 2026-04-24, 1d

    section M4 覆盖 medium
    DataProfiler (华)               :m4_profiler, 2026-04-25, 2d
    PDF tool (吴)                   :m4_pdf, 2026-04-25, 2d
    RAG tool (吴)                   :m4_rag, 2026-04-26, 2d
    本地 Scorer 完善 (华)            :m4_scorer, 2026-04-26, 2d
    Agent 集成 Profiler+PDF (华+吴) :m4_integ, 2026-04-27, 2d
    Text-to-SQL prompt 调优 (华+吴) :m4_prompt, 2026-04-27, 2d
    v2 提交 (华+吴)                 :crit, m4_v2, 2026-04-29, 1d
    gap 分析 (华+吴)                :m4_gap, 2026-04-30, 1d

    section M5 hard/extreme
    Plan-Execute 实现 (华+吴)       :crit, m5_planexec, 2026-05-01, 3d
    Image/Chart tool (吴)           :m5_image, 2026-05-01, 3d
    Image/Chart 测试 (华)           :m5_image_test, 2026-05-03, 2d
    Reflection 层 (华+吴)           :m5_reflect, 2026-05-04, 2d
    Context Folding (华+吴)         :m5_fold, 2026-05-05, 2d
    v3 提交 (华+吴)                 :crit, m5_v3, 2026-05-07, 1d
    Prompt 实验 E1-E5 (华+吴)       :m5_exp, 2026-05-08, 2d
    v4/v5 提交 (华+吴)              :crit, m5_v45, 2026-05-10, 1d

    section M6 冲榜
    鲁棒性加固 (华)                  :m6_robust, 2026-05-11, 4d
    Prompt 精调 (华+吴)             :m6_prompt, 2026-05-11, 4d
    Context Folding 调参 (华+吴)    :m6_fold2, 2026-05-15, 4d
    云 16C64G dry run (华)          :m6_cloud, 2026-05-16, 1d
    并行度调优 (吴)                  :m6_parallel, 2026-05-15, 3d
    v6-v20 迭代提交                  :m6_iter, 2026-05-11, 8d
    冷冻期 prompt 微调 (华+吴)       :m6_freeze, 2026-05-19, 2d
    最终回归测 (华+吴)               :m6_regress, 2026-05-20, 2d
    最终提交 v25-v28                 :crit, m6_final, 2026-05-21, 2d
    Phase 1 截止                     :milestone, m6_deadline, 2026-05-23, 0d
```

## 13.2 两人并行视图

```mermaid
gantt
    title 华 vs 吴 — 并行任务分配
    dateFormat YYYY-MM-DD
    axisFormat %m/%d

    section 华
    环境+注册+API             :done, h1, 2026-04-17, 2d
    下载数据+Discord Q        :h2, 2026-04-18, 2d
    跑 baseline+数据探索      :h3, 2026-04-20, 2d
    结构化 Tools 改进          :h4, 2026-04-22, 2d
    Dockerfile+eval_local     :crit, h5, 2026-04-23, 1d
    OutputWriter+归一化        :h6, 2026-04-22, 1d
    v1 提交                    :crit, h7, 2026-04-24, 1d
    DataProfiler               :h8, 2026-04-25, 2d
    Scorer 完善                :h9, 2026-04-26, 2d
    Image tool 测试            :h10, 2026-05-03, 2d
    鲁棒性加固                 :h11, 2026-05-11, 4d
    云 dry run                 :h12, 2026-05-16, 1d
    回归+冷冻                  :h13, 2026-05-19, 4d

    section 吴
    环境+clone+读文档          :w1, 2026-04-17, 3d
    跑 baseline+metrics        :w2, 2026-04-20, 2d
    非结构化 Tools 原型        :w3, 2026-04-22, 2d
    PDF tool                   :w4, 2026-04-25, 2d
    RAG tool                   :w5, 2026-04-26, 2d
    Image/Chart tool           :w6, 2026-05-01, 3d
    并行度调优                 :w7, 2026-05-15, 3d

    section 华+吴 (一起)
    Agent 骨架+LLM Client      :crit, j1, 2026-04-22, 2d
    M2 集成测试                :crit, j2, 2026-04-23, 1d
    Agent 集成 Profiler+PDF    :j3, 2026-04-27, 2d
    Prompt 调优 (text-to-sql)  :j4, 2026-04-27, 2d
    v2 提交                    :crit, j5, 2026-04-29, 1d
    Plan-Execute 实现          :crit, j6, 2026-05-01, 3d
    Reflection                 :j7, 2026-05-04, 2d
    Context Folding            :j8, 2026-05-05, 2d
    v3 提交                    :crit, j9, 2026-05-07, 1d
    Prompt 实验 E1-E5          :j10, 2026-05-08, 2d
    v4/v5 提交                 :crit, j11, 2026-05-10, 1d
    Prompt 精调                :j12, 2026-05-11, 8d
    Context Folding 调参       :j13, 2026-05-15, 4d
    最终提交                   :crit, j14, 2026-05-21, 2d
    Phase 1 截止               :milestone, deadline, 2026-05-23, 0d
```

## 13.3 关键里程碑

```mermaid
gantt
    title 关键里程碑与提交节奏
    dateFormat YYYY-MM-DD
    axisFormat %m/%d

    section 里程碑
    M0 环境就绪        :milestone, 2026-04-19, 0d
    M1 Baseline 跑通   :milestone, 2026-04-22, 0d
    M2 Docker 化完成   :milestone, 2026-04-23, 0d
    Phase 1 开赛       :milestone, 2026-04-24, 0d
    M4 medium 覆盖     :milestone, 2026-04-30, 0d
    M5 hard 覆盖       :milestone, 2026-05-10, 0d
    Phase 1 截止       :milestone, 2026-05-23, 0d

    section 提交时间线 (≤30 次)
    v1 baseline 占位          :crit, 2026-04-24, 1d
    v2 medium 覆盖            :crit, 2026-04-29, 1d
    v3 hard+plan-exec         :crit, 2026-05-07, 1d
    v4-v5 实验验证             :crit, 2026-05-10, 1d
    v6-v20 迭代                :2026-05-11, 8d
    v21-v25 精调               :2026-05-19, 3d
    v26-v28 最终               :crit, 2026-05-21, 2d
```

## 13.4 注意事项

- `crit` 标记 = 关键路径上的任务，延迟会影响后续所有里程碑
- `done` = 已完成（注册 ✅）
- `active` = 进行中
- **华+吴 一起做的任务**集中在 Agent 核心循环、Plan-Execute、Reflection、Prompt 调优——这些需要两人对齐设计思路，不宜各写各的
- 甘特图的日期是**目标日期**，实际可能偏移 1-2 天；每周日复盘时校准
