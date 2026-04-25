# Changelog

所有发往组委会的提交版本记录。本地迭代（非提交）不记录此处。
格式参照 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)。

---

## [Unreleased]

### Added
- 项目开发计划与架构（12 份 docs，~3700 行）
- GitHub 私有仓库初始化

### Changed
- **配置加载迁移到 Pydantic v2**：`load_app_config` 现在用 `AppConfigModel` 校验 YAML，
  随后实例化原有的冻结 dataclass 返回。对外接口（返回类型 `AppConfig`、抛出的
  `ValueError` 文本）保持字节兼容；新增模块 `data_agent_baseline.config_models`
  作为内部实现细节。
- `AgentConfig.tool_protocol` 从 `str` 收紧为 `Literal["native", "text"]`，
  与 runner / `ReActAgent` 端到端一致；删除 `runner.py` 上唯一的
  `# type: ignore[arg-type]`。

### Behavior changes (intentional tightening)
- **严格 bool/int 校验**：YAML 里**带引号**的布尔/整数（`"false"`, `"0"`, `"16384"`）
  从此被拒收。Pydantic strict 模式杜绝
  `bool("false") == True` 这类静默反向陷阱。继续支持的写法是无引号的
  YAML 标量（`false`, `0`, `16384`）。

### Planning
- M0 环境/注册（截止 4/23）
- M1 Fork starter kit 并跑通
- M2 Docker 化
- M3 首次提交（目标 4/24 开赛当天）

---

## Version template（正式提交后用）

```markdown
## v<N> — YYYY-MM-DD (AoE)

### Git / Image
- Git tag: v<N>
- Commit: <sha>
- Image SHA256: <sha256>
- Archive: team<NNNN>_v<N>.tar.gz (<size>)

### Added / Changed / Fixed
- ...

### Local demo score (λ=0.1)
- Total: 0.XX (vs v<N-1>: 0.YY)
- By difficulty:
  - easy: 0.XX
  - medium: 0.XX
  - hard: 0.XX
  - extreme: 0.XX

### Leaderboard score
- TBD（待组委会 24h 内回报）

### Notes
- 本次变更的核心假设与验证结果
```
