# Changelog

所有发往组委会的提交版本记录。本地迭代（非提交）不记录此处。
格式参照 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)。

---

## [Unreleased]

### Added
- 项目开发计划与架构（12 份 docs，~3700 行）
- GitHub 私有仓库初始化

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
