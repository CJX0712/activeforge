# Changelog

All notable changes to ActiveForge are documented here. 作者：晨星。

## [0.1.0] - 2026-09-27

### Added
- 模块化主动学习系统，分层架构 `core / data / al / eval / pipeline`，单向无环依赖。
- 5 种查询策略：`random`（强基线）、`uncertainty`（entropy/margin/least_confident）、
  `committee`（QBC + BALD 近似）、`coreset`（k-center）、`hybrid`（批感知「不确定性 × 多样性」，自研创新点）。
- 5 个基准数据集：gaussian2d / moons / blobs20 / spiral / digits700（合成 + sklearn 内置，零下载）。
- 端到端 demo → `benchmark.json`（多 seed 基线 + 消融 + 失败案例 + 确定性校验 + 环境信息）。
- pytest 单测 12 项、CLI（`bench` / `selftest`）、Dockerfile、Makefile、GitHub Actions CI。
- 文档：`README.md`（五徽章）、`docs/architecture.md`、`docs/model_card.md`。

### Benchmark (3 seeds, budget=20%)
- 最佳主动策略 vs 随机：平均 Δacc **+0.0396**，5/5 数据集通过显著性门槛 → 质量等级 **S**。
- 端到端 47.6s（≤60s 预算），内存 < 2GB。
- 确定性：同 seed 两次运行 `benchmark.json` 除 `elapsed_sec` 外逐位一致。

### Fixed（本次实测踩坑）
- `np.trapz` 在 NumPy 2.x 已移除 → 改用 `np.trapezoid`（保留 <2.0 兼容）。
- 委员会 bootstrap 漏类导致 `predict_proba` 列数不一致 → 每成员保证见全类 + 对齐全局类别。
- BALD 广播错误：`_entropy` 对 `(M,U,C)` 误在 axis=1 求和 → 逐成员计算。
- `n_jobs=-1` 在受限环境触发 joblib 崩溃 → RF 固定 `n_jobs=1`。
- 消融/失败案例口径与主基准不一致 → 统一 `BUDGET=0.2 / BATCH=0.03`。
- 失败案例叙事与数字矛盾 → 全部改为从 `results` 真实派生。
