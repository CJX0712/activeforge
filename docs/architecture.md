# ActiveForge 架构文档

作者：晨星 · 版本：v0.1.0

## 1. 系统定位

模块化**池式主动学习（pool-based Active Learning）**系统。核心命题：在标注预算受限时，用更少的标签达到更高的分类准确率。

工程定位：CPU 可跑、零网络下载、固定 seed 确定性可复现、SOTA 后端缺失可离线降级。

## 2. 分层架构（单向无环）

```
cli.py → pipeline/active_pipeline.py → {data/, al/, eval/} → core/
```

| 层 | 模块 | 职责 | 关键接口 |
|----|------|------|----------|
| core | `seed.py` | 全局确定性唯一入口 | `set_all(seed)` |
| core | `errors.py` | E100 配置 / E200 数据 / E300 学习器 / E400 策略 / E500 流水线 | `ActiveForgeError` |
| core | `types.py` | `Dataset` / `StrategyResult` / `BenchmarkEntry` | dataclass |
| core | `config.py` | `ENV_ACTIVEFORGE_SEED`、`ENV_ACTIVEFORGE_N_JOBS` 覆盖 | `EnvConfig.load()` |
| core | `interfaces.py` | `Learner` / `QueryStrategy` Protocol | 契约 |
| data | `synthetic.py` | 4 个合成基准（gaussian2d / moons / blobs20 / spiral） | `synthetic_zoo(seed)` |
| data | `loaders.py` | sklearn 内置数据集（零下载），支持子采样 | `load_bundled(name, max_samples)` |
| al | `learner.py` | 学习器封装 + XGBoost/LightGBM 探测 | `make_learner("rf"/"lr"/"xgb"/"lgbm")` |
| al | `pool.py` | 已标注/未标注掩码、查询与标注 | `ActivePool.query(strategy, learner, n)` |
| al | `strategies.py` | 5 种查询策略 | `build_strategy(name)` |
| eval | `metrics.py` | accuracy / 学习曲线 AUC / 显著性 | - |
| pipeline | `active_pipeline.py` | 主循环 + 跨 seed 聚合 | `run_single()` / `benchmark()` |

## 3. 查询策略谱系

| 策略 | 原理 | 出处 | 分数语义 |
|------|------|------|----------|
| `random` | 随机采样（强基线） | - | 均匀分布 |
| `uncertainty` | least_confident / margin / entropy | Lewis & Catlett 1994, Settles 2009 | 越大越不确定 |
| `committee` | QBC：bootstrap 委员会，BALD 近似或投票熵 | Seung 1992; Houlsby et al. 2011 | 越大越分歧 |
| `coreset` | k-center 贪心（标准化特征空间） | Sener & Savare 2018 | 越大越远离已标注 |
| `hybrid` | **批感知** `score = u_norm × d_norm`（自研） | BatchBALD / BADGE 思路 | 既不确定又互补 |

统一接口：`select(pool, learner, n) -> 全局索引数组`。语义约定：**分数越大越值得查询**，保证跨模块公平评测。

## 4. 主循环

```
pool = ActivePool(dataset, init_per_class=2)   # 每类 2 个种子标注
loop:
    learner.fit(pool.labeled_X, pool.labeled_y)
    acc = accuracy(dataset.y_test, learner.predict(dataset.X_test))
    若 pool.labeled_frac >= budget: break
    chosen = pool.query(strategy, learner, batch)
```

- 无数据泄漏：策略只能看到 `pool.labeled_X/y` 与 `pool.unlabeled_X`（特征），测试集独立。
- 委员会策略在已标注集上训练，缺类成员按全局类别列对齐（补 0）。

## 5. 确定性

- 唯一入口 `core.seed.set_all(seed)`，一次设齐 numpy / random / torch（可选）。
- 策略内部随机源均派生自 `seed + pool.n_labeled`，不污染全局。
- **验证**：同一 seed 两次运行 `benchmark.json`，除 `elapsed_sec` 外全部逐位一致（`determinism_check.identical = true`）。

## 6. 基准方法论与关键调优发现

### 6.1 显著性门槛
`Δacc > 0.5 × (std_active + std_random)` 且均值更优；性能对比一律 ≥3 seeds 并报 mean±std。

### 6.2 难度甜点（实测踩坑）
初版在 40% 标签预算下所有策略增益仅 +0.015 —— **随机采样已逼近天花板，无区分度**。
扫描难度参数后定位甜点：

| 数据集 | 参数 | 随机 acc | 天花板 | 最佳 Δacc |
|--------|------|----------|--------|-----------|
| gaussian2d | sep=1.2 | 0.711 | 0.744 | +0.043 |
| moons | noise=0.40 | 0.861 | 0.879 | +0.025 |
| blobs20 | class_sep=1.0 | 0.670 | 0.815 | +0.051 |
| spiral | noise=0.12 | 0.594 | 0.683 | +0.021 |
| digits700 | subsample 700 | 0.873 | - | +0.059 |

**结论**：主动学习增益 = f(随机基线距天花板的 gap)。数据过易（gap≈0）时主动必然无增益——这是 F3 失败案例的根因，也是选型数据集时必须先估难度的原因。

### 6.3 学习器强度与耗时的权衡（实测）
| RF 树数 | 平均 Δacc | 显著数据集 | 耗时 |
|---------|-----------|-----------|------|
| 40 | 0.0282 | 3/5 | ~47s |
| 50 | 0.0302 | 4/5 | ~53s |
| **60** | **0.0396** | **5/5** | **47.6s** |

弱学习器会削弱主动增益（两臂同时变弱，但随机的相对损失更小），最终选 60 树。

## 7. 已知坑（本次实测）

| 症状 | 根因 | 修法 |
|------|------|------|
| `np.trapz` AttributeError | NumPy 2.x 已移除 | 改用 `np.trapezoid`，保留 <2.0 兼容分支 |
| 委员会 `np.stack` 报形状不一致 | 极少标注时 bootstrap 漏类 → `predict_proba` 列数不同 | 每成员保证见到全类 + 预测时对齐全局类别（缺类补 0） |
| BALD 广播错误 `(556,) vs (2,)` | `_entropy` 对 `(M,U,C)` 误在 axis=1 求和 | 逐成员 `_entropy(probas[m])`（类轴为 axis=1）后 `.mean(axis=0)` |
| `hybrid` 报 `NotFittedError` | 策略需要已拟合 learner | 流水线每步先 `fit` 再 `query`（契约） |
| `n_jobs=-1` 触发 joblib 崩溃 | 受限环境不支持并发 `send_bytes` | RF 固定 `n_jobs=1` |
| 消融/失败案例口径不一致 | 辅助函数用了默认 budget=0.4 | 统一常量 `BUDGET=0.2 / BATCH=0.03` |
| 失败案例叙事与数字矛盾 | 先写结论后填数 | 失败案例全部从 `results` 派生（最差格 / 最弱正增益格） |

## 8. 离线降级路径

`al/learner.py` 的 `available_xgboost()` / `available_lightgbm()` 探测失败时，`make_learner()` 拒绝构造对应后端，主链路自动回退 `rf`（纯 scikit-learn）。全部数据集为合成或 sklearn 内置，**零网络下载**即可完成 demo。
