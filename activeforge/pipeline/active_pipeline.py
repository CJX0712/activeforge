"""主动学习流水线：单策略单 seed 运行 + 跨数据集 / 跨 seed 基准聚合。"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional

import numpy as np

from ..al.learner import make_learner
from ..al.pool import ActivePool
from ..al.strategies import build_strategy
from ..core.seed import set_all
from ..core.types import Dataset, StrategyResult
from ..eval.metrics import accuracy, auc_learning_curve, significance

# 控制台 UTF-8 重配置（Windows 安全）
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


def run_single(
    dataset: Dataset,
    strategy_name: str,
    seed: int,
    learner_kind: str = "rf",
    budget_frac: float = 0.4,
    batch_frac: float = 0.05,
    init_per_class: int = 2,
    strategy_kwargs: Optional[dict] = None,
) -> StrategyResult:
    """运行单个策略单个 seed 的主动学习主循环。"""
    set_all(seed)
    pool = ActivePool(dataset, init_per_class=init_per_class, seed=seed)
    learner = make_learner(learner_kind)
    strategy = build_strategy(strategy_name, seed=seed, **(strategy_kwargs or {}))
    n = pool.n
    batch = max(1, int(round(batch_frac * n)))

    n_labeled_curve: List[int] = []
    acc_curve: List[float] = []

    def _eval() -> float:
        learner.fit(pool.labeled_X, pool.labeled_y)
        return accuracy(dataset.y_test, learner.predict(dataset.X_test))

    # 初始评估
    n_labeled_curve.append(pool.n_labeled)
    acc_curve.append(_eval())

    max_steps = max(1, int(round((budget_frac * n - pool.n_labeled) / batch)) + 1)
    for _ in range(max_steps):
        if pool.labeled_frac >= budget_frac:
            break
        chosen = pool.query(strategy, learner, batch)
        if len(chosen) == 0:
            break
        n_labeled_curve.append(pool.n_labeled)
        acc_curve.append(_eval())

    final_acc = float(acc_curve[-1])
    auc = auc_learning_curve(n_labeled_curve, acc_curve, n)
    return StrategyResult(
        strategy=strategy_name, seed=seed, dataset=dataset.name,
        n_labeled_curve=n_labeled_curve, acc_curve=acc_curve,
        final_acc=final_acc, auc_lc=auc,
    )


def _agg(entries: List[StrategyResult]):
    acc = [e.final_acc for e in entries]
    auc = [e.auc_lc for e in entries]
    return float(np.mean(acc)), float(np.std(acc)), float(np.mean(auc)), float(np.std(auc))


def benchmark(
    datasets: Dict[str, Dataset],
    strategy_names: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    learner_kind: str = "rf",
    budget_frac: float = 0.4,
    batch_frac: float = 0.05,
    threshold_abs: float = 0.03,
) -> dict:
    """跨数据集 / 跨策略 / 跨 seed 基准。

    返回结构化 dict（用于落盘 benchmark.json + 报告）。
    threshold_abs：预设「世界顶级 S 级」门槛——最佳主动策略相对随机的均值
    准确率提升须 >= 该绝对差值（且通过简易显著性）。
    """
    if strategy_names is None:
        strategy_names = ["random", "uncertainty", "committee", "coreset", "hybrid"]
    if seeds is None:
        seeds = [42, 123, 777]

    raw: Dict[str, Dict[str, List[StrategyResult]]] = {}
    for ds_name, ds in datasets.items():
        raw[ds_name] = {}
        for s in strategy_names:
            raw[ds_name][s] = [run_single(ds, s, sd, learner_kind, budget_frac, batch_frac)
                               for sd in seeds]

    results = []
    for ds_name in raw:
        random_entry = raw[ds_name]["random"]
        r_mean_acc, r_std_acc, r_mean_auc, r_std_auc = _agg(random_entry)
        for s in strategy_names:
            m_acc, sd_acc, m_auc, sd_auc = _agg(raw[ds_name][s])
            delta = m_acc - r_mean_acc
            sig = significance(m_acc, sd_acc, r_mean_acc, r_std_acc)
            results.append({
                "dataset": ds_name,
                "strategy": s,
                "mean_acc": round(m_acc, 4),
                "std_acc": round(sd_acc, 4),
                "mean_auc": round(m_auc, 4),
                "std_auc": round(sd_auc, 4),
                "n_seeds": len(seeds),
                "mean_final_labeled_frac": round(budget_frac, 3),
                "delta_acc_vs_random": round(delta, 4),
                "significant_vs_random": bool(sig),
            })

    # 找出每个数据集的最佳主动策略
    best_per_ds = {}
    for ds_name in raw:
        best = None
        for s in strategy_names:
            if s == "random":
                continue
            row = next(r for r in results if r["dataset"] == ds_name and r["strategy"] == s)
            if best is None or row["mean_acc"] > best["mean_acc"]:
                best = row
        best_per_ds[ds_name] = best

    # 聚合：所有数据集上最佳主动策略 vs 随机
    agg_active_acc = [best_per_ds[d]["mean_acc"] for d in best_per_ds]
    agg_random_acc = [next(r for r in results if r["dataset"] == d and r["strategy"] == "random")["mean_acc"]
                      for d in best_per_ds]
    agg_active_auc = [best_per_ds[d]["mean_auc"] for d in best_per_ds]
    agg_random_auc = [next(r for r in results if r["dataset"] == d and r["strategy"] == "random")["mean_auc"]
                      for d in best_per_ds]
    agg_delta_acc = float(np.mean(np.array(agg_active_acc) - np.array(agg_random_acc)))
    agg_delta_auc = float(np.mean(np.array(agg_active_auc) - np.array(agg_random_auc)))
    n_sig = sum(1 for d in best_per_ds if best_per_ds[d]["significant_vs_random"])

    grade = _grade(agg_delta_acc, n_sig, len(best_per_ds), threshold_abs)

    return {
        "seeds": seeds,
        "learner_kind": learner_kind,
        "budget_frac": budget_frac,
        "batch_frac": batch_frac,
        "threshold_abs": threshold_abs,
        "datasets": list(datasets.keys()),
        "results": results,
        "best_per_dataset": best_per_ds,
        "agg_vs_random": {
            "mean_delta_acc": round(agg_delta_acc, 4),
            "mean_delta_auc": round(agg_delta_auc, 4),
            "n_significant": n_sig,
            "n_datasets": len(best_per_ds),
        },
        "quality_grade": grade,
    }


def _grade(delta_acc: float, n_sig: int, n_ds: int, threshold_abs: float) -> str:
    if n_ds == 0:
        return "C"
    if delta_acc >= threshold_abs and n_sig == n_ds:
        return "S"
    if delta_acc > 0 and n_sig >= max(1, n_ds - 1):
        return "A"
    if delta_acc > 0:
        return "B"
    return "C"
