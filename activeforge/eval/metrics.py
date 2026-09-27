"""评测指标。

语义约定：准确率越大越好；auc_lc（标签效率）越大越好，归一化到 [0,1]，
表示「以更少标签更快逼近高准确率」的程度。
"""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn.metrics import accuracy_score as _sk_accuracy_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(_sk_accuracy_score(np.asarray(y_true), np.asarray(y_pred)))


def auc_learning_curve(n_labeled_curve: List[int], acc_curve: List[float],
                       n_total: int) -> float:
    """学习曲线下面积（标签效率），归一化到 [0,1]。

    横轴 = 标签占比 f = n_labeled / n_total；纵轴 = test acc。
    以 (f - f0) 宽度为权，面积除以 (1 - f0) 得到 0..1 的效率分。
    """
    if len(acc_curve) < 2 or n_total <= 0:
        return float(acc_curve[-1]) if acc_curve else 0.0
    f = np.array(n_labeled_curve, dtype=float) / n_total
    a = np.array(acc_curve, dtype=float)
    order = np.argsort(f)
    f, a = f[order], a[order]
    f0 = f[0]
    if hasattr(np, "trapezoid"):
        area = float(np.trapezoid(a, f))
    else:  # numpy < 2.0 兼容
        area = float(np.trapz(a, f))
    denom = max(1e-9, 1.0 - f0)
    return float(np.clip(area / denom, 0.0, 1.0))


def efficiency_gain(active_auc: float, random_auc: float) -> float:
    """主动策略相对随机的标签效率增益（绝对差）。"""
    return float(active_auc - random_auc)


def significance(mean_a: float, std_a: float, mean_b: float, std_b: float) -> bool:
    """简易显著性门槛：均值更优且差值 > 0.5*(std_a + std_b)。"""
    return mean_a > mean_b and (mean_a - mean_b) > 0.5 * (std_a + std_b)
