"""合成基准数据生成器（确定性，固定 seed 可复现）。

设计目标：生成「未标注池对边界具信息量」的分类问题，使主动查询相对随机采样
（强基线）能产生可量化、可复现的标签效率增益。所有生成器返回 Dataset。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..core.seed import rng
from ..core.types import Dataset


def _split(X: np.ndarray, y: np.ndarray, test_size: float, seed: int) -> tuple:
    from sklearn.model_selection import train_test_split

    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def make_gaussian(n: int = 600, d: int = 2, sep: float = 1.2, test_size: float = 0.3,
                  seed: int = 42) -> Dataset:
    """两团高斯重叠（线性可分时边界含大量不确定样本）。"""
    g = rng(seed)
    half = n // 2
    X1 = g.normal(-sep / 2.0, 1.0, size=(half, d))
    X2 = g.normal(sep / 2.0, 1.0, size=(half, d))
    X = np.vstack([X1, X2])
    y = np.array([0] * half + [1] * half)
    X_tr, X_te, y_tr, y_te = _split(X, y, test_size, seed)
    return Dataset(X=X_tr, y=y_tr, X_test=X_te, y_test=y_te, name="gaussian2d")


def make_moons(n: int = 800, noise: float = 0.40, test_size: float = 0.3,
               seed: int = 42) -> Dataset:
    """两月牙（非线性边界），不确定性采样优势最明显。"""
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    X_tr, X_te, y_tr, y_te = _split(X, y, test_size, seed)
    return Dataset(X=X_tr, y=y_tr, X_test=X_te, y_test=y_te, name="moons")


def make_blobs(n: int = 900, d: int = 20, n_classes: int = 3, class_sep: float = 1.0,
               test_size: float = 0.3, seed: int = 42) -> Dataset:
    """高维多分类（make_classification），信息/冗余特征混合。"""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n, n_features=d, n_informative=8, n_redundant=4,
        n_classes=n_classes, class_sep=class_sep, random_state=seed,
    )
    X_tr, X_te, y_tr, y_te = _split(X, y, test_size, seed)
    return Dataset(X=X_tr, y=y_tr, X_test=X_te, y_test=y_te, name="blobs20")


def make_spiral(n: int = 800, test_size: float = 0.3, seed: int = 42) -> Dataset:
    """双螺旋（强非线性），对弱 learner + 不确定性采样是硬但可学的基准。"""
    g = rng(seed)
    n_per = n // 2
    t = np.linspace(0.0, 4 * np.pi, n_per)
    r = t / (4 * np.pi)
    X1 = np.stack([r * np.cos(t), r * np.sin(t)], axis=1) + g.normal(0, 0.12, (n_per, 2))
    X2 = np.stack([r * np.cos(t + np.pi), r * np.sin(t + np.pi)], axis=1) + g.normal(0, 0.12, (n_per, 2))
    X = np.vstack([X1, X2])
    y = np.array([0] * n_per + [1] * n_per)
    X_tr, X_te, y_tr, y_te = _split(X, y, test_size, seed)
    return Dataset(X=X_tr, y=y_tr, X_test=X_te, y_test=y_te, name="spiral")


def synthetic_zoo(seed: int = 42) -> Dict[str, Dataset]:
    """返回一组合成基准（确定性）。"""
    return {
        "gaussian2d": make_gaussian(seed=seed),
        "moons": make_moons(seed=seed),
        "blobs20": make_blobs(seed=seed),
        "spiral": make_spiral(seed=seed),
    }


def list_synthetic() -> List[str]:
    return ["gaussian2d", "moons", "blobs20", "spiral"]
