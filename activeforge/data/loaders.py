"""内置数据集载入（sklearn 自带，零下载离线可跑）。"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..core.types import Dataset


def _load(name: str, max_samples: int = 0, seed: int = 42) -> Dataset:
    from sklearn.datasets import (
        load_breast_cancer, load_digits, load_iris, load_wine,
    )

    if name == "digits":
        d = load_digits()
    elif name == "iris":
        d = load_iris()
    elif name == "wine":
        d = load_wine()
    elif name == "breast_cancer":
        d = load_breast_cancer()
    else:
        raise ValueError(f"未知内置数据集: {name}")
    X, y = d.data.astype(np.float64), d.target
    # 可选子采样（CPU 时间预算控制），分层保持类别比例
    if max_samples and max_samples < len(y):
        from sklearn.model_selection import train_test_split

        X, _, y, _ = train_test_split(X, y, train_size=max_samples,
                                      random_state=seed, stratify=y)
    # 内置数据集较小，分层切分避免池过薄
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    tag = name if not max_samples else f"{name}{max_samples}"
    return Dataset(X=X_tr, y=y_tr, X_test=X_te, y_test=y_te, name=tag)


def load_bundled(name: str, max_samples: int = 0) -> Dataset:
    return _load(name, max_samples=max_samples)


def bundled_zoo() -> Dict[str, Dataset]:
    out: Dict[str, Dataset] = {}
    for n in list_bundled():
        try:
            out[n] = _load(n)
        except Exception:
            continue
    return out


def list_bundled() -> List[str]:
    return ["digits", "iris", "wine", "breast_cancer"]
