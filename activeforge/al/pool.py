"""主动学习池：维护已标注 / 未标注掩码，执行查询与标注。"""
from __future__ import annotations

from typing import List

import numpy as np

from ..core.seed import rng
from ..core.types import Dataset


class ActivePool:
    """池式主动学习：初始随机标注每类若干样本，后续按策略查询。"""

    def __init__(self, dataset: Dataset, init_per_class: int = 2, seed: int = 42):
        self.X = np.asarray(dataset.X, dtype=np.float64)
        self.y = np.asarray(dataset.y)
        self.n = int(self.X.shape[0])
        self.labeled_mask = np.zeros(self.n, dtype=bool)
        self.rng = rng(seed)
        init: List[int] = []
        for c in np.unique(self.y):
            idx = np.where(self.y == c)[0]
            k = min(init_per_class, len(idx))
            chosen = self.rng.choice(idx, size=k, replace=False)
            init.extend(chosen.tolist())
        self.labeled_mask[init] = True
        self.initial_labeled = len(init)

    @property
    def unlabeled_global(self) -> np.ndarray:
        return np.where(~self.labeled_mask)[0]

    @property
    def labeled_X(self) -> np.ndarray:
        return self.X[self.labeled_mask]

    @property
    def labeled_y(self) -> np.ndarray:
        return self.y[self.labeled_mask]

    @property
    def unlabeled_X(self) -> np.ndarray:
        return self.X[~self.labeled_mask]

    @property
    def n_labeled(self) -> int:
        return int(self.labeled_mask.sum())

    @property
    def labeled_frac(self) -> float:
        return self.n_labeled / self.n

    def query(self, strategy, learner, n: int) -> np.ndarray:
        """执行一次查询：返回被选中的全局索引并标记为已标注。"""
        ux = self.unlabeled_global
        if len(ux) == 0 or n <= 0:
            return np.array([], dtype=int)
        k = min(n, len(ux))
        chosen = np.asarray(strategy.select(self, learner, k), dtype=int)
        chosen = chosen[np.isin(chosen, ux)]  # 仅保留未标注子集
        if len(chosen) > k:
            chosen = chosen[:k]
        self.labeled_mask[chosen] = True
        return chosen
