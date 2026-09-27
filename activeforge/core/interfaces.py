"""接口契约（Protocol）。

调用单向无环：cli -> pipeline -> {data, al, eval} -> core。
跨模块公平评测的语义约定：
  - Learner.predict_proba 返回 (n_samples, n_classes) 概率，行和为 1。
  - QueryStrategy.select 返回「全局索引」数组（pool 中未标注子集的子集）。
  - 分数语义：越大越「值得查询」（越不确定 / 越具多样性）。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Learner(Protocol):
    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Learner":
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...


@runtime_checkable
class QueryStrategy(Protocol):
    name: str

    def select(self, pool, learner, n: int) -> np.ndarray:
        """返回全局索引（未标注子集的子集），数量 <= n。"""
        ...
