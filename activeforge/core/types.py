"""核心数据类型（dataclass），跨模块共享。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Dataset:
    """一个主动学习基准数据集：有标签初始由 Pool 管理，这里给出完整池 + 测试集。"""

    X: np.ndarray
    y: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    name: str = "dataset"
    feature_names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        self.y = np.asarray(self.y)
        self.X_test = np.asarray(self.X_test, dtype=np.float64)
        self.y_test = np.asarray(self.y_test)
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X/y 行数不一致")
        if self.X.shape[1] != self.X_test.shape[1]:
            raise ValueError("训练/测试特征维度不一致")
        self.classes_: np.ndarray = np.unique(self.y)
        self.n_classes_: int = int(len(self.classes_))

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


@dataclass
class QueryStep:
    """单步查询记录。"""

    iteration: int
    strategy: str
    queried_idx: List[int]
    n_labeled: int
    test_acc: float


@dataclass
class StrategyResult:
    """单个策略、单个 seed 的运行结果。"""

    strategy: str
    seed: int
    dataset: str
    n_labeled_curve: List[int] = field(default_factory=list)
    acc_curve: List[float] = field(default_factory=list)
    final_acc: float = 0.0
    auc_lc: float = 0.0  # 学习曲线下面积（标签效率，归一化到 [0,1]）

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "dataset": self.dataset,
            "n_labeled_curve": self.n_labeled_curve,
            "acc_curve": self.acc_curve,
            "final_acc": self.final_acc,
            "auc_lc": self.auc_lc,
        }


@dataclass
class BenchmarkEntry:
    """跨 seed 聚合后的对照条目（均值 ± 标准差）。"""

    dataset: str
    strategy: str
    mean_acc: float
    std_acc: float
    mean_auc: float
    std_auc: float
    n_seeds: int
    mean_final_labeled_frac: float = 0.0
    significant_vs_random: bool = False
    delta_acc_vs_random: float = 0.0
