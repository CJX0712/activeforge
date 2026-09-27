"""学习器封装：默认 sklearn，可选 XGBoost / LightGBM 强后端（自动探测）。

语义约定：predict_proba 行和为 1；对外标签保持整数（与输入一致）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def available_xgboost() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


def available_lightgbm() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:
        return False


class SklearnLearner:
    """包装任意 sklearn 兼容估计器，统一 fit/predict/predict_proba。"""

    def __init__(self, estimator=None, name: str = "sklearn"):
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=120, random_state=0, n_jobs=1)
        self.estimator = clone(estimator)
        self.name = name
        self._classes: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnLearner":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        # 标签编码仅在 backend 边界做；本系统标签已为整数，直接拟合
        self.estimator.fit(X, y)
        self._classes = np.asarray(sorted(np.unique(y)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(np.asarray(X, dtype=np.float64))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(np.asarray(X, dtype=np.float64)).astype(np.float64)

    @property
    def classes_(self) -> np.ndarray:
        return self._classes if self._classes is not None else np.array([])


def make_learner(kind: str = "rf") -> SklearnLearner:
    """按种类构造学习器。kind: rf / lr / xgb / lgbm。"""
    if kind == "rf":
        return SklearnLearner(
            RandomForestClassifier(n_estimators=60, random_state=0, n_jobs=1),
            name="RandomForest",
        )
    if kind == "lr":
        return SklearnLearner(
            LogisticRegression(max_iter=2000, random_state=0), name="LogisticRegression"
        )
    if kind == "xgb":
        if not available_xgboost():
            raise RuntimeError("xgboost 不可用")
        import xgboost as xgb  # type: ignore

        return SklearnLearner(
            xgb.XGBClassifier(n_estimators=120, random_state=0, verbosity=0,
                              tree_method="exact"),
            name="XGBoost",
        )
    if kind == "lgbm":
        if not available_lightgbm():
            raise RuntimeError("lightgbm 不可用")
        import lightgbm as lgb  # type: ignore

        return SklearnLearner(
            lgb.LGBMClassifier(n_estimators=120, random_state=0, verbose=-1,
                              n_jobs=1),
            name="LightGBM",
        )
    raise ValueError(f"未知 learner: {kind}")
