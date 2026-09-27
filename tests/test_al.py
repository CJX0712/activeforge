import numpy as np

from activeforge.al.learner import (
    available_lightgbm,
    available_xgboost,
    make_learner,
)
from activeforge.al.pool import ActivePool
from activeforge.al.strategies import (
    CoresetStrategy,
    HybridStrategy,
    RandomStrategy,
    UncertaintyStrategy,
    build_strategy,
)
from activeforge.core.types import Dataset


def _pool():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    ds = Dataset(X=X, y=y, X_test=X[:20], y_test=y[:20])
    return ActivePool(ds, init_per_class=2, seed=0)


def test_learner_proba_sums_to_one():
    pool = _pool()
    lr = make_learner("lr")
    lr.fit(pool.labeled_X, pool.labeled_y)
    p = lr.predict_proba(pool.unlabeled_X)
    assert p.shape[0] == pool.unlabeled_X.shape[0]
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)


def test_backends_available_flag():
    # 至少 sklearn 后端可用；xgb/lgbm 取决于安装
    assert isinstance(available_xgboost(), bool)
    assert isinstance(available_lightgbm(), bool)


def test_all_strategies_return_valid_indices():
    pool = _pool()
    lr = make_learner("lr")
    lr.fit(pool.labeled_X, pool.labeled_y)
    for name in ["random", "uncertainty", "committee", "coreset", "hybrid"]:
        strat = build_strategy(name, seed=0)
        idx = strat.select(pool, lr, 10)
        idx = np.asarray(idx, dtype=int)
        # 必须是未标注子集、不重复、数量 <= 请求
        assert len(idx) <= 10
        assert len(np.unique(idx)) == len(idx)
        assert set(idx.tolist()).issubset(set(pool.unlabeled_global.tolist()))


def test_uncertainty_entropy_scores_most_uncertain():
    pool = _pool()
    lr = make_learner("lr")
    lr.fit(pool.labeled_X, pool.labeled_y)
    u = UncertaintyStrategy(mode="entropy")
    idx = u.select(pool, lr, 5)
    assert len(idx) == 5


def test_committee_sizes():
    pool = _pool()
    for k in (3, 9):
        c = build_strategy("committee", seed=0, n_members=k)
        idx = c.select(pool, make_learner("lr"), 8)
        assert len(idx) == 8


def test_coreset_and_hybrid_dont_crash():
    pool = _pool()
    lr = make_learner("lr")
    lr.fit(pool.labeled_X, pool.labeled_y)  # 流水线中每步先 fit 再 query
    for s in (CoresetStrategy(seed=0), HybridStrategy(seed=0)):
        idx = s.select(pool, lr, 6)
        assert 1 <= len(idx) <= 6
