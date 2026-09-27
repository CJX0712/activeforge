import numpy as np

from activeforge.core.seed import get_seed, set_all
from activeforge.core.types import Dataset


def test_set_all_idempotent():
    a = set_all(123)
    b = set_all(123)
    assert a == b == 123
    assert get_seed() == 123


def test_dataset_validates_shape():
    X = np.random.randn(10, 3)
    y = np.arange(10)
    ds = Dataset(X=X, y=y, X_test=X[:3], y_test=y[:3])
    assert ds.n_samples == 10 and ds.n_features == 3
    # 行数不一致应报错
    try:
        Dataset(X=X, y=y[:9], X_test=X, y_test=y)
        raise AssertionError("应触发校验错误")
    except ValueError:
        pass


def test_unlabeled_global_excludes_labeled():
    from activeforge.al.pool import ActivePool

    ds = Dataset(X=np.random.randn(50, 2), y=np.array([0] * 25 + [1] * 25),
                 X_test=np.random.randn(10, 2), y_test=np.array([0] * 5 + [1] * 5))
    pool = ActivePool(ds, init_per_class=2, seed=1)
    assert pool.n_labeled == 4
    assert len(pool.unlabeled_global) == 46
