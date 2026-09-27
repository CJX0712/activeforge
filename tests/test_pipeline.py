import numpy as np

from activeforge.data.synthetic import make_moons
from activeforge.pipeline.active_pipeline import benchmark, run_single


def test_run_single_basic():
    ds = make_moons(seed=42)
    res = run_single(ds, "random", seed=42, learner_kind="rf")
    assert 0.0 <= res.final_acc <= 1.0
    assert len(res.acc_curve) >= 2
    assert res.auc_lc >= 0.0


def test_determinism_hybrid():
    ds = make_moons(seed=42)
    r1 = run_single(ds, "hybrid", seed=42, learner_kind="rf")
    r2 = run_single(ds, "hybrid", seed=42, learner_kind="rf")
    assert np.allclose(r1.acc_curve, r2.acc_curve, atol=1e-12)
    assert r1.final_acc == r2.final_acc


def test_benchmark_small():
    from activeforge.data.synthetic import synthetic_zoo

    ds = {"moons": make_moons(seed=1), "gaussian2d": synthetic_zoo(1)["gaussian2d"]}
    rep = benchmark(ds, seeds=[42, 123], learner_kind="rf", budget_frac=0.3)
    assert rep["quality_grade"] in ("S", "A", "B", "C")
    assert len(rep["results"]) == len(ds) * 5
    # 至少有一个主动策略不劣于随机
    moons = [r for r in rep["results"] if r["dataset"] == "moons"]
    best = max(moons, key=lambda r: r["mean_acc"] if r["strategy"] != "random" else -1)
    assert best["mean_acc"] >= next(r["mean_acc"] for r in moons if r["strategy"] == "random") - 0.05
