from __future__ import annotations

import math

import numpy as np

from bskt_hll import BsktHllRing, JointStatistic, joint_from_statistic


def stream(rng, N, n_groups, size, pool=None):
    pool = pool or 5 * size
    return [(rng.integers(1, n_groups + 1, size=size).astype(np.uint64),
             rng.integers(1, pool, size=size).astype(np.uint64)) for _ in range(N + 1)]


def test_views_and_estimates() -> None:
    rng = np.random.default_rng(0)
    N, l, d = 4, 9, 3                                   # small pool, heavy sharing
    data = stream(rng, N, 30, 400)
    r = BsktHllRing(l=l, p=5, d=d, N=N, seed=7)
    ref = np.zeros((N + 1, l, r.m), dtype=np.uint8)
    for t, (g, x) in enumerate(data):
        r.process_period(g, x, t)
        arrays = r.arrays_of(g)
        j, rank = r.register_rank(x)
        for n in range(len(g)):
            for a in arrays[n]:
                ref[t, a, j[n]] = max(ref[t, a, j[n]], rank[n])
    groups = np.unique(data[-1][0])
    A = r.views(groups)
    arrays = r.arrays_of(groups)
    for i in range(N + 1):
        want = np.min(ref[N - i][arrays], axis=1)
        assert np.array_equal(A[i], want), i
    ks = [1, 2, 4]
    est = r.query(groups, ks)
    for gi in range(len(groups)):
        for ki, k in enumerate(ks):
            rh = np.max(A[1:k + 1, gi], axis=0)
            la, _, _, ok = joint_from_statistic(JointStatistic(A[0, gi].astype(np.int64), rh.astype(np.int64), r.q))
            want = la if ok else math.nan
            assert (math.isnan(est[ki, gi]) and math.isnan(want)) or math.isclose(est[ki, gi], want), (gi, k)
    print(f"shared pool (l={l}, d={d}): views and estimates of {len(groups)} groups match the reference  ok")


def test_accuracy_without_sharing() -> None:
    # two groups, a large pool: almost no sharing, so the estimate is plain HLL-Joint
    N, true = 2, 2000
    errs = []
    for seed in range(40):
        r = BsktHllRing(l=4096, p=7, d=2, N=N, seed=seed)
        old = np.arange(1, 6001, dtype=np.uint64)
        for t in range(N):
            r.process_period(np.ones(len(old), dtype=np.uint64), old, t)
        cur = np.concatenate([old[:3000], np.arange(10**6, 10**6 + true, dtype=np.uint64)])
        r.process_period(np.ones(len(cur), dtype=np.uint64), cur, N)
        errs.append(r.query(np.array([1], dtype=np.uint64), [2])[0, 0] / true - 1)
    errs = np.array(errs)
    assert abs(errs.mean()) < 0.15 and np.isfinite(errs).all(), errs
    print(f"no sharing, true 2000 new of 5000: mean rel. error {errs.mean():+.3f}, rel. stdev {errs.std():.3f}  ok")


def test_memory() -> None:
    r = BsktHllRing(l=100, p=7, d=4, N=12)
    assert r.memory_bits() == 13 * 100 * 128 * 5 + 2 * 128 * 5
    assert BsktHllRing.l_for_budget(2**20, 7, 12) == (8 * 2**20 - 1280) // (13 * 640)
    print("memory accounting  ok")


if __name__ == "__main__":
    test_views_and_estimates()
    test_accuracy_without_sharing()
    test_memory()
    print("all tests passed")
