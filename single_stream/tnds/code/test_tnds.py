"""Tests for tnds.py."""

from __future__ import annotations

import math

import sys
from pathlib import Path

import numpy as np

from tnds import Tnds

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common" / "code"))

from hashing import U64  # noqa: E402


def test_example() -> None:
    # worked example: t = 6, k = 2, m = 8, M = 16
    tn = Tnds(m=8, K=[2], N=7)
    assert tn.M == 16
    tn.t = 6
    tn.T = np.array([1, 4, 3, 6, 5, 0, 6, 2], dtype=np.int64)
    ages = tn.ages()
    assert ages.tolist() == [5, 2, 3, 0, 1, 6, 0, 4]
    tn.z[2] = int((ages > 2).sum())
    tn.G = np.bincount(ages[ages <= tn.N], minlength=tn.N + 1)
    assert tn.z[2] == 4
    tn._process_cells(np.array([2, 2, 5, 1]))          # a, a (repeat), b, c
    assert math.isclose(tn.query(2), 8 / 4 + 8 / 3), tn.query(2)
    assert tn.z[2] == 2 and tn.ages(np.array([2, 5, 1])).tolist() == [0, 0, 0]
    print(f"example: C_2 = {tn.query(2):.4f} (8/4 + 8/3)  ok")


def reference(stream: list[np.ndarray], m: int, K: list[int], cells_of) -> list[dict]:
    """Algorithm 2 without wrap-around, z_k recounted before every item."""
    T = np.full(m, -10**9, dtype=np.int64)
    out = []
    for t, items in enumerate(stream):
        C = {k: 0.0 for k in K}
        v = {k: True for k in K}
        for j in cells_of(items).tolist():
            a = t - T[j]
            for k in K:
                z = int((t - T > k).sum())
                if z == 0:
                    v[k] = False
                if a > k and v[k]:
                    C[k] += m / z
            T[j] = t
        out.append({k: (C[k] if v[k] else None) for k in K})
    return out


def test_matches_unwrapped_reference() -> None:
    rng = np.random.default_rng(1)
    for m, N, n_epochs, size in [(64, 3, 30, 25), (200, 5, 40, 60), (40, 2, 25, 50), (12, 2, 40, 30)]:
        K = list(range(1, N + 1))
        # small label pool, so epochs share many labels
        stream = [rng.integers(0, 3 * size, size=int(rng.integers(0, size))).astype(U64)
                  for _ in range(n_epochs)]
        tn = Tnds(m=m, K=K, N=N, seed=5)
        got = []
        for t, items in enumerate(stream):
            tn.process_period(items, t)
            got.append({k: tn.query(k) for k in K})
        want = reference(stream, m, K, tn._cells)
        for t, (g, w) in enumerate(zip(got, want)):
            for k in K:
                assert (g[k] is None) == (w[k] is None), (m, N, t, k, g[k], w[k])
                if g[k] is not None:
                    assert math.isclose(g[k], w[k], rel_tol=1e-9, abs_tol=1e-9), (m, N, t, k, g[k], w[k])
        n_unavail = sum(w[k] is None for w in want for k in K)
        print(f"m={m} N={N} (M={tn.M}): {n_epochs} epochs identical to the unwrapped reference "
              f"({n_unavail} unavailable estimates)  ok")


def test_unbiased() -> None:
    # Theorem 3: the estimate is unbiased while stale cells remain
    hist = [np.arange(i * 1000, i * 1000 + 1500, dtype=U64) for i in range(4)]
    cur = np.concatenate([np.arange(3000, 3600, dtype=U64),          # seen in epochs 2 and 3
                          np.arange(10**6, 10**6 + 800, dtype=U64)])   # new
    true = {1: 800, 3: 800}
    ests = {1: [], 3: []}
    for s in range(300):
        tn = Tnds(m=4096, K=[1, 3], N=4, seed=s)
        for t, h in enumerate(hist):
            tn.process_period(h, t)
        tn.process_period(np.concatenate([cur, cur]), 4)
        for k in ests:
            ests[k].append(tn.query(k))
    for k, e in ests.items():
        bias = np.mean(e) / true[k] - 1
        se = np.std(e) / true[k] / math.sqrt(len(e))
        assert abs(bias) < 4 * se, (k, bias, se)
        print(f"k={k}: true {true[k]}, mean {np.mean(e):.1f} (bias {100 * bias:+.2f}%, se {100 * se:.2f}%)  ok")


def test_unavailable_when_saturated() -> None:
    tn = Tnds(m=16, K=[1], N=2, seed=0)
    tn.process_period(np.arange(1000, dtype=U64), 0)
    assert tn.query(1) is None
    print("saturated epoch reports unavailable  ok")


if __name__ == "__main__":
    test_example()
    test_matches_unwrapped_reference()
    test_unbiased()
    test_unavailable_when_saturated()
    print("all tests passed")
