"""Tests for theta_ring.py."""

from __future__ import annotations

import math

import sys
from pathlib import Path

import numpy as np

from theta_ring import TWO64, ThetaRing

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common" / "code"))

from hashing import U64  # noqa: E402


def h(v: float) -> int:
    return int(v * TWO64)


def test_example() -> None:
    # theta_t = 0.6, E_t = {0.1, 0.3, 0.5}; theta_H = 0.4, E_H = {0.1, 0.2, 0.35}:
    # theta = 0.4, only 0.3 remains, estimate 1/0.4 = 2.5
    r = ThetaRing(c=3, N=1)
    r.advance(1)
    r.E[r._slot(1)] = np.array([h(0.1), h(0.3), h(0.5)], dtype=U64)
    r.theta[r._slot(1)] = h(0.6)
    r.E[r._slot(0)] = np.array([h(0.1), h(0.2), h(0.35)], dtype=U64)
    r.theta[r._slot(0)] = h(0.4)
    est = r.query(1)
    assert math.isclose(est, 2.5, rel_tol=1e-9), est
    print(f"example: estimate {est:.3f} (expected 2.5)  ok")


def test_exact_mode_and_order_independence() -> None:
    r = ThetaRing(c=1000, N=2)
    r.process_period(np.arange(1, 301, dtype=U64), 0)
    r.process_period(np.arange(200, 501, dtype=U64), 1)
    r.process_period(np.arange(450, 701, dtype=U64), 2)       # novel vs {1..500}: 501..700
    assert r.query(2) == 200.0, r.query(2)
    assert r.query(1) == 700 - 500, r.query(1)                 # vs {200..500}: 501..700
    # pieces in shuffled order give the same sketch as one batch
    a, b = ThetaRing(c=50, N=1, seed=3), ThetaRing(c=50, N=1, seed=3)
    items = np.random.default_rng(0).permutation(np.arange(1, 5001, dtype=U64))
    a.process_period(items, 0)
    for chunk in np.array_split(items[::-1], 37):
        b.process_period(chunk, 0)
    assert np.array_equal(a.E[0], b.E[0]) and a.theta[0] == b.theta[0]
    print("exact mode is exact; chunked/reordered recording gives the same sketch  ok")


def test_accuracy() -> None:
    hist = np.arange(1, 40_001, dtype=U64)
    novel = np.arange(10**6, 10**6 + 4000, dtype=U64)
    ests = []
    for s in range(200):
        r = ThetaRing(c=1024, N=2, seed=s)
        r.process_period(hist[:20_000], 0)
        r.process_period(hist[20_000:], 1)
        r.process_period(np.concatenate([novel, hist[::3]]), 2)
        ests.append(r.query(2))
    bias = np.mean(ests) / 4000 - 1
    se = np.std(ests) / 4000 / math.sqrt(len(ests))
    assert abs(bias) < 4 * se + 0.01, (bias, se)
    print(f"Theta-Difference, true 4000: mean {np.mean(ests):.0f} (bias {100 * bias:+.2f}%, "
          f"rel. stdev {np.std(ests) / 4000:.3f})  ok")


if __name__ == "__main__":
    test_example()
    test_exact_mode_and_order_independence()
    test_accuracy()
    print("all tests passed")
