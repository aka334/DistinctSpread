"""Tests for hll_ring.py."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from hll_ring import (HllRing, JointLogLikelihood, JointStatistic, U64, bit_length_u64,
                      joint_estimate, ml_estimate)


def registers(items: np.ndarray, p: int, salt: int = 12345) -> np.ndarray:
    ring = HllRing(p=p, N=1, seed=salt)
    ring.process_period(items, 0)
    return ring.ring[ring._slot(0)].copy()


def test_bit_length() -> None:
    v = np.array([0, 1, 2, 3, 2**52 - 1, 2**53 + 1, 2**63, 2**64 - 1], dtype=U64)
    assert bit_length_u64(v).tolist() == [int(x).bit_length() for x in v.tolist()]
    print("exact uint64 bit length  ok")


def test_ml_matches_root_of_eq42() -> None:
    p = 10
    q, m = 64 - p, 1 << p
    for n in [0, 1, 7, 100, 1000, 10_000, 200_000]:
        c = np.bincount(registers(np.arange(1, n + 1, dtype=U64), p), minlength=q + 2)
        est = ml_estimate(c, p, q)
        if n == 0:
            assert est == 0.0
            continue
        h = lambda x: 0.0 if x <= 0 else (1.0 if x > 700 else 1.0 - x / math.expm1(x))
        f = lambda x: (x * sum(c[k] / 2**k for k in range(q + 1))
                       + sum(c[k] * h(x / 2**k) for k in range(1, q + 1))
                       + c[q + 1] * h(x / 2**q) - (m - c[0]))
        root = brentq(f, 1e-12, 1e6) * m
        assert abs(est - root) <= 2e-3 * root + 1e-9, (n, est, root)
    print("Algorithm 8 port agrees with the root of Eq. 42  ok")


def test_ml_unbiased() -> None:
    p, n = 8, 3000
    ests = [ml_estimate(np.bincount(registers(np.arange(n, dtype=U64) + 1, p, s), minlength=64 - p + 2),
                        p, 64 - p) for s in range(300)]
    bias = np.mean(ests) / n - 1
    se = np.std(ests) / n / math.sqrt(len(ests))
    assert abs(bias) < 4 * se + 0.005, (bias, se)
    print(f"single-sketch ML: bias {100 * bias:+.2f}% (se {100 * se:.2f}%), "
          f"rel. stdev {np.std(ests) / n:.3f} vs 1.04/sqrt(m)={1.04 / math.sqrt(1 << p):.3f}  ok")


def test_joint_gradient() -> None:
    p = 8
    a = np.arange(1, 501, dtype=U64)
    b = np.arange(10_001, 10_301, dtype=U64)
    x = np.arange(20_001, 22_001, dtype=U64)
    k1, k2 = registers(np.concatenate([a, x]), p), registers(np.concatenate([b, x]), p)
    ll = JointLogLikelihood(JointStatistic(k1, k2, 64 - p))
    phi = np.log([400.0, 350.0, 1800.0])
    f0, g = ll.evaluate(phi)
    for i in range(3):
        d = np.zeros(3)
        d[i] = 1e-6
        num = (ll.evaluate(phi + d)[0] - ll.evaluate(phi - d)[0]) / 2e-6
        assert abs(num - g[i]) <= 1e-4 * max(1.0, abs(g[i])), (i, num, g[i])
    print("joint log-likelihood gradient matches finite differences  ok")


def test_joint_beats_inclusion_exclusion() -> None:
    # small novelty against a large shared history: the setting HLL-Joint targets
    p, (na, nb, nx), runs = 10, (300, 2000, 8000), 200
    q = 64 - p
    err_j, err_ie, fails = [], [], 0
    for s in range(runs):
        base = s * 10**7
        a = np.arange(base + 1, base + na + 1, dtype=U64)
        b = np.arange(base + 10**6, base + 10**6 + nb, dtype=U64)
        x = np.arange(base + 2 * 10**6, base + 2 * 10**6 + nx, dtype=U64)
        k1, k2 = registers(np.concatenate([a, x]), p, s), registers(np.concatenate([b, x]), p, s)
        la, _, _, ok = joint_estimate(k1, k2, q)
        st = JointStatistic(k1, k2, q)
        ie = ml_estimate(st.counts_max(), p, q) - ml_estimate(st.counts2(), p, q)
        if not ok:
            fails += 1
            continue
        err_j.append(la / na - 1)
        err_ie.append(ie / na - 1)
    rj, ri = math.sqrt(np.mean(np.square(err_j))), math.sqrt(np.mean(np.square(err_ie)))
    assert fails == 0 and rj < ri, (rj, ri, fails)
    print(f"|A| estimate rmse: joint {rj:.3f} vs inclusion-exclusion {ri:.3f} "
          f"(mean {np.mean(err_j):+.3f} vs {np.mean(err_ie):+.3f}), failed solves {fails}  ok")


def test_ring_novelty() -> None:
    ring = HllRing(p=12, N=3, seed=7)
    hist = np.arange(1, 20_001, dtype=U64)
    novel = np.arange(10**6, 10**6 + 3000, dtype=U64)
    ring.process_period(hist[:10_000], 0)
    ring.process_period(hist[10_000:], 1)
    ring.process_period(np.concatenate([novel, hist[5000:15_000], novel]), 2)
    clip, raw = ring.query_difference(2)
    joint = ring.query_joint(2)
    assert abs(joint - 3000) < 0.15 * 3000, joint
    print(f"ring, true novelty 3000: HLL-Difference {clip:.0f} (raw {raw:.0f}), HLL-Joint {joint:.0f}  ok")


if __name__ == "__main__":
    test_bit_length()
    test_ml_matches_root_of_eq42()
    test_ml_unbiased()
    test_joint_gradient()
    test_joint_beats_inclusion_exclusion()
    test_ring_novelty()
    print("all tests passed")
