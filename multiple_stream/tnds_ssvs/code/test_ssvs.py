"""Tests for ssvs.py: the counter port against the original Java
VariableCounter, and unbiasedness of the sum query on a synthetic grouped load.

    javac -d build java_harness/Harness.java VariableCounter.java
    python test_ssvs.py build

VariableCounter.java is from https://github.com/DimitrisMel/SSVS.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numba
import numpy as np

from ssvs import Ssvs, _seed, _step, _value


@numba.njit(cache=True)
def apply_ops(state, v0, v1, ex, units, idxs, deltas, reps):
    for n in range(units.shape[0]):
        for _ in range(reps[n]):
            _step(state, v0, v1, ex, units[n], idxs[n], deltas[n])


def run_java(jdir: Path, w: int, ops: np.ndarray) -> np.ndarray:
    text = "\n".join(" ".join(map(str, r)) for r in ops.tolist()) + "\n"
    out = subprocess.run(["java", "-cp", str(jdir), "Harness", str(w)], input=text,
                         capture_output=True, text=True, check=True).stdout
    return np.array([[int(x) for x in line.split()] for line in out.strip().splitlines()])


def run_port(w: int, ops: np.ndarray, seed: int = 0) -> np.ndarray:
    state = np.zeros(w, dtype=np.uint8)
    v0, v1, ex = (np.zeros(w, dtype=np.int64) for _ in range(3))
    _seed(seed)
    apply_ops(state, v0, v1, ex, ops[:, 0].copy(), ops[:, 1].copy(), ops[:, 2].copy(), ops[:, 3].copy())
    return np.array([[u, state[u], int(_value(state, v0, v1, ex, u, 0)), int(_value(state, v0, v1, ex, u, 1))]
                     for u in range(w)])


def test_deterministic_states(jdir: Path) -> None:
    rng = np.random.default_rng(0)
    w, n_ops = 400, 60_000
    units = rng.integers(0, w, n_ops)
    idxs = rng.integers(0, 2, n_ops)
    drift = np.where(units % 3 == 0, 0.8, 0.5)               # a third of the units climb past 127
    deltas = np.where(rng.random(n_ops) < drift, 1, -1)
    reps = rng.integers(1, 40, n_ops)
    ops = np.stack([units, idxs, deltas, reps], axis=1)
    java, port = run_java(jdir, w, ops), run_port(w, ops)
    assert java[:, 1].max() <= 1, "test meant for states 0/1 only"
    assert np.array_equal(java, port), np.flatnonzero((java != port).any(axis=1))[:5]
    merged = int((java[:, 1] == 1).sum())
    print(f"states 0/1: {w} units x {n_ops:,} ops identical to Java ({merged} units merged 8->16 bit, "
          f"values up to {np.abs(java[:, 2:]).max():,})  ok")


def test_active_counters(jdir: Path) -> None:
    # every unit gets the same ops; units are independent trials of the random states
    w = 300
    for target in (100_000, 3_000_000):
        ops = np.array([[u, 0, 1, target] for u in range(w)] + [[u, 1, -1, 50] for u in range(w)])
        exact = target - 50
        java = run_java(jdir, w, ops)
        port = run_port(w, ops, seed=1)
        for name, res in (("java", java), ("port", port)):
            vals = res[:, 2]
            se = vals.std(ddof=1) / math.sqrt(w)
            assert abs(vals.mean() - exact) < 4 * se + 1, (name, target, vals.mean(), exact, se)
        print(f"active counter, exact value {exact:,}: Java mean {java[:, 2].mean():,.0f}, "
              f"port mean {port[:, 2].mean():,.0f}, states {sorted(set(java[:, 1]))}/{sorted(set(port[:, 1]))}  ok")


def test_sketch_unbiased() -> None:
    # 2,000 groups, 0..60 weighted samples each; plain-sum query is unbiased per group
    rng = np.random.default_rng(5)
    G = 2000
    n_s = rng.integers(0, 60, G)
    groups = np.repeat(np.arange(G), n_s)
    weights = rng.uniform(1.0, 6.0, len(groups))
    true = np.bincount(groups, weights=weights, minlength=G)
    est = []
    for seed in range(60):
        sk = Ssvs(w=1500, n_groups_ids=np.arange(G) + 7, d=4, seed=seed)
        sk.add(groups[rng.permutation(len(groups))], weights)
        est.append(sk.query()[0])
    est = np.array(est)
    bias = est.mean(0) - true
    se = est.std(0, ddof=1) / math.sqrt(len(est))
    frac = np.mean(np.abs(bias) < 3 * se)
    assert frac > 0.97, frac
    print(f"SSVS-2 plain-sum query: per-group bias within 3 se for {100 * frac:.1f}% of {G} groups "
          f"(mean relative noise {np.mean(est.std(0)) / true.mean():.2f})  ok")


if __name__ == "__main__":
    jdir = Path(sys.argv[1])
    test_deterministic_states(jdir)
    test_active_counters(jdir)
    test_sketch_unbiased()
    print("all tests passed")
