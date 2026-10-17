"""SSVS (Melissourgos et al., PVLDB 2023) as the counter layer of shared TNDS.
Port of SSVS_2.java and VariableCounter.java from
https://github.com/DimitrisMel/SSVS.

Each 18-bit unit has a 2-bit state and 16 data bits:
  state 0  two signed 8-bit counters; reaching |v| = 128 merges them into
           state 1 with value v_other + v_new
  state 1  one signed 16-bit counter; reaching |v| = 32768 moves to state 2
           with mantissa 2048 and exponent 4
  state 2  value mant * 2^exp (12-bit mantissa, exp <= 7), each step applied
           with probability 2^-exp; |mant| = 4096 increments exp (mant 2048),
           or at exp 7 moves to state 3 (mant 512, exp 10)
  state 3  10-bit mantissa, 5-bit exponent; |mant| = 1024 gives mant 512, exp + 1
An update of weight delta is delta single steps. Unlike the Java code, the
sign of an active counter follows its value when the mantissa crosses zero.

A group has d positions from d seeded hashes h_i(g): unit (h >> 2) mod w,
sub-counter (h >> 1) & 1, sign +1 if h & 1 else -1. Each sample updates one
random position. We keep one sketch per history length k; a sample <g, k, w>
adds sign * round(w), rounded up with probability w - floor(w).

Queries:
  sum    sum of sign_i * value_i (unbiased)
  ssvs2  the SSVS-2 estimator: the closest pair of position values plus the
         values within noise/4, scaled by d/used, clipped at 0; the noise is
         estimated from 20,000 random counters
"""

from __future__ import annotations

import sys
from pathlib import Path

import numba
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "common" / "code"))

from hashing import U64, mix64, salt_from_seed  # noqa: E402

UNIT_BITS = 18


@numba.njit(cache=True)
def _step(state, v0, v1, ex, u, idx, delta):
    """One +-1 step on sub-counter idx of unit u."""
    s = state[u]
    if s == 0:
        if idx == 0:
            nv = v0[u] + delta
            other = v1[u]
        else:
            nv = v1[u] + delta
            other = v0[u]
        if nv >= 128 or nv <= -128:
            state[u] = 1
            v0[u] = other + nv
            v1[u] = 0
        elif idx == 0:
            v0[u] = nv
        else:
            v1[u] = nv
    elif s == 1:
        nv = v0[u] + delta
        if nv >= 32768 or nv <= -32768:
            state[u] = 2
            ex[u] = 4
            v0[u] = 2048 if nv > 0 else -2048
        else:
            v0[u] = nv
    elif s == 2:
        if np.random.random() * (1 << ex[u]) < 1.0:
            nv = v0[u] + delta
            if abs(nv) == 4096:
                if ex[u] == 7:
                    state[u] = 3
                    ex[u] = 10
                    v0[u] = 512 if nv > 0 else -512
                else:
                    ex[u] += 1
                    v0[u] = 2048 if nv > 0 else -2048
            else:
                v0[u] = nv
    else:
        if np.random.random() * (2.0 ** ex[u]) < 1.0:
            nv = v0[u] + delta
            if abs(nv) == 1024:
                ex[u] += 1
                v0[u] = 512 if nv > 0 else -512
            else:
                v0[u] = nv


@numba.njit(cache=True)
def _value(state, v0, v1, ex, u, idx):
    s = state[u]
    if s == 0:
        return float(v0[u] if idx == 0 else v1[u])
    if s == 1:
        return float(v0[u])
    return float(v0[u]) * (2.0 ** ex[u])


@numba.njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@numba.njit(cache=True)
def _add_samples(state, v0, v1, ex, units, idxs, signs, groups, weights):
    """units, idxs, signs: (G, d) per group; groups, weights: samples in order."""
    d = units.shape[1]
    for n in range(groups.shape[0]):
        g = groups[n]
        w = weights[n]
        steps = int(np.floor(w))
        if np.random.random() < w - steps:
            steps += 1
        r = np.random.randint(0, d)
        u, idx, sg = units[g, r], idxs[g, r], signs[g, r]
        for _ in range(steps):
            _step(state, v0, v1, ex, u, idx, sg)


@numba.njit(cache=True)
def _query(state, v0, v1, ex, units, idxs, signs, noise):
    G, d = units.shape
    out_sum = np.empty(G)
    out_s2 = np.empty(G)
    vals = np.empty(d)
    for g in range(G):
        tot = 0.0
        for r in range(d):
            vals[r] = signs[g, r] * _value(state, v0, v1, ex, units[g, r], idxs[g, r])
            tot += vals[r]
        out_sum[g] = tot
        # SSVS-2 estimate, integer arithmetic as in the original
        v = np.sort(np.floor(vals))
        if d > 2:
            mi, gap = 0, np.inf
            for i in range(d - 1):
                if v[i + 1] - v[i] < gap:
                    gap = v[i + 1] - v[i]
                    mi = i
            est = v[mi] + v[mi + 1]
            used = 2
            q = noise // 4
            for i in range(mi):
                if v[mi] - q < v[i] < v[mi] + q:
                    est += v[i]
                    used += 1
            for i in range(d - 1, mi + 1, -1):
                if v[mi + 1] - q < v[i] < v[mi + 1] + q:
                    est += v[i]
                    used += 1
            est = (est * d) // used
        elif d == 2:
            est = v[0] + v[1]
        else:
            est = v[0]
        out_s2[g] = max(est, 0.0)
    return out_sum, out_s2


@numba.njit(cache=True)
def _noise(state, v0, v1, ex, w, trials):
    """Mean of max(0, +-value) over random counters."""
    tot = 0
    for _ in range(trials):
        u = np.random.randint(0, w)
        sg = 1 if np.random.randint(0, 2) == 1 else -1
        e = int(sg * _value(state, v0, v1, ex, u, np.random.randint(0, 2)))
        tot += max(e, 0)
    return tot // trials


class Ssvs:
    """One sketch of w 18-bit units with d positions per group."""

    def __init__(self, w: int, n_groups_ids: np.ndarray, d: int = 4, seed: int = 0) -> None:
        self.w, self.d = w, d
        self.state = np.zeros(w, dtype=np.uint8)
        self.v0 = np.zeros(w, dtype=np.int64)
        self.v1 = np.zeros(w, dtype=np.int64)
        self.ex = np.zeros(w, dtype=np.int64)
        rng = np.random.default_rng(seed)
        row_salts = [salt_from_seed(int(s)) for s in rng.integers(0, 2**31, size=d)]
        ids = np.asarray(n_groups_ids, dtype=U64)
        h = np.stack([mix64(ids, s) for s in row_salts], axis=1)            # (G, d)
        self.units = ((h >> U64(2)) % U64(w)).astype(np.int64)
        self.idxs = ((h >> U64(1)) & U64(1)).astype(np.int64)
        self.signs = np.where((h & U64(1)) == U64(1), 1, -1).astype(np.int64)
        _seed(int(rng.integers(0, 2**31)))

    def add(self, groups: np.ndarray, weights: np.ndarray) -> None:
        _add_samples(self.state, self.v0, self.v1, self.ex, self.units, self.idxs, self.signs,
                     groups.astype(np.int64), weights.astype(np.float64))

    def query(self) -> tuple[np.ndarray, np.ndarray]:
        noise = _noise(self.state, self.v0, self.v1, self.ex, self.w, 20_000)
        return _query(self.state, self.v0, self.v1, self.ex, self.units, self.idxs, self.signs,
                      float(noise))

    @staticmethod
    def units_for_bits(bits: int) -> int:
        return max(bits // UNIT_BITS, 1)
