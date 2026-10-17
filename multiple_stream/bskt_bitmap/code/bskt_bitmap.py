"""bSkt(HB): the HB estimator over a bSkt-style shared pool of bitmaps.

Each epoch has a pool of l bitmaps of b bits; a ring keeps the pools of the
current and the previous N epochs. Group g maps to d bitmaps h_1(g)..h_d(g),
and item <g, x> sets bit H(x) in all of them.

Query (g, k) at epoch t:
    A_u = AND_i pool_u[h_i(g)]
    B   = A_{t-1} | ... | A_{t-k},  F = A_t & ~B
    z = zeros of B, s = ones of F, estimate -b ln(1 - s/z)
    unavailable if z = 0 or s = z

Memory: (N + 1) l b bits plus two b-bit query buffers. Bitmaps are stored in
ceil(b/64) words; padding bits in the last word are never set.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "common" / "code"))

from hashing import U64, mix64, salt_from_seed  # noqa: E402


class BsktBitmapRing:
    def __init__(self, l: int, b: int, d: int, N: int, seed: int = 0x9E3779B9) -> None:
        if b < 1 or l < 1 or d < 1:
            raise ValueError("need l >= 1, d >= 1 and b >= 1")
        self.l, self.b, self.d, self.N = l, b, d, N
        self.words = -(-b // 64)
        self.bit_salt = salt_from_seed(seed)                     # H(x), as in the single-stream HB
        rng = np.random.default_rng([seed, 1])
        self.row_salts = [salt_from_seed(int(s)) for s in rng.integers(0, 2**31, size=d)]
        self.pools = np.zeros((N + 1, l, self.words), dtype=np.uint64)
        self.t: int | None = None

    def bitmaps_of(self, groups: np.ndarray) -> np.ndarray:
        g = np.asarray(groups, dtype=U64)
        return np.stack([(mix64(g, s) % U64(self.l)).astype(np.int64) for s in self.row_salts], axis=1)

    def bit_of(self, elems: np.ndarray) -> np.ndarray:
        return (mix64(np.asarray(elems, dtype=U64), self.bit_salt) % U64(self.b)).astype(np.int64)

    def _slot(self, u: int) -> int:
        return u % (self.N + 1)

    def advance(self, t: int) -> None:
        if self.t is not None and t < self.t:
            raise ValueError("epochs must be nondecreasing")
        if self.t == t:
            return
        start = t if self.t is None else self.t + 1
        for u in range(max(start, t - self.N), t + 1):
            self.pools[self._slot(u)] = 0
        self.t = t

    def process_period(self, groups: np.ndarray, elems: np.ndarray, t: int) -> None:
        """Record items <groups[i], elems[i]> of epoch t."""
        self.advance(t)
        if not len(groups):
            return
        maps = self.bitmaps_of(groups)                              # (n, d)
        bit = self.bit_of(elems)
        flat = (maps * (64 * self.words) + bit[:, None]).ravel()     # bit index in the pool
        flat = np.unique(flat)
        pool = self.pools[self._slot(t)].reshape(-1)
        np.bitwise_or.at(pool, flat // 64, U64(1) << (flat % 64).astype(U64))

    def query(self, groups: np.ndarray, ks: list[int], chunk: int = 2048, saturated: str = "nan"):
        """Estimates of shape (len(ks), len(groups)). Saturated groups (z = 0 or
        s = z) get NaN, or with saturated="max" the largest estimate the bitmap
        can give: b ln z if s = z, b ln b if z = 0. With "max" the saturated
        mask is returned as well."""
        if any(not 1 <= k <= self.N for k in ks):
            raise ValueError(f"k must be in 1..N={self.N}")
        maps_all = self.bitmaps_of(groups)
        out = np.full((len(ks), len(groups)), np.nan)
        sat = np.zeros((len(ks), len(groups)), dtype=bool)
        kmax = max(ks)
        for c0 in range(0, len(groups), chunk):
            maps = maps_all[c0:c0 + chunk]

            def anded(u: int) -> np.ndarray:
                if u < 0:
                    return np.zeros((len(maps), self.words), dtype=np.uint64)
                pool = self.pools[self._slot(u)]
                acc = pool[maps[:, 0]].copy()
                for i in range(1, self.d):
                    acc &= pool[maps[:, i]]
                return acc

            cur = anded(self.t)
            hist = np.zeros_like(cur)
            for k in range(1, kmax + 1):
                hist |= anded(self.t - k)
                if k in ks:
                    # signed popcounts, so that b - count cannot wrap
                    z = self.b - np.bitwise_count(hist).sum(axis=1, dtype=np.int64)
                    s = np.bitwise_count(cur & ~hist).sum(axis=1, dtype=np.int64)
                    ok = (z > 0) & (s < z)
                    est = np.full(len(maps), np.nan)
                    est[ok] = -self.b * np.log1p(-s[ok] / z[ok])
                    if saturated == "max":
                        est[~ok] = np.where(z[~ok] > 0, self.b * np.log(np.maximum(z[~ok], 1)), self.b * math.log(self.b))
                    out[ks.index(k), c0:c0 + len(maps)] = est
                    sat[ks.index(k), c0:c0 + len(maps)] = ~ok
        return (out, sat) if saturated == "max" else out

    def memory_bits(self, workspace: bool = True) -> int:
        return (self.N + 1) * self.l * self.b + (2 * self.b if workspace else 0)

    @staticmethod
    def l_for_budget(nbytes: int, b: int, N: int) -> int:
        return (8 * nbytes - 2 * b) // ((N + 1) * b)
