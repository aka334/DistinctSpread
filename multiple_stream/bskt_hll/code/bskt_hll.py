"""bSkt(HLL-Joint): the HLL-Joint estimator over a bSkt-style shared pool of
HLL arrays.

Each epoch has a pool of l HLL arrays of m = 2^p registers (5 bits each, a
32-bit hash: register = top p bits, rank from the next q = 32 - p bits); a
ring keeps the pools of the current and the previous N epochs. Group g maps
to d arrays h_1(g)..h_d(g), and item <g, x> updates its register in all of
them.

Query (g, k) at epoch t:
    A_u = register-wise min over pool_u[h_i(g)], i = 1..d
    R_t = A_t,  R_H = max(A_{t-1}, ..., A_{t-k})
    estimate lambda_a of the HLL-Joint fit on (R_t, R_H); NaN if the fit fails

Memory: (N + 1) l m 5 bits plus two m-register query buffers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "common" / "code"))
sys.path.insert(0, str(ROOT / "single_stream" / "hll" / "code"))

from hashing import U64, mix64, salt_from_seed  # noqa: E402
from hll_ring import JointStatistic, bit_length_u64, joint_from_statistic  # noqa: E402

REGISTER_BITS = 5


class BsktHllRing:
    def __init__(self, l: int, p: int = 7, d: int = 4, N: int = 12, seed: int = 0x9E3779B9) -> None:
        if l < 1 or d < 1:
            raise ValueError("need l >= 1 and d >= 1")
        self.l, self.p, self.d, self.N = l, p, d, N
        self.m, self.q = 1 << p, 32 - p
        self.item_salt = salt_from_seed(seed)
        rng = np.random.default_rng([seed, 2])
        self.row_salts = [salt_from_seed(int(s)) for s in rng.integers(0, 2**31, size=d)]
        self.pools = np.zeros((N + 1, l, self.m), dtype=np.uint8)
        self.t: int | None = None
        self.cache: dict[bytes, float] = {}

    def arrays_of(self, groups: np.ndarray) -> np.ndarray:
        g = np.asarray(groups, dtype=U64)
        return np.stack([(mix64(g, s) % U64(self.l)).astype(np.int64) for s in self.row_salts], axis=1)

    def register_rank(self, elems: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mix64(np.asarray(elems, dtype=U64), self.item_salt) >> U64(32)      # 32-bit hash
        j = (h >> U64(self.q)).astype(np.int64)
        w = h & U64((1 << self.q) - 1)
        rank = (self.q - bit_length_u64(w) + 1).astype(np.uint8)                # q + 1 if w == 0
        return j, rank

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
        arrays = self.arrays_of(groups)                                      # (n, d)
        j, rank = self.register_rank(elems)
        flat = (arrays * self.m + j[:, None]).ravel()
        np.maximum.at(self.pools[self._slot(t)].reshape(-1), flat, np.repeat(rank, self.d))

    def views(self, groups: np.ndarray) -> np.ndarray:
        """A_u for u = t, t-1, ..., t-N: shape (N + 1, len(groups), m)."""
        arrays = self.arrays_of(groups)
        out = np.zeros((self.N + 1, len(groups), self.m), dtype=np.uint8)
        for i in range(self.N + 1):
            u = self.t - i
            if u < 0:
                continue
            pool = self.pools[self._slot(u)]
            acc = pool[arrays[:, 0]].copy()
            for r in range(1, self.d):
                np.minimum(acc, pool[arrays[:, r]], out=acc)
            out[i] = acc
        return out

    def joint(self, rt: np.ndarray, rh: np.ndarray) -> float:
        st = JointStatistic(rt.astype(np.int64), rh.astype(np.int64), self.q)
        key = st.key()
        if key not in self.cache:
            la, _, _, ok = joint_from_statistic(st)
            self.cache[key] = la if ok else np.nan
        return self.cache[key]

    def query(self, groups: np.ndarray, ks: list[int]) -> np.ndarray:
        """Estimates of shape (len(ks), len(groups))."""
        if any(not 1 <= k <= self.N for k in ks):
            raise ValueError(f"k must be in 1..N={self.N}")
        A = self.views(groups)
        out = np.full((len(ks), len(groups)), np.nan)
        for gi in range(len(groups)):
            rh = np.zeros(self.m, dtype=np.uint8)
            for k in range(1, max(ks) + 1):
                np.maximum(rh, A[k, gi], out=rh)
                if k in ks:
                    out[ks.index(k), gi] = self.joint(A[0, gi], rh)
        return out

    def memory_bits(self, workspace: bool = True) -> int:
        return (self.N + 1) * self.l * self.m * REGISTER_BITS + (2 * self.m * REGISTER_BITS if workspace else 0)

    @staticmethod
    def l_for_budget(nbytes: int, p: int, N: int) -> int:
        m = 1 << p
        return (8 * nbytes - 2 * m * REGISTER_BITS) // ((N + 1) * m * REGISTER_BITS)
