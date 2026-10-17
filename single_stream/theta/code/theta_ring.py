"""Theta-Difference baseline for one stream: a ring of N + 1 KMV sketches with
capacity c.

The sketch of epoch u keeps the distinct normalized hashes eta(x)/2^64 below
its threshold theta_u, at most c of them; when a (c+1)-th would be kept,
theta_u drops to the largest one, which is removed. The result does not
depend on arrival order (the c smallest hashes, theta_u = the (c+1)-th
smallest, or 1 with at most c labels), so an epoch is recorded in one step.

Query k at epoch t:
    history: theta_H = min of the thresholds of epochs t-1..t-k, union of
             their hashes below theta_H, cut to the c smallest
    estimate: theta = min(theta_t, theta_H),
              |{y in E_t : y < theta, y not in E_H}| / theta

Hashes are 64-bit integers; a threshold of None means theta = 1.
"""

from __future__ import annotations

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common" / "code"))

from hashing import U64, mix64, salt_from_seed  # noqa: E402


TWO64 = float(2**64)


def _cap(hashes: np.ndarray, c: int) -> tuple[np.ndarray, int | None]:
    """Sorted distinct hashes to (at most c smallest, threshold or None)."""
    if len(hashes) <= c:
        return hashes, None
    return hashes[:c], int(hashes[c])


class ThetaRing:
    def __init__(self, c: int, N: int, seed: int = 0x9E3779B9) -> None:
        if c < 2:
            raise ValueError("capacity c must be >= 2")
        self.c, self.N = c, N
        self.salt = salt_from_seed(seed)
        self.E: list[np.ndarray] = [np.empty(0, dtype=U64) for _ in range(N + 1)]
        self.theta: list[int | None] = [None] * (N + 1)
        self.t: int | None = None

    def _slot(self, u: int) -> int:
        return u % (self.N + 1)

    def advance(self, t: int) -> None:
        if self.t is not None and t < self.t:
            raise ValueError("epochs must be nondecreasing")
        if self.t == t:
            return
        start = t if self.t is None else self.t + 1
        for u in range(max(start, t - self.N), t + 1):
            s = self._slot(u)
            self.E[s], self.theta[s] = np.empty(0, dtype=U64), None
        self.t = t

    def process_period(self, elem: np.ndarray, t: int) -> None:
        self.advance(t)
        if not len(elem):
            return
        s = self._slot(t)
        new = np.unique(mix64(elem, self.salt))
        if self.theta[s] is not None:
            new = new[new < U64(self.theta[s])]
        merged = np.union1d(self.E[s], new)
        self.E[s], th = _cap(merged, self.c)
        if th is not None:                          # thresholds only decrease
            self.theta[s] = th if self.theta[s] is None else min(th, self.theta[s])

    def _hist_union(self, k: int) -> tuple[np.ndarray, int | None]:
        if not 1 <= k <= self.N:
            raise ValueError(f"k must be in 1..N={self.N}")
        slots = [self._slot(self.t - i) for i in range(1, k + 1)]
        ths = [self.theta[s] for s in slots if self.theta[s] is not None]
        th_h = min(ths) if ths else None
        parts = [self.E[s] if th_h is None else self.E[s][self.E[s] < U64(th_h)] for s in slots]
        union = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=U64)
        e_h, th_cap = _cap(union, self.c)
        if th_cap is not None:
            th_h = th_cap if th_h is None else min(th_h, th_cap)
        return e_h, th_h

    def query(self, k: int) -> float:
        e_h, th_h = self._hist_union(k)
        s = self._slot(self.t)
        cands = [v for v in (self.theta[s], th_h) if v is not None]
        th = min(cands) if cands else None
        e_t = self.E[s] if th is None else self.E[s][self.E[s] < U64(th)]
        novel = np.setdiff1d(e_t, e_h, assume_unique=True)
        theta = 1.0 if th is None else th / TWO64
        return len(novel) / theta

    def memory_bits(self, workspace: bool = True) -> int:
        """N + 1 sketches of c hashes and a threshold, plus the history union."""
        stored = (self.N + 1) * (self.c * 64 + 64)
        return stored + ((self.c * 64 + 64) if workspace else 0)
