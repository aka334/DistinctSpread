"""

m cells hold w-bit wrapped timestamps, w = ceil(log2(2N + 2)), M = 2^w. The
age of cell j at epoch t is (t - T[j]) mod M; the cell is stale for history
length k if its age is > k. Cells start at T = M - N - 1, i.e. stale for
every k at epoch 0. G[u] counts cells of age u (0 <= u <= N); for each k we
keep the stale count z_k, the validity flag v_k and the estimate C_k.

Item x at epoch t, j = h(x), a = age of j before the hit:
    for k in K:  if z_k = 0: v_k = 0
                 if a > k: (if v_k: C_k += m / z_k); z_k -= 1
    if a > 0: G[a] -= 1 (if a <= N); G[0] += 1; T[j] = t mod M
End of epoch t:
    z_k += G[k], v_k = 1
    every N + 1 epochs, cells of age >= N are reset to T = (t - N) mod M
    shift G by one bin

Only the first hit of a cell in an epoch can be admitted and each admission
lowers z_k by one, so an epoch is processed in one vectorized step: the n-th
admission for k has weight m / (z_k - n).
"""

from __future__ import annotations

import math

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common" / "code"))

from hashing import U64, mix64, salt_from_seed  # noqa: E402


class Tnds:
    def __init__(self, m: int, K: list[int], N: int | None = None, seed: int = 0x9E3779B9) -> None:
        self.K = sorted(K)
        self.N = max(self.K) if N is None else N
        if self.K[0] < 1 or self.K[-1] > self.N:
            raise ValueError("history lengths must lie in 1..N")
        self.m = m
        self.w = math.ceil(math.log2(2 * self.N + 2))
        self.M = 1 << self.w
        self.salt = salt_from_seed(seed)
        self.T = np.full(m, self.M - self.N - 1, dtype=np.int64)
        self.G = np.zeros(self.N + 1, dtype=np.int64)
        self.z = {k: m for k in self.K}
        self.v = {k: True for k in self.K}
        self.C = {k: 0.0 for k in self.K}
        self.t = 0

    def ages(self, cells: np.ndarray | None = None) -> np.ndarray:
        T = self.T if cells is None else self.T[cells]
        return (self.t - T) % self.M

    def _cells(self, elem: np.ndarray) -> np.ndarray:
        return (mix64(elem, self.salt) % U64(self.m)).astype(np.int64)

    def _advance_one(self) -> None:
        t = self.t
        for k in self.K:
            self.z[k] += int(self.G[k])
            self.v[k] = True
        if (t + 1) % (self.N + 1) == 0:                    # periodic reset of old cells
            old = self.ages() >= self.N
            self.T[old] = (t - self.N) % self.M
        self.G[1:] = self.G[:-1].copy()
        self.G[0] = 0
        for k in self.K:
            self.C[k] = 0.0
        self.t = t + 1

    def advance(self, t: int) -> None:
        if t < self.t:
            raise ValueError(f"epochs must be nondecreasing (at {self.t}, got {t})")
        while self.t < t:
            self._advance_one()

    def process_period(self, elem: np.ndarray, t: int) -> None:
        """Items of epoch t in arrival order. Can be called several times per epoch."""
        self.advance(t)
        if len(elem):
            self._process_cells(self._cells(elem))

    def _process_cells(self, j: np.ndarray) -> None:
        n_items = len(j)
        if not n_items:
            return
        _, first = np.unique(j, return_index=True)
        first = np.sort(first)                              # first hit of each cell
        cells = j[first]
        a = self.ages(cells)                                # ages before the hits

        for k in self.K:
            pos = first[a > k]                              # admitted items
            n_adm, z0 = len(pos), self.z[k]
            # invalid if an item arrives while z_k = 0
            if z0 == 0 or (n_adm == z0 and pos[-1] < n_items - 1):
                self.v[k] = False
            if n_adm:
                if self.v[k]:
                    self.C[k] += float((self.m / np.arange(z0, z0 - n_adm, -1, dtype=np.float64)).sum())
                self.z[k] = z0 - n_adm

        hit = a > 0
        tracked = a[hit] <= self.N
        np.subtract.at(self.G, a[hit][tracked], 1)
        self.G[0] += int(hit.sum())
        self.T[cells[hit]] = self.t % self.M

    def query(self, k: int) -> float | None:
        """Estimate for the current epoch, or None if unavailable."""
        return self.C[k] if self.v[k] else None

    def memory_bits(self, workspace: bool = True) -> int:
        """Timestamps, histogram, and per k a stale count, a flag and a 64-bit estimate."""
        cnt = math.ceil(math.log2(self.m + 1))
        return self.m * self.w + (self.N + 1) * cnt + len(self.K) * (cnt + 1 + 64)
