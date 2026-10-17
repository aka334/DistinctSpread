"""Bitmap baseline HB for one stream (Algorithm 1).

A ring of N + 1 m-bit bitmaps, one per epoch; all epochs use the same hash
h(x) in [0, m).

    entering epoch t:  clear B_t
    item x:            B_t[h(x)] = 1
    query k:           B = B_{t-1} | ... | B_{t-k}
                       z = zeros of B, s = ones of B_t & ~B
                       unavailable if z = 0 or s = z, else -m ln(1 - s/z)


"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common" / "code"))

from hashing import U64, mix64, salt_from_seed  # noqa: E402


class EpochBitmapRing:
    def __init__(self, m: int, N: int, seed: int = 0x9E3779B9) -> None:
        if m <= 1:
            raise ValueError("m must be > 1")
        self.m = m
        self.N = N
        self.salt = salt_from_seed(seed)
        self.ring = np.zeros((N + 1, m), dtype=bool)  # slot u % (N + 1) holds B_u
        self.t: int | None = None

    def _bit_index(self, elem: np.ndarray) -> np.ndarray:
        return (mix64(elem, self.salt) % U64(self.m)).astype(np.int64)

    def _slot(self, u: int) -> int:
        return u % (self.N + 1)

    def advance(self, t: int) -> None:
        """Move to epoch t and clear its slot and those of skipped epochs."""
        if self.t is not None and t < self.t:
            raise ValueError(f"epochs must be nondecreasing (at {self.t}, got {t})")
        if self.t == t:
            return
        start = t if self.t is None else self.t + 1
        for u in range(max(start, t - self.N), t + 1):
            self.ring[self._slot(u)] = False
        self.t = t

    def process_period(self, elem: np.ndarray, t: int) -> None:
        """Record the items of epoch t."""
        self.advance(t)
        if len(elem):
            self.ring[self._slot(t), self._bit_index(elem)] = True

    def counts(self, k: int) -> tuple[int, int]:
        """(z, s) for history length k at the current epoch."""
        if self.t is None:
            raise ValueError("no epoch recorded yet")
        if not 1 <= k <= self.N:
            raise ValueError(f"k must be in 1..N={self.N}")
        hist = np.zeros(self.m, dtype=bool)
        for i in range(1, k + 1):
            hist |= self.ring[self._slot(self.t - i)]
        z = self.m - int(hist.sum())
        s = int((self.ring[self._slot(self.t)] & ~hist).sum())
        return z, s

    def query(self, k: int) -> float | None:
        """Estimate for the current epoch, or None if unavailable."""
        z, s = self.counts(k)
        if z == 0 or s == z:
            return None
        return -self.m * math.log1p(-s / z)

    def memory_bits(self, workspace: bool = True) -> int:
        """N + 1 bitmaps plus the m-bit mask built by a query."""
        return (self.N + 1 + (1 if workspace else 0)) * self.m
