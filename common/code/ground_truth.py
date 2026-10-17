"""Exact k-novel distinct counts per flow.

A pair (f, x) seen in epoch t counts toward flow f's k-novel count if its
previous occurrence, if any, is more than k epochs earlier. One map from pair
to last epoch serves every k at once.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


class ExactDistinctSpreadTracker:
    def __init__(self, windows: list[int]) -> None:
        self.windows = windows
        self._last_seen: dict[int, int] = {}

    def process_period(self, flow: np.ndarray, label: np.ndarray, period: int) -> dict[int, dict[int, int]]:
        """{k: {flow: k-novel count}} for this epoch."""
        keys = (flow.astype(np.uint64) << np.uint64(32)) | label.astype(np.uint64)
        uniq, first = np.unique(keys, return_index=True)
        gid = flow[first]

        result = {n: defaultdict(int) for n in self.windows}
        last_seen = self._last_seen
        for k, f in zip(uniq.tolist(), gid.tolist()):
            prev = last_seen.get(k)
            for n in self.windows:
                if prev is None or period - prev > n:
                    result[n][f] += 1
            last_seen[k] = period
        return {n: dict(d) for n, d in result.items()}
