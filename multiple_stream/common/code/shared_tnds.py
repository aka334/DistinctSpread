"""Shared TNDS filter for grouped streams: one array of m timestamp cells for
all groups, indexed by the hash of the pair <g, x>, with the update and
expiration of Algorithm 2 (reuses the Tnds class).

samples() returns, for the evaluation epoch N and each k, the weighted
samples <g, k, w> in arrival order, which feed the counter layer (hash table
or SSVS):
  - only the first hit of a cell in the epoch can be admitted, and it is
    admitted for k if the cell's age before the epoch is > k;
  - the n-th admission for k (n = 0, 1, ...) has weight m / (z_k - n), where
    z_k is the number of stale cells at the start of the epoch;
  - the sample goes to the group of the item that first hit the cell;
  - k is unavailable for all groups if an item arrives while z_k = 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "common" / "code"))
sys.path.insert(0, str(ROOT / "single_stream" / "tnds" / "code"))

from hashing import U64, mix64  # noqa: E402
from tnds import Tnds  # noqa: E402


def pair_cells(keys: np.ndarray, m: int, salt: int) -> np.ndarray:
    """Cell of each pair key g << 32 | x."""
    return (mix64(keys, salt) % U64(m)).astype(np.int64)


class SharedTnds(Tnds):
    """Tnds over pair keys, returning the samples of the evaluation epoch."""

    def samples(self, d: dict) -> tuple[dict, dict]:
        """Run epochs 0..N-1, then return ({k: (groups, weights)}, {k: valid})
        for epoch N. d comes from grouped_streams.build()."""
        N = int(d["N"])
        for u in range(N):
            self.advance(u)
            self._process_cells(pair_cells(d[f"hist_{u}"], self.m, self.salt))
        self.advance(N)

        # first hit of each cell, in arrival order, and its group
        cells = pair_cells(d["cur_key"], self.m, self.salt)
        order = np.lexsort((d["cur_first"], cells))
        cells_s, first_s, group_s = cells[order], d["cur_first"][order], d["cur_group"][order]
        head = np.ones(len(cells_s), dtype=bool)
        head[1:] = cells_s[1:] != cells_s[:-1]
        hit_cell, hit_pos, hit_group = cells_s[head], first_s[head], group_s[head]
        arrival = np.argsort(hit_pos, kind="stable")
        hit_cell, hit_pos, hit_group = hit_cell[arrival], hit_pos[arrival], hit_group[arrival]
        age = self.ages(hit_cell)
        last_item = int(d["n_packets"]) - 1

        out, valid = {}, {}
        for k in self.K:
            adm = age > k
            groups, pos = hit_group[adm], hit_pos[adm]
            z0 = self.z[k]
            n = len(groups)
            valid[k] = not (z0 == 0 or (n == z0 and pos[-1] < last_item))
            weights = self.m / (z0 - np.arange(n, dtype=np.float64)) if n else np.empty(0)
            out[k] = (groups, weights)
        return out, valid
