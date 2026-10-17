"""Sparse simulation of one m-cell array per flow, all flows at once, with the
cell of a label given by h(x) = eta(x) mod m.

For history length k at the evaluation epoch t:
    z_k  m minus the cells touched in epochs t-k .. t-1
         (bitmap: zeros of the history mask; TNDS: stale cells)
    s_k  cells touched at t and in none of those epochs
         (bitmap: ones of B_t & ~B; TNDS: admitted samples)
Both estimators depend only on (m, z_k, s_k); TNDS validity also needs the
position of the last admission. Only touched cells are stored.
"""

from __future__ import annotations

import numpy as np

from flow_streams import split_key
from hashing import U64, mix64

SHIFT = U64(32)


def _cell_keys(key: np.ndarray, m: int, salt: int) -> np.ndarray:
    fi, src = split_key(key)
    cell = mix64(src, salt) % U64(m)
    return (fi.astype(U64) << SHIFT) | cell


def cell_counts(d: dict, m: int, salt: int):
    """z, s, last_adm, each of shape (N, F)."""
    N, F = int(d["N"]), len(d["flows"])
    t = N

    # last epoch before t that touched each (flow, cell)
    hk = [np.unique(_cell_keys(d[f"hist_{u}"], m, salt)) for u in range(t)]
    keys = np.concatenate(hk)
    ep = np.concatenate([np.full(len(h), u, dtype=np.int64) for u, h in enumerate(hk)])
    order = np.lexsort((ep, keys))
    keys, ep = keys[order], ep[order]
    is_last = np.ones(len(keys), dtype=bool)
    is_last[:-1] = keys[1:] != keys[:-1]
    hist_key, hist_last = keys[is_last], ep[is_last]
    hist_f = (hist_key >> SHIFT).astype(np.int64)

    # cells touched at t, each at the position of its first hit
    ck = _cell_keys(d["cur_key"], m, salt)
    order = np.lexsort((d["cur_first"], ck))
    ck, cfirst = ck[order], d["cur_first"][order]
    first = np.ones(len(ck), dtype=bool)
    first[1:] = ck[1:] != ck[:-1]
    cur_key, cur_pos = ck[first], cfirst[first]
    cur_f = (cur_key >> SHIFT).astype(np.int64)
    pos = np.searchsorted(hist_key, cur_key)
    pos_c = np.minimum(pos, len(hist_key) - 1)
    found = (pos < len(hist_key)) & (hist_key[pos_c] == cur_key)
    cur_last = np.where(found, hist_last[pos_c], -1)

    z = np.empty((N, F), dtype=np.int64)
    s = np.empty((N, F), dtype=np.int64)
    last_adm = np.full((N, F), -1, dtype=np.int64)
    for k in range(1, N + 1):
        z[k - 1] = m - np.bincount(hist_f[hist_last >= t - k], minlength=F)
        adm = cur_last < t - k
        s[k - 1] = np.bincount(cur_f[adm], minlength=F)
        np.maximum.at(last_adm[k - 1], cur_f[adm], cur_pos[adm])
    return z, s, last_adm
