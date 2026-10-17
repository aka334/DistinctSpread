"""Tests for bskt_bitmap.py: against a set-based reference, and against the
single-stream HB when every group has its own bitmap."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "single_stream" / "hb" / "code"))

from bskt_bitmap import BsktBitmapRing 
from epoch_bitmap import EpochBitmapRing 


def stream(rng, N, n_groups, size):
    return [(rng.integers(1, n_groups + 1, size=size).astype(np.uint64),
             rng.integers(1, 5 * size, size=size).astype(np.uint64)) for _ in range(N + 1)]


def test_against_set_reference(b: int = 128) -> None:
    rng = np.random.default_rng(2)
    N, l, d = 4, 12, 3                              # small pool, heavy sharing
    st = stream(rng, N, 40, 150)
    r = BsktBitmapRing(l=l, b=b, d=d, N=N, seed=9)
    sets = [dict() for _ in range(N + 1)]           # epoch -> bitmap -> set of bits
    for t, (g, x) in enumerate(st):
        r.process_period(g, x, t)
        for gi, bi in zip(r.bitmaps_of(g).tolist(), r.bit_of(x).tolist()):
            for m in gi:
                sets[t].setdefault(m, set()).add(bi)
    groups = np.unique(st[-1][0])
    ks = list(range(1, N + 1))
    got = r.query(groups, ks)
    for col, (gid, maps) in enumerate(zip(groups.tolist(), r.bitmaps_of(groups).tolist())):
        A = [set.intersection(*(sets[u].get(m, set()) for m in maps)) for u in range(N + 1)]
        for k in ks:
            B = set().union(*A[N - k:N])
            z, s = b - len(B), len(A[N] - B)
            want = math.nan if (z == 0 or s == z) else -b * math.log1p(-s / z)
            g_ = got[k - 1, col]
            assert (math.isnan(want) and math.isnan(g_)) or math.isclose(g_, want, rel_tol=1e-12), (gid, k)
    print(f"shared pool (l={l}, b={b}, d={d}): {len(groups)} groups x {N} histories match the "
          f"set-based AND/OR reference ({int(np.isnan(got).sum())} unavailable)  ok")


def test_reduces_to_private_bitmap() -> None:
    rng = np.random.default_rng(4)
    N, b, d, seed = 5, 256, 3, 21
    st = stream(rng, N, 15, 300)
    r = BsktBitmapRing(l=10**6, b=b, d=d, N=N, seed=seed)
    groups = np.unique(st[-1][0])
    maps = r.bitmaps_of(np.unique(np.concatenate([g for g, _ in st])))
    assert len(np.unique(maps)) == maps.size, "test needs no shared bitmaps"
    for t, (g, x) in enumerate(st):
        r.process_period(g, x, t)
    got = r.query(groups, list(range(1, N + 1)))
    for col, gid in enumerate(groups.tolist()):
        priv = EpochBitmapRing(m=b, N=N, seed=seed)
        for t, (g, x) in enumerate(st):
            priv.process_period(x[g == gid], t)
        for k in range(1, N + 1):
            want = priv.query(k)
            g_ = got[k - 1, col]
            assert (want is None and math.isnan(g_)) or math.isclose(g_, want, rel_tol=1e-12), (gid, k, g_, want)
    print(f"no shared bitmaps: identical to the single-stream HB (m = b = {b}) "
          f"for {len(groups)} groups x {N} histories  ok")


if __name__ == "__main__":
    for b in (128, 100, 5000):                      # 100, 5000: padded last word
        test_against_set_reference(b)
    test_reduces_to_private_bitmap()
    print("all tests passed")
