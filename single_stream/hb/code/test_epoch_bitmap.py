"""Tests for epoch_bitmap.py."""

from __future__ import annotations

import math

import numpy as np

from epoch_bitmap import EpochBitmapRing


def bits(s: str) -> np.ndarray:
    return np.array([c == "1" for c in s], dtype=bool)


def test_example() -> None:
    # worked example: k = 2, m = 8 gives z = 5, s = 2
    bm = EpochBitmapRing(m=8, N=2)
    bm.advance(2)
    bm.ring[bm._slot(0)] = bits("10001000")  # B_{t-2}
    bm.ring[bm._slot(1)] = bits("01001000")  # B_{t-1}
    bm.ring[bm._slot(2)] = bits("01100100")  # B_t
    z, s = bm.counts(2)
    assert (z, s) == (5, 2), (z, s)
    expected = -8 * math.log(1 - 2 / 5)
    assert math.isclose(bm.query(2), expected), (bm.query(2), expected)
    print(f"example: z={z} s={s} estimate={bm.query(2):.4f}  ok")


def test_bottom_cases() -> None:
    bm = EpochBitmapRing(m=4, N=1)
    bm.advance(1)
    bm.ring[bm._slot(0)] = bits("1111")  # z = 0
    assert bm.query(1) is None
    bm.ring[bm._slot(0)] = bits("1100")
    bm.ring[bm._slot(1)] = bits("0011")  # s = z
    assert bm.query(1) is None
    bm.ring[bm._slot(1)] = bits("1100")  # s = 0
    assert bm.query(1) == 0.0
    print("bottom (z=0, s=z) and zero cases  ok")


def test_recurrence_and_return() -> None:
    m, N = 1 << 16, 3
    old = np.arange(1, 501, dtype=np.uint64)
    bm = EpochBitmapRing(m=m, N=N)
    bm.process_period(old, 0)
    bm.process_period(np.array([], dtype=np.uint64), 1)
    bm.process_period(np.array([], dtype=np.uint64), 2)
    bm.process_period(np.concatenate([old, old]), 3)
    # absent for 2 epochs: new for k = 1, 2, not for k = 3
    for k, true in [(1, 500), (2, 500), (3, 0)]:
        est = bm.query(k)
        assert abs(est - true) <= 0.05 * max(true, 1) + 1, (k, est, true)
    print("returns after absence are novel; recurrences are filtered  ok")


def test_skipped_epochs_are_cleared() -> None:
    bm = EpochBitmapRing(m=1024, N=2)
    bm.process_period(np.arange(100, dtype=np.uint64), 0)
    # skip to epoch 3: slots of the silent epochs 1 and 2 must be empty
    bm.process_period(np.arange(100, dtype=np.uint64), 3)
    z, s = bm.counts(2)
    assert z == 1024, z
    print("epochs without traffic clear their ring slots  ok")


def test_theorem1_occupancy() -> None:
    # E[r/z] = (1 - 1/m)^d given the history mask
    m, N, d, h = 2048, 2, 600, 900
    ratios, ests = [], []
    for seed in range(400):
        bm = EpochBitmapRing(m=m, N=N, seed=seed)
        hist = np.arange(1, h + 1, dtype=np.uint64)
        novel = np.arange(10**6, 10**6 + d, dtype=np.uint64)
        bm.process_period(hist[: h // 2], 0)
        bm.process_period(hist[h // 2:], 1)
        # current epoch: d new labels plus all old ones, repeated
        bm.process_period(np.concatenate([novel, hist, novel]), 2)
        z, s = bm.counts(2)
        ratios.append((z - s) / z)
        ests.append(bm.query(2))
    expected = (1 - 1 / m) ** d
    se = np.std(ratios, ddof=1) / math.sqrt(len(ratios))
    assert abs(np.mean(ratios) - expected) < 4 * se, (np.mean(ratios), expected, se)
    bias = np.mean(ests) / d - 1
    print(f"theorem 1: mean r/z={np.mean(ratios):.4f} vs (1-1/m)^d={expected:.4f} "
          f"(se {se:.4f}); estimator mean/true - 1 = {100 * bias:+.2f}%  ok")


if __name__ == "__main__":
    test_example()
    test_bottom_cases()
    test_recurrence_and_return()
    test_skipped_epochs_are_cleared()
    test_theorem1_occupancy()
    print("all tests passed")
