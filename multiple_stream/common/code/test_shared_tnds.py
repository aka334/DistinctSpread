"""Tests for shared_tnds.py: the samples summed per group (Algorithm 3 with an
exact hash table) must match a per-item reference of Algorithms 2 and 3
without timestamp wrap-around."""

from __future__ import annotations

import math

import numpy as np

from grouped_streams import pair_keys
from shared_tnds import SharedTnds, pair_cells


def as_grouped(stream, flows):
    """Per-epoch (dst, src) arrays to the dict returned by grouped_streams.build()."""
    N = len(stream) - 1
    d = {"N": np.array(N), "flows": flows}
    for u, (dst, src) in enumerate(stream[:-1]):
        d[f"hist_{u}"] = np.unique(pair_keys(dst, src))
    dst, src = stream[-1]
    key = pair_keys(dst, src)
    ukey, first = np.unique(key, return_index=True)
    d["cur_key"], d["cur_first"] = ukey, first.astype(np.int64)
    d["cur_group"] = np.searchsorted(flows, ukey >> np.uint64(32))
    d["n_packets"] = np.array(len(key))
    return d


def reference(stream, flows, m, K, cells_of):
    N = len(stream) - 1
    T = np.full(m, -10**9, dtype=np.int64)
    for t, (dst, src) in enumerate(stream):
        C = {k: np.zeros(len(flows)) for k in K}
        v = {k: True for k in K}
        cells = cells_of(pair_keys(dst, src)).tolist()
        groups = np.searchsorted(flows, dst).tolist() if t == N else [0] * len(dst)
        for j, g in zip(cells, groups):
            a = t - T[j]
            for k in K:
                z = int((t - T > k).sum())
                if z == 0:
                    v[k] = False
                if a > k and v[k] and t == N:
                    C[k][g] += m / z
            T[j] = t
    return C, v


def main() -> None:
    rng = np.random.default_rng(7)
    for m, N, n_groups, size in [(400, 4, 30, 120), (3000, 6, 80, 900), (60, 3, 10, 80), (25, 3, 6, 60)]:
        K = list(range(1, N + 1))
        stream = []
        for _ in range(N + 1):
            n = int(rng.integers(size // 2, size))
            dst = rng.integers(1, n_groups + 1, size=n).astype(np.uint32)
            src = rng.integers(1, 4 * size, size=n).astype(np.uint32)
            stream.append((dst, src))
        flows = np.unique(stream[-1][0]).astype(np.uint64)
        d = as_grouped([(a.astype(np.uint64), b.astype(np.uint64)) for a, b in stream], flows)
        st = SharedTnds(m=m, K=K, N=N, seed=11)
        samples, valid = st.samples(d)
        C_ref, v_ref = reference([(a.astype(np.uint64), b.astype(np.uint64)) for a, b in stream],
                                 flows, m, K, lambda keys: pair_cells(keys, m, st.salt))
        n_invalid = 0
        for k in K:
            assert valid[k] == v_ref[k], (m, k, valid[k], v_ref[k])
            if not valid[k]:
                n_invalid += 1
                continue
            groups, w = samples[k]
            got = np.bincount(groups, weights=w, minlength=len(flows))
            assert np.allclose(got, C_ref[k], rtol=1e-9, atol=1e-9), (m, k)
        print(f"m={m:>5} N={N}: {len(flows)} groups x {N} histories identical to the brute-force "
              f"reference ({n_invalid} unavailable histories)  ok")
    unbiased()
    print("all tests passed")


def unbiased() -> None:
    """Theorem 4: per-group estimates are unbiased while stale cells remain."""
    rng = np.random.default_rng(3)
    N, K = 3, [1, 3]
    stream = []
    for _ in range(N + 1):
        dst = rng.integers(1, 6, size=600).astype(np.uint64)
        src = rng.integers(1, 700, size=600).astype(np.uint64)
        stream.append((dst, src))
    flows = np.unique(stream[-1][0])
    d = as_grouped(stream, flows)
    cur = set(zip(stream[-1][0].tolist(), stream[-1][1].tolist()))
    true = {}
    for k in K:
        hist = set()
        for u in range(N - k, N):
            hist |= set(zip(stream[u][0].tolist(), stream[u][1].tolist()))
        true[k] = np.array([sum(1 for (g, x) in cur if g == f and (g, x) not in hist) for f in flows.tolist()])
    sums = {k: [] for k in K}
    for seed in range(300):
        smp, valid = SharedTnds(m=20_000, K=K, N=N, seed=seed).samples(d)
        for k in K:
            assert valid[k]
            g, w = smp[k]
            sums[k].append(np.bincount(g, weights=w, minlength=len(flows)))
    for k in K:
        arr = np.array(sums[k])
        z = (arr.mean(0) - true[k]) / (arr.std(0, ddof=1) / math.sqrt(len(arr)))
        assert np.all(np.abs(z) < 4), (k, z)
        print(f"k={k}: per-group mean over 300 seeds vs truth, max |z| = {np.abs(z).max():.2f} "
              f"(truth {true[k].tolist()})  ok")


if __name__ == "__main__":
    main()
