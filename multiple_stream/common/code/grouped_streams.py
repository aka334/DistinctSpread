"""A trace as one grouped stream of items <g, x> = <destination, source>
(E-commerce: <product, user>). Every group's k-novel count is estimated from
one shared structure, so the history keeps the pairs of all groups, and the
evaluation epoch keeps arrival order.

build() returns (cached):
    flows      groups active at epoch N
    true       (N, F) exact k-novel counts
    hist_u     epoch u < N: sorted distinct pair keys (g << 32 | x)
    cur_key    epoch N: sorted distinct pair keys
    cur_first  stream position of each pair's first item
    cur_group  index of each pair's group in flows
    n_packets  items in epoch N
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "common" / "code"))

import flow_streams  # noqa: E402
import traces  # noqa: E402
from paths import CACHE, CAIDA  # noqa: E402
from reader import load_period, period_files  # noqa: E402

U64 = np.uint64
BUDGETS_MB = {"caida": [0.5, 1, 2, 5, 10, 20, 50, 100],
              "mawi": [1, 2, 5, 10, 20, 50, 100, 200, 500],
              "ecommerce": [0.5, 1, 2, 5, 10, 20, 50, 100]}


def result_dir(method_result: Path, dataset: str, seed: int) -> Path:
    """result/seed<S> for CAIDA, result/<dataset>/seed<S> otherwise."""
    return method_result / (f"seed{seed}" if dataset == "caida" else f"{dataset}/seed{seed}")


def pair_keys(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    return (dst.astype(U64) << U64(32)) | src.astype(U64)


def truth(hist: list[np.ndarray], ukey: np.ndarray, groups: np.ndarray, F: int) -> np.ndarray:
    """Exact k-novel counts (N, F): a pair of epoch N counts for k if it does
    not occur in epochs N-k .. N-1."""
    N = len(hist)
    last = np.full(len(ukey), -1, dtype=np.int64)
    for u, h in enumerate(hist):
        i = np.minimum(np.searchsorted(h, ukey), len(h) - 1)
        last[h[i] == ukey] = u
    return np.stack([np.bincount(groups[last < N - k], minlength=F) for k in range(1, N + 1)])


def build(N: int = 12, use_cache: bool = True, dataset: str = "caida") -> dict:
    if dataset != "caida":
        return build_trace(dataset, N, use_cache)
    cache = CACHE / f"grouped_streams_caida_N{N}.npz"
    if use_cache and cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}

    fs = flow_streams.build(N)
    out = {"flows": fs["flows"], "true": fs["true"], "N": np.array(N)}
    for t, path in enumerate(period_files(CAIDA, N + 1)):
        src, dst = load_period(path)
        key = pair_keys(dst, src)
        if t < N:
            out[f"hist_{t}"] = np.unique(key)
        else:
            ukey, first = np.unique(key, return_index=True)
            out["cur_key"], out["cur_first"] = ukey, first.astype(np.int64)
            out["cur_group"] = np.searchsorted(fs["flows"], (ukey >> U64(32)))
            out["n_packets"] = np.array(len(key))
        print(f"epoch {t}/{N}", flush=True)
    assert np.array_equal(fs["flows"][out["cur_group"]], out["cur_key"] >> U64(32))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def build_trace(dataset: str, N: int, use_cache: bool = True) -> dict:
    """build() for MAWI and E-commerce."""
    cache = CACHE / f"grouped_streams_{dataset}_N{N}.npz"
    if use_cache and cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    out, hist = {"N": np.array(N)}, []
    for t, (src, dst) in enumerate(traces.epochs(dataset, N + 1)):
        key = pair_keys(dst, src)
        if t < N:
            hist.append(np.unique(key))
            out[f"hist_{t}"] = hist[-1]
        else:
            ukey, first = np.unique(key, return_index=True)
            flows = np.unique(dst).astype(U64)
            out.update(flows=flows, cur_key=ukey, cur_first=first.astype(np.int64),
                       cur_group=np.searchsorted(flows, ukey >> U64(32)), n_packets=np.array(len(key)))
        print(f"{dataset} epoch {t}/{N}: {len(key):,} items", flush=True)
    out["true"] = truth(hist, out["cur_key"], out["cur_group"], len(out["flows"]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "caida"
    d = build(12, dataset=name)
    pairs = [len(d[f"hist_{u}"]) for u in range(12)] + [len(d["cur_key"])]
    t = d["true"]
    print(f"{name}: {len(d['flows']):,} groups at epoch 12, distinct pairs per epoch "
          f"{min(pairs):,}..{max(pairs):,}, max count k=1 {t[0].max():,}, k=12 {t[-1].max():,}")
