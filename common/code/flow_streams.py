"""Single-stream inputs on CAIDA: each destination IP is its own stream of
source IPs, with its own structure. Estimates are made at epoch t = N, the
first epoch with a full N-epoch history.

build() returns (cached) for the flows active at epoch N:
    flows      (F,) flow ids, sorted
    true       (N, F) exact k-novel distinct counts, k = 1..N
    hist_u     epoch u < N: distinct (flow index << 32 | src) keys
    cur_key    epoch N: distinct keys
    cur_first  stream position of each key's first packet
    cur_last   stream position of each flow's last packet
"""

from __future__ import annotations

import numpy as np

from ground_truth import ExactDistinctSpreadTracker
from paths import CACHE, CAIDA
from reader import load_period, period_files


def build(N: int = 12, flow: str = "dst", use_cache: bool = True) -> dict:
    cache = CACHE / f"flow_streams_N{N}_{flow}.npz"
    if use_cache and cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}

    periods = []
    gt = ExactDistinctSpreadTracker(windows=list(range(1, N + 1)))
    for t, path in enumerate(period_files(CAIDA, N + 1)):
        src, dst = load_period(path)
        f, e = (dst, src) if flow == "dst" else (src, dst)
        res = gt.process_period(f, e, t)
        periods.append((f.astype(np.uint64), e.astype(np.uint64)))
        print(f"epoch {t}/{N}", flush=True)

    f_cur, e_cur = periods[N]
    flows = np.unique(f_cur)
    true = np.array([[res[k].get(int(x), 0) for x in flows.tolist()] for k in range(1, N + 1)], dtype=np.int64)

    out = {"flows": flows, "true": true, "N": np.array(N)}
    for u in range(N):
        f, e = periods[u]
        keep = np.isin(f, flows)
        out[f"hist_{u}"] = np.unique((np.searchsorted(flows, f[keep]).astype(np.uint64) << np.uint64(32)) | e[keep])

    fi = np.searchsorted(flows, f_cur).astype(np.uint64)
    key = (fi << np.uint64(32)) | e_cur
    ukey, first = np.unique(key, return_index=True)
    out["cur_key"], out["cur_first"] = ukey, first.astype(np.int64)
    last = np.full(len(flows), -1, dtype=np.int64)
    np.maximum.at(last, fi.astype(np.int64), np.arange(len(f_cur), dtype=np.int64))
    out["cur_last"] = last
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def split_key(key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (key >> np.uint64(32)).astype(np.int64), key & np.uint64(0xFFFFFFFF)


def epoch_labels(d: dict) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per epoch u = 0..N: (flow index, src, offsets); the labels of flow i
    in epoch u are src[offsets[i]:offsets[i + 1]]."""
    N, F = int(d["N"]), len(d["flows"])
    out = []
    for u in range(N + 1):
        key = d[f"hist_{u}"] if u < N else d["cur_key"]
        fi, src = split_key(key)
        out.append((fi, src, np.searchsorted(fi, np.arange(F + 1))))
    return out
