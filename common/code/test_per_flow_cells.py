"""Check that the sparse all-flows simulation (per_flow_cells and the
bitmap/TNDS sweep formulas) matches the per-flow classes run on each flow's
actual packet sequence."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for sub in ("hb", "tnds"):
    sys.path.insert(0, str(ROOT / "single_stream" / sub / "code"))

import flow_streams  # noqa: E402
from epoch_bitmap import EpochBitmapRing  # noqa: E402
from hashing import salt_from_seed  # noqa: E402
from per_flow_cells import cell_counts  # noqa: E402
from reader import load_period, period_files  # noqa: E402
from tnds import Tnds  # noqa: E402

import importlib.util  # noqa: E402


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bitmap_sweep = load_module(ROOT / "single_stream" / "hb" / "code" / "sweep_single_stream.py", "bitmap_sweep")
tnds_sweep = load_module(ROOT / "single_stream" / "tnds" / "code" / "sweep_single_stream.py", "tnds_sweep")


def main() -> None:
    N, seed = 12, 3
    d = flow_streams.build(N)
    true1 = d["true"][0]
    rng = np.random.default_rng(0)
    sample = []
    for lo, hi, n in [(0, 2, 40), (2, 10, 40), (10, 100, 40), (100, 1000, 30), (1000, 10**9, 10)]:
        idx = np.flatnonzero((true1 >= lo) & (true1 < hi))
        sample += rng.choice(idx, size=min(n, len(idx)), replace=False).tolist()
    sample = np.array(sorted(sample))
    flows = d["flows"][sample]

    # packets of the sampled flows per epoch, in arrival order
    packets = []
    for t, path in enumerate(period_files(flow_streams.CAIDA, N + 1)):
        src, dst = load_period(path)
        keep = np.isin(dst, flows.astype(dst.dtype))
        packets.append((dst[keep].astype(np.uint64), src[keep].astype(np.uint64)))

    salt = salt_from_seed(seed)
    for m in (16, 64, 2048):
        z, s, last_adm = cell_counts(d, m, salt)
        bm = bitmap_sweep.bitmap_estimates(m, z, s)[:, sample]
        tn = tnds_sweep.tnds_estimates(m, z, s, last_adm, d["cur_last"])[:, sample]
        n_nan = 0
        for col, f in enumerate(flows.tolist()):
            ring = EpochBitmapRing(m=m, N=N, seed=seed)
            tds = Tnds(m=m, K=list(range(1, N + 1)), N=N, seed=seed)
            for t, (dst, src) in enumerate(packets):
                elems = src[dst == f]
                ring.process_period(elems, t)
                tds.process_period(elems, t)
            for k in range(1, N + 1):
                for got, want, name in ((bm[k - 1, col], ring.query(k), "bitmap"),
                                        (tn[k - 1, col], tds.query(k), "tnds")):
                    if want is None:
                        assert math.isnan(got), (name, m, f, k, got)
                        n_nan += 1
                    else:
                        assert math.isclose(got, want, rel_tol=1e-6, abs_tol=1e-6), (name, m, f, k, got, want)
        print(f"m={m:>5}: {len(flows)} flows x {N} histories identical for bitmap and TNDS "
              f"({n_nan} unavailable)  ok")
    print("all tests passed")


if __name__ == "__main__":
    main()
