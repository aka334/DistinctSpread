"""bSkt(HB) with b = 5000 bits and d = 4 bitmaps per group at every budget.
    python run_max.py --dataset caida
Writes ../result/max/<dataset>/seed1/<B>MB.npz: est (N, F) and unavail (N, F).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common" / "code"))

import grouped_streams as gs  # noqa: E402
from bskt_bitmap import BsktBitmapRing  # noqa: E402

B, D, SEED, N = 5000, 4, 1, 12
G: dict = {}


def run(mb: float):
    d = G["d"]
    l = BsktBitmapRing.l_for_budget(int(mb * 2**20), B, N)
    if l < 1:
        return mb, l, None, None
    r = BsktBitmapRing(l=l, b=B, d=D, N=N, seed=SEED)
    U32 = np.uint64(0xFFFFFFFF)
    for u in range(N):
        key = d[f"hist_{u}"]
        r.process_period(key >> np.uint64(32), key & U32, u)
    key = d["cur_key"]
    r.process_period(key >> np.uint64(32), key & U32, N)
    est, sat = r.query(d["flows"], list(range(1, N + 1)), saturated="max")
    return mb, l, est, sat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="caida")
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args()
    G["d"] = gs.build(N, dataset=args.dataset)
    out = HERE.parent / "result" / "max" / args.dataset / f"seed{SEED}"
    out.mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(args.workers) as pool:
        for mb, l, est, sat in pool.imap_unordered(run, gs.BUDGETS_MB[args.dataset]):
            if est is None:
                print(f"{args.dataset} {mb:g} MB: below one bitmap per epoch, skipped", flush=True)
                continue
            np.savez_compressed(out / f"{mb:g}MB.npz", est=est.astype(np.float32), unavail=sat)
            print(f"{args.dataset} {mb:g} MB: {l:,} bitmaps per epoch, saturated estimates {sat.mean():.1%}, "
                  f"largest estimate {np.nanmax(est):,.0f}", flush=True)


if __name__ == "__main__":
    main()
