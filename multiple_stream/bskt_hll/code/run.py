"""bSkt(HLL-Joint) on a grouped stream: 128 registers of 5 bits per HLL array,
d = 4 arrays per group, one hash seed, estimates at epoch N for k = 1..N at
every total budget of the dataset.

    python run.py --dataset caida [--workers 20]
Writes ../result/<dataset>/seed1/<B>MB.npz (est (N, F), NaN = fit failed) and
memory.tsv.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common" / "code"))

import grouped_streams as gs  # noqa: E402
from bskt_hll import BsktHllRing  # noqa: E402

P, D, SEED, N = 7, 4, 1, 12
G: dict = {}


def query_chunk(bounds):
    a, b = bounds
    r = G["ring"]
    est = r.query(G["flows"][a:b], list(range(1, N + 1)))
    return a, b, est, len(r.cache)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="caida")
    ap.add_argument("--budgets-mb", type=float, nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=4000)
    args = ap.parse_args()
    d = gs.build(N, dataset=args.dataset)
    flows = d["flows"]
    F = len(flows)
    out = HERE.parent / "result" / args.dataset / f"seed{SEED}"
    out.mkdir(parents=True, exist_ok=True)
    mem_path = out / "memory.tsv"
    mem = mem_path.read_text().splitlines() if mem_path.exists() else ["budget_MB\tl\tbits_used\tbits_budget\tfailed_fits"]
    U32 = np.uint64(0xFFFFFFFF)
    for mb in args.budgets_mb or gs.BUDGETS_MB[args.dataset]:
        if (out / f"{mb:g}MB.npz").exists():
            print(f"{args.dataset} {mb:g} MB: done already", flush=True)
            continue
        t0 = time.time()
        bits = int(mb * 2**20 * 8)
        l = BsktHllRing.l_for_budget(bits // 8, P, N)
        if l < 1:
            print(f"{args.dataset} {mb:g} MB: below one array per epoch, skipped", flush=True)
            continue
        ring = BsktHllRing(l=l, p=P, d=D, N=N, seed=SEED)
        for u in range(N):
            key = d[f"hist_{u}"]
            ring.process_period(key >> np.uint64(32), key & U32, u)
        key = d["cur_key"]
        ring.process_period(key >> np.uint64(32), key & U32, N)
        G.update(ring=ring, flows=flows)
        est = np.full((N, F), np.nan, dtype=np.float32)
        bounds = [(a, min(a + args.chunk, F)) for a in range(0, F, args.chunk)]
        done = 0
        with mp.get_context("fork").Pool(args.workers) as pool:
            for a, b, e, n_cache in pool.imap_unordered(query_chunk, bounds):
                est[:, a:b] = e
                done += b - a
                if done * 10 // F != (done - (b - a)) * 10 // F:
                    print(f"  {args.dataset} {mb:g} MB: {done:,}/{F:,} groups, {time.time() - t0:.0f} s", flush=True)
        np.savez_compressed(out / f"{mb:g}MB.npz", est=est)
        failed = int(np.isnan(est).sum())
        mem.append(f"{mb:g}\t{l}\t{ring.memory_bits()}\t{bits}\t{failed}")
        mem_path.write_text("\n".join(mem) + "\n")
        print(f"{args.dataset} {mb:g} MB: {l:,} arrays per epoch, failed fits {failed:,}, "
              f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
