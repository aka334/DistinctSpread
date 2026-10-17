"""
    python sweep.py --dataset caida --seed 1 --per-n --fracs 0.1 0.2 ... 0.9 --ds 1 2 4
Writes ../result/[pern/][<dataset>/]seed<S>/<B>MB_ts<f%>_d<d>.npz with the
"sum" and "ssvs2" estimates (NaN = unavailable), and memory.tsv.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common" / "code"))
sys.path.insert(0, str(HERE.parents[1] / "tnds_ht" / "code"))

import grouped_streams  # noqa: E402
from shared_tnds import SharedTnds  # noqa: E402
from ssvs import Ssvs, UNIT_BITS  # noqa: E402
from sweep import m_for_bits, mb_label, tnds_bits  # noqa: E402  (from tnds_ht)

RESULT = HERE.parent / "result"
G: dict = {}


def run(job):
    mb, frac, dd, seed = job
    d, N = G["d"], G["N"]
    K = list(range(1, N + 1))
    F = len(d["flows"])
    bits = int(mb * 2**20 * 8)
    m = m_for_bits(int(bits * frac), N, len(K))
    # per_n: the sketch of each k gets all counter memory, otherwise 1/|K| each
    w = Ssvs.units_for_bits((bits - tnds_bits(m, N, len(K))) // (1 if G["per_n"] else len(K)))
    samples, valid = SharedTnds(m=m, K=K, N=N, seed=seed).samples(d)
    est_sum = np.full((N, F), np.nan)
    est_s2 = np.full((N, F), np.nan)
    for k in K:
        if not valid[k]:
            continue
        sk = Ssvs(w=w, n_groups_ids=d["flows"], d=dd, seed=seed * 1000 + k)
        g, wt = samples[k]
        sk.add(g, wt)
        est_sum[k - 1], est_s2[k - 1] = sk.query()
    used = tnds_bits(m, N, len(K)) + (1 if G["per_n"] else len(K)) * w * UNIT_BITS
    bad = [k for k in K if not valid[k]]
    return mb, frac, dd, m, w, used, bits, bad, est_sum, est_s2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--budgets-mb", type=float, nargs="+", default=None,
                    help="default: grouped_streams.BUDGETS_MB[dataset]")
    ap.add_argument("--dataset", default="caida", choices=grouped_streams.traces.DATASETS)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--ds", type=int, nargs="+", default=[1, 2, 4],
                    help="SSVS positions per group")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per-n", action="store_true",
                    help="charge each k separately (results under result/pern/)")
    args = ap.parse_args()

    G.update(d=grouped_streams.build(args.N, dataset=args.dataset), N=args.N, per_n=args.per_n)
    args.budgets_mb = args.budgets_mb or grouped_streams.BUDGETS_MB[args.dataset]
    out = grouped_streams.result_dir(RESULT / "pern" if args.per_n else RESULT, args.dataset, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(mb, f, dd, args.seed) for mb in args.budgets_mb for f in args.fracs for dd in args.ds]
    rows = []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for mb, f, dd, m, w, used, bits, bad, es, e2 in pool.imap_unordered(run, jobs):
            np.savez_compressed(out / f"{mb_label(mb)}_ts{round(100 * f)}_d{dd}.npz",
                                sum=es.astype(np.float32), ssvs2=e2.astype(np.float32))
            rows.append((mb, f, dd, f"{mb:g}\t{round(100 * f)}\t{dd}\t{m}\t{w}\t{used}\t{bits}\t"
                                    f"{','.join(map(str, bad)) or '-'}"))
            print(f"{mb:>6g} MB ts {round(100 * f)}% d={dd}: m = {m:,} cells, SSVS w = {w:,} units per k; "
                  f"unavailable k: {bad or 'none'}", flush=True)
    rows.sort()
    (out / "memory.tsv").write_text(
        "budget_MB\tts_pct\td\tm\tssvs_units_per_k\tbits_used\tbits_budget\tunavailable_k\n"
        + "\n".join(r[3] for r in rows) + "\n")


if __name__ == "__main__":
    main()
