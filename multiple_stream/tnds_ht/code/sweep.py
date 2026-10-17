"""Shared TNDS with an exact hash table (Algorithm 3) on a grouped stream,
estimates at epoch N for k = 1..N, for each total memory budget.

The budget covers the filter (Tnds.memory_bits) and the hash table, charged
as one entry per group that received a sample: a 32-bit key plus --counters
32-bit counters (default one per k). The table size depends on m, so m is
found by bisection as the largest filter with filter + table <= budget.
With --ht-offchip the whole budget goes to the filter and the table is not
charged.

    python sweep.py --dataset caida --seed 1 --counters 1
Writes ../result/[ht<C>c/][<dataset>/]seed<S>/<B>MB.npz and memory.tsv.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "common" / "code"))

import grouped_streams  # noqa: E402
from shared_tnds import SharedTnds  # noqa: E402

RESULT = HERE.parent / "result"
DEFAULT_MB = [0.5, 1, 2, 5, 10, 20, 50, 100]


def tnds_bits(m: int, N: int, n_k: int) -> int:
    w = math.ceil(math.log2(2 * N + 2))
    cnt = math.ceil(math.log2(m + 1))
    return m * w + (N + 1) * cnt + n_k * (cnt + 1 + 64)


def m_for_bits(bits: int, N: int, n_k: int) -> int:
    m = bits // math.ceil(math.log2(2 * N + 2))
    while m > 1 and tnds_bits(m, N, n_k) > bits:
        m -= 1
    return m


HT_COUNTERS = None          # counters per entry, None = one per k


def ht_entry_bits(n_k: int) -> int:
    return 32 + 32 * (HT_COUNTERS or n_k)


def run_ht(d: dict, m: int, K: list[int], N: int, seed: int):
    """Estimates (N, F), number of groups in the table, unavailable k."""
    F = len(d["flows"])
    samples, valid = SharedTnds(m=m, K=K, N=N, seed=seed).samples(d)
    est = np.full((N, F), np.nan)
    sampled = np.zeros(F, dtype=bool)
    for k in K:
        g, w = samples[k]
        sampled[g] = True
        if valid[k]:
            est[k - 1] = np.bincount(g, weights=w, minlength=F)
    return est, int(sampled.sum()), [k for k in K if not valid[k]]


def fit_charged(d: dict, bits: int, K: list[int], N: int, seed: int, rel_tol: float = 0.01):
    """Largest m (to rel_tol) with tnds_bits(m) + hash table <= bits."""
    fits = lambda m, n_ht: tnds_bits(m, N, len(K)) + n_ht * ht_entry_bits(len(K)) <= bits
    hi = m_for_bits(bits, N, len(K))
    res = run_ht(d, hi, K, N, seed)
    if fits(hi, res[1]):
        return hi, res
    lo, best = 1, None                               # hi never fits
    while hi - lo > max(1, int(rel_tol * hi)):
        mid = (lo + hi) // 2
        res = run_ht(d, mid, K, N, seed)
        if fits(mid, res[1]):
            lo, best = mid, (mid, res)
        else:
            hi = mid
    return best if best is not None else (lo, run_ht(d, lo, K, N, seed))


def mb_label(mb: float) -> str:
    return f"{mb:g}MB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--budgets-mb", type=float, nargs="+", default=None,
                    help="default: grouped_streams.BUDGETS_MB[dataset]")
    ap.add_argument("--dataset", default="caida", choices=grouped_streams.traces.DATASETS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ht-offchip", action="store_true", help="do not charge the hash table")
    ap.add_argument("--counters", type=int, default=None, help="32-bit counters per table entry (default: one per k)")
    args = ap.parse_args()

    d = grouped_streams.build(args.N, dataset=args.dataset)
    args.budgets_mb = args.budgets_mb or grouped_streams.BUDGETS_MB[args.dataset]
    K = list(range(1, args.N + 1))
    F = len(d["flows"])
    global HT_COUNTERS
    HT_COUNTERS = args.counters
    base = RESULT if args.counters is None else RESULT / f"ht{args.counters}c"
    out = grouped_streams.result_dir(base, args.dataset, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    mem = ["budget_MB\tm\tfilter_bits\tht_groups\tht_bits\tht_charged\tbits_used\tbits_budget\tunavailable_k"]
    for mb in args.budgets_mb:
        bits = int(mb * 2**20 * 8)
        if args.ht_offchip:
            m = m_for_bits(bits, args.N, len(K))
            est, n_ht, bad = run_ht(d, m, K, args.N, args.seed)
        else:
            m, (est, n_ht, bad) = fit_charged(d, bits, K, args.N, args.seed)
        np.savez_compressed(out / f"{mb_label(mb)}.npz", est=est.astype(np.float32))
        fb, hb = tnds_bits(m, args.N, len(K)), n_ht * ht_entry_bits(len(K))
        mem.append(f"{mb:g}\t{m}\t{fb}\t{n_ht}\t{hb}\t{int(not args.ht_offchip)}\t"
                   f"{fb + (0 if args.ht_offchip else hb)}\t{bits}\t{','.join(map(str, bad)) or '-'}")
        print(f"{mb:>6g} MB: m = {m:,} cells ({fb / bits:.0%} of budget), hash table {n_ht:,} groups "
              f"({hb / bits:.0%}{', off-chip' if args.ht_offchip else ''}); unavailable k: {bad or 'none'}", flush=True)
    (out / "memory.tsv").write_text("\n".join(mem) + "\n")


if __name__ == "__main__":
    main()
