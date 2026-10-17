"""Theta-Difference in the single-stream setting: every CAIDA destination has
its own ring of N + 1 KMV sketches. Estimates at epoch N for k = 1..N, for
each per-flow memory budget.

c is the largest value with (N + 2) * 64 * (c + 1) bits <= budget (N + 1
sketches and the history union, each with a threshold). A flow with at most
c labels over all epochs stays in exact mode, so its estimate equals the true
count; those flows are filled in directly (a sample is checked against
ThetaRing) and all others are run through ThetaRing.

    python sweep_single_stream.py --seed 1
Writes ../result/single_stream/seed<S>/<B>KB.npz and memory.tsv.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common" / "code"))

import flow_streams  # noqa: E402
from theta_ring import ThetaRing  # noqa: E402

RESULT = HERE.parent / "result" / "single_stream"
DEFAULT_KB = [1] + list(range(10, 101, 10))

G: dict = {}


def c_for_budget(nbytes: int, N: int) -> int:
    return (8 * nbytes) // (64 * (N + 2)) - 1


def run_flows(job):
    c, seed, idx = job
    N, el = G["N"], G["el"]
    out = np.empty((N, len(idx)))
    for col, i in enumerate(idx.tolist()):
        r = ThetaRing(c=c, N=N, seed=seed)
        for u, (_, src, off) in enumerate(el):
            r.process_period(src[off[i]:off[i + 1]], u)
        out[:, col] = [r.query(k) for k in range(1, N + 1)]
    return idx, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--budgets-kb", type=int, nargs="+", default=DEFAULT_KB)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--check-exact", type=int, default=2000,
                    help="exact-mode flows re-run through ThetaRing as a check")
    args = ap.parse_args()

    d = flow_streams.build(args.N)
    el = flow_streams.epoch_labels(d)
    G.update(N=args.N, el=el)
    F = len(d["flows"])
    # labels per flow over all epochs
    allkeys = np.unique(np.concatenate([d[f"hist_{u}"] for u in range(args.N)] + [d["cur_key"]]))
    union = np.bincount(flow_streams.split_key(allkeys)[0], minlength=F)

    out = RESULT / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    mem = ["budget_KB\tc\tbits_used\tbits_budget\tflows_simulated"]
    rng = np.random.default_rng(0)
    with mp.get_context("fork").Pool(args.workers) as pool:
        for kb in args.budgets_kb:
            nbytes = kb * 1024
            c = c_for_budget(nbytes, args.N)
            est = d["true"].astype(np.float64).copy()
            sim = np.flatnonzero(union > c)
            exact = np.flatnonzero(union <= c)
            check = rng.choice(exact, size=min(args.check_exact, len(exact)), replace=False)
            jobs = [(c, args.seed, chunk) for chunk in np.array_split(np.concatenate([sim, check]), 4 * args.workers)
                    if len(chunk)]
            for idx, vals in pool.imap_unordered(run_flows, jobs):
                in_check = np.isin(idx, check)
                if in_check.any():
                    assert np.array_equal(vals[:, in_check], est[:, idx[in_check]]), "exact-mode shortcut mismatch"
                est[:, idx[~in_check]] = vals[:, ~in_check]
            np.savez_compressed(out / f"{kb}KB.npz", est=est.astype(np.float32))
            mem.append(f"{kb}\t{c}\t{(args.N + 2) * 64 * (c + 1)}\t{8 * nbytes}\t{len(sim)}")
            print(f"{kb:>4} KB: c = {c}, simulated {len(sim):,} flows, exact-mode check on "
                  f"{len(check):,} flows passed", flush=True)
    (out / "memory.tsv").write_text("\n".join(mem) + "\n")


if __name__ == "__main__":
    main()
