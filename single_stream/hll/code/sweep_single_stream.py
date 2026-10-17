"""HLL-Difference and HLL-Joint in the single-stream setting: every CAIDA
destination has its own ring of N + 1 HLL arrays. Estimates at epoch N for
k = 1..N, for each per-flow memory budget.

m is the largest power of two with (N + 3) m registers of 6 bits within the
budget (N + 1 arrays plus R_H and R_U); budgets with the same m share a run.
Registers are stored sparsely per flow, and the single-sketch and joint ML
estimates are cached by their sufficient statistics, which repeat often
across small flows.

    python sweep_single_stream.py --seed 1
Writes ../result/single_stream/seed<S>/<B>KB.npz with "difference",
"difference_raw" (before clipping at 0) and "joint" (NaN = fit failed), and
memory.tsv.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common" / "code"))

import flow_streams  # noqa: E402
from hashing import U64, mix64, salt_from_seed  # noqa: E402
from hll_ring import JointStatistic, bit_length_u64, joint_from_statistic, ml_estimate  # noqa: E402

RESULT = HERE.parent / "result" / "single_stream"
DEFAULT_KB = [1] + list(range(10, 101, 10))
REG_BITS = 6

G: dict = {}


def p_for_budget(nbytes: int, N: int) -> int:
    m_max = (8 * nbytes) // ((N + 3) * REG_BITS)
    return int(math.floor(math.log2(m_max)))


def epoch_registers(el, p: int, salt: int):
    """Per epoch: (j, rank, offsets) with each flow's nonzero registers (max rank)."""
    q = 64 - p
    F = len(el[0][2]) - 1
    out = []
    for fi, src, _ in el:
        eta = mix64(src, salt)
        j = (eta >> U64(q)).astype(np.int64)
        rank = (q - bit_length_u64(eta & U64((1 << q) - 1)) + 1).astype(np.int64)
        key = fi * (1 << p) + j
        order = np.lexsort((-rank, key))
        key, rank = key[order], rank[order]
        first = np.ones(len(key), dtype=bool)
        first[1:] = key[1:] != key[:-1]
        key, rank = key[first], rank[first]                    # max rank per (flow, register)
        f = key >> p
        out.append((key & ((1 << p) - 1), rank, np.searchsorted(f, np.arange(F + 1))))
    return out


def run_flows(idx: np.ndarray):
    p, N = G["p"], G["N"]
    q, m = 64 - p, 1 << p
    regs = G["regs"]
    ml_memo, joint_memo = {}, {}

    def ml(counts: np.ndarray) -> float:
        key = counts.tobytes()
        v = ml_memo.get(key)
        if v is None:
            v = ml_memo[key] = ml_estimate(counts, p, q)
        return v

    def counts_of(vals: np.ndarray, n_regs: int) -> np.ndarray:
        c = np.bincount(vals, minlength=q + 2)
        c[0] += m - n_regs
        return c

    diff = np.empty((N, len(idx)))
    raw = np.empty((N, len(idx)))
    joint = np.empty((N, len(idx)))
    fails = 0
    for col, i in enumerate(idx.tolist()):
        j_t, r_t, off = regs[N]
        cur = dict(zip(j_t[off[i]:off[i + 1]].tolist(), r_t[off[i]:off[i + 1]].tolist()))
        hist: dict[int, int] = {}
        for k in range(1, N + 1):
            j_u, r_u, off_u = regs[N - k]
            for jj, rr in zip(j_u[off_u[i]:off_u[i + 1]].tolist(), r_u[off_u[i]:off_u[i + 1]].tolist()):
                if rr > hist.get(jj, 0):
                    hist[jj] = rr
            union = list(cur.keys() | hist.keys())
            k1 = np.array([cur.get(x, 0) for x in union], dtype=np.int64)
            k2 = np.array([hist.get(x, 0) for x in union], dtype=np.int64)
            e_u = ml(counts_of(np.maximum(k1, k2), len(union)))
            e_h = ml(counts_of(np.fromiter(hist.values(), dtype=np.int64, count=len(hist)), len(hist)))
            raw[k - 1, col] = e_u - e_h
            diff[k - 1, col] = max(0.0, e_u - e_h)
            st = JointStatistic(k1, k2, q, m)
            key = st.key()
            res = joint_memo.get(key)
            if res is None:
                la, _, _, ok = joint_from_statistic(st)
                res = joint_memo[key] = la if ok else math.nan
            joint[k - 1, col] = res
            fails += math.isnan(res)
    return idx, diff, raw, joint, fails


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--budgets-kb", type=int, nargs="+", default=DEFAULT_KB)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=18)
    args = ap.parse_args()

    d = flow_streams.build(args.N)
    el = flow_streams.epoch_labels(d)
    salt = salt_from_seed(args.seed)
    F = len(d["flows"])
    out = RESULT / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    by_p: dict[int, list[int]] = {}
    for kb in args.budgets_kb:
        by_p.setdefault(p_for_budget(kb * 1024, args.N), []).append(kb)

    mem = ["budget_KB\tm\tbits_used\tbits_budget\tjoint_failed"]
    for p, kbs in sorted(by_p.items()):
        G.update(p=p, N=args.N, regs=epoch_registers(el, p, salt))
        # spread the few heavy flows across chunks by interleaving
        chunks = [np.arange(s, F, 8 * args.workers) for s in range(8 * args.workers)]
        diff = np.empty((args.N, F))
        raw = np.empty((args.N, F))
        joint = np.empty((args.N, F))
        fails = 0
        with mp.get_context("fork").Pool(args.workers) as pool:
            for idx, dd, rr, jj, ff in pool.imap_unordered(run_flows, chunks):
                diff[:, idx], raw[:, idx], joint[:, idx] = dd, rr, jj
                fails += ff
        m = 1 << p
        for kb in kbs:
            np.savez_compressed(out / f"{kb}KB.npz", difference=diff.astype(np.float32),
                                difference_raw=raw.astype(np.float32), joint=joint.astype(np.float32))
            mem.append(f"{kb}\t{m}\t{(args.N + 3) * m * REG_BITS}\t{8 * kb * 1024}\t{fails}")
        print(f"m = {m:>5} (p={p}) for {kbs} KB: joint solves failed {fails:,}", flush=True)
    rows = [mem[0]] + sorted(mem[1:], key=lambda line: int(line.split()[0]))
    (out / "memory.tsv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
