"""TNDS in the single-stream setting: every CAIDA destination has its own
m-cell timestamp array. Estimates at epoch N for k = 1..N, for each per-flow
memory budget.

m is the largest value with Tnds.memory_bits(m) <= budget. With a private
array the s_k admissions at epoch N have weights m/z_k, m/(z_k - 1), ..., so

    C_k = m (psi(z_k + 1) - psi(z_k - s_k + 1)),

and the estimate is unavailable (NaN) if an item arrives while z_k = 0.

    python sweep_single_stream.py --seed 1
Writes ../result/single_stream/seed<S>/<B>KB.npz and memory.tsv.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import digamma

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common" / "code"))

import flow_streams  # noqa: E402
from hashing import salt_from_seed  # noqa: E402
from per_flow_cells import cell_counts  # noqa: E402

RESULT = Path(__file__).resolve().parents[1] / "result" / "single_stream"
DEFAULT_KB = [1] + list(range(10, 101, 10))


def tnds_bits(m: int, N: int, n_k: int) -> int:
    """Same accounting as tnds.Tnds.memory_bits."""
    w = math.ceil(math.log2(2 * N + 2))
    cnt = math.ceil(math.log2(m + 1))
    return m * w + (N + 1) * cnt + n_k * (cnt + 1 + 64)


def m_for_budget(nbytes: int, N: int, n_k: int) -> int:
    w = math.ceil(math.log2(2 * N + 2))
    m = (8 * nbytes) // w
    while m > 1 and tnds_bits(m, N, n_k) > 8 * nbytes:
        m -= 1
    return m


def tnds_estimates(m: int, z: np.ndarray, s: np.ndarray, last_adm: np.ndarray,
                   cur_last: np.ndarray) -> np.ndarray:
    est = m * (digamma(z + 1.0) - digamma(z - s + 1.0))
    est = np.where(s == 0, 0.0, est)
    invalid = (z == 0) | ((s == z) & (last_adm < cur_last[None, :]))
    return np.where(invalid, np.nan, est)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--budgets-kb", type=int, nargs="+", default=DEFAULT_KB)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    d = flow_streams.build(args.N)
    salt = salt_from_seed(args.seed)
    out = RESULT / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    mem = ["budget_KB\tm\tbits_used\tbits_budget"]
    for kb in args.budgets_kb:
        nbytes = kb * 1024
        m = m_for_budget(nbytes, args.N, args.N)
        z, s, last_adm = cell_counts(d, m, salt)
        est = tnds_estimates(m, z, s, last_adm, d["cur_last"])
        np.savez_compressed(out / f"{kb}KB.npz", est=est.astype(np.float32))
        mem.append(f"{kb}\t{m}\t{tnds_bits(m, args.N, args.N)}\t{8 * nbytes}")
        print(f"{kb:>4} KB: m = {m:,} cells, unavailable {np.isnan(est).sum():,}", flush=True)
    (out / "memory.tsv").write_text("\n".join(mem) + "\n")


if __name__ == "__main__":
    main()
