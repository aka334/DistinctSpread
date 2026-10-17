"""HB in the single-stream setting: every CAIDA destination has its own ring
of N + 1 m-bit bitmaps. Estimates at epoch N for k = 1..N, for each per-flow
memory budget. The ring and the query mask take (N + 2) m bits, so
m = floor(8 * bytes / (N + 2)). Unavailable estimates are NaN.

    python sweep_single_stream.py --seed 1
Writes ../result/single_stream/seed<S>/<B>KB.npz and memory.tsv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common" / "code"))

import flow_streams  # noqa: E402
from hashing import salt_from_seed  # noqa: E402
from per_flow_cells import cell_counts  # noqa: E402

RESULT = Path(__file__).resolve().parents[1] / "result" / "single_stream"
DEFAULT_KB = [1] + list(range(10, 101, 10))


def m_for_budget(nbytes: int, N: int) -> int:
    return (8 * nbytes) // (N + 2)


def bitmap_estimates(m: int, z: np.ndarray, s: np.ndarray) -> np.ndarray:
    est = np.full(z.shape, np.nan)
    ok = (z > 0) & (s < z)
    est[ok] = -m * np.log1p(-s[ok] / z[ok])
    return est


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
        m = m_for_budget(nbytes, args.N)
        z, s, _ = cell_counts(d, m, salt)
        est = bitmap_estimates(m, z, s)
        np.savez_compressed(out / f"{kb}KB.npz", est=est.astype(np.float32))
        mem.append(f"{kb}\t{m}\t{(args.N + 2) * m}\t{8 * nbytes}")
        print(f"{kb:>4} KB: m = {m:,} bits/bitmap, unavailable {np.isnan(est).sum():,}", flush=True)
    (out / "memory.tsv").write_text("\n".join(mem) + "\n")


if __name__ == "__main__":
    main()
