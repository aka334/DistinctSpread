"""Check that the sparse sweep in sweep_single_stream.py gives the same
estimates as HllRing on a sample of CAIDA flows."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "common" / "code"))

import flow_streams  # noqa: E402
from hashing import salt_from_seed  # noqa: E402
from hll_ring import HllRing  # noqa: E402

spec = importlib.util.spec_from_file_location("hll_sweep", HERE / "sweep_single_stream.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def main() -> None:
    N, seed = 12, 4
    d = flow_streams.build(N)
    el = flow_streams.epoch_labels(d)
    true1 = d["true"][0]
    rng = np.random.default_rng(1)
    sample = []
    for lo, hi, n in [(0, 2, 40), (2, 10, 40), (10, 100, 40), (100, 1000, 30), (1000, 10**9, 10)]:
        idx = np.flatnonzero((true1 >= lo) & (true1 < hi))
        sample += rng.choice(idx, size=min(n, len(idx)), replace=False).tolist()
    sample = np.array(sorted(sample))
    salt = salt_from_seed(seed)
    for p in (6, 10):
        sweep.G.update(p=p, N=N, regs=sweep.epoch_registers(el, p, salt))
        _, diff, raw, joint, _ = sweep.run_flows(sample)
        for col, i in enumerate(sample.tolist()):
            ring = HllRing(p=p, N=N, seed=seed)
            for u, (_, src, off) in enumerate(el):
                ring.process_period(src[off[i]:off[i + 1]], u)
            for k in range(1, N + 1):
                clip, r = ring.query_difference(k)
                jt = ring.query_joint(k)
                assert math.isclose(raw[k - 1, col], r, rel_tol=1e-9, abs_tol=1e-9), (p, i, k)
                assert math.isclose(diff[k - 1, col], clip, rel_tol=1e-9, abs_tol=1e-9), (p, i, k)
                if jt is None:
                    assert math.isnan(joint[k - 1, col])
                else:
                    assert math.isclose(joint[k - 1, col], jt, rel_tol=1e-9, abs_tol=1e-9), (p, i, k, joint[k - 1, col], jt)
        print(f"p={p:>2}: {len(sample)} flows x {N} histories identical to HllRing "
              f"(difference, raw difference, joint)  ok")
    print("all tests passed")


if __name__ == "__main__":
    main()
