"""Traces as a sequence of epochs of (src, dst) uint32 pairs.

    caida      one file oN.txt per one-minute epoch
    mawi       MAWI samplepoint-F, 2026-10-12 14:00-14:15 JST
               (https://mawi.wide.ad.jp/mawi/samplepoint-F/2026/202610121400.html),
               IPv4 only, 60 s epochs from the first packet
    ecommerce  REES46 multi-category store, October 2019 (https://data.rees46.com),
               record <time, user_id, product_id>: group = product, label = user,
               one-day epochs from 2019-10-01 00:00 UTC

MAWI is converted with pcap2bin.c and E-commerce with ecommerce2bin.py into
12-byte records <ts_sec, src, dst>.
"""

from __future__ import annotations

import numpy as np

from paths import CAIDA, TRACES
from reader import load_period, period_files

# file, epoch length (s)
BIN = {"mawi": (TRACES / "mawi" / "202610121400.bin", 60),
       "ecommerce": (TRACES / "ecommerce" / "2019-Oct.bin", 86400)}
DATASETS = ("caida", "mawi", "ecommerce")


def records(name: str) -> np.ndarray:
    return np.memmap(BIN[name][0], dtype="<u4", mode="r").reshape(-1, 3)


def epochs(name: str, n_epochs: int):
    """Yield (src, dst) for epochs 0 .. n_epochs-1, in arrival order."""
    if name == "caida":
        for path in period_files(CAIDA, n_epochs):
            yield load_period(path)
        return
    r = records(name)
    sec = BIN[name][1]
    # epoch from each record's own timestamp (the pcap has a few 1 s steps back)
    ep = (r[:, 0].astype(np.int64) - int(r[0, 0])) // sec
    assert ep.max() >= n_epochs, f"{name}: fewer than {n_epochs} full epochs"
    for t in range(n_epochs):
        rec = np.asarray(r[ep == t])
        yield rec[:, 1].copy(), rec[:, 2].copy()
