"""Loader for the CAIDA epoch files oN.txt (one "src dst" pair of dotted-quad
addresses per packet). Each file is parsed once and cached as .npz."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paths import CACHE


def _ip_to_u32(col: pd.Series) -> np.ndarray:
    parts = col.str.split(".", expand=True).astype(np.uint32).to_numpy()
    return (parts[:, 0] << 24) | (parts[:, 1] << 16) | (parts[:, 2] << 8) | parts[:, 3]


def load_period(path: str | Path, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(src, dst) as uint32 arrays, one entry per packet, in file order."""
    path = Path(path)
    cache = CACHE / "caida" / (path.stem + ".npz")
    if use_cache and cache.exists():
        data = np.load(cache)
        return data["src"], data["dst"]

    df = pd.read_csv(path, sep=r"\s+", header=None, names=["src", "dst"], dtype=str, engine="c")
    src, dst = _ip_to_u32(df["src"]), _ip_to_u32(df["dst"])
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, src=src, dst=dst)
    return src, dst


def period_files(caida_dir: str | Path, count: int = 62) -> list[Path]:
    """o1.txt .. o<count>.txt in numeric order."""
    return [Path(caida_dir) / f"o{i}.txt" for i in range(1, count + 1)]
