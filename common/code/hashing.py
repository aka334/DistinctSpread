"""64-bit label hash shared by all methods."""

from __future__ import annotations

import numpy as np

U64 = np.uint64


def mix64(x: np.ndarray, salt: int) -> np.ndarray:
    """splitmix64 finalizer of x ^ salt."""
    z = (np.asarray(x, dtype=U64) ^ U64(salt)) + U64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> U64(30))) * U64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> U64(27))) * U64(0x94D049BB133111EB)
    return z ^ (z >> U64(31))


def salt_from_seed(seed: int) -> int:
    # Draw a full 64-bit salt from the seed. XOR-ing a small seed into small
    # labels would give nearly the same key set for different seeds.
    rng = np.random.default_rng(seed)
    hi, lo = rng.integers(0, 2**32, size=2, dtype=U64)
    return int((hi << U64(32)) | lo)
