"""HLL-Difference and HLL-Joint baselines for one stream.

A ring of N + 1 HLL arrays of m = 2^p registers, one per epoch; entering an
epoch clears its slot (and those of skipped epochs). For item x with 64-bit
hash eta(x), the register is the top p bits and the rank is 1 + the leading
zeros of the remaining q = 64 - p bits (q + 1 if they are all zero).

Query k at epoch t:
    R_H = max(R_{t-1}, ..., R_{t-k}),  R_U = max(R_t, R_H)
    HLL-Difference:  max(0, E(R_U) - E(R_H))
    HLL-Joint:       lambda_a of the joint ML fit on (R_t, R_H)

E() is Ertl's maximum-likelihood estimator for one sketch, and the joint
estimator follows his two-sketch ML estimation (same initial values,
likelihood and stopping rule), with SciPy's BFGS. Both are ported from the
C++ code of O. Ertl, "New cardinality estimation algorithms for HyperLogLog
sketches", arXiv:1702.01284 (github.com/oertl/hyperloglog-sketch-estimation-paper).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common" / "code"))

from hashing import U64, mix64, salt_from_seed  # noqa: E402



def bit_length_u64(w: np.ndarray) -> np.ndarray:
    """Bit length of uint64 values, without going through floats."""
    w = w.astype(U64).copy()
    n = np.zeros(w.shape, dtype=np.int64)
    for s in (32, 16, 8, 4, 2, 1):
        big = w >= (U64(1) << U64(s))
        n[big] += s
        w[big] >>= U64(s)
    return n + (w > 0)



def ml_estimate(c: np.ndarray, p: int, q: int) -> float:
    """Ertl's single-sketch ML estimate from the register histogram
    c = (C_0, ..., C_{q+1})."""
    m = 1 << p
    c = [int(v) for v in c]
    if c[q + 1] == m:
        return math.inf
    k_min = next(k for k in range(q + 2) if c[k] > 0)
    k_min_p = max(1, k_min)
    k_max = next(k for k in range(q + 1, -1, -1) if c[k] > 0)
    k_max_p = min(q, k_max)

    z = 0.0
    for k in range(k_max_p, k_min_p - 1, -1):
        z = 0.5 * z + c[k]
    z = math.ldexp(z, -k_min_p)

    c_prime = c[q + 1] + (c[k_max_p] if q >= 1 else 0)
    g_prev = 0.0
    a = z + c[0]
    m_prime = m - c[0]
    b = z + math.ldexp(c[q + 1], -q)
    if b <= 1.5 * a:
        x = m_prime / (0.5 * b + a)               # weak lower bound (47)
    else:
        x = (m_prime / b) * math.log1p(b / a)     # strong lower bound (46)

    rel_lim = 1e-2 / math.sqrt(m)
    dx = x
    while dx > x * rel_lim:
        _, kappa_m1 = math.frexp(x)
        xp = math.ldexp(x, -max(k_max_p + 1, kappa_m1 + 2))
        xp2 = xp * xp
        h = xp - xp2 / 3 + (xp2 * xp2) * (1.0 / 45.0 - xp2 / 472.5)
        for _ in range(kappa_m1, k_max_p - 1, -1):
            hp = 1.0 - h
            h = (xp + h * hp) / (xp + hp)
            xp += xp
        g = c_prime * h
        for k in range(k_max_p - 1, k_min_p - 1, -1):
            hp = 1.0 - h
            h = (xp + h * hp) / (xp + hp)
            xp += xp
            g += c[k] * h
        g += x * a
        if g_prev < g <= m_prime:
            dx *= (g - m_prime) / (g_prev - g)
        else:
            dx = 0.0
        x += dx
        g_prev = g
    return x * m



ZERO_PHI = 2 * math.log(np.finfo(float).tiny)   # exp(ZERO_PHI) == 0, as in Ertl


class JointStatistic:
    """Sufficient statistic (Eq. 71) of a register pair (K1, K2): counts of
    k = K1 = K2 (equal), k = K1 < K2 (smaller1), k = K1 > K2 (larger1), etc."""

    def __init__(self, k1: np.ndarray, k2: np.ndarray, q: int, m: int | None = None) -> None:
        """k1, k2: register values of the same indices in both sketches. With m
        given, k1/k2 may list only registers nonzero in either sketch; the other
        m - len(k1) registers are zero in both (counted as equal at 0)."""
        size = q + 2
        lt, gt, eq = k1 < k2, k1 > k2, k1 == k2
        bc = lambda v: np.bincount(v, minlength=size).astype(np.int64)
        self.smaller1, self.larger2 = bc(k1[lt]), bc(k2[lt])
        self.larger1, self.smaller2 = bc(k1[gt]), bc(k2[gt])
        self.equal = bc(k1[eq])
        self.q, self.m = q, len(k1) if m is None else m
        self.equal[0] += self.m - len(k1)
        self.p = int(round(math.log2(self.m)))

    def key(self) -> bytes:
        """Hashable key, for caching joint estimates."""
        return np.concatenate([self.equal, self.smaller1, self.larger1,
                               self.smaller2, self.larger2]).tobytes()

    def counts1(self):
        return self.equal + self.smaller1 + self.larger1

    def counts2(self):
        return self.equal + self.smaller2 + self.larger2

    def counts_max(self):
        return self.equal + self.larger1 + self.larger2

    def counts_min(self):
        return self.equal + self.smaller1 + self.smaller2


def _calc_exp(x: np.ndarray):
    """y = e^-x, z = 1 - y, xy = x * y."""
    small = x < math.log(2)
    z = np.where(small, -np.expm1(-x), 1.0 - np.exp(-x))
    y = np.where(small, 1.0 - z, np.exp(-x))
    xy = np.where(y > 0, x * y, 0.0)
    return y, z, xy


class JointLogLikelihood:
    """Joint log-likelihood f(phi_a, phi_b, phi_x) and its gradient."""

    def __init__(self, st: JointStatistic) -> None:
        q, p = st.q, st.p
        self.st = st
        ks = np.arange(1, q + 2)
        self.ks = ks
        # pow2k[k] = 1/(m 2^min(k,q))
        self.pow2k = np.ldexp(1.0, -(np.minimum(ks, q) + p))
        w = np.ldexp(1.0, -np.arange(q + 1))            # 2^-k for k = 0..q
        m = st.m
        self.lin_a = float((st.counts1()[: q + 1] * w).sum()) / m
        self.lin_b = float((st.counts2()[: q + 1] * w).sum()) / m
        self.lin_x = float((st.counts_min()[: q + 1] * w).sum()) / m
        self.s1, self.l1 = st.smaller1[ks], st.larger1[ks]
        self.s2, self.l2 = st.smaller2[ks], st.larger2[ks]
        self.eq = st.equal[ks]

    def evaluate(self, phi: np.ndarray):
        ea, eb, ex = np.exp(phi)
        ya, za, xaya = _calc_exp(ea * self.pow2k)
        yb, zb, xbyb = _calc_exp(eb * self.pow2k)
        yx, zx, xxyx = _calc_exp(ex * self.pow2k)
        f = fa = fb = fx = 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.s1.any():
                arg = zx + yx * za
                sel = self.s1 > 0
                f += float((self.s1[sel] * np.log(arg[sel])).sum())
                tmp = np.where(sel, self.s1 / arg, 0.0)
                fa += float((tmp * yx * xaya).sum())
                fx += float((tmp * ya * xxyx).sum())
            if self.l1.any():
                sel = self.l1 > 0
                f += float((self.l1[sel] * np.log(za[sel])).sum())
                fa += float(np.where(sel, self.l1 * xaya / za, 0.0).sum())
            if self.s2.any():
                arg = zx + yx * zb
                sel = self.s2 > 0
                f += float((self.s2[sel] * np.log(arg[sel])).sum())
                tmp = np.where(sel, self.s2 / arg, 0.0)
                fb += float((tmp * yx * xbyb).sum())
                fx += float((tmp * yb * xxyx).sum())
            if self.l2.any():
                sel = self.l2 > 0
                f += float((self.l2[sel] * np.log(zb[sel])).sum())
                fb += float(np.where(sel, self.l2 * xbyb / zb, 0.0).sum())
            if self.eq.any():
                arg = za * zb * yx + zx
                sel = self.eq > 0
                f += float((self.eq[sel] * np.log(arg[sel])).sum())
                tmp = np.where(sel, self.eq / arg, 0.0)
                fa += float((tmp * yx * zb * xaya).sum())
                fb += float((tmp * yx * za * xbyb).sum())
                fx += float((tmp * (ya + yb * za) * xxyx).sum())
        la, lb, lx = self.lin_a * ea, self.lin_b * eb, self.lin_x * ex
        f -= la + lb + lx
        return f, np.array([fa - la, fb - lb, fx - lx])


class _Converged(Exception):
    pass


def joint_estimate(k1: np.ndarray, k2: np.ndarray, q: int):
    """(lambda_a, lambda_b, lambda_x, ok) for S1 (registers k1), S2 (k2):
    lambda_a = |S1 \\ S2|, lambda_b = |S2 \\ S1|, lambda_x = |S1 & S2|.
    ok = False when the solve fails (reported as unavailable)."""
    return joint_from_statistic(JointStatistic(k1, k2, q))


def joint_from_statistic(st: JointStatistic):
    """joint_estimate on a prebuilt sufficient statistic."""
    p, m, q = st.p, st.m, st.q
    ax = ml_estimate(st.counts1(), p, q)
    bx = ml_estimate(st.counts2(), p, q)

    # initial values
    if st.counts_min()[0] == m:                       # disjoint: exact optimum
        return ax, bx, 0.0, True
    if not st.larger1.any():
        a, b, x = 0.0, max(0.0, bx - ax), ax
    elif not st.larger2.any():
        a, b, x = max(0.0, ax - bx), 0.0, bx
    else:
        abx = ml_estimate(st.counts_max(), p, q)
        a, b, x = max(1.0, abx - bx), max(1.0, abx - ax), max(1.0, ax + bx - abx)
    if not all(math.isfinite(v) for v in (a, b, x)):
        return math.nan, math.nan, math.nan, False
    phi0 = np.array([math.log(v) if v > 0 else ZERO_PHI for v in (a, b, x)])

    ll = JointLogLikelihood(st)
    delta = 1e-2 / math.sqrt(m)
    prev = {"x": None}

    def neg(phi):
        f, g = ll.evaluate(phi)
        return -f, -g

    def stop(intermediate_result):
        xk = intermediate_result.x
        if prev["x"] is not None and np.max(np.abs(xk - prev["x"])) <= delta:
            raise StopIteration
        prev["x"] = xk.copy()

    try:
        res = minimize(neg, phi0, jac=True, method="BFGS", callback=stop,
                       options={"gtol": 0.0, "maxiter": 1000})
        phi = res.x
    except Exception:
        return math.nan, math.nan, math.nan, False
    if not np.all(np.isfinite(phi)):
        return math.nan, math.nan, math.nan, False
    la, lb, lx = (float(v) for v in np.exp(phi))
    return la, lb, lx, True



class HllRing:
    """Ring of N+1 epoch HLL arrays (m = 2^p registers, 64-bit hash)."""

    def __init__(self, p: int, N: int, seed: int = 0x9E3779B9) -> None:
        if not 4 <= p <= 30:
            raise ValueError("p out of range")
        self.p, self.m, self.N = p, 1 << p, N
        self.q = 64 - p
        self.nu = self.q + 1                       # max rank, = 65 - log2 m
        self.salt = salt_from_seed(seed)
        self.ring = np.zeros((N + 1, self.m), dtype=np.uint8)
        self.t: int | None = None

    def _slot(self, u: int) -> int:
        return u % (self.N + 1)

    def advance(self, t: int) -> None:
        if self.t is not None and t < self.t:
            raise ValueError("epochs must be nondecreasing")
        if self.t == t:
            return
        start = t if self.t is None else self.t + 1
        for u in range(max(start, t - self.N), t + 1):
            self.ring[self._slot(u)] = 0
        self.t = t

    def process_period(self, elem: np.ndarray, t: int) -> None:
        self.advance(t)
        if not len(elem):
            return
        eta = mix64(elem, self.salt)
        j = (eta >> U64(self.q)).astype(np.int64)
        w = eta & U64((1 << self.q) - 1)
        rank = (self.q - bit_length_u64(w) + 1).astype(np.uint8)   # q + 1 if w == 0
        np.maximum.at(self.ring[self._slot(t)], j, rank)

    def _hist_union(self, k: int) -> np.ndarray:
        if not 1 <= k <= self.N:
            raise ValueError(f"k must be in 1..N={self.N}")
        rh = np.zeros(self.m, dtype=np.uint8)
        for i in range(1, k + 1):
            np.maximum(rh, self.ring[self._slot(self.t - i)], out=rh)
        return rh

    def counts(self, regs: np.ndarray) -> np.ndarray:
        return np.bincount(regs, minlength=self.q + 2)

    def query_difference(self, k: int) -> tuple[float, float]:
        """HLL-Difference: (max(0, raw), raw) with raw = E(R_U) - E(R_H)."""
        rt = self.ring[self._slot(self.t)]
        rh = self._hist_union(k)
        ru = np.maximum(rt, rh)
        raw = ml_estimate(self.counts(ru), self.p, self.q) - ml_estimate(self.counts(rh), self.p, self.q)
        return max(0.0, raw), raw

    def query_joint(self, k: int) -> float | None:
        """HLL-Joint: lambda_1 = |A(t) \\ H_k(t)|; None if the solve fails."""
        rt = self.ring[self._slot(self.t)]
        rh = self._hist_union(k)
        la, _, _, ok = joint_estimate(rt, rh, self.q)
        return la if ok else None

    def register_bits(self) -> int:
        return math.ceil(math.log2(self.nu + 1))

    def memory_bits(self, workspace: bool = True) -> int:
        stored = (self.N + 1) * self.m * self.register_bits()
        return stored + (2 * self.m * self.register_bits() if workspace else 0)   # R_H, R_U
