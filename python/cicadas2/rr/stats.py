"""Uncertainty for every reported number (R&R task 5, R1 M7).

Survival is exact per patient, so the remaining Monte Carlo error is patient
sampling (plus within-run seizure-count randomness, which is part of each
patient's value). MC-SE = SD/sqrt(n). Strategy contrasts are paired: the same
patients, the same uniform tape, resampled jointly in the bootstrap.
"""
from __future__ import annotations

import numpy as np

Z95 = 1.959963984540054


def mean_ci(x):
    x = np.asarray(x, float)
    m = 100 * x.mean()
    se = 100 * x.std(ddof=1) / np.sqrt(x.size)
    return dict(est=m, se=se, lo=m - Z95 * se, hi=m + Z95 * se)


def paired(a, b, B=2000, seed=1):
    a = np.asarray(a, float); b = np.asarray(b, float)
    D = a - b
    n = D.size
    est = 100 * D.mean()
    sd = 100 * D.std(ddof=1)
    se = sd / np.sqrt(n)
    idx = np.random.default_rng(seed).integers(0, n, size=(B, n))
    bm = 100 * D[idx].mean(axis=1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return dict(est=est, se=se, sd=sd, lo_n=est - Z95 * se, hi_n=est + Z95 * se, lo=lo, hi=hi, n=n, B=B)


def bootstrap_stat(fn, n, B=2000, seed=1):
    """Percentile CI of any statistic fn(idx) over patient resamples."""
    g = np.random.default_rng(seed)
    vals = np.array([fn(g.integers(0, n, size=n)) for _ in range(B)])
    return dict(lo=float(np.percentile(vals, 2.5)), hi=float(np.percentile(vals, 97.5)),
                se=float(vals.std(ddof=1)))


def p_best(S, B=2000, seed=1):
    """Share of patient resamples in which each column has the highest mean."""
    S = np.asarray(S, float)
    n, K = S.shape
    idx = np.random.default_rng(seed).integers(0, n, size=(B, n))
    w = np.zeros(K)
    for b in range(B):
        w[np.argmax(S[idx[b]].mean(axis=0))] += 1
    return w / B


def evalue_rr(rr):
    """VanderWeele-Ding E-value for a risk ratio (applied to the ratio of death risks)."""
    rr = float(rr)
    if rr < 1:
        rr = 1.0 / rr
    return rr + np.sqrt(rr * (rr - 1.0))


def n_for_halfwidth(sd_pp, hw_pp=1.0):
    return int(np.ceil((Z95 * sd_pp / hw_pp) ** 2))
