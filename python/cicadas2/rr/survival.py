"""Exact survival under the daily logistic hazard (R&R task 5, G10, R1 M2b).

mortality.run draws death day by day with probability h_d, so the mean of
its 0/1 output is exactly S = prod_d (1 - h_d). Averaging M draws estimates
S with extra noise sd = sqrt(S(1-S)/M) and nothing else; computing S directly
removes that noise (R1 M7: no more "100 vs 1,000 mortality draws").

n_days defaults to 7 (Tmax, as the paper defines). The original loop runs
ceil(Nt*dt) = 8 hazard days because Nt*dt = 7.0035 (G10); pass n_days=8 to
reproduce the submitted numbers.

Seizure-burden shapes g(u), u = L_cum/35, all equal to 1 at u = 1 (5 sz/day
for 7 days) so the coefficients keep their meaning across shapes:
  linear      g = u                        (the paper's model)
  squared     g = u^2                      (seizure term only; the submitted stress test
                                            squared BOTH terms: use drug_shape="squared" too)
  threshold   g = max(0, u - 0.5) / 0.5    (no harm below half the reference burden)
  saturating  g = 2u / (1 + u)             (harm flattens at high burden)
  piecewise   g = 0.5u below u = 0.5, slope 1.5 above (kink at half the reference)
"""
from __future__ import annotations

import numpy as np

L_NORM, X_NORM, TMAX = 35.0, 5000.0, 7.0

SHAPES = {
    "linear":     lambda u: u,
    "squared":    lambda u: u ** 2,
    "threshold":  lambda u: np.maximum(0.0, u - 0.5) / 0.5,
    "saturating": lambda u: 2.0 * u / (1.0 + u),
    "piecewise":  lambda u: np.where(u < 0.5, 0.5 * u, 0.25 + 1.5 * (u - 0.5)),
}


def hazard_logit(Lday, Xday, age, sofa, theta, shape="linear", drug_shape="linear",
                 n_days=7, z_extra=None, time_power=2):
    """Daily log-odds z[..., d] for d = 1..n_days. Lday/Xday: (n, D) or (n, K, D)."""
    Lday = np.asarray(Lday, float)[..., :n_days]
    Xday = np.asarray(Xday, float)[..., :n_days]
    d = np.arange(1, n_days + 1)
    extra_dims = Lday.ndim - 1
    sofa_n = np.asarray(sofa, float).reshape((-1,) + (1,) * extra_dims) / 10
    age_n = np.asarray(age, float).reshape((-1,) + (1,) * extra_dims) / 65
    gL = SHAPES[shape] if isinstance(shape, str) else shape
    gX = SHAPES[drug_shape] if isinstance(drug_shape, str) else drug_shape
    z = (theta[0] + theta[1] * (d / TMAX) ** time_power
         + (theta[2] + theta[3] * sofa_n) * gL(Lday / L_NORM)
         + (theta[4] + theta[5] * age_n) * gX(Xday / X_NORM))
    if z_extra is not None:
        z = z + np.asarray(z_extra, float).reshape((-1,) + (1,) * extra_dims)
    return z


def exact(Lday, Xday, age, sofa, theta, shape="linear", drug_shape="linear", n_days=7, z_extra=None, time_power=2):
    z = hazard_logit(Lday, Xday, age, sofa, theta, shape, drug_shape, n_days, z_extra, time_power)
    h = 1.0 / (1.0 + np.exp(-z))
    return np.prod(1.0 - h, axis=-1)


def draw_death_day(Lday, Xday, age, sofa, theta, u, shape="linear", n_days=7, z_extra=None, time_power=2, drug_shape="linear"):
    """One observed death per patient (0 = survived), from per-patient uniforms u (n, n_days)."""
    z = hazard_logit(Lday, Xday, age, sofa, theta, shape, drug_shape, n_days, z_extra, time_power)
    h = 1.0 / (1.0 + np.exp(-z))
    died = u[:, :n_days] < h
    first = np.where(died.any(axis=1), died.argmax(axis=1) + 1, 0)
    return first
