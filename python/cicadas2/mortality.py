"""Port of ``functions/fcn_mortality.m``.

Discrete-time daily logistic hazard mortality model::

    z_d = theta0 + theta1*(d/Tmax)**2
          + (theta2 + theta3 * SOFA/10) * (L_cum(d)/L_norm)
          + (theta4 + theta5 * Age/65)  * (X_cum(d)/X_norm)
    h_d = 1 / (1 + exp(-z_d))

At each day, draw u ~ Uniform(0,1) and declare death if u < h_d.

The MATLAB version uses ``rand()`` internally. To get bit-identical
output on the Python side, callers can pass a pre-drawn uniform tape
via the ``u_draws`` argument; see ``_matlab_compat.py`` for the
rationale.
"""

from __future__ import annotations

import numpy as np

from .params import default_params


def run(
    lambda_obs: np.ndarray,
    x:          np.ndarray,
    dt:         float,
    theta:      np.ndarray,
    age:        float,
    sofa:       float,
    mort_params: dict | None = None,
    u_draws:     np.ndarray | None = None,
) -> tuple[int, float]:
    """Simulate per-patient survival under the daily-hazard model.

    Parameters
    ----------
    lambda_obs
        Observed seizure rate, ``[Nt]``.
    x
        Drug concentration, ``[Nt]``.
    dt
        Time step (days).
    theta
        Mortality coefficients, length 6.
    age
        Patient age (years).
    sofa
        SOFA score (0-24).
    mort_params
        Optional dict with ``Tmax``, ``L_norm``, ``X_norm``; defaults
        from ``params.default_params()`` if omitted.
    u_draws
        Optional length-``n_days`` array of uniforms for RNG injection.
        If omitted, draws fresh uniforms via ``np.random.uniform``.

    Returns
    -------
    alive
        1 if survived through ``Tmax``, 0 if died.
    death_time
        Day of death (1-based, mirroring MATLAB), ``nan`` if survived.
    """
    if mort_params is None:
        _, _, _, mort_params = default_params()

    Nt = lambda_obs.size
    theta0, theta1, theta2, theta3, theta4, theta5 = theta

    Tmax   = mort_params["Tmax"]
    L_norm = mort_params["L_norm"]
    X_norm = mort_params["X_norm"]

    sofa_norm = sofa / 10
    age_norm  = age / 65

    L_cum = np.cumsum(lambda_obs) * dt
    X_cum = np.cumsum(x) * dt

    n_days = int(np.ceil(Nt * dt))

    if u_draws is None:
        u_draws = np.random.uniform(size=n_days)
    else:
        u_draws = np.asarray(u_draws)

    return _run_hazard_loop(
        L_cum, X_cum, dt, theta, sofa_norm, age_norm,
        Tmax, L_norm, X_norm, u_draws, n_days, burden_power=1,
    )


def run_squared(
    lambda_obs: np.ndarray,
    x:          np.ndarray,
    dt:         float,
    theta:      np.ndarray,
    age:        float,
    sofa:       float,
    mort_params: dict | None = None,
    u_draws:     np.ndarray | None = None,
) -> tuple[int, float]:
    """Squared-burden mortality variant for misspecification experiments.

    Identical to :func:`run` except the burden terms enter the
    log-odds as ``(L_cum/L_norm)**2`` and ``(X_cum/X_norm)**2`` rather
    than linearly. Mirrors MATLAB's ``fcn_mortality_squared.m``.
    """
    if mort_params is None:
        _, _, _, mort_params = default_params()

    Nt = lambda_obs.size
    Tmax   = mort_params["Tmax"]
    L_norm = mort_params["L_norm"]
    X_norm = mort_params["X_norm"]

    sofa_norm = sofa / 10
    age_norm  = age / 65

    L_cum = np.cumsum(lambda_obs) * dt
    X_cum = np.cumsum(x) * dt

    n_days = int(np.ceil(Nt * dt))
    if u_draws is None:
        u_draws = np.random.uniform(size=n_days)
    else:
        u_draws = np.asarray(u_draws)

    return _run_hazard_loop(
        L_cum, X_cum, dt, theta, sofa_norm, age_norm,
        Tmax, L_norm, X_norm, u_draws, n_days, burden_power=2,
    )


def _run_hazard_loop(L_cum, X_cum, dt, theta, sofa_norm, age_norm,
                     Tmax, L_norm, X_norm, u_draws, n_days,
                     burden_power):
    """Day-by-day hazard loop shared by :func:`run` and :func:`run_squared`.

    ``burden_power`` is 1 (linear, the default model) or 2 (squared,
    the misspecification variant). Kept as one function rather than
    duplicating the loop body so the two MATLAB-side variants stay in
    visible sync.
    """
    theta0, theta1, theta2, theta3, theta4, theta5 = theta
    Nt = L_cum.size

    for d in range(1, n_days + 1):           # MATLAB: for d = 1:n_days
        idx_matlab = min(int(round(d / dt)), Nt)
        idx = idx_matlab - 1                  # MATLAB 1-indexed → Python 0-indexed

        t_frac = (d / Tmax) ** 2

        L_term = (L_cum[idx] / L_norm) ** burden_power
        X_term = (X_cum[idx] / X_norm) ** burden_power

        z = (theta0
             + theta1 * t_frac
             + (theta2 + theta3 * sofa_norm) * L_term
             + (theta4 + theta5 * age_norm)  * X_term)

        h = 1.0 / (1.0 + np.exp(-z))

        if u_draws[d - 1] < h:
            return 0, float(d)

    return 1, float("nan")
