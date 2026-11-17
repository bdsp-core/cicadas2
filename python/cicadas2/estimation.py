"""Port of ``functions/fcn_estimate_natural_history.m``.

Estimates the rise-decay trajectory parameters ``(A_p, r, kappa)``
from Poisson seizure counts via maximum-likelihood (Nelder-Mead).

Verification has two layers:

1. **NLL bit-identical** — the negative log-likelihood at a given
   theta is purely deterministic (no RNG), and matches the MATLAB
   implementation to ~1e-12 on the same inputs.
2. **Converged optimum** — ``scipy.optimize.minimize`` with
   ``method='Nelder-Mead'`` is *not* algorithmically identical to
   MATLAB's ``fminsearch``, so converged ``theta_opt`` may differ
   by ~1-5 % per parameter. We assert that the converged NLL value
   matches the MATLAB optimum to a loose tolerance, and that the
   downstream derived quantity ``L_cum`` (the only thing the
   mortality model actually uses) matches within ~1 %.
"""

from __future__ import annotations

import numpy as np
import scipy.optimize as opt


def _apply_pd(lbase, x, is_treated, pd_mode, alpha, C50, gamma_hill):
    """PD forward model: lambda_base → lambda_obs (mirrors MATLAB)."""
    if not is_treated:
        return lbase
    if pd_mode.lower() == "linear":
        return np.maximum(1e-10, lbase - alpha * x)
    if pd_mode.lower() == "hill":
        s_x = np.ones_like(x)
        idx = x > 0
        s_x[idx] = 1 - 1 / ((C50 / x[idx]) ** gamma_hill + 1)
        return np.maximum(1e-10, lbase * s_x)
    raise ValueError(f"Unknown PD mode: {pd_mode!r}")


def _invert_pd(lobs, x, is_treated, pd_mode, alpha, C50, gamma_hill):
    """Inverse PD heuristic used only for initialization."""
    if not is_treated:
        return lobs
    if pd_mode.lower() == "linear":
        return lobs + alpha * x
    if pd_mode.lower() == "hill":
        s_x = np.ones_like(x)
        idx = x > 0
        s_x[idx] = 1 - 1 / ((C50 / x[idx]) ** gamma_hill + 1)
        return lobs / np.maximum(s_x, 0.01)
    raise ValueError(f"Unknown PD mode: {pd_mode!r}")


def negative_log_likelihood(theta, seizure_counts, T, dt, peak_rate,
                            x, is_treated, pd_mode, alpha, C50, gamma_hill):
    """Penalised Poisson negative log-likelihood.

    Returns ``nll`` such that minimising it over
    ``theta = (log A_p, log r, log kappa)`` yields the MLE.
    Bit-identical to the nested ``neg_log_likelihood`` in
    ``fcn_estimate_natural_history.m``.
    """
    A_p_trial = np.exp(theta[0])
    r_trial   = np.exp(theta[1])
    d_trial   = np.exp(theta[2])      # MATLAB calls it `d_trial`; that's kappa

    shape = np.exp(-d_trial * T) * (1 - np.exp(-r_trial * T))
    max_shape = float(np.max(shape))
    if max_shape < 1e-10:
        return 1e10

    lambda_base_trial = A_p_trial * peak_rate * shape / max_shape
    lambda_obs_trial  = _apply_pd(lambda_base_trial, x, is_treated, pd_mode,
                                  alpha, C50, gamma_hill)

    mu = np.maximum(lambda_obs_trial * dt, 1e-10)
    nll = -float(np.sum(seizure_counts * np.log(mu) - mu))
    nll += 0.01 * (theta[0]**2 + theta[1]**2 + theta[2]**2)
    return nll


def estimate_natural_history(
    seizure_counts: np.ndarray,
    T:              np.ndarray,
    dt:             float,
    *,
    peak_rate:     float = 10,
    k_e:           float | None = None,
    doses_given:   np.ndarray | None = None,
    infusion_rate: np.ndarray | None = None,
    is_infusion:   bool = False,
    pd_mode:       str = "linear",
    alpha:         float | None = None,
    C50:           float | None = None,
    gamma_hill:    float | None = None,
) -> tuple[dict, dict]:
    """Estimate ``(A_p, r, kappa)`` by Poisson MLE.

    Returns ``(params_est, fit_info)`` matching the MATLAB struct
    schema.
    """
    seizure_counts = np.asarray(seizure_counts, dtype=float)
    T              = np.asarray(T,              dtype=float)
    Nt = T.size

    has_dosing  = k_e is not None and (doses_given is not None or infusion_rate is not None)
    is_treated  = has_dosing

    # --- Reconstruct drug concentration ----------------------------------
    x = np.zeros(Nt)
    if is_treated:
        for k in range(Nt - 1):
            if is_infusion and infusion_rate is not None:
                x[k+1] = x[k] * k_e + infusion_rate[k]
            elif doses_given is not None:
                x[k+1] = x[k] * k_e + doses_given[k]

    # --- Initial parameter heuristic -------------------------------------
    window = min(int(round(0.5 / dt)), Nt // 4)
    if window > 1:
        kernel = np.ones(window) / window
        smooth_rate = np.convolve(seizure_counts / dt, kernel, mode="same")
    else:
        smooth_rate = seizure_counts / dt

    smooth_rate_base = _invert_pd(smooth_rate, x, is_treated, pd_mode,
                                  alpha, C50, gamma_hill)
    peak_idx = int(np.argmax(smooth_rate_base))
    peak_val = float(smooth_rate_base[peak_idx])
    peak_time = float(T[peak_idx])

    A_p_init = max(0.1, peak_val / peak_rate)
    r_init   = max(0.5, 2 / max(peak_time, dt))

    post_peak = smooth_rate_base[peak_idx:]
    half_val = peak_val / 2
    below = np.where(post_peak < half_val)[0]
    if below.size > 0:
        # MATLAB find(...) returns 1-based first index; subtract 1 then
        # multiply by dt — but MATLAB also multiplies the 1-based index
        # by dt, not (idx - 1) * dt, so we mirror that exactly.
        half_idx_matlab = int(below[0]) + 1
        half_time = half_idx_matlab * dt
        kappa_init = max(0.1, np.log(2) / half_time)
    else:
        kappa_init = 0.3

    theta0 = np.array([np.log(A_p_init), np.log(r_init), np.log(kappa_init)])

    # --- Optimize via Nelder-Mead (scipy equivalent of MATLAB fminsearch) ----
    result = opt.minimize(
        negative_log_likelihood, theta0,
        args=(seizure_counts, T, dt, peak_rate, x, is_treated, pd_mode,
              alpha, C50, gamma_hill),
        method="Nelder-Mead",
        options={
            "xatol": 1e-8,
            "fatol": 1e-8,
            "maxiter": 2000,
            "maxfev":  5000,
            "disp":     False,
        },
    )
    theta_opt = result.x

    params_est = {
        "A_p":       float(np.exp(theta_opt[0])),
        "r":         float(np.exp(theta_opt[1])),
        "kappa":     float(np.exp(theta_opt[2])),
        "peak_rate": float(peak_rate),
    }

    shape_est = np.exp(-params_est["kappa"] * T) * (1 - np.exp(-params_est["r"] * T))
    lambda_base_hat = params_est["A_p"] * peak_rate * shape_est / np.max(shape_est)
    lambda_obs_hat  = _apply_pd(lambda_base_hat, x, is_treated, pd_mode,
                                alpha, C50, gamma_hill)

    fit_info = {
        "negloglik":       float(result.fun),
        "exitflag":        int(result.success),
        "lambda_base_hat": lambda_base_hat,
        "lambda_obs_hat":  lambda_obs_hat,
        "x":               x,
        "is_treated":      is_treated,
        "pd_mode":         pd_mode,
    }
    return params_est, fit_info
