"""Port of ``scripts/s13_closed_loop_personalized.m``.

Compares three controllers: uniform omega=2, fixed regime-optimal
omega*, and a dynamic omega(t) controller that adjusts omega every
24 hours based on the harm-balance ratio.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import strategies, mortality as mortality_mod, patient as patient_mod
from .._matlab_compat import mat_save
from ..params import default_params
from ..pd import observed_rate
from .s5_quadrant_analysis import REGIMES


def _run_dynamic_adaptive(lambda_base, dp, cp, T, dt, theta, age, sofa,
                          rng):
    """Dynamic omega(t) controller. Mirrors MATLAB's `run_dynamic_adaptive`."""
    Nt = lambda_base.size
    k_e = dp["k_e"]
    dosing_interval_days = dp["dosing_interval"] / 24

    sz_coeff   = theta[2] + theta[3] * sofa / 10
    drug_coeff = theta[4] + theta[5] * age / 65
    L_norm, X_norm = 35, 5000

    pd_struct = {
        "mode":  "hill",
        "C50":   dp.get("patient_C50",   dp["C50"]),
        "gamma": dp.get("patient_gamma", dp["gamma_hill"]),
    }

    tau_x, q = 0.5, 1e-4
    phi   = float(np.exp(-dt / tau_x))
    ekf_x = 0.0
    ekf_P = 1.0

    r_pop, kappa_pop, peak_pop = 1.5, 0.3, 5
    shape_pop = np.exp(-kappa_pop * T) * (1 - np.exp(-r_pop * T))
    mu_prior  = np.maximum(0.1, peak_pop * shape_pop / np.max(shape_pop))

    dose_indices = set()
    for i in range(Nt):
        tsd = T[i] % dosing_interval_days
        if tsd < dt / 2 or (dosing_interval_days - tsd) < dt / 2:
            dose_indices.add(i)

    x = np.zeros(Nt)
    lambda_obs     = np.zeros(Nt)
    lambda_hat     = np.zeros(Nt)
    doses_given    = np.zeros(Nt)
    seizure_counts = np.zeros(Nt)
    omega_history  = np.zeros(Nt)

    harm_ratio = drug_coeff / max(sz_coeff, 0.01)
    omega = max(0.5, min(10.0, 2 * harm_ratio))

    last_update_day = 0
    daily_lambda_sum = 0.0
    daily_x_sum = 0.0
    daily_count = 0

    for k in range(Nt):
        lambda_obs[k] = float(observed_rate(
            np.array([lambda_base[k]]), np.array([x[k]]), pd_struct
        )[0])
        seizure_counts[k] = float(rng.poisson(lambda_obs[k] * dt))

        ekf_x = phi * ekf_x
        ekf_P = phi**2 * ekf_P + q
        lam_pred = mu_prior[k] * np.exp(ekf_x)
        yhat = lam_pred * dt
        H = yhat
        R = max(yhat, 1e-3)
        K = (ekf_P * H) / (H * ekf_P * H + R)
        ekf_x = ekf_x + K * (seizure_counts[k] - yhat)
        ekf_P = (1 - K * H) * ekf_P * (1 - K * H) + K * R * K
        lambda_hat[k] = mu_prior[k] * np.exp(ekf_x)

        daily_lambda_sum += lambda_hat[k]
        daily_x_sum      += x[k]
        daily_count      += 1

        current_day = int(np.floor(T[k]))
        if current_day > last_update_day and daily_count > 0:
            mean_lam = daily_lambda_sum / daily_count
            mean_x   = daily_x_sum      / daily_count
            marg_sz   = sz_coeff   * mean_lam / L_norm
            marg_drug = drug_coeff * mean_x   / X_norm
            ratio = marg_drug / max(marg_sz, 1e-6)
            if ratio > 1.5:
                omega = min(10.0, omega * 1.3)
            elif ratio < 0.5:
                omega = max(0.5, omega * 0.7)
            last_update_day = current_day
            daily_lambda_sum = 0.0
            daily_x_sum = 0.0
            daily_count = 0
        omega_history[k] = omega

        if k in dose_indices:
            if   lambda_hat[k] < 0.5 * omega: doses_given[k] = 0
            elif lambda_hat[k] < 1.0 * omega: doses_given[k] = cp["dose_low"]
            elif lambda_hat[k] < 1.5 * omega: doses_given[k] = cp["dose_intermediate"]
            elif lambda_hat[k] < 2.0 * omega: doses_given[k] = cp["dose_medium"]
            else:                              doses_given[k] = cp["dose_high"]
        if k < Nt - 1:
            x[k + 1] = x[k] * k_e + doses_given[k]

    return {
        "lambda_obs":      lambda_obs,
        "lambda_hat":      lambda_hat,
        "x":               x,
        "doses_given":     doses_given,
        "omega_history":   omega_history,
        "expected_burden": float(np.sum(lambda_obs) * dt),
        "drug_exposure":   float(np.sum(x) * dt),
        "total_dose":      float(np.sum(doses_given)),
    }


def run(Npat: int = 200, Nruns: int = 200, use_ekf: bool = True,
        omega_optimal=(0.5, 1.5, 0.5, 8.0),
        seed: int = 42, out_path: str | Path | None = None) -> dict:
    rng = np.random.default_rng(seed)
    time, drug_p, ctrl_p, mort_p = default_params()
    T, Nt, dt = time["T"], time["Nt"], time["dt"]

    patients = []
    for _ in range(Npat):
        u_pat = {k: rng.uniform() for k in
                 ("age", "sofa", "A_base", "r", "kappa",
                  "alpha", "C50", "gamma_hill")}
        patients.append(patient_mod.generate(
            T, {"peak_rate": 10, "drug_params": drug_p}, u_draws=u_pat))

    n_reg = len(REGIMES)
    surv_results = np.zeros((n_reg, 3))
    n_days = int(np.ceil(Nt * dt))

    for ri, reg in enumerate(REGIMES):
        theta = np.asarray(reg["theta"])
        surv_pat = np.zeros((Npat, 3))
        for p in range(Npat):
            pat = patients[p]
            dp = dict(drug_p, alpha=pat["alpha"])
            if "C50" in pat:
                dp["patient_C50"]   = pat["C50"]
                dp["patient_gamma"] = pat["gamma_hill"]

            cp1 = dict(ctrl_p, omega=2)
            u_p1 = rng.uniform(size=Nt)
            r1 = strategies.run_strategy(
                "adaptive", pat["lambda_base"], dp, cp1, T, dt,
                use_ekf=True, u_draws={"poisson": u_p1})

            cp2 = dict(ctrl_p, omega=omega_optimal[ri])
            u_p2 = rng.uniform(size=Nt)
            s = "none" if np.isinf(omega_optimal[ri]) else "adaptive"
            r2 = strategies.run_strategy(
                s, pat["lambda_base"], dp, cp2, T, dt,
                use_ekf=True, u_draws={"poisson": u_p2})

            r3 = _run_dynamic_adaptive(
                pat["lambda_base"], dp, ctrl_p, T, dt, theta,
                pat["age"], pat["sofa"], rng)

            for si, r in enumerate((r1, r2, r3)):
                alive = 0
                for _ in range(Nruns):
                    u_m = rng.uniform(size=n_days)
                    a, _t = mortality_mod.run(
                        r["lambda_obs"], r["x"], dt, theta,
                        pat["age"], pat["sofa"], mort_p, u_draws=u_m)
                    alive += a
                surv_pat[p, si] = alive / Nruns
        surv_results[ri] = surv_pat.mean(axis=0) * 100

    out = {
        "surv_results":  surv_results,
        "regimes":       np.array([{"name": r["name"], "theta": np.asarray(r["theta"])}
                                   for r in REGIMES], dtype=object),
        "omega_optimal": np.asarray(omega_optimal),
        "Npat":          Npat,
        "Nruns":         Nruns,
    }
    if out_path is not None:
        mat_save(out_path, **out)
    return out
