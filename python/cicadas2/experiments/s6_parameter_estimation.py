"""Port of ``scripts/s6_parameter_estimation.m``.

Recovers (A_p, r, kappa) from simulated observations for Npat
patients, half treated (fixed dose q8h) and half untreated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import strategies, estimation as estimation_mod, patient as patient_mod
from .._matlab_compat import mat_save
from ..params import default_params


def run(Npat: int = 200, frac_treated: float = 0.5,
        seed: int = 42, out_path: str | Path | None = None) -> dict:
    rng = np.random.default_rng(seed)

    time, drug_p, ctrl_p, mort_p = default_params()
    T, Nt, dt = time["T"], time["Nt"], time["dt"]

    N_treated   = round(Npat * frac_treated)
    N_untreated = Npat - N_treated

    true_Ap    = np.zeros(Npat)
    true_r     = np.zeros(Npat)
    true_kappa = np.zeros(Npat)
    est_Ap     = np.zeros(Npat)
    est_r      = np.zeros(Npat)
    est_kappa  = np.zeros(Npat)
    is_treated   = np.zeros(Npat, dtype=bool)
    fit_exitflag = np.zeros(Npat, dtype=int)
    patient_lambdas = np.zeros((Npat, Nt))
    true_Lcum = np.zeros(Npat)
    est_Lcum  = np.zeros(Npat)

    for p in range(Npat):
        u_pat = {k: rng.uniform() for k in
                 ("age", "sofa", "A_base", "r", "kappa",
                  "alpha", "C50", "gamma_hill")}
        pat = patient_mod.generate(T, {"peak_rate": 10, "drug_params": drug_p},
                                   u_draws=u_pat)
        dp = dict(drug_p, alpha=pat["alpha"])
        if "C50" in pat:
            dp["patient_C50"]   = pat["C50"]
            dp["patient_gamma"] = pat["gamma_hill"]

        true_Ap[p]    = pat["A_patient"]
        true_r[p]     = pat["r"]
        true_kappa[p] = pat["kappa"]
        patient_lambdas[p, :] = pat["lambda_base"]

        if p < N_untreated:
            counts = rng.poisson(pat["lambda_base"] * dt)
            params_est, fit_info = estimation_mod.estimate_natural_history(
                counts, T, dt, peak_rate=10,
            )
        else:
            is_treated[p] = True
            u_poisson = rng.uniform(size=Nt)
            r = strategies.run_strategy(
                "fixed", pat["lambda_base"], dp, ctrl_p, T, dt,
                use_ekf=False, u_draws={"poisson": u_poisson},
            )
            kwargs = dict(peak_rate=10, k_e=dp["k_e"],
                          doses_given=r["doses_given"])
            if drug_p.get("pd_mode", "").lower() == "hill":
                kwargs.update(pd_mode="hill",
                              C50=dp["patient_C50"],
                              gamma_hill=dp["patient_gamma"])
            else:
                kwargs.update(pd_mode="linear", alpha=dp["alpha"])
            params_est, fit_info = estimation_mod.estimate_natural_history(
                r["seizure_counts"], T, dt, **kwargs,
            )

        est_Ap[p]    = params_est["A_p"]
        est_r[p]     = params_est["r"]
        est_kappa[p] = params_est["kappa"]
        fit_exitflag[p] = fit_info["exitflag"]
        true_Lcum[p] = float(np.sum(pat["lambda_base"]) * dt)
        est_Lcum[p]  = float(np.sum(fit_info["lambda_base_hat"]) * dt)

    # Derived quantities (peak height + timing)
    true_peak_height = patient_lambdas.max(axis=1)
    true_peak_time   = T[patient_lambdas.argmax(axis=1)]

    est_peak_height = np.zeros(Npat)
    est_peak_time   = np.zeros(Npat)
    for p in range(Npat):
        shape_est = np.exp(-est_kappa[p] * T) * (1 - np.exp(-est_r[p] * T))
        max_shape = shape_est.max()
        if max_shape < 1e-10:
            lam = np.zeros_like(T)
        else:
            lam = est_Ap[p] * 10 * shape_est / max_shape
        est_peak_height[p] = lam.max()
        est_peak_time[p]   = T[lam.argmax()]

    out = {
        "true_Ap":          true_Ap,
        "true_r":           true_r,
        "true_kappa":       true_kappa,
        "est_Ap":           est_Ap,
        "est_r":            est_r,
        "est_kappa":        est_kappa,
        "true_Lcum":        true_Lcum,
        "est_Lcum":         est_Lcum,
        "true_peak_height": true_peak_height,
        "est_peak_height":  est_peak_height,
        "true_peak_time":   true_peak_time,
        "est_peak_time":    est_peak_time,
        "is_treated":       is_treated.astype(int),
        "fit_exitflag":     fit_exitflag,
        "patient_lambdas":  patient_lambdas,
        "Npat":             Npat,
        "frac_treated":     frac_treated,
    }
    if out_path is not None:
        mat_save(out_path, **out)
    return out
