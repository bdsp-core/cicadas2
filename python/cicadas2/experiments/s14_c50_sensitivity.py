"""Port of ``scripts/s14_c50_sensitivity.m`` (C50 uncertainty sweep).

Structurally parallel to s11 — same logic, sweeps C50 instead of
the drug half-life.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import strategies, estimation as estimation_mod, mortality as mortality_mod, patient as patient_mod
from .._matlab_compat import mat_save
from ..params import default_params
from .s5_quadrant_analysis import REGIMES, STRATEGIES, STRAT_LABELS


def run(Npat: int = 200, Nruns: int = 100, use_ekf: bool = True,
        c50_multipliers=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0,
                         1.05, 1.1, 1.2, 1.3, 1.5),
        seed: int = 42, out_path: str | Path | None = None) -> dict:
    rng = np.random.default_rng(seed)
    time, drug_p, ctrl_p, mort_p = default_params()
    T, Nt, dt = time["T"], time["Nt"], time["dt"]
    true_C50 = drug_p["C50"]

    # --- Patients + fixed strategy at true PKPD ------------------------
    patients = []
    true_lambda_base = np.zeros((Npat, Nt))
    true_Lcum = np.zeros(Npat)
    fixed_results = []
    for p in range(Npat):
        u_pat = {k: rng.uniform() for k in
                 ("age", "sofa", "A_base", "r", "kappa",
                  "alpha", "C50", "gamma_hill")}
        pat = patient_mod.generate(T, {"peak_rate": 10, "drug_params": drug_p},
                                   u_draws=u_pat)
        patients.append(pat)
        true_lambda_base[p] = pat["lambda_base"]
        true_Lcum[p] = float(np.sum(pat["lambda_base"]) * dt)
        dp = dict(drug_p, alpha=pat["alpha"])
        if "C50" in pat:
            dp["patient_C50"]   = pat["C50"]
            dp["patient_gamma"] = pat["gamma_hill"]
        u_p = rng.uniform(size=Nt)
        fixed_results.append(strategies.run_strategy(
            "fixed", pat["lambda_base"], dp, ctrl_p, T, dt,
            use_ekf=False, u_draws={"poisson": u_p}))

    # --- Sweep C50 error levels ---------------------------------------
    c50_multipliers = np.asarray(c50_multipliers, dtype=float)
    n_levels = c50_multipliers.size
    corr_Lcum = np.zeros(n_levels)
    corr_peak = np.zeros(n_levels)
    mape_Lcum = np.zeros(n_levels)
    mean_Lcum_bias = np.zeros(n_levels)
    est_lambda_all = np.zeros((Npat, Nt, n_levels))
    tested_C50 = np.zeros(n_levels)

    for li, mul in enumerate(c50_multipliers):
        c50_test = true_C50 * mul
        tested_C50[li] = c50_test

        est_Lcum_lvl = np.zeros(Npat)
        est_peak_lvl = np.zeros(Npat)
        true_peak_lvl = np.zeros(Npat)
        for p in range(Npat):
            r = fixed_results[p]
            _, fit = estimation_mod.estimate_natural_history(
                r["seizure_counts"], T, dt,
                peak_rate=10, k_e=drug_p["k_e"],
                doses_given=r["doses_given"],
                pd_mode="hill", C50=c50_test,
                gamma_hill=drug_p["gamma_hill"],
            )
            est_lambda_all[p, :, li] = fit["lambda_base_hat"]
            est_Lcum_lvl[p] = float(np.sum(fit["lambda_base_hat"]) * dt)
            est_peak_lvl[p] = float(fit["lambda_base_hat"].max())
            true_peak_lvl[p] = float(true_lambda_base[p].max())

        corr_Lcum[li]      = float(np.corrcoef(true_Lcum, est_Lcum_lvl)[0, 1])
        corr_peak[li]      = float(np.corrcoef(true_peak_lvl, est_peak_lvl)[0, 1])
        mape_Lcum[li]      = float(np.mean(np.abs(true_Lcum - est_Lcum_lvl) /
                                            np.maximum(true_Lcum, 0.1)) * 100)
        mean_Lcum_bias[li] = float(np.mean(est_Lcum_lvl - true_Lcum))

    # --- Winner matching (same pattern as s11) -------------------------
    n_strats = len(STRATEGIES)
    n_reg    = len(REGIMES)
    n_days = int(np.ceil(Nt * dt))

    true_strat = [[None]*n_strats for _ in range(Npat)]
    for p in range(Npat):
        pat = patients[p]
        dp = dict(drug_p, alpha=pat["alpha"])
        if "C50" in pat:
            dp["patient_C50"]   = pat["C50"]
            dp["patient_gamma"] = pat["gamma_hill"]
        for si, s in enumerate(STRATEGIES):
            u_p = rng.uniform(size=Nt)
            true_strat[p][si] = strategies.run_strategy(
                s, true_lambda_base[p], dp, ctrl_p, T, dt,
                use_ekf=use_ekf, u_draws={"poisson": u_p})

    surv_true = np.zeros((n_reg, n_strats))
    for ri, reg in enumerate(REGIMES):
        theta = np.asarray(reg["theta"])
        for p in range(Npat):
            for si in range(n_strats):
                alive = 0
                for _ in range(Nruns):
                    u_m = rng.uniform(size=n_days)
                    a, _t = mortality_mod.run(
                        true_strat[p][si]["lambda_obs"], true_strat[p][si]["x"],
                        dt, theta, patients[p]["age"], patients[p]["sofa"],
                        mort_p, u_draws=u_m)
                    alive += a
                surv_true[ri, si] += alive / Nruns
    surv_true = surv_true / Npat * 100
    true_winners = surv_true.argmax(axis=1)

    winner_matches = np.zeros(n_levels, dtype=int)
    surv_est_all   = np.zeros((n_reg, n_strats, n_levels))
    max_surv_diff  = np.zeros(n_levels)

    for li in range(n_levels):
        est_strat = [[None]*n_strats for _ in range(Npat)]
        for p in range(Npat):
            pat = patients[p]
            dp = dict(drug_p, alpha=pat["alpha"])
            if "C50" in pat:
                dp["patient_C50"]   = pat["C50"]
                dp["patient_gamma"] = pat["gamma_hill"]
            for si, s in enumerate(STRATEGIES):
                u_p = rng.uniform(size=Nt)
                est_strat[p][si] = strategies.run_strategy(
                    s, est_lambda_all[p, :, li], dp, ctrl_p, T, dt,
                    use_ekf=use_ekf, u_draws={"poisson": u_p})

        surv_est = np.zeros((n_reg, n_strats))
        for ri, reg in enumerate(REGIMES):
            theta = np.asarray(reg["theta"])
            for p in range(Npat):
                for si in range(n_strats):
                    alive = 0
                    for _ in range(Nruns):
                        u_m = rng.uniform(size=n_days)
                        a, _t = mortality_mod.run(
                            est_strat[p][si]["lambda_obs"], est_strat[p][si]["x"],
                            dt, theta, patients[p]["age"], patients[p]["sofa"],
                            mort_p, u_draws=u_m)
                        alive += a
                    surv_est[ri, si] += alive / Nruns
        surv_est = surv_est / Npat * 100
        surv_est_all[:, :, li] = surv_est
        est_winners = surv_est.argmax(axis=1)
        winner_matches[li] = int(np.sum(est_winners == true_winners))
        max_surv_diff[li]  = float(np.max(np.abs(surv_est - surv_true)))

    out = {
        "c50_multipliers": c50_multipliers,
        "tested_C50":     tested_C50,
        "corr_Lcum":      corr_Lcum,
        "corr_peak":      corr_peak,
        "mape_Lcum":      mape_Lcum,
        "mean_Lcum_bias": mean_Lcum_bias,
        "winner_matches": winner_matches,
        "max_surv_diff":  max_surv_diff,
        "surv_true":      surv_true,
        "surv_est_all":   surv_est_all,
        "true_C50":       true_C50,
        "regimes":        np.array([{"name": r["name"], "theta": np.asarray(r["theta"])}
                                    for r in REGIMES], dtype=object),
        "strat_labels":   np.array(STRAT_LABELS, dtype=object),
        "Npat":           Npat,
        "Nruns":          Nruns,
    }
    if out_path is not None:
        mat_save(out_path, **out)
    return out
