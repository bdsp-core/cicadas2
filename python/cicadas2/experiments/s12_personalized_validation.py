"""Port of ``scripts/s12_personalized_validation.m``.

Tests whether the per-patient omega* can be recovered from
estimated disease parameters (pop-mean PKPD).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import strategies, estimation as estimation_mod, mortality as mortality_mod, patient as patient_mod
from .._matlab_compat import mat_save
from ..params import default_params
from .s5_quadrant_analysis import REGIMES
from .s10_smart_adaptive import OMEGA_VALS


def run(Npat: int = 200, Nruns: int = 200, frac_treated: float = 0.5,
        use_ekf: bool = True,
        seed: int = 42, out_path: str | Path | None = None) -> dict:
    rng = np.random.default_rng(seed)
    time, drug_p, ctrl_p, mort_p = default_params()
    T, Nt, dt = time["T"], time["Nt"], time["dt"]

    N_treated   = round(Npat * frac_treated)
    N_untreated = Npat - N_treated

    n_omega = OMEGA_VALS.size
    n_reg   = len(REGIMES)

    # Step 1: Generate patients
    patients = []
    for _ in range(Npat):
        u_pat = {k: rng.uniform() for k in
                 ("age", "sofa", "A_base", "r", "kappa",
                  "alpha", "C50", "gamma_hill")}
        patients.append(patient_mod.generate(
            T, {"peak_rate": 10, "drug_params": drug_p}, u_draws=u_pat))

    # Step 2: Estimate disease parameters (pop-mean PKPD)
    true_lambda_base = np.zeros((Npat, Nt))
    est_lambda_base  = np.zeros((Npat, Nt))
    for p in range(Npat):
        pat = patients[p]
        dp = dict(drug_p, alpha=pat["alpha"])
        if "C50" in pat:
            dp["patient_C50"]   = pat["C50"]
            dp["patient_gamma"] = pat["gamma_hill"]
        true_lambda_base[p] = pat["lambda_base"]
        if p < N_untreated:
            counts = rng.poisson(pat["lambda_base"] * dt)
            _, fit = estimation_mod.estimate_natural_history(
                counts, T, dt, peak_rate=10)
        else:
            u_p = rng.uniform(size=Nt)
            r = strategies.run_strategy(
                "fixed", pat["lambda_base"], dp, ctrl_p, T, dt,
                use_ekf=False, u_draws={"poisson": u_p})
            _, fit = estimation_mod.estimate_natural_history(
                r["seizure_counts"], T, dt, peak_rate=10,
                k_e=dp["k_e"], doses_given=r["doses_given"],
                pd_mode="hill", C50=drug_p["C50"], gamma_hill=drug_p["gamma_hill"])
        est_lambda_base[p] = fit["lambda_base_hat"]

    corr_Lcum = float(np.corrcoef(true_lambda_base.sum(1) * dt,
                                  est_lambda_base.sum(1) * dt)[0, 1])

    # Step 3: omega sweep on true and estimated trajectories
    strat_true = [[None]*n_omega for _ in range(Npat)]
    strat_est  = [[None]*n_omega for _ in range(Npat)]
    for p in range(Npat):
        pat = patients[p]
        dp = dict(drug_p, alpha=pat["alpha"])
        if "C50" in pat:
            dp["patient_C50"]   = pat["C50"]
            dp["patient_gamma"] = pat["gamma_hill"]
        for oi, om in enumerate(OMEGA_VALS):
            cp = dict(ctrl_p, omega=om)
            s = "none" if np.isinf(om) else "adaptive"
            u_t = rng.uniform(size=Nt)
            u_e = rng.uniform(size=Nt)
            strat_true[p][oi] = strategies.run_strategy(
                s, true_lambda_base[p], dp, cp, T, dt,
                use_ekf=use_ekf, u_draws={"poisson": u_t})
            strat_est[p][oi]  = strategies.run_strategy(
                s, est_lambda_base[p], dp, cp, T, dt,
                use_ekf=use_ekf, u_draws={"poisson": u_e})

    # Step 4: mortality at each omega per regime
    surv_true_grid = np.zeros((Npat, n_omega, n_reg))
    surv_est_grid  = np.zeros((Npat, n_omega, n_reg))
    n_days = int(np.ceil(Nt * dt))
    for ri, reg in enumerate(REGIMES):
        theta = np.asarray(reg["theta"])
        for p in range(Npat):
            for oi in range(n_omega):
                a_t = a_e = 0
                for _ in range(Nruns):
                    u_mt = rng.uniform(size=n_days)
                    u_me = rng.uniform(size=n_days)
                    a1, _ = mortality_mod.run(
                        strat_true[p][oi]["lambda_obs"], strat_true[p][oi]["x"],
                        dt, theta, patients[p]["age"], patients[p]["sofa"],
                        mort_p, u_draws=u_mt)
                    a2, _ = mortality_mod.run(
                        strat_est[p][oi]["lambda_obs"], strat_est[p][oi]["x"],
                        dt, theta, patients[p]["age"], patients[p]["sofa"],
                        mort_p, u_draws=u_me)
                    a_t += a1
                    a_e += a2
                surv_true_grid[p, oi, ri] = a_t / Nruns
                surv_est_grid[p, oi, ri]  = a_e / Nruns

    # Step 5: optimal omega per patient/regime
    omega_true       = np.zeros((Npat, n_reg))
    omega_est        = np.zeros((Npat, n_reg))
    surv_at_true_opt = np.zeros((Npat, n_reg))
    surv_at_est_opt  = np.zeros((Npat, n_reg))
    for ri in range(n_reg):
        for p in range(Npat):
            it = int(surv_true_grid[p, :, ri].argmax())
            ie = int(surv_est_grid[p, :, ri].argmax())
            surv_at_true_opt[p, ri] = surv_true_grid[p, it, ri]
            surv_at_est_opt[p, ri]  = surv_est_grid[p,  ie, ri]
            omega_true[p, ri] = OMEGA_VALS[it]
            omega_est[p, ri]  = OMEGA_VALS[ie]

    out = {
        "omega_vals":         OMEGA_VALS,
        "omega_true":         omega_true,
        "omega_est":          omega_est,
        "surv_true_grid":     surv_true_grid,
        "surv_est_grid":      surv_est_grid,
        "surv_at_true_opt":   surv_at_true_opt,
        "surv_at_est_opt":    surv_at_est_opt,
        "regimes":            np.array([{"name": r["name"], "theta": np.asarray(r["theta"])}
                                        for r in REGIMES], dtype=object),
        "Npat":               Npat,
        "Nruns":              Nruns,
        "corr_Lcum":          corr_Lcum,
    }
    if out_path is not None:
        mat_save(out_path, **out)
    return out
