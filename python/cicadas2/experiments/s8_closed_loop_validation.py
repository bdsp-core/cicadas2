"""Port of ``scripts/s8_closed_loop_validation.m``.

End-to-end pipeline test: estimate (A_p, r, kappa) from observations,
run strategies on both true and estimated trajectories, compare
mortality rankings under the quadrant regimes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import strategies, estimation as estimation_mod, mortality as mortality_mod, patient as patient_mod
from .._matlab_compat import mat_save
from ..params import default_params
from .s5_quadrant_analysis import REGIMES, STRATEGIES, STRAT_LABELS


def run(Npat: int = 300, Nruns: int = 100, frac_treated: float = 0.5,
        use_ekf: bool = True, pkpd_mode: str = "pop_mean",
        seed: int = 42, out_path: str | Path | None = None) -> dict:
    rng = np.random.default_rng(seed)

    time, drug_p, ctrl_p, mort_p = default_params()
    T, Nt, dt = time["T"], time["Nt"], time["dt"]

    N_treated   = round(Npat * frac_treated)
    N_untreated = Npat - N_treated

    # --- Step 1: Generate patients -----------------------------------
    patients = []
    for _ in range(Npat):
        u_pat = {k: rng.uniform() for k in
                 ("age", "sofa", "A_base", "r", "kappa",
                  "alpha", "C50", "gamma_hill")}
        patients.append(patient_mod.generate(
            T, {"peak_rate": 10, "drug_params": drug_p}, u_draws=u_pat))

    # --- Step 2: Estimate parameters ---------------------------------
    true_lambda_base = np.zeros((Npat, Nt))
    est_lambda_base  = np.zeros((Npat, Nt))
    true_Lcum = np.zeros(Npat)
    est_Lcum  = np.zeros(Npat)

    for p in range(Npat):
        pat = patients[p]
        dp = dict(drug_p, alpha=pat["alpha"])
        if "C50" in pat:
            dp["patient_C50"]   = pat["C50"]
            dp["patient_gamma"] = pat["gamma_hill"]

        true_lambda_base[p, :] = pat["lambda_base"]
        true_Lcum[p] = float(np.sum(pat["lambda_base"]) * dt)

        if p < N_untreated:
            counts = rng.poisson(pat["lambda_base"] * dt)
            _, fit = estimation_mod.estimate_natural_history(
                counts, T, dt, peak_rate=10)
        else:
            u_poisson = rng.uniform(size=Nt)
            r = strategies.run_strategy(
                "fixed", pat["lambda_base"], dp, ctrl_p, T, dt,
                use_ekf=False, u_draws={"poisson": u_poisson})
            if pkpd_mode == "oracle":
                est_C50, est_gamma = dp["patient_C50"], dp["patient_gamma"]
            else:
                est_C50, est_gamma = drug_p["C50"], drug_p["gamma_hill"]
            kwargs = dict(peak_rate=10, k_e=dp["k_e"],
                          doses_given=r["doses_given"])
            if drug_p.get("pd_mode", "").lower() == "hill":
                kwargs.update(pd_mode="hill", C50=est_C50, gamma_hill=est_gamma)
            else:
                kwargs.update(pd_mode="linear", alpha=dp["alpha"])
            _, fit = estimation_mod.estimate_natural_history(
                r["seizure_counts"], T, dt, **kwargs)
        est_lambda_base[p, :] = fit["lambda_base_hat"]
        est_Lcum[p] = float(np.sum(fit["lambda_base_hat"]) * dt)

    corr_Lcum = float(np.corrcoef(true_Lcum, est_Lcum)[0, 1])

    # --- Step 3: Run strategies on TRUE and ESTIMATED ----------------
    n_strats = len(STRATEGIES)
    true_strat = [[None] * n_strats for _ in range(Npat)]
    est_strat  = [[None] * n_strats for _ in range(Npat)]
    for p in range(Npat):
        pat = patients[p]
        dp = dict(drug_p, alpha=pat["alpha"])
        if "C50" in pat:
            dp["patient_C50"]   = pat["C50"]
            dp["patient_gamma"] = pat["gamma_hill"]
        for si, s in enumerate(STRATEGIES):
            u_t = rng.uniform(size=Nt)
            true_strat[p][si] = strategies.run_strategy(
                s, true_lambda_base[p], dp, ctrl_p, T, dt,
                use_ekf=use_ekf, u_draws={"poisson": u_t})
            u_e = rng.uniform(size=Nt)
            est_strat[p][si]  = strategies.run_strategy(
                s, est_lambda_base[p], dp, ctrl_p, T, dt,
                use_ekf=use_ekf, u_draws={"poisson": u_e})

    # --- Step 4: Mortality under each regime -------------------------
    n_reg = len(REGIMES)
    surv_true = np.zeros((n_reg, n_strats))
    surv_est  = np.zeros((n_reg, n_strats))
    n_days = int(np.ceil(Nt * dt))
    for ri, reg in enumerate(REGIMES):
        theta = np.asarray(reg["theta"])
        sp_t = np.zeros((Npat, n_strats))
        sp_e = np.zeros((Npat, n_strats))
        for p in range(Npat):
            pat = patients[p]
            for si in range(n_strats):
                rt = true_strat[p][si]
                re = est_strat[p][si]
                a_t = a_e = 0
                for _ in range(Nruns):
                    u_mt = rng.uniform(size=n_days)
                    u_me = rng.uniform(size=n_days)
                    a1, _ = mortality_mod.run(
                        rt["lambda_obs"], rt["x"], dt, theta,
                        pat["age"], pat["sofa"], mort_p, u_draws=u_mt)
                    a2, _ = mortality_mod.run(
                        re["lambda_obs"], re["x"], dt, theta,
                        pat["age"], pat["sofa"], mort_p, u_draws=u_me)
                    a_t += a1
                    a_e += a2
                sp_t[p, si] = a_t / Nruns
                sp_e[p, si] = a_e / Nruns
        surv_true[ri] = sp_t.mean(axis=0) * 100
        surv_est[ri]  = sp_e.mean(axis=0) * 100

    out = {
        "surv_true":    surv_true,
        "surv_est":     surv_est,
        "regimes":      np.array([{"name": r["name"], "theta": np.asarray(r["theta"])}
                                  for r in REGIMES], dtype=object),
        "strategies":   np.array(STRATEGIES,   dtype=object),
        "strat_labels": np.array(STRAT_LABELS, dtype=object),
        "corr_Lcum":    corr_Lcum,
        "Npat":         Npat,
        "Nruns":        Nruns,
        "pkpd_mode":    pkpd_mode,
    }
    if out_path is not None:
        mat_save(out_path, **out)
    return out
