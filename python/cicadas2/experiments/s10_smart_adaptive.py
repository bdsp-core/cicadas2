"""Port of ``scripts/s10_smart_adaptive.m``.

Per-patient ω* optimization across the four quadrant regimes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import strategies, mortality as mortality_mod, patient as patient_mod
from .._matlab_compat import mat_save
from ..params import default_params
from .s5_quadrant_analysis import REGIMES


OMEGA_VALS = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf])


def run(Npat: int = 200, Nruns: int = 200, use_ekf: bool = True,
        omega_uniform: float = 2.0,
        seed: int = 42, out_path: str | Path | None = None) -> dict:
    rng = np.random.default_rng(seed)

    time, drug_p, ctrl_p, mort_p = default_params()
    T, Nt, dt = time["T"], time["Nt"], time["dt"]

    n_omega   = OMEGA_VALS.size
    n_regimes = len(REGIMES)

    # --- Patients ----------------------------------------------------
    patients   = []
    ages       = np.zeros(Npat)
    sofas      = np.zeros(Npat)
    peak_rates = np.zeros(Npat)
    for p in range(Npat):
        u_pat = {k: rng.uniform() for k in
                 ("age", "sofa", "A_base", "r", "kappa",
                  "alpha", "C50", "gamma_hill")}
        pat = patient_mod.generate(T, {"peak_rate": 10, "drug_params": drug_p},
                                   u_draws=u_pat)
        patients.append(pat)
        ages[p]       = pat["age"]
        sofas[p]      = pat["sofa"]
        peak_rates[p] = float(pat["lambda_base"].max())

    # --- Strategy sweep over omega per patient -----------------------
    strat_results = [[None] * n_omega for _ in range(Npat)]
    for p in range(Npat):
        pat = patients[p]
        dp = dict(drug_p, alpha=pat["alpha"])
        if "C50" in pat:
            dp["patient_C50"]   = pat["C50"]
            dp["patient_gamma"] = pat["gamma_hill"]
        for oi, om in enumerate(OMEGA_VALS):
            cp = dict(ctrl_p, omega=om)
            u_poisson = rng.uniform(size=Nt)
            if np.isinf(om):
                strat_results[p][oi] = strategies.run_strategy(
                    "none", pat["lambda_base"], dp, cp, T, dt,
                    use_ekf=use_ekf, u_draws={"poisson": u_poisson})
            else:
                strat_results[p][oi] = strategies.run_strategy(
                    "adaptive", pat["lambda_base"], dp, cp, T, dt,
                    use_ekf=use_ekf, u_draws={"poisson": u_poisson})

    # --- Mortality per (p, oi, reg) ----------------------------------
    surv_grid = np.zeros((Npat, n_omega, n_regimes))
    n_days = int(np.ceil(Nt * dt))
    for ri, reg in enumerate(REGIMES):
        theta = np.asarray(reg["theta"])
        for p in range(Npat):
            for oi in range(n_omega):
                r = strat_results[p][oi]
                alive_sum = 0
                for _run in range(Nruns):
                    u_mort = rng.uniform(size=n_days)
                    a, _ = mortality_mod.run(
                        r["lambda_obs"], r["x"], dt, theta,
                        patients[p]["age"], patients[p]["sofa"],
                        mort_p, u_draws=u_mort)
                    alive_sum += a
                surv_grid[p, oi, ri] = alive_sum / Nruns

    # --- Optimal omega per patient per regime ------------------------
    idx_uniform = int(np.argmin(np.abs(OMEGA_VALS - omega_uniform)))
    omega_star   = np.zeros((Npat, n_regimes))
    surv_star    = np.zeros((Npat, n_regimes))
    surv_uniform = np.zeros((Npat, n_regimes))
    for ri in range(n_regimes):
        for p in range(Npat):
            best_oi = int(np.argmax(surv_grid[p, :, ri]))
            surv_star[p, ri]    = surv_grid[p, best_oi, ri]
            omega_star[p, ri]   = OMEGA_VALS[best_oi]
            surv_uniform[p, ri] = surv_grid[p, idx_uniform, ri]

    out = {
        "omega_vals":     OMEGA_VALS,
        "omega_star":     omega_star,
        "surv_grid":      surv_grid,
        "surv_star":      surv_star,
        "surv_uniform":   surv_uniform,
        "ages":           ages,
        "sofas":          sofas,
        "peak_rates":     peak_rates,
        "regimes":        np.array([{"name": r["name"], "theta": np.asarray(r["theta"])}
                                    for r in REGIMES], dtype=object),
        "Npat":           Npat,
        "Nruns":          Nruns,
        "omega_uniform":  omega_uniform,
    }
    if out_path is not None:
        mat_save(out_path, **out)
    return out
