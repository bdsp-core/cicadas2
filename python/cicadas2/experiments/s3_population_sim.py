"""Port of ``scripts/s3_population_sim.m`` (population survival simulation)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import strategies, mortality as mortality_mod, patient as patient_mod
from .._matlab_compat import mat_save
from ..params import default_params


def run(Npat: int = 500, Nruns: int = 1000, use_ekf: bool = True,
        seed: int = 42, out_path: str | Path | None = None) -> dict:
    rng = np.random.default_rng(seed)
    time, drug_p, ctrl_p, mort_p = default_params()
    T, Nt, dt = time["T"], time["Nt"], time["dt"]

    results = {k: np.zeros(Npat) for k in (
        "age", "sofa", "A_patient", "peak_rate", "alpha",
        "surv_none", "surv_fixed", "surv_prn", "surv_adaptive",
        "burden_none", "burden_fixed", "burden_prn", "burden_adaptive",
        "drug_none",   "drug_fixed",   "drug_prn",   "drug_adaptive",
    )}

    strategies_list = ["none", "fixed", "prn", "adaptive"]
    n_days = int(np.ceil(Nt * dt))

    for p in range(Npat):
        u_pat = {k: rng.uniform() for k in
                 ("age", "sofa", "A_base", "r", "kappa",
                  "alpha", "C50", "gamma_hill")}
        pat = patient_mod.generate(T, {"peak_rate": 10, "drug_params": drug_p},
                                   u_draws=u_pat)
        results["age"][p]       = pat["age"]
        results["sofa"][p]      = pat["sofa"]
        results["A_patient"][p] = pat["A_patient"]
        results["peak_rate"][p] = float(pat["lambda_base"].max())
        results["alpha"][p]     = pat["alpha"]

        dp = dict(drug_p, alpha=pat["alpha"])
        if "C50" in pat:
            dp["patient_C50"]   = pat["C50"]
            dp["patient_gamma"] = pat["gamma_hill"]

        runs = {}
        for s in strategies_list:
            u_poisson = rng.uniform(size=Nt)
            runs[s] = strategies.run_strategy(
                s, pat["lambda_base"], dp, ctrl_p, T, dt,
                use_ekf=use_ekf, u_draws={"poisson": u_poisson})
            results[f"burden_{s}"][p] = runs[s]["burden"]
            results[f"drug_{s}"][p]   = runs[s]["total_dose"]

        for s in strategies_list:
            alive_sum = 0
            for _ in range(Nruns):
                u_mort = rng.uniform(size=n_days)
                a, _t = mortality_mod.run(
                    runs[s]["lambda_obs"], runs[s]["x"], dt, mort_p["theta"],
                    pat["age"], pat["sofa"], mort_p, u_draws=u_mort)
                alive_sum += a
            results[f"surv_{s}"][p] = alive_sum / Nruns

    out = {"results": results, "Npat": Npat, "Nruns": Nruns,
           "use_ekf": int(use_ekf)}
    if out_path is not None:
        mat_save(out_path, **out)
    return out
