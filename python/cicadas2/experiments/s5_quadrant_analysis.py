"""Port of ``scripts/s5_quadrant_analysis.m``.

Tests how the optimal treatment strategy changes across four
harm-phenotype regimes. Each patient is simulated under all four
strategies once; mortality is then re-evaluated per regime with
that regime's `theta` vector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import strategies, mortality as mortality_mod, patient as patient_mod
from .._matlab_compat import mat_save
from ..params import default_params


THETA4_HIGH = 1.25
THETA4_LOW  = 0.05

REGIMES = [
    {"name": "High Sz / Low Drug",  "theta": [-5, 0.5, 3.0, 1.0, 0.05,       0.02]},
    {"name": "High Sz / High Drug", "theta": [-5, 0.5, 3.0, 1.0, THETA4_HIGH, 0.3 ]},
    {"name": "Low Sz / Low Drug",   "theta": [-5, 0.5, 0.5, 0.2, 0.05,       0.02]},
    {"name": "Low Sz / High Drug",  "theta": [-5, 0.5, 0.5, 0.2, THETA4_HIGH, 0.3 ]},
]
STRATEGIES   = ["none", "fixed", "prn", "adaptive"]
STRAT_LABELS = ["None", "Fixed", "PRN", "Adaptive"]


def run(Npat: int = 500, Nruns: int = 1000, use_ekf: bool = True,
        seed: int = 42, out_path: str | Path | None = None) -> dict:
    rng = np.random.default_rng(seed)

    time, drug_p, ctrl_p, mort_p = default_params()
    T, dt = time["T"], time["dt"]

    # --- Generate patients + run each strategy once per patient ---------
    patients      = []
    strat_results = []
    burden_pat    = np.zeros((Npat, len(STRATEGIES)))
    drug_pat      = np.zeros((Npat, len(STRATEGIES)))
    ages          = np.zeros(Npat)
    sofas         = np.zeros(Npat)

    for p in range(Npat):
        u_pat = {k: rng.uniform() for k in
                 ("age", "sofa", "A_base", "r", "kappa",
                  "alpha", "C50", "gamma_hill")}
        pat = patient_mod.generate(T, {"peak_rate": 10, "drug_params": drug_p},
                                   u_draws=u_pat)
        patients.append(pat)
        ages[p]  = pat["age"]
        sofas[p] = pat["sofa"]

        dp = dict(drug_p, alpha=pat["alpha"])
        if "C50" in pat:
            dp["patient_C50"]   = pat["C50"]
            dp["patient_gamma"] = pat["gamma_hill"]

        row = []
        for si, strat in enumerate(STRATEGIES):
            u_poisson = rng.uniform(size=time["Nt"])
            r = strategies.run_strategy(strat, pat["lambda_base"], dp, ctrl_p,
                                        T, dt, use_ekf=use_ekf,
                                        u_draws={"poisson": u_poisson})
            row.append(r)
            burden_pat[p, si] = r["expected_burden"]
            drug_pat[p, si]   = r["drug_exposure"]
        strat_results.append(row)

    # --- Evaluate mortality across each regime --------------------------
    n_reg, n_s = len(REGIMES), len(STRATEGIES)
    surv_rates   = np.zeros((n_reg, n_s))
    burden_means = np.zeros((n_reg, n_s))
    drug_means   = np.zeros((n_reg, n_s))
    all_results  = []

    n_days = int(np.ceil(time["Nt"] * dt))
    for ri, reg in enumerate(REGIMES):
        theta = np.asarray(reg["theta"])
        surv_pat = np.zeros((Npat, n_s))
        for p in range(Npat):
            pat = patients[p]
            for si in range(n_s):
                r = strat_results[p][si]
                alive_sum = 0
                for _run in range(Nruns):
                    u_mort = rng.uniform(size=n_days)
                    a, _ = mortality_mod.run(
                        r["lambda_obs"], r["x"], dt, theta,
                        pat["age"], pat["sofa"], mort_p, u_draws=u_mort,
                    )
                    alive_sum += a
                surv_pat[p, si] = alive_sum / Nruns

        surv_rates[ri]   = np.mean(surv_pat, axis=0) * 100
        burden_means[ri] = np.mean(burden_pat, axis=0)
        drug_means[ri]   = np.mean(drug_pat,   axis=0)
        all_results.append({
            "surv_pat":   surv_pat,
            "burden_pat": burden_pat,
            "ages":       ages,
            "sofas":      sofas,
        })

    out = {
        "regimes":      np.array([{"name": r["name"], "theta": np.asarray(r["theta"])}
                                  for r in REGIMES], dtype=object),
        "surv_rates":   surv_rates,
        "burden_means": burden_means,
        "drug_means":   drug_means,
        "all_results":  np.array(all_results, dtype=object),
        "strategies":   np.array(STRATEGIES,   dtype=object),
        "strat_labels": np.array(STRAT_LABELS, dtype=object),
    }
    if out_path is not None:
        mat_save(out_path, **out)
    return out
