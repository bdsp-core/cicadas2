"""Port of ``scripts/s15_ekf_sensitivity.m`` (EKF tuning sweep).

Sweeps each EKF parameter individually and records the resulting
adaptive-strategy survival.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import strategies, mortality as mortality_mod, patient as patient_mod
from .._matlab_compat import mat_save
from ..params import default_params
from ..strategies import run_adaptive_custom


SWEEPS = [
    {"name": "q (process noise)",                "param": "q",
     "values": [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
     "default": 1e-4},
    {"name": "tau_x (mean-reversion, days)",     "param": "tau_x",
     "values": [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0],
     "default": 0.5},
    {"name": "P_pop (prior peak, sz/day)",       "param": "P_pop",
     "values": [1, 2, 3, 5, 7, 10, 15, 20],
     "default": 5},
    {"name": "floor (prior minimum)",            "param": "floor",
     "values": [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
     "default": 0.1},
    {"name": "omega (dose threshold, sz/day)",   "param": "omega",
     "values": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0],
     "default": 2.0},
]


def run(Npat: int = 300, Nruns_mort: int = 200,
        seed: int = 42, out_path: str | Path | None = None) -> dict:
    rng = np.random.default_rng(seed)
    time, drug_p, ctrl_p, mort_p = default_params()
    T, Nt, dt = time["T"], time["Nt"], time["dt"]
    theta_mort = mort_p["theta"]
    n_days = int(np.ceil(Nt * dt))

    # Patients shared across all sweeps
    patients = []
    for _ in range(Npat):
        u_pat = {k: rng.uniform() for k in
                 ("age", "sofa", "A_base", "r", "kappa",
                  "alpha", "C50", "gamma_hill")}
        patients.append(patient_mod.generate(
            T, {"peak_rate": 10, "drug_params": drug_p}, u_draws=u_pat))

    # Baselines
    surv_none  = np.zeros(Npat)
    surv_fixed = np.zeros(Npat)
    for p in range(Npat):
        pat = patients[p]
        dp = dict(drug_p, alpha=pat["alpha"])
        if "C50" in pat:
            dp["patient_C50"]   = pat["C50"]
            dp["patient_gamma"] = pat["gamma_hill"]
        for sname, surv_arr in (("none", surv_none), ("fixed", surv_fixed)):
            u_p = rng.uniform(size=Nt)
            r = strategies.run_strategy(
                sname, pat["lambda_base"], dp, ctrl_p, T, dt,
                use_ekf=True, u_draws={"poisson": u_p})
            alive = 0
            for _ in range(Nruns_mort):
                u_m = rng.uniform(size=n_days)
                a, _t = mortality_mod.run(
                    r["lambda_obs"], r["x"], dt, theta_mort,
                    pat["age"], pat["sofa"], mort_p, u_draws=u_m)
                alive += a
            surv_arr[p] = alive / Nruns_mort
    baseline_none  = float(np.mean(surv_none))  * 100
    baseline_fixed = float(np.mean(surv_fixed)) * 100

    # Sweeps
    results = []
    defaults = {sw["param"]: sw["default"] for sw in SWEEPS}
    for sw in SWEEPS:
        vals = np.asarray(sw["values"], dtype=float)
        surv_adaptive = np.zeros(vals.size)
        for vi, val in enumerate(vals):
            settings = dict(defaults)
            settings[sw["param"]] = val
            surv_pat = np.zeros(Npat)
            for p in range(Npat):
                pat = patients[p]
                dp = dict(drug_p, alpha=pat["alpha"])
                if "C50" in pat:
                    dp["patient_C50"]   = pat["C50"]
                    dp["patient_gamma"] = pat["gamma_hill"]
                u_p = rng.uniform(size=Nt)
                u_m = rng.uniform(size=(Nruns_mort, n_days))
                surv_pat[p] = run_adaptive_custom(
                    pat, dp, ctrl_p, T, dt, theta_mort,
                    Nruns_mort,
                    q_val=settings["q"], tau_x_val=settings["tau_x"],
                    P_pop_val=settings["P_pop"], floor_val=settings["floor"],
                    omega_val=settings["omega"],
                    u_draws={"poisson": u_p, "mortality": u_m},
                )
            surv_adaptive[vi] = float(np.mean(surv_pat)) * 100
        results.append({
            "name":     sw["name"],
            "param":    sw["param"],
            "values":   vals,
            "survival": surv_adaptive,
        })

    out = {
        "results":         np.array(results, dtype=object),
        "sweeps":          np.array(SWEEPS, dtype=object),
        "baseline_none":   baseline_none,
        "baseline_fixed":  baseline_fixed,
        "Npat":            Npat,
        "Nruns_mort":      Nruns_mort,
    }
    if out_path is not None:
        mat_save(out_path, **out)
    return out
