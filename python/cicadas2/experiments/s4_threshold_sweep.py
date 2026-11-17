"""Port of ``scripts/s4_threshold_sweep.m``.

Sweeps the PRN threshold and the adaptive ``omega`` to find optimal
values. Writes ``results/threshold_sweep_results.mat`` with the same
schema as the MATLAB version, so ``make_paper_figures.py`` works on
either output unchanged.

Verification: distributional equivalence within MC tolerance against
the MATLAB cached output (see ``tests/test_s4_threshold_sweep.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import strategies, mortality as mortality_mod, patient as patient_mod
from .._matlab_compat import mat_save
from ..params import default_params


def run(
    Npat: int = 200,
    Nruns: int = 50,
    use_ekf: bool = True,
    seed: int = 42,
    prn_thresholds=(2, 3, 5, 7, 10, 15),
    omega_values=(1, 2, 3, 5, 8, 10),
    out_path: str | Path | None = None,
) -> dict:
    """Run the threshold sweep and return / save the results dict.

    Parameters mirror the MATLAB script's defaults exactly. ``seed``
    is the master RNG seed used for patient generation and all
    stochastic draws (Poisson, mortality).
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)   # for any cicadas2.* code that calls np.random.*

    time, drug_p, ctrl_p, mort_p = default_params()
    T  = time["T"]
    Nt = time["Nt"]
    dt = time["dt"]

    prn_thresholds = np.asarray(prn_thresholds, dtype=float)
    omega_values   = np.asarray(omega_values,   dtype=float)

    # --- Generate patient population once ------------------------------
    patient_params = {"peak_rate": 10, "drug_params": drug_p}
    patients = []
    for _ in range(Npat):
        # Inject per-patient uniforms so the population is reproducible
        u = {
            "age":        rng.uniform(),
            "sofa":       rng.uniform(),
            "A_base":     rng.uniform(),
            "r":          rng.uniform(),
            "kappa":      rng.uniform(),
            "alpha":      rng.uniform(),
            "C50":        rng.uniform(),
            "gamma_hill": rng.uniform(),
        }
        patients.append(patient_mod.generate(T, patient_params, u_draws=u))

    def _drug_params_for(pat):
        dp = dict(drug_p)
        dp["alpha"] = pat["alpha"]
        if "C50" in pat:
            dp["patient_C50"]   = pat["C50"]
            dp["patient_gamma"] = pat["gamma_hill"]
        return dp

    def _sweep_one_strategy(strategy: str, params_for_value, value_list):
        """For each value in `value_list`, run the strategy on every
        patient with Nruns mortality draws and return the summary
        struct.
        """
        n = len(value_list)
        surv_rate = np.zeros(n)
        burden    = np.zeros(n)
        drug      = np.zeros(n)
        surv_std  = np.zeros(n)
        for i, value in enumerate(value_list):
            ctrl = params_for_value(value)
            surv_pat   = np.zeros(Npat)
            burden_pat = np.zeros(Npat)
            drug_pat   = np.zeros(Npat)
            for p in range(Npat):
                pat = patients[p]
                dp  = _drug_params_for(pat)
                u_poisson = rng.uniform(size=Nt)
                r = strategies.run_strategy(
                    strategy, pat["lambda_base"], dp, ctrl,
                    T, dt, use_ekf=use_ekf,
                    u_draws={"poisson": u_poisson},
                )
                burden_pat[p] = r["burden"]
                drug_pat[p]   = r["total_dose"]

                n_days = int(np.ceil(Nt * dt))
                alive_sum = 0
                for _run in range(Nruns):
                    u_mort = rng.uniform(size=n_days)
                    a, _ = mortality_mod.run(
                        r["lambda_obs"], r["x"], dt, mort_p["theta"],
                        pat["age"], pat["sofa"], mort_p, u_draws=u_mort,
                    )
                    alive_sum += a
                surv_pat[p] = alive_sum / Nruns

            surv_rate[i] = float(np.mean(surv_pat)) * 100
            burden[i]    = float(np.mean(burden_pat))
            drug[i]      = float(np.mean(drug_pat))
            surv_std[i]  = float(np.std(surv_pat, ddof=1)) * 100 / np.sqrt(Npat)
        return surv_rate, burden, drug, surv_std

    # --- Baseline (no treatment) ---------------------------------------
    base_surv = np.zeros(Npat)
    base_burden = np.zeros(Npat)
    for p in range(Npat):
        pat = patients[p]
        dp  = _drug_params_for(pat)
        u_poisson = rng.uniform(size=Nt)
        r = strategies.run_strategy(
            "none", pat["lambda_base"], dp, ctrl_p,
            T, dt, use_ekf=use_ekf,
            u_draws={"poisson": u_poisson},
        )
        base_burden[p] = r["burden"]

        n_days = int(np.ceil(Nt * dt))
        alive_sum = 0
        for _run in range(Nruns):
            u_mort = rng.uniform(size=n_days)
            a, _ = mortality_mod.run(
                r["lambda_obs"], r["x"], dt, mort_p["theta"],
                pat["age"], pat["sofa"], mort_p, u_draws=u_mort,
            )
            alive_sum += a
        base_surv[p] = alive_sum / Nruns

    results_baseline = {
        "surv_rate": float(np.mean(base_surv)) * 100,
        "burden":    float(np.mean(base_burden)),
    }

    # --- Sweep PRN thresholds ------------------------------------------
    prn_surv, prn_burden, prn_drug, prn_std = _sweep_one_strategy(
        "prn",
        lambda thr: {**ctrl_p, "prn_threshold": thr},
        prn_thresholds,
    )
    results_prn = {
        "threshold": prn_thresholds,
        "surv_rate": prn_surv,
        "burden":    prn_burden,
        "drug":      prn_drug,
        "surv_std":  prn_std,
    }

    # --- Sweep omega (adaptive) ----------------------------------------
    om_surv, om_burden, om_drug, om_std = _sweep_one_strategy(
        "adaptive",
        lambda om: {**ctrl_p, "omega": om},
        omega_values,
    )
    results_omega = {
        "omega":     omega_values,
        "surv_rate": om_surv,
        "burden":    om_burden,
        "drug":      om_drug,
        "surv_std":  om_std,
    }

    out = {
        "results_prn":      results_prn,
        "results_omega":    results_omega,
        "results_baseline": results_baseline,
        "prn_thresholds":   prn_thresholds,
        "omega_values":     omega_values,
        "Npat":             Npat,
        "Nruns":            Nruns,
    }

    if out_path is not None:
        mat_save(out_path, **out)

    return out
