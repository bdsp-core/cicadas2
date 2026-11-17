"""Port of ``scripts/s7_estimation_robustness.m``.

Tests how parameter estimation degrades as the treatment fraction
increases (0% → 100% treated).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import strategies, estimation as estimation_mod, patient as patient_mod
from .._matlab_compat import mat_save
from ..params import default_params


def _build_lambda(psi, T, peak_rate):
    Ap, r, d = np.exp(psi)
    shape = np.exp(-d * T) * (1 - np.exp(-r * T))
    ms = float(shape.max())
    if ms < 1e-10:
        return np.zeros_like(T)
    return Ap * peak_rate * shape / ms


def run(Npat: int = 200, treatment_fracs=(0.0, 0.25, 0.5, 0.75, 1.0),
        seed: int = 42, out_path: str | Path | None = None) -> dict:
    rng = np.random.default_rng(seed)

    time, drug_p, ctrl_p, _ = default_params()
    T, Nt, dt = time["T"], time["Nt"], time["dt"]

    # Generate patient population once
    patients = []
    for _ in range(Npat):
        u_pat = {k: rng.uniform() for k in
                 ("age", "sofa", "A_base", "r", "kappa",
                  "alpha", "C50", "gamma_hill")}
        patients.append(patient_mod.generate(
            T, {"peak_rate": 10, "drug_params": drug_p}, u_draws=u_pat))

    treatment_fracs = np.asarray(treatment_fracs, dtype=float)
    n_fracs = treatment_fracs.size

    results = {
        "frac":           treatment_fracs,
        "corr_Lcum":      np.zeros(n_fracs),
        "corr_peak":      np.zeros(n_fracs),
        "corr_Lcum_ut":   np.zeros(n_fracs),
        "corr_Lcum_tx":   np.zeros(n_fracs),
        "corr_Ap":        np.zeros(n_fracs),
        "corr_r":         np.zeros(n_fracs),
        "corr_kappa":     np.zeros(n_fracs),
        "mape_Lcum":      np.zeros(n_fracs),
        "fim_cond":       np.zeros(n_fracs),
        "convergence":    np.zeros(n_fracs),
    }

    for fi, frac in enumerate(treatment_fracs):
        N_treated   = round(Npat * frac)
        N_untreated = Npat - N_treated

        sub_rng = np.random.default_rng(seed + fi + 1)
        order = sub_rng.permutation(Npat)

        true_Ap    = np.zeros(Npat)
        true_r     = np.zeros(Npat)
        true_kappa = np.zeros(Npat)
        est_Ap     = np.zeros(Npat)
        est_r      = np.zeros(Npat)
        est_kappa  = np.zeros(Npat)
        true_Lcum  = np.zeros(Npat)
        est_Lcum   = np.zeros(Npat)
        true_peak  = np.zeros(Npat)
        est_peak   = np.zeros(Npat)
        is_treated = np.zeros(Npat, dtype=bool)
        exitflags  = np.zeros(Npat, dtype=int)
        fim_conds  = np.full(Npat, np.inf)

        for p in range(Npat):
            pat = patients[order[p]]
            dp = dict(drug_p, alpha=pat["alpha"])
            if "C50" in pat:
                dp["patient_C50"]   = pat["C50"]
                dp["patient_gamma"] = pat["gamma_hill"]

            true_Ap[p]    = pat["A_patient"]
            true_r[p]     = pat["r"]
            true_kappa[p] = pat["kappa"]
            true_Lcum[p]  = float(np.sum(pat["lambda_base"]) * dt)
            true_peak[p]  = float(pat["lambda_base"].max())

            if p < N_untreated:
                counts = rng.poisson(pat["lambda_base"] * dt)
                params_est, fit_info = estimation_mod.estimate_natural_history(
                    counts, T, dt, peak_rate=10)
            else:
                is_treated[p] = True
                u_poisson = rng.uniform(size=Nt)
                r = strategies.run_strategy(
                    "fixed", pat["lambda_base"], dp, ctrl_p, T, dt,
                    use_ekf=False, u_draws={"poisson": u_poisson})
                kwargs = dict(peak_rate=10, k_e=dp["k_e"],
                              doses_given=r["doses_given"])
                if drug_p.get("pd_mode", "").lower() == "hill":
                    kwargs.update(pd_mode="hill",
                                  C50=dp["patient_C50"],
                                  gamma_hill=dp["patient_gamma"])
                else:
                    kwargs.update(pd_mode="linear", alpha=dp["alpha"])
                params_est, fit_info = estimation_mod.estimate_natural_history(
                    r["seizure_counts"], T, dt, **kwargs)

            est_Ap[p]    = params_est["A_p"]
            est_r[p]     = params_est["r"]
            est_kappa[p] = params_est["kappa"]
            exitflags[p] = fit_info["exitflag"]
            est_Lcum[p]  = float(np.sum(fit_info["lambda_base_hat"]) * dt)
            est_peak[p]  = float(fit_info["lambda_base_hat"].max())

            # FIM condition number
            psi = np.array([np.log(params_est["A_p"]),
                            np.log(params_est["r"]),
                            np.log(params_est["kappa"])])
            shape_est = np.exp(-params_est["kappa"] * T) * (1 - np.exp(-params_est["r"] * T))
            if shape_est.max() > 1e-10:
                lam_est = params_est["A_p"] * 10 * shape_est / shape_est.max()
                h_step = 0.01
                J = np.zeros((3, Nt))
                for dim in range(3):
                    plus  = psi.copy(); plus[dim]  += h_step
                    minus = psi.copy(); minus[dim] -= h_step
                    J[dim] = (_build_lambda(plus,  T, 10)
                              - _build_lambda(minus, T, 10)) / (2 * h_step)
                FIM = np.zeros((3, 3))
                for k in range(Nt):
                    if lam_est[k] > 1e-6:
                        FIM += (1 / lam_est[k]) * np.outer(J[:, k], J[:, k]) * dt
                eigs = np.sort(np.linalg.eigvalsh(FIM))[::-1]
                if eigs[-1] > 1e-15:
                    fim_conds[p] = float(eigs[0] / eigs[-1])

        results["corr_Lcum"][fi]   = float(np.corrcoef(true_Lcum,  est_Lcum)[0, 1])
        results["corr_peak"][fi]   = float(np.corrcoef(true_peak,  est_peak)[0, 1])
        results["corr_Ap"][fi]     = float(np.corrcoef(true_Ap,    est_Ap)[0, 1])
        results["corr_r"][fi]      = float(np.corrcoef(true_r,     est_r)[0, 1])
        results["corr_kappa"][fi]  = float(np.corrcoef(true_kappa, est_kappa)[0, 1])
        results["mape_Lcum"][fi]   = float(np.mean(
            np.abs(true_Lcum - est_Lcum) / np.maximum(true_Lcum, 0.1)) * 100)
        finite = fim_conds[np.isfinite(fim_conds)]
        results["fim_cond"][fi] = float(np.median(finite)) if finite.size else float("inf")
        results["convergence"][fi] = float(np.mean(exitflags == 1) * 100)
        idx_ut = ~is_treated
        idx_tx = is_treated
        results["corr_Lcum_ut"][fi] = (float(np.corrcoef(true_Lcum[idx_ut], est_Lcum[idx_ut])[0, 1])
                                       if idx_ut.sum() > 2 else float("nan"))
        results["corr_Lcum_tx"][fi] = (float(np.corrcoef(true_Lcum[idx_tx], est_Lcum[idx_tx])[0, 1])
                                       if idx_tx.sum() > 2 else float("nan"))

    out = {"results": results, "Npat": Npat,
           "treatment_fracs": treatment_fracs}
    if out_path is not None:
        mat_save(out_path, **out)
    return out
