"""Port of ``scripts/s9_gformula_causal.m`` (the g-formula centerpiece).

Three estimates of the same average treatment effect (ATE):
  - RCT simulation (ground truth)
  - Naive observational analysis (biased by confounding by indication)
  - Parametric g-formula (debiased)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import strategies, estimation as estimation_mod, mortality as mortality_mod, patient as patient_mod
from .._matlab_compat import mat_save
from ..params import default_params


def _early_features(lam, peak_rate, early_steps):
    half = early_steps // 2
    L_mean   = float(np.mean(lam))
    L_max    = float(np.max(lam))
    raw_g    = (lam[-1] - lam[0]) / (lam[0] + 0.1)
    feat_mean   = L_mean / peak_rate
    feat_max    = L_max  / (2 * peak_rate)
    feat_growth = float(np.log1p(max(0.0, raw_g)))
    p_fit = np.polyfit(np.arange(1, early_steps + 1), lam, 1)
    slope = float(p_fit[0])
    feat_slope = slope * early_steps / max(L_max, 0.1)
    if half > 1:
        p1 = np.polyfit(np.arange(1, half + 1), lam[:half], 1)
        p2 = np.polyfit(np.arange(half + 1, early_steps + 1), lam[half:], 1)
        feat_accel = float((p2[0] - p1[0])) * early_steps / max(L_max, 0.1)
    else:
        feat_accel = 0.0
    return feat_mean, feat_max, feat_growth, feat_slope, feat_accel


def run(Npat: int = 1000, Nruns_mort: int = 500,
        use_ekf: bool = True, use_cicadas_dgp: bool = False,
        seed: int = 42, out_path: str | Path | None = None) -> dict:
    rng = np.random.default_rng(seed)

    time, drug_p, ctrl_p, _ = default_params()
    T, Nt, dt = time["T"], time["Nt"], time["dt"]

    theta_mort = (np.array([-4, 0.8, 3.0, 0.8, 0.8, 0.3])
                  if use_cicadas_dgp
                  else np.array([-5, 0.5, 1.5, 0.5, 0.5, 0.2]))

    # --- Step 1: Generate patient population --------------------------
    patients = []
    for _ in range(Npat):
        u_pat = {k: rng.uniform() for k in
                 ("age", "sofa", "A_base", "r", "kappa",
                  "alpha", "C50", "gamma_hill")}
        if use_cicadas_dgp:
            # SDE generator also needs t_switch + a long sde tape
            u_pat["t_switch"] = rng.uniform()
            u_pat["sde"]      = rng.uniform(size=Nt - 1)
            patients.append(patient_mod.generate_cicadas(
                T, {"peak_rate": 10, "drug_params": drug_p}, u_draws=u_pat))
        else:
            patients.append(patient_mod.generate(
                T, {"peak_rate": 10, "drug_params": drug_p}, u_draws=u_pat))

    # --- Step 2: RCT simulation ---------------------------------------
    rct_rng = np.random.default_rng(100)
    rct_assignment = rct_rng.uniform(size=Npat) > 0.5

    surv_rct_treated   = np.zeros(Npat)
    surv_rct_untreated = np.zeros(Npat)
    n_days = int(np.ceil(Nt * dt))

    def _run_pat(strat, pat, dp):
        u_poisson = rng.uniform(size=Nt)
        return strategies.run_strategy(
            strat, pat["lambda_base"], dp, ctrl_p, T, dt,
            use_ekf=use_ekf, u_draws={"poisson": u_poisson})

    def _mortality_mc(r, pat, theta, n_runs):
        alive = 0
        for _ in range(n_runs):
            u_mort = rng.uniform(size=n_days)
            a, _t = mortality_mod.run(
                r["lambda_obs"], r["x"], dt, theta,
                pat["age"], pat["sofa"], None, u_draws=u_mort)
            alive += a
        return alive

    for p in range(Npat):
        pat = patients[p]
        dp = dict(drug_p, alpha=pat["alpha"])
        if "C50" in pat:
            dp["patient_C50"]   = pat["C50"]
            dp["patient_gamma"] = pat["gamma_hill"]
        strat = "adaptive" if rct_assignment[p] else "none"
        r = _run_pat(strat, pat, dp)
        alive_sum = _mortality_mc(r, pat, theta_mort, Nruns_mort)
        if rct_assignment[p]:
            surv_rct_treated[p]   = alive_sum / Nruns_mort
        else:
            surv_rct_untreated[p] = alive_sum / Nruns_mort

    rct_surv_treated   = float(np.mean(surv_rct_treated[rct_assignment])) * 100
    rct_surv_untreated = float(np.mean(surv_rct_untreated[~rct_assignment])) * 100
    rct_ate = rct_surv_treated - rct_surv_untreated

    # --- Step 3: Observational simulation (confounding by indication) -
    early_steps = min(int(round(1.0 / dt)), Nt)
    obs_assignment = np.zeros(Npat, dtype=bool)
    obs_treat_prob = np.zeros(Npat)

    for p in range(Npat):
        pat = patients[p]
        lam = pat["lambda_base"][:early_steps]
        fm, fmax, fg, fsl, fa = _early_features(lam, 10, early_steps)
        sofa, age = pat["sofa"], pat["age"]
        if use_cicadas_dgp:
            z = (-1.5 + 1.5*fm + 1.2*fmax + 0.6*fg + 0.4*fsl + 0.3*fa
                 + 0.5*(sofa/10) - 0.3*(age/65) + rng.standard_normal()*0.3)
        else:
            z = (-0.5 + 0.8*fm + 0.6*fmax + 0.4*fg + 0.3*fsl + 0.3*fa
                 + 0.5*(sofa/10) - 0.3*(age/65) + rng.standard_normal()*0.3)
        obs_treat_prob[p]  = 1.0 / (1.0 + np.exp(-z))
        obs_assignment[p]  = bool(rng.uniform() < obs_treat_prob[p])

    # Naive observational analysis
    surv_obs_treated   = np.zeros(Npat)
    surv_obs_untreated = np.zeros(Npat)
    for p in range(Npat):
        pat = patients[p]
        dp = dict(drug_p, alpha=pat["alpha"])
        if "C50" in pat:
            dp["patient_C50"]   = pat["C50"]
            dp["patient_gamma"] = pat["gamma_hill"]
        strat = "adaptive" if obs_assignment[p] else "none"
        r = _run_pat(strat, pat, dp)
        alive_sum = _mortality_mc(r, pat, theta_mort, Nruns_mort)
        if obs_assignment[p]:
            surv_obs_treated[p]   = alive_sum / Nruns_mort
        else:
            surv_obs_untreated[p] = alive_sum / Nruns_mort

    naive_surv_treated   = float(np.mean(surv_obs_treated[obs_assignment])) * 100
    naive_surv_untreated = float(np.mean(surv_obs_untreated[~obs_assignment])) * 100
    naive_ate = naive_surv_treated - naive_surv_untreated

    # --- Step 4: G-formula ---------------------------------------------
    est_Ap    = np.zeros(Npat)
    est_r     = np.zeros(Npat)
    est_kappa = np.zeros(Npat)
    est_lambda_base = np.zeros((Npat, Nt))

    for p in range(Npat):
        pat = patients[p]
        dp = dict(drug_p, alpha=pat["alpha"])
        if "C50" in pat:
            dp["patient_C50"]   = pat["C50"]
            dp["patient_gamma"] = pat["gamma_hill"]
        if obs_assignment[p]:
            r = _run_pat("adaptive", pat, dp)
            kwargs = dict(peak_rate=10, k_e=dp["k_e"],
                          doses_given=r["doses_given"])
            if drug_p.get("pd_mode", "").lower() == "hill":
                kwargs.update(pd_mode="hill",
                              C50=dp["patient_C50"],
                              gamma_hill=dp["patient_gamma"])
            else:
                kwargs.update(pd_mode="linear", alpha=dp["alpha"])
            params_est, fit = estimation_mod.estimate_natural_history(
                r["seizure_counts"], T, dt, **kwargs)
        else:
            counts = rng.poisson(pat["lambda_base"] * dt)
            params_est, fit = estimation_mod.estimate_natural_history(
                counts, T, dt, peak_rate=10)
        est_Ap[p]    = params_est["A_p"]
        est_r[p]     = params_est["r"]
        est_kappa[p] = params_est["kappa"]
        est_lambda_base[p] = fit["lambda_base_hat"]

    gform_surv_treated   = np.zeros(Npat)
    gform_surv_untreated = np.zeros(Npat)
    for p in range(Npat):
        pat = patients[p]
        dp = dict(drug_p, alpha=pat["alpha"])
        if "C50" in pat:
            dp["patient_C50"]   = pat["C50"]
            dp["patient_gamma"] = pat["gamma_hill"]
        lam_base_est = est_lambda_base[p]
        u_t = rng.uniform(size=Nt)
        u_n = rng.uniform(size=Nt)
        r_treat = strategies.run_strategy(
            "adaptive", lam_base_est, dp, ctrl_p, T, dt,
            use_ekf=use_ekf, u_draws={"poisson": u_t})
        r_none = strategies.run_strategy(
            "none", lam_base_est, dp, ctrl_p, T, dt,
            use_ekf=use_ekf, u_draws={"poisson": u_n})
        gform_surv_treated[p]   = _mortality_mc(r_treat, pat, theta_mort, Nruns_mort) / Nruns_mort
        gform_surv_untreated[p] = _mortality_mc(r_none,  pat, theta_mort, Nruns_mort) / Nruns_mort

    gform_surv_treat_pct   = float(np.mean(gform_surv_treated))   * 100
    gform_surv_untreat_pct = float(np.mean(gform_surv_untreated)) * 100
    gform_ate = gform_surv_treat_pct - gform_surv_untreat_pct

    out = {
        "rct_surv_treated":       rct_surv_treated,
        "rct_surv_untreated":     rct_surv_untreated,
        "rct_ate":                rct_ate,
        "naive_surv_treated":     naive_surv_treated,
        "naive_surv_untreated":   naive_surv_untreated,
        "naive_ate":              naive_ate,
        "gform_surv_treat_pct":   gform_surv_treat_pct,
        "gform_surv_untreat_pct": gform_surv_untreat_pct,
        "gform_ate":              gform_ate,
        "obs_assignment":         obs_assignment.astype(int),
        "rct_assignment":         rct_assignment.astype(int),
        "obs_treat_prob":         obs_treat_prob,
        "est_Ap":                 est_Ap,
        "est_r":                  est_r,
        "est_kappa":              est_kappa,
        "gform_surv_treated":     gform_surv_treated,
        "gform_surv_untreated":   gform_surv_untreated,
        "Npat":                   Npat,
        "Nruns_mort":             Nruns_mort,
        "theta_mort":             theta_mort,
        "use_cicadas_dgp":        int(use_cicadas_dgp),
    }
    if out_path is not None:
        mat_save(out_path, **out)
    return out
