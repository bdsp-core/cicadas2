"""Port of ``scripts/s16_misspecification.m``.

Three misspecification experiments:
  1. Three-layer decomposition (EKF only / trajectory only / joint).
  2. Graded trajectory misspecification (mixture sweep).
  3. Mortality model misspecification (squared-burden DGP, linear estimation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import strategies, estimation as estimation_mod, mortality as mortality_mod, patient as patient_mod
from .._matlab_compat import mat_save
from ..params import default_params


def _patient_dp(pat, drug_p):
    dp = dict(drug_p, alpha=pat["alpha"])
    if "C50" in pat:
        dp["patient_C50"]   = pat["C50"]
        dp["patient_gamma"] = pat["gamma_hill"]
    return dp


def _assign_treatment(pat, early_steps, half_steps, peak_rate, rng):
    """8-feature confounding-by-indication logistic. Same as s9."""
    L_vec = pat["lambda_base"][:early_steps]
    L_max = float(np.max(L_vec))
    feat_mean   = float(np.mean(L_vec)) / peak_rate
    feat_max    = L_max / (2 * peak_rate)
    raw_g = (L_vec[-1] - L_vec[0]) / (L_vec[0] + 0.1)
    feat_growth = float(np.log1p(max(0.0, raw_g)))
    p_fit = np.polyfit(np.arange(1, early_steps + 1), L_vec, 1)
    feat_slope = float(p_fit[0]) * early_steps / max(L_max, 0.1)
    if half_steps > 1:
        p1 = np.polyfit(np.arange(1, half_steps + 1), L_vec[:half_steps], 1)
        p2 = np.polyfit(np.arange(half_steps + 1, early_steps + 1), L_vec[half_steps:], 1)
        feat_accel = (float(p2[0]) - float(p1[0])) * early_steps / max(L_max, 0.1)
    else:
        feat_accel = 0.0
    z = (-0.5 + 0.8*feat_mean + 0.6*feat_max + 0.4*feat_growth
         + 0.3*feat_slope + 0.3*feat_accel
         + 0.5*(pat["sofa"]/10) - 0.3*(pat["age"]/65)
         + rng.standard_normal()*0.3)
    return rng.uniform() < 1.0 / (1.0 + np.exp(-z))


def _mc_survival(r, dt, theta, pat, Nruns, mort_p, rng, use_squared=False):
    n_days = int(np.ceil(r["lambda_obs"].size * dt))
    alive = 0
    fn = mortality_mod.run_squared if use_squared else mortality_mod.run
    for _ in range(Nruns):
        u_m = rng.uniform(size=n_days)
        a, _t = fn(r["lambda_obs"], r["x"], dt, theta,
                   pat["age"], pat["sofa"], mort_p, u_draws=u_m)
        alive += a
    return alive / Nruns


def _run_gformula(patients, drug_p, ctrl_p, T, dt, theta_mort,
                  Npat, Nruns_mort, peak_rate, ekf_override,
                  rng, mort_p):
    Nt = T.size
    use_ekf = True
    use_custom = ekf_override is not None

    # ---- RCT --------------------------------------------------------
    rct_rng = np.random.default_rng(100)
    rct_assignment = rct_rng.uniform(size=Npat) > 0.5
    surv_rct_t = np.zeros(Npat)
    surv_rct_u = np.zeros(Npat)
    for p in range(Npat):
        pat = patients[p]
        dp  = _patient_dp(pat, drug_p)
        if rct_assignment[p]:
            if use_custom:
                u_p = rng.uniform(size=Nt)
                r = strategies.run_adaptive_with_ekf(
                    pat, dp, ctrl_p, T, dt, ekf_override,
                    u_draws={"poisson": u_p})
            else:
                u_p = rng.uniform(size=Nt)
                r = strategies.run_strategy(
                    "adaptive", pat["lambda_base"], dp, ctrl_p, T, dt,
                    use_ekf=use_ekf, u_draws={"poisson": u_p})
        else:
            u_p = rng.uniform(size=Nt)
            r = strategies.run_strategy(
                "none", pat["lambda_base"], dp, ctrl_p, T, dt,
                use_ekf=use_ekf, u_draws={"poisson": u_p})
        surv = _mc_survival(r, dt, theta_mort, pat, Nruns_mort, mort_p, rng)
        if rct_assignment[p]:
            surv_rct_t[p] = surv
        else:
            surv_rct_u[p] = surv
    rct_ate = (surv_rct_t[rct_assignment].mean()
               - surv_rct_u[~rct_assignment].mean()) * 100

    # ---- Observational ---------------------------------------------
    obs_rng = np.random.default_rng(200)
    early_steps = min(int(round(1.0 / dt)), Nt)
    half_steps  = early_steps // 2
    obs_assignment = np.array(
        [_assign_treatment(p, early_steps, half_steps, peak_rate, obs_rng)
         for p in patients],
        dtype=bool,
    )

    surv_obs_t = np.zeros(Npat)
    surv_obs_u = np.zeros(Npat)
    for p in range(Npat):
        pat = patients[p]
        dp  = _patient_dp(pat, drug_p)
        if obs_assignment[p]:
            if use_custom:
                u_p = rng.uniform(size=Nt)
                r = strategies.run_adaptive_with_ekf(
                    pat, dp, ctrl_p, T, dt, ekf_override,
                    u_draws={"poisson": u_p})
            else:
                u_p = rng.uniform(size=Nt)
                r = strategies.run_strategy(
                    "adaptive", pat["lambda_base"], dp, ctrl_p, T, dt,
                    use_ekf=use_ekf, u_draws={"poisson": u_p})
        else:
            u_p = rng.uniform(size=Nt)
            r = strategies.run_strategy(
                "none", pat["lambda_base"], dp, ctrl_p, T, dt,
                use_ekf=use_ekf, u_draws={"poisson": u_p})
        surv = _mc_survival(r, dt, theta_mort, pat, Nruns_mort, mort_p, rng)
        if obs_assignment[p]:
            surv_obs_t[p] = surv
        else:
            surv_obs_u[p] = surv
    naive_ate = (surv_obs_t[obs_assignment].mean()
                 - surv_obs_u[~obs_assignment].mean()) * 100

    # ---- G-formula --------------------------------------------------
    est_lambda_base = np.zeros((Npat, Nt))
    for p in range(Npat):
        pat = patients[p]
        dp  = _patient_dp(pat, drug_p)
        if obs_assignment[p]:
            if use_custom:
                u_p = rng.uniform(size=Nt)
                r = strategies.run_adaptive_with_ekf(
                    pat, dp, ctrl_p, T, dt, ekf_override,
                    u_draws={"poisson": u_p})
            else:
                u_p = rng.uniform(size=Nt)
                r = strategies.run_strategy(
                    "adaptive", pat["lambda_base"], dp, ctrl_p, T, dt,
                    use_ekf=use_ekf, u_draws={"poisson": u_p})
            kw = dict(peak_rate=peak_rate, k_e=dp["k_e"],
                      doses_given=r["doses_given"])
            if drug_p.get("pd_mode", "").lower() == "hill":
                kw.update(pd_mode="hill", C50=dp["patient_C50"],
                          gamma_hill=dp["patient_gamma"])
            else:
                kw.update(pd_mode="linear", alpha=dp["alpha"])
            _, fi = estimation_mod.estimate_natural_history(
                r["seizure_counts"], T, dt, **kw)
        else:
            counts = rng.poisson(pat["lambda_base"] * dt)
            _, fi = estimation_mod.estimate_natural_history(
                counts, T, dt, peak_rate=peak_rate)
        est_lambda_base[p] = fi["lambda_base_hat"]

    gform_surv_t = np.zeros(Npat)
    gform_surv_u = np.zeros(Npat)
    for p in range(Npat):
        pat = patients[p]
        dp  = _patient_dp(pat, drug_p)
        lam_est = est_lambda_base[p]
        u_t = rng.uniform(size=Nt)
        u_n = rng.uniform(size=Nt)
        r_t = strategies.run_strategy(
            "adaptive", lam_est, dp, ctrl_p, T, dt,
            use_ekf=use_ekf, u_draws={"poisson": u_t})
        r_n = strategies.run_strategy(
            "none", lam_est, dp, ctrl_p, T, dt,
            use_ekf=use_ekf, u_draws={"poisson": u_n})
        gform_surv_t[p] = _mc_survival(r_t, dt, theta_mort, pat, Nruns_mort, mort_p, rng)
        gform_surv_u[p] = _mc_survival(r_n, dt, theta_mort, pat, Nruns_mort, mort_p, rng)

    gform_ate = (gform_surv_t.mean() - gform_surv_u.mean()) * 100
    return rct_ate, naive_ate, gform_ate


def _run_gformula_mort_misspec(patients, drug_p, ctrl_p, T, dt,
                               theta_gen, theta_est, Npat, Nruns_mort,
                               peak_rate, rng, mort_p):
    """RCT + naive + g-formula where DGP uses squared-burden mortality
    but the g-formula estimates with linear mortality."""
    Nt = T.size

    # RCT (squared mortality)
    rct_rng = np.random.default_rng(100)
    rct_assignment = rct_rng.uniform(size=Npat) > 0.5
    surv_rct_t = np.zeros(Npat)
    surv_rct_u = np.zeros(Npat)
    for p in range(Npat):
        pat = patients[p]
        dp  = _patient_dp(pat, drug_p)
        u_p = rng.uniform(size=Nt)
        strat = "adaptive" if rct_assignment[p] else "none"
        r = strategies.run_strategy(strat, pat["lambda_base"], dp, ctrl_p, T, dt,
                                    use_ekf=True, u_draws={"poisson": u_p})
        surv = _mc_survival(r, dt, theta_gen, pat, Nruns_mort, mort_p, rng,
                            use_squared=True)
        if rct_assignment[p]: surv_rct_t[p] = surv
        else:                 surv_rct_u[p] = surv
    rct_ate = (surv_rct_t[rct_assignment].mean()
               - surv_rct_u[~rct_assignment].mean()) * 100

    # Observational (squared mortality)
    obs_rng = np.random.default_rng(200)
    early_steps = min(int(round(1.0 / dt)), Nt)
    half_steps  = early_steps // 2
    obs_assignment = np.array(
        [_assign_treatment(p, early_steps, half_steps, peak_rate, obs_rng)
         for p in patients], dtype=bool)

    surv_obs_t = np.zeros(Npat)
    surv_obs_u = np.zeros(Npat)
    for p in range(Npat):
        pat = patients[p]
        dp  = _patient_dp(pat, drug_p)
        u_p = rng.uniform(size=Nt)
        strat = "adaptive" if obs_assignment[p] else "none"
        r = strategies.run_strategy(strat, pat["lambda_base"], dp, ctrl_p, T, dt,
                                    use_ekf=True, u_draws={"poisson": u_p})
        surv = _mc_survival(r, dt, theta_gen, pat, Nruns_mort, mort_p, rng,
                            use_squared=True)
        if obs_assignment[p]: surv_obs_t[p] = surv
        else:                 surv_obs_u[p] = surv
    naive_ate = (surv_obs_t[obs_assignment].mean()
                 - surv_obs_u[~obs_assignment].mean()) * 100

    # G-formula counterfactual with LINEAR mortality (misspec)
    est_lambda_base = np.zeros((Npat, Nt))
    for p in range(Npat):
        pat = patients[p]
        dp  = _patient_dp(pat, drug_p)
        if obs_assignment[p]:
            u_p = rng.uniform(size=Nt)
            r = strategies.run_strategy(
                "adaptive", pat["lambda_base"], dp, ctrl_p, T, dt,
                use_ekf=True, u_draws={"poisson": u_p})
            kw = dict(peak_rate=peak_rate, k_e=dp["k_e"],
                      doses_given=r["doses_given"])
            if drug_p.get("pd_mode", "").lower() == "hill":
                kw.update(pd_mode="hill", C50=dp["patient_C50"],
                          gamma_hill=dp["patient_gamma"])
            else:
                kw.update(pd_mode="linear", alpha=dp["alpha"])
            _, fi = estimation_mod.estimate_natural_history(
                r["seizure_counts"], T, dt, **kw)
        else:
            counts = rng.poisson(pat["lambda_base"] * dt)
            _, fi = estimation_mod.estimate_natural_history(
                counts, T, dt, peak_rate=peak_rate)
        est_lambda_base[p] = fi["lambda_base_hat"]

    gform_surv_t = np.zeros(Npat)
    gform_surv_u = np.zeros(Npat)
    for p in range(Npat):
        pat = patients[p]
        dp  = _patient_dp(pat, drug_p)
        lam_est = est_lambda_base[p]
        u_t = rng.uniform(size=Nt)
        u_n = rng.uniform(size=Nt)
        r_t = strategies.run_strategy(
            "adaptive", lam_est, dp, ctrl_p, T, dt,
            use_ekf=True, u_draws={"poisson": u_t})
        r_n = strategies.run_strategy(
            "none", lam_est, dp, ctrl_p, T, dt,
            use_ekf=True, u_draws={"poisson": u_n})
        gform_surv_t[p] = _mc_survival(r_t, dt, theta_est, pat, Nruns_mort, mort_p, rng,
                                       use_squared=False)
        gform_surv_u[p] = _mc_survival(r_n, dt, theta_est, pat, Nruns_mort, mort_p, rng,
                                       use_squared=False)

    gform_ate = (gform_surv_t.mean() - gform_surv_u.mean()) * 100
    return rct_ate, naive_ate, gform_ate


def run(Npat: int = 500, Nruns_mort: int = 300,
        seed: int = 42, out_path: str | Path | None = None) -> dict:
    rng = np.random.default_rng(seed)
    time, drug_p, ctrl_p, mort_p = default_params()
    T, Nt, dt = time["T"], time["Nt"], time["dt"]
    peak_rate = 10

    theta_mort = np.array([-5, 0.5, 1.5, 0.5, 0.5, 0.2])
    ekf_misspec = {"P_pop": 20, "tau_x": 2.0, "q": 1e-2, "floor": 1.0}

    # Generate two patient populations
    patients_sde = []
    patients_rd  = []
    for _ in range(Npat):
        u_pat = {k: rng.uniform() for k in
                 ("age", "sofa", "A_base", "r", "kappa",
                  "alpha", "C50", "gamma_hill")}
        u_sde = dict(u_pat, t_switch=rng.uniform(), sde=rng.uniform(size=Nt-1))
        patients_sde.append(patient_mod.generate_cicadas(
            T, {"peak_rate": peak_rate, "drug_params": drug_p}, u_draws=u_sde))
        u_rd = {k: rng.uniform() for k in
                ("age", "sofa", "A_base", "r", "kappa",
                 "alpha", "C50", "gamma_hill")}
        patients_rd.append(patient_mod.generate(
            T, {"peak_rate": peak_rate, "drug_params": drug_p}, u_draws=u_rd))

    # EXP 1: three-layer decomposition
    ref_rct, _, ref_gform = _run_gformula(
        patients_rd, drug_p, ctrl_p, T, dt, theta_mort,
        Npat, Nruns_mort, peak_rate, None, rng, mort_p)
    ref_bias = abs(ref_gform - ref_rct)
    A_rct, _, A_gform = _run_gformula(
        patients_rd, drug_p, ctrl_p, T, dt, theta_mort,
        Npat, Nruns_mort, peak_rate, ekf_misspec, rng, mort_p)
    A_bias = abs(A_gform - A_rct)
    B_rct, _, B_gform = _run_gformula(
        patients_sde, drug_p, ctrl_p, T, dt, theta_mort,
        Npat, Nruns_mort, peak_rate, None, rng, mort_p)
    B_bias = abs(B_gform - B_rct)
    C_rct, _, C_gform = _run_gformula(
        patients_sde, drug_p, ctrl_p, T, dt, theta_mort,
        Npat, Nruns_mort, peak_rate, ekf_misspec, rng, mort_p)
    C_bias = abs(C_gform - C_rct)

    # EXP 2: graded mixture
    mix_fractions = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                              0.7, 0.8, 0.9, 1.0])
    graded_rct   = np.zeros(mix_fractions.size)
    graded_gform = np.zeros(mix_fractions.size)
    for mi, frac in enumerate(mix_fractions):
        n_sde = round(frac * Npat)
        n_rd  = Npat - n_sde
        mix_rng = np.random.default_rng(seed + mi + 1)
        idx = mix_rng.permutation(Npat)
        mixed = ([patients_rd[i] for i in range(n_rd)]
                 + [patients_sde[i] for i in range(n_sde)])
        mixed = [mixed[i] for i in idx]
        r_ate, _, g_ate = _run_gformula(
            mixed, drug_p, ctrl_p, T, dt, theta_mort,
            Npat, Nruns_mort, peak_rate, None, rng, mort_p)
        graded_rct[mi]   = r_ate
        graded_gform[mi] = g_ate
    graded_bias = np.abs(graded_gform - graded_rct)

    # EXP 3: mortality misspecification
    theta_squared = np.array([-4, 0.5, 1.0, 0.7, 0.10, 0.05])
    mort_rct, _, mort_gform = _run_gformula_mort_misspec(
        patients_rd, drug_p, ctrl_p, T, dt,
        theta_squared, theta_mort, Npat, Nruns_mort, peak_rate, rng, mort_p)
    mort_bias = abs(mort_gform - mort_rct)

    out = {
        "ref_rct_ate":      ref_rct,
        "ref_gform_ate":    ref_gform,
        "ref_gform_bias":   ref_bias,
        "layerA_rct_ate":   A_rct,
        "layerA_gform_ate": A_gform,
        "layerA_gform_bias": A_bias,
        "layerB_rct_ate":   B_rct,
        "layerB_gform_ate": B_gform,
        "layerB_gform_bias": B_bias,
        "layerC_rct_ate":   C_rct,
        "layerC_gform_ate": C_gform,
        "layerC_gform_bias": C_bias,
        "mix_fractions":    mix_fractions,
        "graded_rct":       graded_rct,
        "graded_gform":     graded_gform,
        "graded_bias":      graded_bias,
        "mort_rct_ate":     mort_rct,
        "mort_gform_ate":   mort_gform,
        "mort_gform_bias":  mort_bias,
        "Npat":             Npat,
        "Nruns_mort":       Nruns_mort,
        "ekf_misspec":      ekf_misspec,
        "theta_squared":    theta_squared,
    }
    if out_path is not None:
        mat_save(out_path, **out)
    return out
