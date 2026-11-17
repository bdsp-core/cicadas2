"""Port of ``functions/fcn_generate_patient.m``.

Stochastic patient generator. Bit-identical verification requires
RNG injection — see PYTHON_PORT_PLAN.md §4.2. The MATLAB
``fcn_generate_patient(T, params, u_draws)`` and the Python
``generate(T, params, u_draws=...)`` both consume the same uniform
tape and apply the matching inverse CDF, so identical ``u_draws``
yield identical patients.

``u_draws`` is a dict-like object with the following fields (only
the ones actually needed for the current call have to be present):

* ``age``         — Beta(3, 5) draw for age   (skipped if ``params['age']`` given)
* ``sofa``        — Gamma(2, 1.5) draw for SOFA (skipped if ``params['sofa']`` given)
* ``A_base``      — LogN(0, 0.8) draw for severity multiplier
* ``r``           — LogN(ln 1.5, 0.3) draw for rise rate
* ``kappa``       — LogN(ln 0.3, 0.4) draw for decay rate
* ``alpha``       — N(0, 1) draw for log-normal alpha random effect
                    (only if ``params['drug_params']`` is given)
* ``C50``         — N(0, 1) draw for log-normal C50 random effect
                    (only if Hill PD mode)
* ``gamma_hill``  — N(0, 1) draw for log-normal gamma random effect
                    (only if Hill PD mode)
"""

from __future__ import annotations

import numpy as np
import scipy.stats as st


def generate(T, params: dict, u_draws: dict | None = None) -> dict:
    """Return one synthetic patient as a dict matching the MATLAB struct."""
    T = np.asarray(T, dtype=float)
    use_u = u_draws is not None

    if "seed" in params and not use_u:
        np.random.seed(int(params["seed"]))

    peak_rate = float(params.get("peak_rate", 10))

    # ----- Age: Beta(3, 5) scaled to [40, 95], clamped to [18, 100] -----
    if "age" in params:
        age = float(params["age"])
    else:
        if use_u:
            beta_draw = st.beta.ppf(u_draws["age"], 3, 5)
        else:
            beta_draw = np.random.beta(3, 5)
        age = float(40 + 55 * beta_draw)
        age = max(18.0, min(100.0, age))

    # ----- SOFA: Gamma(2, 1.5), clamped to [0, 24], rounded ------------
    if "sofa" in params:
        sofa = float(params["sofa"])
    else:
        if use_u:
            gamma_draw = st.gamma.ppf(u_draws["sofa"], 2, scale=1.5)
        else:
            gamma_draw = np.random.gamma(2, 1.5)
        sofa = float(min(24, max(0, round(float(gamma_draw)))))

    # ----- Disease trajectory params -----------------------------------
    if use_u:
        A_base = st.lognorm.ppf(u_draws["A_base"], s=0.8,  scale=np.exp(0))
        r      = st.lognorm.ppf(u_draws["r"],      s=0.3,  scale=np.exp(np.log(1.5)))
        kappa  = st.lognorm.ppf(u_draws["kappa"],  s=0.4,  scale=np.exp(np.log(0.3)))
    else:
        A_base = np.random.lognormal(0, 0.8)
        r      = np.random.lognormal(np.log(1.5), 0.3)
        kappa  = np.random.lognormal(np.log(0.3), 0.4)

    A_patient = float(A_base * (1 + 0.05 * sofa))

    # Build the rise–decay trajectory
    sh = np.exp(-kappa * T) * (1 - np.exp(-r * T))
    lambda_base = A_patient * peak_rate * sh / np.max(sh)

    patient = {
        "age":         age,
        "sofa":        sofa,
        "A_patient":   A_patient,
        "r":           float(r),
        "kappa":       float(kappa),
        "lambda_base": lambda_base,
        "peak_rate":   peak_rate,
    }

    # ----- Drug sensitivity (linear alpha + optional Hill PD) ----------
    if "drug_params" in params:
        dp = params["drug_params"]
        sofa_norm = sofa / 10
        age_norm  = age / 65
        alpha_mean = (dp["alpha_base"]
                      * (1 + dp["alpha_sofa"] * sofa_norm
                          + dp["alpha_age"]  * age_norm))
        if use_u:
            alpha_random = np.exp(st.norm.ppf(u_draws["alpha"]) * 0.3)
        else:
            alpha_random = np.exp(np.random.randn() * 0.3)
        patient["alpha"] = float(max(0.001, alpha_mean * alpha_random))

        if dp.get("pd_mode", "").lower() == "hill":
            age_star  = (age  - 60) / 10
            sofa_star = (sofa -  3) /  2
            C50_mean = dp["C50"] + 50*age_star*(-0.3) + 50*sofa_star*(0.5)
            if use_u:
                C50_random = np.exp(st.norm.ppf(u_draws["C50"]) * 0.2)
            else:
                C50_random = np.exp(np.random.randn() * 0.2)
            patient["C50"] = float(max(50, C50_mean * C50_random))

            gamma_mean = dp["gamma_hill"] + 0.1*sofa_star + 0.05*age_star
            if use_u:
                gamma_random = np.exp(st.norm.ppf(u_draws["gamma_hill"]) * 0.15)
            else:
                gamma_random = np.exp(np.random.randn() * 0.15)
            patient["gamma_hill"] = float(max(0.5, gamma_mean * gamma_random))
    else:
        patient["alpha"] = 0.02

    return patient


# =====================================================================
# CICADAS SDE generator (port of fcn_generate_patient_cicadas.m)
# =====================================================================
_DEFAULT_POP_PARAMS = {
    "g":           4.0,    # logistic growth rate (1/day)
    "vartheta":    1.0,    # mean reversion strength (1/day)
    "omega_decay": 0.5,    # equilibrium decay rate (1/day)
    "eps_early":   0.5,    # early volatility coefficient
    "eps_late":    0.05,   # late volatility coefficient
    "tau_noise":   1.5,    # noise decay time constant (days)
    "t_switch":    1.5,    # early->late transition (days)
    "equil_frac":  0.2,    # fraction of H for initial late equilibrium
}


def generate_cicadas(T, params: dict, u_draws: dict | None = None) -> dict:
    """CICADAS-style SDE patient generator (Sec 4.1 of the paper).

    Bit-identical to ``fcn_generate_patient_cicadas.m`` when the same
    ``u_draws`` tape is provided. ``u_draws`` is a dict with the
    following fields (only those needed for the current call):

    * ``age``, ``sofa``       — Beta / Gamma demographics (skipped if overridden)
    * ``A_base``              — LogN(0, 0.8) severity multiplier
    * ``t_switch``            — LogN(0, 0.15) jitter on the growth → decay transition
    * ``sde``                 — length ``Nt-1`` uniforms for Euler-Maruyama steps
    * ``alpha``, ``C50``, ``gamma_hill`` — only if ``params['drug_params']``
    """
    T = np.asarray(T, dtype=float)
    use_u = u_draws is not None

    if "seed" in params and not use_u:
        np.random.seed(int(params["seed"]))

    peak_rate = float(params.get("peak_rate", 10))

    # Population SDE parameters with user-supplied overrides
    pop = dict(_DEFAULT_POP_PARAMS)
    if "pop_params" in params:
        pop.update(params["pop_params"])

    # ----- Demographics (same distributions as the rise-decay generator) -----
    if "age" in params:
        age = float(params["age"])
    else:
        if use_u:
            beta_draw = st.beta.ppf(u_draws["age"], 3, 5)
        else:
            beta_draw = np.random.beta(3, 5)
        age = float(40 + 55 * beta_draw)
        age = max(18.0, min(100.0, age))

    if "sofa" in params:
        sofa = float(params["sofa"])
    else:
        if use_u:
            gamma_draw = st.gamma.ppf(u_draws["sofa"], 2, scale=1.5)
        else:
            gamma_draw = np.random.gamma(2, 1.5)
        sofa = float(min(24, max(0, round(float(gamma_draw)))))

    # ----- Severity & switching-time jitter -----
    if use_u:
        A_base   = st.lognorm.ppf(u_draws["A_base"],   s=0.8,  scale=1.0)
        t_jitter = st.lognorm.ppf(u_draws["t_switch"], s=0.15, scale=1.0)
    else:
        A_base   = np.random.lognormal(0, 0.8)
        t_jitter = np.random.lognormal(0, 0.15)

    A_patient = float(A_base * (1 + 0.05 * sofa))
    H = A_patient * peak_rate

    t_switch = pop["t_switch"] * t_jitter
    t_switch = max(0.5, min(3.0, t_switch))

    # ----- Euler-Maruyama SDE -----
    dt = T[1] - T[0]
    Nt = T.size
    L = np.zeros(Nt)
    L[0] = 0.01 * H

    if use_u:
        sde_u = np.asarray(u_draws["sde"])
        sde_z = st.norm.ppf(sde_u)
    else:
        sde_z = np.random.randn(Nt - 1)

    sqrt_dt = np.sqrt(dt)
    for k in range(Nt - 1):
        t = T[k]
        Lk = L[k]

        if t < t_switch:
            mu = pop["g"] * Lk * (1 - Lk / H)
        else:
            equil = H * np.exp(-pop["omega_decay"] * (t - t_switch)) * pop["equil_frac"]
            mu = -pop["vartheta"] * (Lk - equil)
            mu = mu - pop["vartheta"] * max(0.0, Lk - H)

        vol = ((pop["eps_early"] * np.exp(-t / pop["tau_noise"]) + pop["eps_late"])
               * max(0.001, Lk))

        L[k + 1] = max(0.0, Lk + mu * dt + vol * sqrt_dt * sde_z[k])

    patient = {
        "age":         age,
        "sofa":        sofa,
        "A_patient":   A_patient,
        "r":           float("nan"),    # no parametric shape — SDE generated
        "kappa":       float("nan"),
        "lambda_base": L,
        "peak_rate":   peak_rate,
        "gen_mode":    "cicadas",
        "cicadas_pop":      pop,
        "cicadas_H":        H,
        "cicadas_t_switch": t_switch,
    }

    # ----- Drug sensitivity (identical block to the rise-decay generator) -----
    if "drug_params" in params:
        dp = params["drug_params"]
        sofa_norm = sofa / 10
        age_norm  = age / 65
        alpha_mean = (dp["alpha_base"]
                      * (1 + dp["alpha_sofa"] * sofa_norm
                          + dp["alpha_age"]  * age_norm))
        if use_u:
            alpha_random = np.exp(st.norm.ppf(u_draws["alpha"]) * 0.3)
        else:
            alpha_random = np.exp(np.random.randn() * 0.3)
        patient["alpha"] = float(max(0.001, alpha_mean * alpha_random))

        if dp.get("pd_mode", "").lower() == "hill":
            age_star  = (age  - 60) / 10
            sofa_star = (sofa -  3) /  2
            C50_mean = dp["C50"] + 50*age_star*(-0.3) + 50*sofa_star*(0.5)
            if use_u:
                C50_random = np.exp(st.norm.ppf(u_draws["C50"]) * 0.2)
            else:
                C50_random = np.exp(np.random.randn() * 0.2)
            patient["C50"] = float(max(50, C50_mean * C50_random))

            gamma_mean = dp["gamma_hill"] + 0.1*sofa_star + 0.05*age_star
            if use_u:
                gamma_random = np.exp(st.norm.ppf(u_draws["gamma_hill"]) * 0.15)
            else:
                gamma_random = np.exp(np.random.randn() * 0.15)
            patient["gamma_hill"] = float(max(0.5, gamma_mean * gamma_random))
    else:
        patient["alpha"] = 0.02

    return patient
