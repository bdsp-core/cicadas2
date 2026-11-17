"""Port of ``functions/fcn_run_strategy.m``.

Central per-patient simulator. Couples PK, PD, Poisson observations,
the EKF, and one of seven treatment strategies into a single loop.

This is the largest single function in the MATLAB codebase (271 LOC)
and the most-called: every experiment script invokes
``fcn_run_strategy`` per patient per strategy.

Strategies supported (mirrored 1:1 from the MATLAB switch):

* ``none``           — no treatment (baseline)
* ``fixed``          — constant q8h bolus of ``dose_standard``
* ``prn``            — q8h bolus if rolling 4-h seizure rate > ``prn_threshold``
                       (legacy, default) or, with ``prn_legacy=False``, if the
                       seizure count in the last ``prn_window_hours`` >= ``prn_min_count``
* ``prn_ekf``        — q8h bolus if EKF rate estimate > ``prn_threshold``
* ``adaptive``       — five-tier q8h bolus on ``lambda_hat`` vs ``omega``
* ``fixed_infusion`` — constant infusion rate (continuous PK input)
* ``pi``             — PI controller on ``lambda_hat`` → infusion rate

RNG note: the only stochastic step is the Poisson draw of seizure
counts at each time step. The function accepts an optional
``u_draws={'poisson': <Nt-length uniform tape>}`` argument; when
supplied, every Poisson draw becomes inverse-CDF on the matching
``u_draws['poisson'][k]``, giving bit-identical results to MATLAB's
``poissinv``.
"""

from __future__ import annotations

import numpy as np
import scipy.stats as st

from . import mortality as mortality_mod
from ._matlab_compat import poisson_ppf_scalar
from .pd import observed_rate


def run_strategy(
    strategy:    str,
    lambda_base: np.ndarray,
    drug_params: dict,
    ctrl_params: dict,
    T:           np.ndarray,
    dt:          float,
    use_ekf:     bool = True,
    u_draws:     dict | None = None,
) -> dict:
    lambda_base = np.asarray(lambda_base, dtype=float)
    T           = np.asarray(T,           dtype=float)
    Nt = lambda_base.size

    inject_u = u_draws is not None and "poisson" in u_draws
    if inject_u:
        u_poisson = np.asarray(u_draws["poisson"], dtype=float)

    # --- Unpack drug parameters --------------------------------------------
    k_e   = drug_params["k_e"]
    alpha = drug_params["alpha"]
    dosing_interval_days = drug_params["dosing_interval"] / 24

    if drug_params.get("pd_mode", "").lower() == "hill":
        pd_struct = {
            "mode":  "hill",
            "C50":   drug_params.get("patient_C50",   drug_params["C50"]),
            "gamma": drug_params.get("patient_gamma", drug_params["gamma_hill"]),
        }
    else:
        pd_struct = {"mode": "linear", "alpha": alpha}

    # Cache pd_struct fields locally for the hot loop (avoid dict lookup per step)
    _pd_is_hill = pd_struct["mode"] == "hill"
    _pd_C50     = pd_struct.get("C50",   0.0)
    _pd_gamma   = pd_struct.get("gamma", 0.0)
    _pd_alpha   = pd_struct.get("alpha", 0.0)

    # --- Unpack control parameters -----------------------------------------
    omega         = ctrl_params["omega"]
    prn_threshold = ctrl_params["prn_threshold"]
    # R&R task 3: legacy (default) = submitted 4-h rate rule; else count rule
    prn_legacy    = ctrl_params.get("prn_legacy", True)
    prn_win_steps = int(round(ctrl_params.get("prn_window_hours", 8) / 24 / dt))
    prn_min_count = ctrl_params.get("prn_min_count", 1)

    k_p   = ctrl_params.get("k_p",   50)
    k_i   = ctrl_params.get("k_i",   10)
    A_max = ctrl_params.get("A_max", 100)

    fixed_infusion_rate = drug_params.get("infusion_rate", 20)

    # --- EKF + PI state ----------------------------------------------------
    tau_x = 0.5
    q     = 1e-4
    phi   = float(np.exp(-dt / tau_x))

    ekf_x = 0.0
    ekf_P = 1.0
    I_k   = 0.0          # PI integral

    # --- Storage ------------------------------------------------------------
    x              = np.zeros(Nt)
    infusion_rate  = np.zeros(Nt)
    doses_given    = np.zeros(Nt)
    lambda_obs     = np.zeros(Nt)
    lambda_hat     = np.zeros(Nt)
    seizure_counts = np.zeros(Nt, dtype=float)

    # --- Population prior for EKF ------------------------------------------
    r_pop     = 1.5
    kappa_pop = 0.3
    peak_pop  = 5
    shape_pop = np.exp(-kappa_pop * T) * (1 - np.exp(-r_pop * T))
    mu_prior  = np.maximum(0.1, peak_pop * shape_pop / np.max(shape_pop))

    # --- Pill dosing time indices (mirrors MATLAB's mod-based scan) --------
    dose_indices = set()
    for i in range(Nt):
        time_since_dose = T[i] % dosing_interval_days
        if (time_since_dose < dt / 2
                or (dosing_interval_days - time_since_dose) < dt / 2):
            dose_indices.add(i)

    is_infusion = strategy.lower() in ("fixed_infusion", "pi")

    # --- Main simulation loop ----------------------------------------------
    for k in range(Nt):
        # Observed rate given current drug concentration (inline scalar PD)
        lb = lambda_base[k]
        xk = x[k]
        if _pd_is_hill:
            if xk > 0:
                s_x = 1.0 - 1.0 / ((_pd_C50 / xk) ** _pd_gamma + 1.0)
            else:
                s_x = 1.0
            lambda_obs[k] = lb * s_x
        else:
            v = lb - _pd_alpha * xk
            lambda_obs[k] = v if v > 0.0 else 0.0

        # Poisson seizure count
        lam_dt = lambda_obs[k] * dt
        if inject_u:
            seizure_counts[k] = float(poisson_ppf_scalar(float(u_poisson[k]), lam_dt))
        else:
            seizure_counts[k] = float(np.random.poisson(lam_dt))

        # EKF update
        if use_ekf:
            ekf_x = phi * ekf_x
            ekf_P = phi**2 * ekf_P + q

            lam_pred = mu_prior[k] * np.exp(ekf_x)
            yhat = lam_pred * dt
            H = yhat
            R = max(yhat, 1e-3)
            K = (ekf_P * H) / (H * ekf_P * H + R)
            ekf_x = ekf_x + K * (seizure_counts[k] - yhat)
            ekf_P = (1 - K * H) * ekf_P * (1 - K * H) + K * R * K

            lambda_hat[k] = mu_prior[k] * np.exp(ekf_x)
        else:
            lambda_hat[k] = lambda_obs[k]

        # Treatment decision (mirror MATLAB switch)
        strat = strategy.lower()
        if strat == "none":
            pass

        elif strat == "fixed":
            if k in dose_indices:
                doses_given[k] = ctrl_params["dose_standard"]

        elif strat == "prn":
            if k in dose_indices:
                if prn_legacy:
                    window_steps = int(round(4 / 24 / dt))
                    start_idx = max(0, k - window_steps)
                    recent_rate = float(np.sum(seizure_counts[start_idx:k + 1])) / max((k - start_idx + 1) * dt, dt)
                    give = recent_rate > prn_threshold
                else:
                    start_idx = max(0, k - prn_win_steps + 1)
                    give = float(np.sum(seizure_counts[start_idx:k + 1])) >= prn_min_count
                if give:
                    doses_given[k] = ctrl_params["dose_standard"]

        elif strat == "fixed_infusion":
            infusion_rate[k] = fixed_infusion_rate

        elif strat == "pi":
            e_k = lambda_hat[k] - omega
            if e_k > 0:
                I_k = I_k + e_k * dt
            else:
                I_k = I_k * 0.99
            I_k = max(0, min(I_k, A_max / k_i))
            A_k = k_p * max(0, e_k) + k_i * I_k
            infusion_rate[k] = min(A_max, max(0, A_k))

        elif strat == "adaptive":
            if k in dose_indices:
                lh = lambda_hat[k]
                if   lh < 0.5 * omega: doses_given[k] = 0
                elif lh < 1.0 * omega: doses_given[k] = ctrl_params["dose_low"]
                elif lh < 1.5 * omega: doses_given[k] = ctrl_params["dose_intermediate"]
                elif lh < 2.0 * omega: doses_given[k] = ctrl_params["dose_medium"]
                else:                  doses_given[k] = ctrl_params["dose_high"]

        elif strat == "prn_ekf":
            if k in dose_indices and lambda_hat[k] > prn_threshold:
                doses_given[k] = ctrl_params["dose_standard"]

        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        # PK update
        if k < Nt - 1:
            if is_infusion:
                x[k + 1] = x[k] * k_e + infusion_rate[k]
            else:
                x[k + 1] = x[k] * k_e + doses_given[k]

    # --- Metrics ------------------------------------------------------------
    burden          = float(np.sum(seizure_counts))
    expected_burden = float(np.sum(lambda_obs) * dt)
    drug_exposure   = float(np.sum(x) * dt)
    if is_infusion:
        total_dose = float(np.sum(infusion_rate))
        n_doses = float("nan")
    else:
        total_dose = float(np.sum(doses_given))
        n_doses = int(np.sum(doses_given > 0))

    results = {
        "strategy":        strategy,
        "is_infusion":     is_infusion,
        "lambda_obs":      lambda_obs,
        "lambda_hat":      lambda_hat,
        "x":               x,
        "doses_given":     doses_given,
        "infusion_rate":   infusion_rate,
        "seizure_counts":  seizure_counts,
        "dose_indices":    np.array(sorted(dose_indices)) + 1,   # 1-based, matches MATLAB
        "burden":          burden,
        "expected_burden": expected_burden,
        "drug_exposure":   drug_exposure,
        "total_dose":      total_dose,
        "n_doses":         n_doses,
        "use_ekf":         use_ekf,
    }
    if strategy.lower() == "pi":
        results["I_final"] = I_k

    return results


# =====================================================================
# Adaptive variants used by the sensitivity / misspecification scripts
# =====================================================================
def _adaptive_ekf_loop(patient, dp, ctrl_params, T, dt,
                      tau_x, q_val, P_pop, floor_val, omega_val,
                      u_draws):
    """Shared inner loop for the two adaptive variants below."""
    lambda_base = np.asarray(patient["lambda_base"], dtype=float)
    T  = np.asarray(T, dtype=float)
    Nt = T.size

    inject_p = u_draws is not None and "poisson" in u_draws
    if inject_p:
        u_poisson = np.asarray(u_draws["poisson"], dtype=float)

    pd_struct = {
        "mode":  "hill",
        "C50":   dp["patient_C50"],
        "gamma": dp["patient_gamma"],
    }
    _pd_C50   = pd_struct["C50"]
    _pd_gamma = pd_struct["gamma"]

    phi   = float(np.exp(-dt / tau_x))
    ekf_x = 0.0
    ekf_P = 1.0

    r_pop     = 1.5
    kappa_pop = 0.3
    shape_pop = np.exp(-kappa_pop * T) * (1 - np.exp(-r_pop * T))
    mu_prior  = np.maximum(floor_val, P_pop * shape_pop / np.max(shape_pop))

    k_e = dp["k_e"]
    dosing_interval_days = dp["dosing_interval"] / 24

    dose_indices = set()
    for i in range(Nt):
        tsd = T[i] % dosing_interval_days
        if tsd < dt / 2 or (dosing_interval_days - tsd) < dt / 2:
            dose_indices.add(i)

    x              = np.zeros(Nt)
    doses_given    = np.zeros(Nt)
    lambda_obs     = np.zeros(Nt)
    seizure_counts = np.zeros(Nt)
    lambda_hat     = np.zeros(Nt)

    for k in range(Nt):
        lb = lambda_base[k]
        xk = x[k]
        if xk > 0:
            s_x = 1.0 - 1.0 / ((_pd_C50 / xk) ** _pd_gamma + 1.0)
        else:
            s_x = 1.0
        lambda_obs[k] = lb * s_x

        lam_dt = lambda_obs[k] * dt
        if inject_p:
            seizure_counts[k] = float(poisson_ppf_scalar(float(u_poisson[k]), lam_dt))
        else:
            seizure_counts[k] = float(np.random.poisson(lam_dt))

        ekf_x = phi * ekf_x
        ekf_P = phi**2 * ekf_P + q_val
        lam_pred = mu_prior[k] * np.exp(ekf_x)
        yhat = lam_pred * dt
        H = yhat
        R = max(yhat, 1e-3)
        K = (ekf_P * H) / (H * ekf_P * H + R)
        ekf_x = ekf_x + K * (seizure_counts[k] - yhat)
        ekf_P = (1 - K * H) * ekf_P * (1 - K * H) + K * R * K
        lambda_hat[k] = mu_prior[k] * np.exp(ekf_x)

        if k in dose_indices:
            lh = lambda_hat[k]
            if   lh < 0.5 * omega_val: doses_given[k] = 0
            elif lh < 1.0 * omega_val: doses_given[k] = ctrl_params["dose_low"]
            elif lh < 1.5 * omega_val: doses_given[k] = ctrl_params["dose_intermediate"]
            elif lh < 2.0 * omega_val: doses_given[k] = ctrl_params["dose_medium"]
            else:                      doses_given[k] = ctrl_params["dose_high"]

        if k < Nt - 1:
            x[k + 1] = x[k] * k_e + doses_given[k]

    return {
        "lambda_obs":     lambda_obs,
        "x":              x,
        "doses_given":    doses_given,
        "seizure_counts": seizure_counts,
        "lambda_hat":     lambda_hat,
    }


def run_adaptive_with_ekf(patient, dp, ctrl_params, T, dt, ekf_ov,
                          u_draws=None):
    """Port of ``fcn_run_adaptive_with_ekf.m``.

    Returns a results dict compatible with the MATLAB output (same
    fields, no mortality MC).
    """
    out = _adaptive_ekf_loop(
        patient, dp, ctrl_params, T, dt,
        tau_x=ekf_ov["tau_x"], q_val=ekf_ov["q"],
        P_pop=ekf_ov["P_pop"], floor_val=ekf_ov["floor"],
        omega_val=ctrl_params["omega"],
        u_draws=u_draws,
    )
    out["strategy"] = "adaptive"
    return out


def run_adaptive_custom(patient, dp, cp, T, dt, theta_mort,
                        Nruns_mort, q_val, tau_x_val, P_pop_val,
                        floor_val, omega_val, u_draws=None):
    """Port of ``fcn_run_adaptive_custom.m``.

    Runs the adaptive simulator once, then averages survival across
    ``Nruns_mort`` independent mortality draws. Returns a single
    survival probability in [0, 1].
    """
    sim = _adaptive_ekf_loop(
        patient, dp, cp, T, dt,
        tau_x=tau_x_val, q_val=q_val,
        P_pop=P_pop_val, floor_val=floor_val,
        omega_val=omega_val,
        u_draws=u_draws,
    )

    inject_m = u_draws is not None and "mortality" in u_draws
    if inject_m:
        u_mort_all = np.atleast_2d(u_draws["mortality"])

    alive_sum = 0
    for run in range(Nruns_mort):
        u_mort = u_mort_all[run] if inject_m else None
        alive, _ = mortality_mod.run(
            lambda_obs=sim["lambda_obs"],
            x=sim["x"],
            dt=dt,
            theta=theta_mort,
            age=patient["age"],
            sofa=patient["sofa"],
            mort_params=None,
            u_draws=u_mort,
        )
        alive_sum += alive
    return alive_sum / Nruns_mort

