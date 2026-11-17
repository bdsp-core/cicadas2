"""Vectorised strategy simulator: every patient advanced together, one time step at a time.

Mirrors ``strategies.run_strategy`` exactly (same PK, PD, EKF, dosing
rules and floating-point expressions) but advances all n patients in
one numpy step, so 1,000 patients cost about as much as a handful run
one at a time. ``tests/test_rr_engine.py`` checks identical seizure counts and doses (continuous quantities to ~1e-15, from scalar vs array pow)
with ``run_strategy`` when both receive the same Poisson uniform tape.

Extra options (all default off, so defaults reproduce run_strategy):
  prn_window_hours / prn_min_count / prn_legacy  - PRN definition (task 3)
  omega as a per-patient array                     - personalised thresholds
  tier_bounds / tier_doses                         - alternative dose menus (R1 minor 3)
  detect_sens / detect_fp_per_day                  - seizure annotation error (R1 M1a)
  ekf = dict(tau_x, q, P_pop, floor)               - EKF settings
  rate_source = "ma", ma_hours = 24                - adaptive tiers driven by a trailing count average
  dynamic = dict(theta, age, sofa)                 - dynamic omega(t) controller (Sec 5.3.3)
  pk2 = dict(k12_per_h, k21_per_h)                 - two-compartment PK (R1 M5)
  c50_drift_per_day                                - drug tolerance (R2 15)
  store = ("lambda_hat", "ekf_P", "counts", "counts_obs", "doses", "lambda_obs", "x")
  loading_dose (mg)                                - fixed strategy: first dose replaced by this amount
  strategy = "harm", harm = dict(...)              - harm-balanced controller (see harm_dose_choice)
  ekf = dict(..., drug_aware=True)                 - EKF tracks the drug-free rate (measurement = mu e^x s(x))
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st

DAY_ENDS = 8  # store 8 day-end values (legacy hazard loop reads 8)


def _poisson_inv(u: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Vectorised Poisson inverse CDF with the same arithmetic as poisson_ppf_scalar."""
    n = np.zeros(u.shape, dtype=float)
    pos = m > 0.0
    if not np.any(pos):
        return n
    p = np.where(pos, np.exp(-np.where(pos, m, 0.0)), 0.0)
    cum = p.copy()
    k = 0
    active = pos & (cum < u) & (u < 1.0)
    while np.any(active):
        k += 1
        p = np.where(active, p * m / k, p)
        cum = np.where(active, cum + p, cum)
        n = np.where(active, k, n)
        active = active & (cum < u)
        if k > 10000:
            break
    n = np.where(pos & (u >= 1.0), 10000, n)
    return n


def dose_index_mask(T: np.ndarray, dt: float, interval_hours: float) -> np.ndarray:
    di = interval_hours / 24
    tsd = T % di
    return (tsd < dt / 2) | ((di - tsd) < dt / 2)


def hill_s(x, C50, gam):
    """Surviving fraction of seizures at concentration x (same Hill form as the PD step)."""
    x = np.asarray(x, float)
    with np.errstate(divide="ignore", over="ignore"):
        return np.where(x > 0, 1.0 / (1.0 + (x / C50) ** gam), 1.0)


def harm_weights(theta, age, sofa):
    """Instantaneous harm weights of the linear-cumulant hazard.

    z_d = c_d + integral_0^{end of day d} f(t) dt with f = a*lambda_base*s(x) + b*x, a = (th2 + th3*SOFA/10)/35 and
    b = (th4 + th5*Age/65)/5000, so lowering f at any instant lowers every later z_d (the Proposition in the
    revision notes). The controller therefore minimises the integral of f over its horizon.
    """
    th = np.asarray(theta, float)
    a = (th[2] + th[3] * np.asarray(sofa, float) / 10) / 35.0
    b = (th[4] + th[5] * np.asarray(age, float) / 65) / 5000.0
    return a, b


def harm_dose_choice(x_now, lam_fc, a, b, C50, gam, k_e, m, d_max, n_grid=31):
    """Dose (mg) minimising the forecast harm over the horizon.

    x_now: (n,) concentration before this step's decay; lam_fc: (n, H) forecast drug-free rate per step of the
    horizon; the profile after a dose D is periodic with peak P = x_now*k_e + D (later doses top the level back up
    to P every m steps), so P is the decision and loading/maintenance follow from the residual. Returns (n,) doses.
    """
    n, H = lam_fc.shape
    D = np.linspace(0.0, 1.0, n_grid)[None, :] * np.asarray(d_max, float).reshape(-1, 1)    # (n, G)
    P = (x_now * k_e)[:, None] + D                                                          # (n, G)
    decay = k_e ** (np.arange(H) % m)                                                       # (H,)
    xs = P[:, :, None] * decay[None, None, :]                                               # (n, G, H)
    C = np.asarray(C50, float).reshape(-1, 1, 1)
    g = np.asarray(gam, float).reshape(-1, 1, 1)
    with np.errstate(divide="ignore", over="ignore"):
        s_x = np.where(xs > 0, 1.0 / (1.0 + (xs / C) ** g), 1.0)
    cost = (np.asarray(a, float).reshape(-1, 1, 1) * lam_fc[:, None, :] * s_x
            + np.asarray(b, float).reshape(-1, 1, 1) * xs).sum(axis=2)                      # (n, G)
    return D[np.arange(n), cost.argmin(axis=1)]


def simulate(spec: dict, LB: np.ndarray, C50: np.ndarray, gam: np.ndarray, U: np.ndarray,
             drug: dict, ctrl: dict, T: np.ndarray, dt: float,
             U_thin: np.ndarray | None = None, U_fp: np.ndarray | None = None,
             store: tuple = ()) -> dict:
    LB = np.atleast_2d(np.asarray(LB, float))
    n, Nt = LB.shape
    C50 = np.broadcast_to(np.asarray(C50, float), (n,))
    gam = np.broadcast_to(np.asarray(gam, float), (n,))
    strat = spec.get("strategy", "none").lower()
    use_ekf = spec.get("use_ekf", True)

    k_e = drug["k_e"]
    is_dose = dose_index_mask(T, dt, drug["dosing_interval"])
    omega = np.broadcast_to(np.asarray(spec.get("omega", ctrl["omega"]), float), (n,))
    prn_thr = spec.get("prn_threshold", ctrl["prn_threshold"])
    prn_legacy = spec.get("prn_legacy", True)   # engine default = submitted rule; experiments set it
    prn_win = int(round(spec.get("prn_window_hours", 8) / 24 / dt))
    prn_min = spec.get("prn_min_count", 1)
    dose_std = spec.get("dose_standard", ctrl["dose_standard"])
    bounds = np.asarray(spec.get("tier_bounds", [0.5, 1.0, 1.5, 2.0]), float)
    doses = np.asarray(spec.get("tier_doses", [0, ctrl["dose_low"], ctrl["dose_intermediate"],
                                                ctrl["dose_medium"], ctrl["dose_high"]]), float)
    sens = spec.get("detect_sens", 1.0)
    fp = spec.get("detect_fp_per_day", 0.0)
    detect = (sens < 1.0) or (fp > 0.0)

    # Dynamic omega(t) controller of Sec 5.3.3 (port of run_dynamic_adaptive in s13)
    dyn = spec.get("dynamic")
    if dyn is not None:
        th = np.asarray(dyn["theta"], float)
        sz_c = th[2] + th[3] * np.asarray(dyn["sofa"], float) / 10
        dr_c = th[4] + th[5] * np.asarray(dyn["age"], float) / 65
        omega = np.clip(2 * dr_c / np.maximum(sz_c, 0.01), 0.5, 10).astype(float)
        r_hi, r_lo = dyn.get("r_hi", 1.5), dyn.get("r_lo", 0.5)
        f_up, f_dn = dyn.get("f_up", 1.3), dyn.get("f_dn", 0.7)
        d_lam = np.zeros(n); d_x = np.zeros(n); d_cnt = 0; last_day = 0
    # Data-generating variants (R1 M5, R2 15): two-compartment PK and drug tolerance
    pk2 = spec.get("pk2")
    if pk2 is not None:
        dth = dt * 24
        f12 = pk2["k12_per_h"] * dth
        f21 = pk2["k21_per_h"] * dth
        y_per = np.zeros(n)
    tol = spec.get("c50_drift_per_day", 0.0)
    load = spec.get("loading_dose")
    first_dose_done = False
    hm = spec.get("harm") if strat == "harm" else None
    if hm is not None:
        h_a, h_b = harm_weights(hm["theta"], hm["age"], hm["sofa"])
        h_a = np.broadcast_to(h_a, (n,)); h_b = np.broadcast_to(h_b, (n,))
        h_oracle = hm.get("oracle")                    # None (estimated), "rate" (true lambda_base), "full" (+ own PD)
        h_C50 = C50 if h_oracle == "full" else np.full(n, hm.get("C50", drug["C50"]), float)
        h_gam = gam if h_oracle == "full" else np.full(n, hm.get("gam", drug["gamma_hill"]), float)
        h_m = int(round(drug["dosing_interval"] / 24 / dt))
        h_H = int(round(hm.get("horizon_h", 24) / 24 / dt))
        h_dmax = np.broadcast_to(np.asarray(hm.get("d_max", 750), float), (n,))
        h_dmax_load = np.broadcast_to(np.asarray(hm.get("d_max_load", hm.get("d_max", 750)), float), (n,))
        h_post = hm.get("posterior", "mean")

    ek = spec.get("ekf", {})
    drug_aware = ek.get("drug_aware", hm is not None and hm.get("oracle") is None)
    tau_x = ek.get("tau_x", 0.5)
    q = ek.get("q", 1e-4)
    P_pop = ek.get("P_pop", 5)
    floor = ek.get("floor", 0.1)
    phi = float(np.exp(-dt / tau_x))
    shape_pop = np.exp(-0.3 * T) * (1 - np.exp(-1.5 * T))
    mu_prior = np.maximum(floor, P_pop * shape_pop / np.max(shape_pop))

    x = np.zeros(n)
    ekf_x = np.zeros(n)
    ekf_P = np.ones(n)
    Lc = np.zeros(n)
    Xc = np.zeros(n)
    day_idx = np.minimum(np.round(np.arange(1, DAY_ENDS + 1) / dt).astype(int), Nt) - 1
    Lday = np.zeros((n, DAY_ENDS))
    Xday = np.zeros((n, DAY_ENDS))
    total_dose = np.zeros(n)
    n_doses = np.zeros(n)
    burden_counts = np.zeros(n)
    want = set(store)
    keep = {k: np.zeros((n, Nt)) for k in want}
    # rolling window of observed counts for PRN
    rate_src = spec.get("rate_source", "ekf")          # "ekf" or "ma" (trailing moving average of counts)
    ma_steps = int(round(spec.get("ma_hours", 24) / 24 / dt))
    cnt_hist = np.zeros((n, Nt)) if (strat == "prn" or "counts_obs" in want or rate_src == "ma") else None
    legacy_steps = int(round(4 / 24 / dt))
    d_ptr = 0

    for k in range(Nt):
        # PD (same expression as run_strategy)
        s_x = np.ones(n)
        pos = x > 0
        if np.any(pos):
            C50k = C50[pos] * (1.0 + tol * T[k]) if tol else C50[pos]
            s_x[pos] = 1.0 - 1.0 / ((C50k / x[pos]) ** gam[pos] + 1.0)
        lam_obs = LB[:, k] * s_x
        counts = _poisson_inv(U[:, k], lam_obs * dt)
        burden_counts += counts
        if detect:
            c_obs = counts.copy()
            if sens < 1.0:
                has = counts > 0
                if np.any(has):
                    c_obs[has] = st.binom.ppf(U_thin[has, k], counts[has].astype(int), sens)
            if fp > 0.0:
                c_obs = c_obs + _poisson_inv(U_fp[:, k], np.full(n, fp * dt))
        else:
            c_obs = counts
        if cnt_hist is not None:
            cnt_hist[:, k] = c_obs

        if use_ekf:
            ekf_x = phi * ekf_x
            ekf_P = phi ** 2 * ekf_P + q
            lam_pred = mu_prior[k] * np.exp(ekf_x)
            if drug_aware:   # measurement model includes the drug: counts ~ Poisson(mu e^x s(x) dt)
                lam_pred = lam_pred * hill_s(x, h_C50 if hm is not None else drug["C50"],
                                             h_gam if hm is not None else drug["gamma_hill"])
            yhat = lam_pred * dt
            H = yhat
            R = np.maximum(yhat, 1e-3)
            K = (ekf_P * H) / (H * ekf_P * H + R)
            ekf_x = ekf_x + K * (c_obs - yhat)
            ekf_P = (1 - K * H) * ekf_P * (1 - K * H) + K * R * K
            lam_hat = mu_prior[k] * np.exp(ekf_x)
        else:
            lam_hat = lam_obs

        if dyn is not None:
            d_lam += lam_hat; d_x += x; d_cnt += 1
            cur_day = int(np.floor(T[k]))
            if cur_day > last_day and d_cnt > 0:
                ratio = (dr_c * (d_x / d_cnt) / 5000) / np.maximum(sz_c * (d_lam / d_cnt) / 35, 1e-6)
                omega = np.where(ratio > r_hi, np.minimum(10, omega * f_up),
                                 np.where(ratio < r_lo, np.maximum(0.5, omega * f_dn), omega))
                last_day = cur_day
                d_lam[:] = 0; d_x[:] = 0; d_cnt = 0

        dose = np.zeros(n)
        if is_dose[k]:
            if strat == "fixed":
                dose[:] = dose_std if (load is None or first_dose_done) else load
                first_dose_done = True
            elif strat == "prn":
                if prn_legacy:
                    s0 = max(0, k - legacy_steps)
                    rate = cnt_hist[:, s0:k + 1].sum(axis=1) / max((k - s0 + 1) * dt, dt)
                    dose[rate > prn_thr] = dose_std
                else:
                    s0 = max(0, k - prn_win + 1)
                    dose[cnt_hist[:, s0:k + 1].sum(axis=1) >= prn_min] = dose_std
            elif strat == "prn_ekf":
                dose[lam_hat > prn_thr] = dose_std
            elif strat == "adaptive":
                if rate_src == "ma":             # same tiers driven by a trailing count average (R1 M3 / R2 9 check)
                    s0 = max(0, k - ma_steps + 1)
                    lam_used = cnt_hist[:, s0:k + 1].sum(axis=1) / ((k - s0 + 1) * dt)
                else:
                    lam_used = lam_hat
                # run_strategy compares lh < b*omega; reproduce that exactly
                tier = np.zeros(n, dtype=int)
                for b in bounds:
                    tier += (lam_used >= b * omega)
                dose = doses[tier]
            elif strat == "harm":
                j1 = np.arange(k + 1, k + 1 + h_H)
                if h_oracle is not None:
                    lam_fc = LB[:, np.minimum(j1, Nt - 1)]
                else:
                    # forecast of the drug-free rate: the OU log-deviation decays by phi per step
                    steps = np.arange(1, h_H + 1)
                    ph = phi ** steps
                    Pj = (ph ** 2)[None, :] * ekf_P[:, None] + q * (1 - ph ** 2)[None, :] / (1 - phi ** 2)
                    logdev = ekf_x[:, None] * ph[None, :] + (0.5 * Pj if h_post == "mean" else 0.0)
                    lam_fc = mu_prior[np.minimum(j1, Nt - 1)][None, :] * np.exp(logdev)
                cap = h_dmax if first_dose_done else h_dmax_load
                dose = harm_dose_choice(x, lam_fc, h_a, h_b, h_C50, h_gam, k_e, h_m, cap)
                first_dose_done = True
            elif strat == "none":
                pass
            else:
                raise ValueError(f"engine: unsupported strategy {strat!r}")
        total_dose += dose
        n_doses += dose > 0

        Lc = Lc + lam_obs          # running sums; times dt at day ends (= cumsum(.)*dt)
        Xc = Xc + x
        while d_ptr < DAY_ENDS and day_idx[d_ptr] == k:
            Lday[:, d_ptr] = Lc * dt
            Xday[:, d_ptr] = Xc * dt
            d_ptr += 1
        for key in want:
            if key == "lambda_hat":
                keep[key][:, k] = lam_hat
            elif key == "ekf_P":
                keep[key][:, k] = ekf_P
            elif key == "counts":
                keep[key][:, k] = counts
            elif key == "counts_obs":
                keep[key][:, k] = c_obs
            elif key == "doses":
                keep[key][:, k] = dose
            elif key == "lambda_obs":
                keep[key][:, k] = lam_obs
            elif key == "x":
                keep[key][:, k] = x
        if k < Nt - 1:
            if pk2 is None:
                x = x * k_e + dose
            else:
                x_new = x * k_e - f12 * x + f21 * y_per + dose
                y_per = y_per + f12 * x - f21 * y_per
                x = x_new
    while d_ptr < DAY_ENDS:          # day ends beyond the grid repeat the last value
        Lday[:, d_ptr] = Lc * dt
        Xday[:, d_ptr] = Xc * dt
        d_ptr += 1

    out = dict(Lday=Lday, Xday=Xday, total_dose=total_dose, n_doses=n_doses,
               burden_counts=burden_counts, expected_burden=Lc * dt, drug_exposure=Xc * dt)
    out.update(keep)
    return out
