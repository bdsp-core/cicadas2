"""G-formula pipeline for the R&R (G5, G11, R1 M2b, R1 M6b, R1 M6c, R2 11).

Differences from the submitted s9 (each is logged in RR_CHANGELOG.md):
  * Ground truth is the ORACLE ATE: every patient simulated under both arms
    with the true trajectory and the true mortality model. The submitted
    "RCT truth" was one coin-flip trial whose own noise is about 2 pp (G5).
  * The mortality model is FITTED (pooled logistic regression on observed
    patient-days) instead of assumed known (G11). The submitted behaviour
    ("theta known") is still computed alongside for comparison.
  * Survival is exact (no mortality draws) and uses 7 daily hazards (G10).
  * Everything per patient is stored, so bootstrap CIs refit theta on each
    resample and cover estimation uncertainty in the outcome model.
  * Optional hidden confounder U (R1 M6b): enters treatment assignment with
    weight gamma_u and the true daily log-odds with weight delta_u, and is
    never seen by any estimation step.
"""
from __future__ import annotations

import numpy as np

from .. import estimation
from ..experiments.s9_gformula_causal import _early_features
from . import engine, survival
from .cohort import uniform_tape

FEATURE_NAMES = ["Mean day-1 rate", "Peak day-1 rate", "Growth", "Slope", "Acceleration", "SOFA", "Age"]


def features(LB, sofa, age, dt, peak_rate=10.0):
    Nt = LB.shape[1]
    early = min(int(round(1.0 / dt)), Nt)
    F = np.zeros((LB.shape[0], 7))
    for i in range(LB.shape[0]):
        F[i, :5] = _early_features(LB[i, :early], peak_rate, early)
    F[:, 5] = sofa
    F[:, 6] = age
    return F


def assignment_logit(F, noise, gamma_u=0.0, U_conf=None):
    """Submitted s9 assignment model (rise-decay coefficients), plus optional hidden confounder."""
    z = (-0.5 + 0.8 * F[:, 0] + 0.6 * F[:, 1] + 0.4 * F[:, 2] + 0.3 * F[:, 3] + 0.3 * F[:, 4]
         + 0.5 * (F[:, 5] / 10) - 0.3 * (F[:, 6] / 65) + 0.3 * noise)
    if U_conf is not None and gamma_u != 0.0:
        z = z + gamma_u * U_conf
    return z


def _design(Lday, Xday, age, sofa, n_days=7):
    n = Lday.shape[0]
    d = np.arange(1, n_days + 1)
    u = Lday[:, :n_days] / survival.L_NORM
    v = Xday[:, :n_days] / survival.X_NORM
    cols = [np.ones((n, n_days)), np.broadcast_to((d / survival.TMAX) ** 2, (n, n_days)),
            u, (sofa[:, None] / 10) * u, v, (age[:, None] / 65) * v]
    return np.stack(cols, axis=-1)  # (n, n_days, 6)


def fit_theta(Lday, Xday, age, sofa, death_day, n_days=7, w=None, max_iter=100, ridge=1e-3):
    """Pooled logistic regression (Newton-Raphson) of daily death on the paper's linear hazard terms.

    Person-days at risk: day 1..death_day for those who died (event on the last),
    day 1..n_days for survivors. Returns (theta_hat, covariance).
    """
    Xd = _design(Lday, Xday, age, sofa, n_days)
    n = Xd.shape[0]
    d = np.arange(1, n_days + 1)[None, :]
    dd = np.where(death_day == 0, n_days + 1, death_day)[:, None]
    at_risk = d <= np.minimum(dd, n_days)
    y = (d == dd).astype(float)
    if w is None:
        w = np.ones(n)
    W = at_risk * w[:, None]
    A = Xd[at_risk]
    yy = y[at_risk]
    ww = W[at_risk]
    beta = np.zeros(6)
    beta[0] = np.log(max(yy.mean(), 1e-6) / max(1 - yy.mean(), 1e-6))
    R = ridge * np.diag([0, 1, 1, 1, 1, 1.0])     # tiny ridge on slopes: guards against separation

    def pll(b):
        eta = A @ b
        return np.sum(ww * (yy * eta - np.logaddexp(0, eta))) - 0.5 * b @ R @ b

    cur = pll(beta)
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-np.clip(A @ beta, -700, 700)))
        g = A.T @ (ww * (yy - p)) - R @ beta
        Hm = (A * (ww * p * (1 - p))[:, None]).T @ A + R
        step = np.linalg.solve(Hm + 1e-10 * np.eye(6), g)
        t = 1.0
        while t > 1e-6:                       # step halving keeps the likelihood increasing
            new = pll(beta + t * step)
            if new >= cur - 1e-12:
                break
            t /= 2
        beta = beta + t * step
        done = abs(new - cur) < 1e-10
        cur = new
        if done:
            break
    cov = np.linalg.inv(Hm + 1e-10 * np.eye(6))
    return beta, cov


def estimate_trajectories(counts, doses, C50, gam, drug, T, dt, treated: bool, peak_rate=10.0):
    """Poisson MLE of the natural history for every patient (submitted estimator, unchanged)."""
    n, Nt = counts.shape
    LBh = np.zeros((n, Nt))
    LOh = np.zeros((n, Nt))
    for i in range(n):
        if treated:
            _, fi = estimation.estimate_natural_history(
                counts[i], T, dt, peak_rate=peak_rate, k_e=drug["k_e"], doses_given=doses[i],
                pd_mode="hill", C50=float(C50[i]), gamma_hill=float(gam[i]))
        else:
            _, fi = estimation.estimate_natural_history(counts[i], T, dt, peak_rate=peak_rate)
        LBh[i] = fi["lambda_base_hat"]
        LOh[i] = fi["lambda_obs_hat"]
    return LBh, LOh


def day_cumsum(L, dt, n_days=8):
    Nt = L.shape[1]
    idx = np.minimum(np.round(np.arange(1, n_days + 1) / dt).astype(int), Nt) - 1
    return (np.cumsum(L, axis=1) * dt)[:, idx]


def prepare(c, drug, ctrl, T, dt, seed, pkpd="oracle", treat="adaptive", dgp=None,
            dose_jitter_h=0.0, dose_drop=0.0, treat_spec=None):
    """Everything that does not depend on the outcome model or the assignment rule.

    Simulates both arms on the true trajectories, estimates each patient's
    natural history from BOTH arms' observations, and simulates both
    counterfactual arms on each estimate. Sweeps over confounding, mortality
    shape or assignment then only reuse these arrays.

    dgp: extra engine options applied ONLY to the true (data-generating) runs,
         e.g. {"detect_sens": 0.75}, {"pk2": {...}}, {"c50_drift_per_day": 0.1}.
         The analyst's estimator and counterfactual simulator keep the standard
         model, which is what makes these misspecification tests.
    treat_spec: extra engine options for the treatment strategy itself (for example a
         miscalibrated EKF), applied in BOTH the true and the counterfactual runs, because
         the analyst knows which strategy was deployed.
    dose_jitter_h / dose_drop: errors in the medication record the estimator
         sees (timestamps shifted by N(0, jitter^2) hours, rounded to the 5-min
         grid; a fraction of doses missing). The true dosing is unchanged.
    """
    n, Nt = c["n"], c["lambda_base"].shape[1]
    U = uniform_tape(n, Nt, seed, stream=1)
    U_cf = uniform_tape(n, Nt, seed, stream=2)
    dgp = dict(dgp or {})
    kw = {}
    if dgp.get("detect_sens", 1.0) < 1.0 or dgp.get("detect_fp_per_day", 0.0) > 0.0:
        kw = dict(U_thin=uniform_tape(n, Nt, seed, stream=3), U_fp=uniform_tape(n, Nt, seed, stream=4))
    st = ("counts", "counts_obs", "doses")
    tspec = dict({"strategy": treat}, **(treat_spec or {}))
    tr = engine.simulate(dict(tspec, **dgp), c["lambda_base"], c["C50"], c["gamma_hill"], U,
                         drug, ctrl, T, dt, store=st, **kw)
    un = engine.simulate(dict({"strategy": "none"}, **dgp), c["lambda_base"], c["C50"], c["gamma_hill"], U,
                         drug, ctrl, T, dt, store=st, **kw)
    if pkpd == "oracle":
        C50e, game = c["C50"], c["gamma_hill"]
    else:  # population-mean PKPD for the inversion (as in Sec 5.3.4)
        C50e, game = np.full(n, float(drug["C50"])), np.full(n, float(drug["gamma_hill"]))
    rec = tr["doses"].copy()
    if dose_jitter_h > 0 or dose_drop > 0:
        g = np.random.default_rng((seed, 77))
        rec = np.zeros_like(tr["doses"])
        for i in range(n):
            for k in np.nonzero(tr["doses"][i])[0]:
                if g.uniform() < dose_drop:
                    continue
                kk = int(np.clip(k + np.round(g.normal(0, dose_jitter_h) * 12), 0, Nt - 1))
                rec[i, kk] += tr["doses"][i, k]
    LBh_t, LOh_t = estimate_trajectories(tr["counts_obs"], rec, C50e, game, drug, T, dt, True)
    LBh_n, LOh_n = estimate_trajectories(un["counts_obs"], un["doses"], C50e, game, drug, T, dt, False)
    cf = {}
    for src, LBh in (("t", LBh_t), ("n", LBh_n)):
        for arm in (treat, "none"):
            o = engine.simulate(tspec if arm == treat else {"strategy": arm}, LBh, C50e, game, U_cf, drug, ctrl, T, dt)  # analyst's model
            cf[(src, arm)] = (o["Lday"], o["Xday"])
    F = features(c["lambda_base"], c["sofa"], c["age"], dt)
    g = np.random.default_rng((seed, 99))
    return dict(n=n, seed=seed, pkpd=pkpd, treat=treat, dgp=dgp, dose_jitter_h=dose_jitter_h, dose_drop=dose_drop, true_t=(tr["Lday"], tr["Xday"]), true_n=(un["Lday"], un["Xday"]),
                obs_hat_t=(day_cumsum(LOh_t, dt), tr["Xday"]), obs_hat_n=(day_cumsum(LOh_n, dt), un["Xday"]),
                cf=cf, F=F, age=c["age"], sofa=c["sofa"],
                noise=g.standard_normal(n), u_assign=g.uniform(size=n), u_death=g.uniform(size=(n, 8)),
                u_rct=g.uniform(size=n), U_conf=g.standard_normal(n),
                Lcum_true_t=tr["Lday"][:, 6], Lcum_true_n=un["Lday"][:, 6],
                Lcum_hat_tt=day_cumsum(LOh_t, dt)[:, 6], Lcum_hat_nn=day_cumsum(LOh_n, dt)[:, 6],
                Lbase_true=day_cumsum(c["lambda_base"], dt)[:, 6],
                Lbase_hat_t=day_cumsum(LBh_t, dt)[:, 6], Lbase_hat_n=day_cumsum(LBh_n, dt)[:, 6])


def evaluate(P, theta_true, shape="linear", gamma_u=0.0, delta_u=0.0, n_days=7, theta_known=None, idx=None, time_power=2, drug_shape="linear"):
    """One scenario: truth, observed data, naive, g-formula (fitted theta and known theta)."""
    if idx is None:
        idx = np.arange(P["n"])
    age, sofa = P["age"][idx], P["sofa"][idx]
    zU = delta_u * P["U_conf"][idx] if delta_u else None
    tL, tX = P["true_t"][0][idx], P["true_t"][1][idx]
    nL, nX = P["true_n"][0][idx], P["true_n"][1][idx]
    S_t = survival.exact(tL, tX, age, sofa, theta_true, shape, drug_shape, n_days=n_days, z_extra=zU, time_power=time_power)
    S_n = survival.exact(nL, nX, age, sofa, theta_true, shape, drug_shape, n_days=n_days, z_extra=zU, time_power=time_power)
    z = assignment_logit(P["F"][idx], P["noise"][idx], gamma_u, P["U_conf"][idx])
    ps_true = 1 / (1 + np.exp(-z))
    A = P["u_assign"][idx] < ps_true
    # observed arm
    oL = np.where(A[:, None], tL, nL)
    oX = np.where(A[:, None], tX, nX)
    death = survival.draw_death_day(oL, oX, age, sofa, theta_true, P["u_death"][idx], shape, n_days, zU, time_power, drug_shape)
    hL = np.where(A[:, None], P["obs_hat_t"][0][idx], P["obs_hat_n"][0][idx])
    th_hat, cov = fit_theta(hL, oX, age, sofa, death, n_days)
    # counterfactual cumulants on each patient's estimate from the arm actually observed
    def cf(arm):
        Lt, Xt = P["cf"][("t", arm)]
        Ln, Xn = P["cf"][("n", arm)]
        return np.where(A[:, None], Lt[idx], Ln[idx]), np.where(A[:, None], Xt[idx], Xn[idx])
    cL_t, cX_t = cf(P["treat"])
    cL_n, cX_n = cf("none")
    G_t = survival.exact(cL_t, cX_t, age, sofa, th_hat, "linear", n_days=n_days)
    G_n = survival.exact(cL_n, cX_n, age, sofa, th_hat, "linear", n_days=n_days)
    tk = theta_true if theta_known is None else theta_known
    K_t = survival.exact(cL_t, cX_t, age, sofa, tk, "linear", n_days=n_days)
    K_n = survival.exact(cL_n, cX_n, age, sofa, tk, "linear", n_days=n_days)
    S_obs = np.where(A, S_t, S_n)
    rct = P["u_rct"][idx] > 0.5
    return dict(idx=idx, A=A, ps_true=ps_true, death=death, theta_hat=th_hat, theta_cov=cov,
                S_t=S_t, S_n=S_n, G_t=G_t, G_n=G_n, K_t=K_t, K_n=K_n,
                oracle_ate=100 * np.mean(S_t - S_n),
                gform_ate=100 * np.mean(G_t - G_n),
                gform_known_ate=100 * np.mean(K_t - K_n),
                naive_ate=100 * (S_obs[A].mean() - S_obs[~A].mean()),
                naive_deaths_ate=100 * ((death[A] == 0).mean() - (death[~A] == 0).mean()),
                rct_ate=100 * (S_t[rct].mean() - S_n[~rct].mean()),
                frac_treated=A.mean())


def bootstrap(P, theta_true, B=300, seed=7, **kw):
    """Patient-resampling bootstrap of the whole estimator (theta refitted on every resample)."""
    g = np.random.default_rng(seed)
    n = P["n"]
    out = np.zeros((B, 4))
    for b in range(B):
        idx = g.integers(0, n, size=n)
        r = evaluate(P, theta_true, idx=idx, **kw)
        out[b] = [r["oracle_ate"], r["gform_ate"], r["gform_ate"] - r["oracle_ate"], r["naive_ate"]]
    lo, hi = np.percentile(out, [2.5, 97.5], axis=0)
    return dict(lo=lo, hi=hi, se=out.std(axis=0, ddof=1), cols=["oracle", "gform", "bias", "naive"])


def propensity_fit(F, A, max_iter=50):
    """Logistic regression of treatment on the 7 recorded features (what an analyst would fit)."""
    Xm = np.column_stack([np.ones(F.shape[0]), (F - F.mean(0)) / F.std(0)])
    b = np.zeros(Xm.shape[1])
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-(Xm @ b)))
        g = Xm.T @ (A - p)
        H = (Xm * (p * (1 - p))[:, None]).T @ Xm
        s = np.linalg.solve(H + 1e-8 * np.eye(len(b)), g)
        b += s
        if np.max(np.abs(s)) < 1e-10:
            break
    return 1 / (1 + np.exp(-(Xm @ b)))


def smd(F, A):
    m1, m0 = F[A].mean(0), F[~A].mean(0)
    s = np.sqrt((F[A].var(0, ddof=1) + F[~A].var(0, ddof=1)) / 2)
    return (m1 - m0) / s
