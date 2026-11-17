"""R&R: the vectorised engine must reproduce strategies.run_strategy exactly."""
import numpy as np
import pytest

from cicadas2 import strategies
from cicadas2.params import default_params
from cicadas2.rr import cohort, engine

TIME, DRUG, CTRL, _ = default_params()
T, dt, Nt = TIME["T"], TIME["dt"], TIME["Nt"]


@pytest.fixture(scope="module")
def small():
    c = cohort.generate(6, seed=11)
    U = cohort.uniform_tape(6, Nt, seed=11, stream=1)
    return c, U


@pytest.mark.parametrize("strat", ["none", "fixed", "prn", "prn_ekf", "adaptive"])
def test_engine_matches_run_strategy(small, strat):
    c, U = small
    spec = {"strategy": strat, "prn_legacy": True}
    out = engine.simulate(spec, c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt,
                          store=("lambda_hat", "counts", "doses", "lambda_obs", "x"))
    for i in range(c["n"]):
        dp = dict(DRUG, alpha=c["alpha"][i], patient_C50=c["C50"][i], patient_gamma=c["gamma_hill"][i])
        r = strategies.run_strategy(strat, c["lambda_base"][i], dp, CTRL, T, dt, True,
                                    u_draws={"poisson": U[i]})
        np.testing.assert_array_equal(out["counts"][i], r["seizure_counts"])
        np.testing.assert_array_equal(out["doses"][i], r["doses_given"])
        np.testing.assert_allclose(out["lambda_obs"][i], r["lambda_obs"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out["lambda_hat"][i], r["lambda_hat"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out["x"][i], r["x"], rtol=1e-12, atol=1e-12)
        idx = np.minimum(np.round(np.arange(1, 9) / dt).astype(int), Nt) - 1
        np.testing.assert_allclose(out["Lday"][i], (np.cumsum(r["lambda_obs"]) * dt)[idx], rtol=1e-12)
        np.testing.assert_allclose(out["Xday"][i], (np.cumsum(r["x"]) * dt)[idx], rtol=1e-12)
        assert out["total_dose"][i] == r["total_dose"]


@pytest.mark.parametrize("win,k", [(8, 1), (8, 3), (24, 2)])
def test_engine_matches_run_strategy_prn_count_rule(small, win, k):
    c, U = small
    out = engine.simulate({"strategy": "prn", "prn_legacy": False, "prn_window_hours": win, "prn_min_count": k},
                          c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt, store=("doses", "counts"))
    cp = dict(CTRL, prn_legacy=False, prn_window_hours=win, prn_min_count=k)
    for i in range(c["n"]):
        dp = dict(DRUG, alpha=c["alpha"][i], patient_C50=c["C50"][i], patient_gamma=c["gamma_hill"][i])
        r = strategies.run_strategy("prn", c["lambda_base"][i], dp, cp, T, dt, True, u_draws={"poisson": U[i]})
        np.testing.assert_array_equal(out["doses"][i], r["doses_given"])
        np.testing.assert_array_equal(out["counts"][i], r["seizure_counts"])


# ---- harm-balanced controller (2026-10-27) ----
def _harm_cohort(n=40):
    from cicadas2.rr import cohort as cm
    from cicadas2.params import default_params
    TIME, DRUG, CTRL, _ = default_params()
    c = cm.generate(n, 11)
    U = cm.uniform_tape(n, TIME["Nt"], 11, stream=1)
    return c, U, TIME, DRUG, CTRL


def test_harm_nests_fixed_max_and_none():
    """Seizure harm only -> every dose is the cap (= fixed at the cap); drug harm only -> no drug (= none)."""
    import numpy as np
    from cicadas2.rr import engine
    c, U, TIME, DRUG, CTRL = _harm_cohort()
    T, dt = TIME["T"], TIME["dt"]
    sim = lambda sp: engine.simulate(sp, c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt)
    sz_only = {"strategy": "harm", "harm": dict(theta=[-5, 0.5, 3.0, 1.0, 0.0, 0.0], age=c["age"], sofa=c["sofa"], oracle="full")}
    dr_only = {"strategy": "harm", "harm": dict(theta=[-5, 0.5, 0.0, 0.0, 1.25, 0.3], age=c["age"], sofa=c["sofa"], oracle="full")}
    o_sz, o_fx = sim(sz_only), sim({"strategy": "fixed", "dose_standard": 750})
    o_dr, o_no = sim(dr_only), sim({"strategy": "none"})
    # the first dose is at t=0 when the true rate is 0: the controller may wait one interval there
    assert np.all(o_sz["total_dose"] >= o_fx["total_dose"] - 750 - 1e-9)
    assert np.allclose(o_dr["total_dose"], 0.0) and np.allclose(o_dr["Lday"], o_no["Lday"])


def test_harm_dose_choice_matches_analytic_optimum():
    """Constant rate, no carry-over limit: the chosen periodic peak minimises the integral of a*lam*s(x) + b*x."""
    import numpy as np
    from cicadas2.rr import engine
    k_e, m, C50, gam = 0.9928057204912689, 96, 500.0, 2.0
    a, b, lam = 0.1, 1e-4, 20.0
    d = engine.harm_dose_choice(np.array([0.0]), np.full((1, 3 * m), lam), np.array([a]), np.array([b]),
                                np.array([C50]), np.array([gam]), k_e, m, np.array([3000.0]), n_grid=601)
    P = np.linspace(0, 3000, 3001)
    dec = k_e ** (np.arange(3 * m) % m)
    xs = P[:, None] * dec[None, :]
    cost = (a * lam / (1 + (xs / C50) ** gam) + b * xs).sum(1)
    assert abs(d[0] - P[cost.argmin()]) <= 5.0


def test_drug_aware_ekf_tracks_untreated_rate_better():
    """Under heavy fixed dosing the drug-aware filter estimates lambda_base, the plain one the treated rate."""
    import numpy as np
    from cicadas2.rr import engine
    c, U, TIME, DRUG, CTRL = _harm_cohort(60)
    T, dt = TIME["T"], TIME["dt"]
    run = lambda aware: engine.simulate({"strategy": "fixed", "dose_standard": 750, "ekf": dict(P_pop=10, tau_x=5, drug_aware=aware)},
                                        c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt, store=("lambda_hat",))
    e_aware = np.abs(run(True)["lambda_hat"] - c["lambda_base"]).mean()
    e_plain = np.abs(run(False)["lambda_hat"] - c["lambda_base"]).mean()
    assert e_aware < e_plain
