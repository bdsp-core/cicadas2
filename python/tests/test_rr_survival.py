"""R&R: exact survival equals the expectation of mortality.run's 0/1 draws (task 5, G10)."""
import numpy as np

from cicadas2 import mortality
from cicadas2.params import default_params
from cicadas2.rr import cohort, engine, survival

TIME, DRUG, CTRL, _ = default_params()
T, dt, Nt = TIME["T"], TIME["dt"], TIME["Nt"]


def test_exact_matches_monte_carlo_legacy_8_days():
    c = cohort.generate(5, seed=3)
    U = cohort.uniform_tape(5, Nt, seed=3, stream=1)
    th = np.array([-5, 0.5, 3.0, 1.0, 1.25, 0.3])
    o = engine.simulate({"strategy": "adaptive"}, c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt,
                        store=("lambda_obs", "x"))
    S8 = survival.exact(o["Lday"], o["Xday"], c["age"], c["sofa"], th, n_days=8)
    rng = np.random.default_rng(0)
    M = 4000
    for i in range(5):
        alive = sum(mortality.run(o["lambda_obs"][i], o["x"][i], dt, th, c["age"][i], c["sofa"][i], None,
                                  u_draws=rng.uniform(size=8))[0] for _ in range(M))
        se = np.sqrt(S8[i] * (1 - S8[i]) / M) + 1e-9
        assert abs(alive / M - S8[i]) < 4.5 * se


def test_seven_days_is_default_and_differs_from_legacy():
    L = np.full((3, 8), 30.0); X = np.full((3, 8), 4000.0)
    a = survival.exact(L, X, np.full(3, 60.0), np.full(3, 3.0), np.array([-5, 0.5, 1.5, 0.5, 0.5, 0.2]))
    b = survival.exact(L, X, np.full(3, 60.0), np.full(3, 3.0), np.array([-5, 0.5, 1.5, 0.5, 0.5, 0.2]), n_days=8)
    assert np.all(a > b)


def test_cohort_is_prefix_stable():
    a = cohort.generate(10, seed=5)
    b = cohort.generate(4, seed=5)
    np.testing.assert_array_equal(a["lambda_base"][:4], b["lambda_base"])
