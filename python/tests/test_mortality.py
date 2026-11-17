"""Tests for ``cicadas2.mortality.run``.

Two layers of verification:

1. **Property tests** (no MATLAB fixture needed). Sanity checks like
   "zero lambda_obs and zero drug should give baseline hazard only",
   "death_time is in [1, Tmax]", "high theta0 kills everyone".
2. **Bit-identical fixture tests** (marked ``needs_matlab_fixture``).
   Load inputs + reference output produced by the MATLAB harness;
   assert the Python port reproduces ``(alive, death_time)`` exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from cicadas2 import mortality
from cicadas2.params import default_params


# ---------------------------------------------------------------------------
# fcn_mortality_squared port (same interface as run; squared burden terms)
# ---------------------------------------------------------------------------
@pytest.mark.needs_matlab_fixture
def test_mortality_squared_matches_matlab(load_fixture):
    f = load_fixture("mortality_squared.mat")
    mp = f["mort_params"]
    mort = {
        "Tmax":   float(mp.Tmax),
        "L_norm": float(mp.L_norm),
        "X_norm": float(mp.X_norm),
    }
    alive, death_time = mortality.run_squared(
        lambda_obs=np.atleast_1d(f["lambda_obs"]),
        x=np.atleast_1d(f["x"]),
        dt=float(f["dt"]),
        theta=np.atleast_1d(f["theta"]),
        age=float(f["age"]),
        sofa=float(f["sofa"]),
        mort_params=mort,
        u_draws=np.atleast_1d(f["u_draws"]),
    )
    assert alive == int(f["alive"])
    if np.isnan(f["death_time"]):
        assert np.isnan(death_time)
    else:
        assert death_time == float(f["death_time"])


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trivial_inputs():
    """A 7-day patient with zero seizures and zero drug exposure."""
    time, _, _, mort = default_params()
    return {
        "lambda_obs": np.zeros(time["Nt"]),
        "x":          np.zeros(time["Nt"]),
        "dt":         time["dt"],
        "theta":      mort["theta"],
        "age":        65,
        "sofa":       3,
        "mort_params": mort,
    }


@pytest.fixture(scope="module")
def n_days(trivial_inputs):
    """Number of hazard evaluations the inner loop will perform.

    Mirrors MATLAB: ``n_days = ceil(Nt * dt)``. With Nt=2017 and
    dt=5/60/24 this equals 8 (not 7) — a quirk where the loop runs
    one day past the nominal Tmax horizon. The port preserves this
    1:1 with the MATLAB reference; flagged for separate discussion.
    """
    Nt = trivial_inputs["lambda_obs"].size
    dt = trivial_inputs["dt"]
    return int(np.ceil(Nt * dt))


def test_zero_load_uses_only_baseline_hazard(trivial_inputs, n_days):
    """With no seizures / no drug, hazard reduces to theta0 + theta1*(d/T)^2."""
    # Force survival by passing all-1 uniforms (u >= h for any h < 1).
    alive, death_time = mortality.run(
        **trivial_inputs, u_draws=np.ones(n_days)
    )
    assert alive == 1
    assert np.isnan(death_time)


def test_certain_death_with_zero_uniforms(trivial_inputs, n_days):
    """All-zero uniforms force death on day 1."""
    alive, death_time = mortality.run(
        **trivial_inputs, u_draws=np.zeros(n_days)
    )
    assert alive == 0
    assert death_time == 1.0


def test_death_time_in_valid_range(trivial_inputs, n_days):
    """death_time, if not NaN, must be an integer day in [1, n_days]."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        u = rng.uniform(size=n_days)
        alive, death_time = mortality.run(**trivial_inputs, u_draws=u)
        if alive == 0:
            assert 1 <= death_time <= n_days
            assert death_time == int(death_time)


def test_high_seizure_burden_kills_more_than_low(trivial_inputs, n_days):
    """Strong baseline rate over the whole window should reduce survival."""
    rng = np.random.default_rng(0)
    n_trials = 500
    low_survival = 0
    high_survival = 0
    for _ in range(n_trials):
        u = rng.uniform(size=n_days)
        low_inputs  = dict(trivial_inputs, lambda_obs=np.full_like(trivial_inputs["lambda_obs"], 0.5))
        high_inputs = dict(trivial_inputs, lambda_obs=np.full_like(trivial_inputs["lambda_obs"], 50.0))
        if mortality.run(**low_inputs,  u_draws=u)[0] == 1: low_survival  += 1
        if mortality.run(**high_inputs, u_draws=u)[0] == 1: high_survival += 1
    # Crude but robust: high-burden patients survive substantially less often.
    assert high_survival < low_survival - 50


# ---------------------------------------------------------------------------
# Bit-identical fixture tests (require MATLAB-generated reference)
# ---------------------------------------------------------------------------
@pytest.mark.needs_matlab_fixture
@pytest.mark.parametrize("fixture_name", [
    "mortality_seed0.mat",        # low-burden patient who survives
    "mortality_high_burden.mat",  # high-burden patient who dies mid-window
])
def test_matches_matlab_reference(load_fixture, fixture_name):
    """End-to-end bit-identical equality against MATLAB-generated fixtures.

    Each fixture (produced by ``tools/matlab_harness.m``) packs the
    inputs (lambda_obs, x, dt, theta, age, sofa, mort_params, u_draws)
    alongside the MATLAB outputs (alive, death_time).  We re-run the
    Python port on the same inputs and assert exact match — including
    the day-of-death integer.
    """
    f = load_fixture(fixture_name)

    # mort_params arrived as a scipy mat_struct; pull the fields we need
    mp = f["mort_params"]
    mort = {
        "Tmax":   float(mp.Tmax),
        "L_norm": float(mp.L_norm),
        "X_norm": float(mp.X_norm),
    }

    alive, death_time = mortality.run(
        lambda_obs=np.atleast_1d(f["lambda_obs"]),
        x=np.atleast_1d(f["x"]),
        dt=float(f["dt"]),
        theta=np.atleast_1d(f["theta"]),
        age=float(f["age"]),
        sofa=float(f["sofa"]),
        mort_params=mort,
        u_draws=np.atleast_1d(f["u_draws"]),
    )

    assert alive == int(f["alive"])
    if np.isnan(f["death_time"]):
        assert np.isnan(death_time)
    else:
        assert death_time == float(f["death_time"])
