"""Tests for ``cicadas2.ekf`` — EKF on log-rate / Poisson observations.

EKF is fully deterministic given (y, mu, dt, tau_x, q), so the
bit-identical fixture is straightforward (no RNG injection needed
on the EKF itself — only on the upstream generators that produce
y for the harness).
"""

from __future__ import annotations

import numpy as np
import pytest

from cicadas2 import ekf


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------
def test_zero_observations_track_prior_decay():
    """With zero counts the posterior log-deviation drifts toward zero."""
    Nt = 100
    dt = 5 / 60 / 24
    mu = np.full(Nt, 2.0)
    lam, xh, Ph = ekf.run(y=np.zeros(Nt), mu=mu, dt=dt, tau_x=0.5, q=1e-4)
    # With zero counts and a state-dependent observation, the filter
    # should drive lambda_hat below the prior mu.
    assert lam[-1] < mu[-1]
    # Variance stays positive and bounded above the process noise floor.
    assert Ph[-1] > 0


def test_high_observations_drive_estimate_up():
    """Many counts in a row should push lambda_hat above the prior."""
    Nt = 500
    dt = 5 / 60 / 24
    mu = np.full(Nt, 1.0)
    # 20 counts per step is way above mu*dt ≈ 0.0035
    y = np.full(Nt, 20, dtype=float)
    lam, _, _ = ekf.run(y=y, mu=mu, dt=dt, tau_x=0.5, q=1e-4)
    assert lam[-1] > 10 * mu[-1]


def test_shapes_match_input_length():
    Nt = 87
    mu = np.linspace(1, 3, Nt)
    lam, xh, Ph = ekf.run(y=np.ones(Nt), mu=mu, dt=0.01, tau_x=0.5, q=1e-4)
    assert lam.shape == (Nt,)
    assert xh.shape  == (Nt,)
    assert Ph.shape  == (Nt,)


# ---------------------------------------------------------------------------
# Bit-identical fixture test
# ---------------------------------------------------------------------------
@pytest.mark.needs_matlab_fixture
def test_matches_matlab_reference(load_fixture):
    f = load_fixture("ekf_lograte_ou.mat")
    lam, xh, Ph = ekf.run(
        y=np.atleast_1d(f["y"]).astype(float),
        mu=np.atleast_1d(f["mu"]).astype(float),
        dt=float(f["dt"]),
        tau_x=float(f["tau_x"]),
        q=float(f["q"]),
    )
    np.testing.assert_allclose(lam, np.atleast_1d(f["lambda_hat"]),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(xh,  np.atleast_1d(f["xh"]),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(Ph,  np.atleast_1d(f["P_hist"]),
                               rtol=0, atol=1e-12)
