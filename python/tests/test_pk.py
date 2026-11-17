"""Tests for ``cicadas2.pk`` — one-compartment PK."""

from __future__ import annotations

import numpy as np
import pytest

from cicadas2 import pk


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------
def test_continuous_zero_infusion_stays_zero():
    """Zero infusion → zero concentration throughout."""
    x = pk.run_continuous(A=np.zeros(100), k_e=0.99, Nt=100)
    np.testing.assert_array_equal(x, np.zeros(100))


def test_continuous_step_infusion_accumulates_then_decays():
    """Step input: concentration rises while infusion is on, decays after."""
    Nt = 200
    A = np.zeros(Nt)
    A[:50] = 1.0
    x = pk.run_continuous(A=A, k_e=0.95, Nt=Nt)
    # Rising phase
    assert x[10] > x[5] > x[1] > 0
    # Peak around the end of the infusion
    assert x[50] > x[150]
    # Decaying phase: monotonically decreasing
    assert all(np.diff(x[60:]) < 0)


def test_pill_first_step_equals_dose():
    """First time step should equal exactly the dose."""
    T = np.linspace(0, 1, 289)
    x = pk.run_pill(dose=500, dosing_interval=8, k_e=0.999, T=T, dt=1/288)
    assert x[0] == 500


def test_pill_decays_between_doses():
    """Between doses, concentration decays geometrically."""
    T = np.linspace(0, 0.3, 87)              # ~7 hours, less than first interval
    dt = T[1] - T[0]
    k_e = 0.99
    x = pk.run_pill(dose=500, dosing_interval=8, k_e=k_e, T=T, dt=dt)
    # x[1] = x[0] * k_e (no additional dose)
    assert x[1] == pytest.approx(x[0] * k_e)
    assert x[2] == pytest.approx(x[1] * k_e)


# ---------------------------------------------------------------------------
# Bit-identical fixture tests
# ---------------------------------------------------------------------------
@pytest.mark.needs_matlab_fixture
def test_continuous_matches_matlab(load_fixture):
    f = load_fixture("pk_continuous.mat")
    py = pk.run_continuous(
        A=np.atleast_1d(f["A"]).astype(float),
        k_e=float(f["k_e"]),
        Nt=int(f["Nt"]),
    )
    np.testing.assert_allclose(py, np.atleast_1d(f["x_out"]), rtol=0, atol=1e-12)


@pytest.mark.needs_matlab_fixture
def test_pill_matches_matlab(load_fixture):
    f = load_fixture("pk_pill.mat")
    py = pk.run_pill(
        dose=float(f["dose"]),
        dosing_interval=float(f["dosing_interval"]),
        k_e=float(f["k_e"]),
        T=np.atleast_1d(f["T"]).astype(float),
        dt=float(f["dt"]),
    )
    np.testing.assert_allclose(py, np.atleast_1d(f["x_out"]), rtol=0, atol=1e-12)
