"""Tests for ``cicadas2.strategies.run_strategy``.

This is the largest single function in the MATLAB codebase, so the
fixture coverage is broad: four strategies (none, fixed, prn,
adaptive), each with a fresh Poisson uniform tape, all asserted
bit-identical to MATLAB.
"""

from __future__ import annotations

import numpy as np
import pytest

from cicadas2 import strategies


def _mat_struct_to_dict(ms):
    if hasattr(ms, "_fieldnames"):
        return {k: getattr(ms, k) for k in ms._fieldnames}
    return dict(ms)


def _scalarize(d):
    """Convert any 0-d arrays in a dict to Python scalars."""
    out = {}
    for k, v in d.items():
        if np.ndim(v) == 0 and not isinstance(v, str):
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------
def test_none_strategy_gives_no_doses():
    """Strategy 'none' should never administer any drug."""
    Nt = 200
    T = np.linspace(0, 7, Nt)
    dt = T[1] - T[0]
    lambda_base = np.ones(Nt)
    drug_p = {
        "k_e": 0.99, "alpha": 0.02, "dosing_interval": 8,
        "pd_mode": "linear", "infusion_rate": 0,
    }
    ctrl_p = {
        "omega": 2, "prn_threshold": 5,
        "dose_standard": 500, "dose_low": 250,
        "dose_intermediate": 375, "dose_medium": 500, "dose_high": 750,
    }
    res = strategies.run_strategy("none", lambda_base, drug_p, ctrl_p,
                                  T, dt, use_ekf=True,
                                  u_draws={"poisson": np.full(Nt, 0.5)})
    assert res["total_dose"] == 0
    assert res["n_doses"] == 0
    np.testing.assert_array_equal(res["x"], np.zeros(Nt))


def test_unknown_strategy_raises():
    Nt = 50
    T = np.linspace(0, 1, Nt)
    lambda_base = np.ones(Nt)
    drug_p = {"k_e": 0.99, "alpha": 0.02, "dosing_interval": 8,
              "pd_mode": "linear", "infusion_rate": 0}
    ctrl_p = {"omega": 2, "prn_threshold": 5, "dose_standard": 500,
              "dose_low": 250, "dose_intermediate": 375,
              "dose_medium": 500, "dose_high": 750}
    with pytest.raises(ValueError, match="Unknown strategy"):
        strategies.run_strategy("bogus", lambda_base, drug_p, ctrl_p,
                                T, T[1] - T[0],
                                u_draws={"poisson": np.full(Nt, 0.5)})


# ---------------------------------------------------------------------------
# Bit-identical fixture tests, one per strategy
# ---------------------------------------------------------------------------
@pytest.mark.needs_matlab_fixture
@pytest.mark.parametrize("strategy", ["none", "fixed", "prn", "adaptive"])
def test_strategy_matches_matlab(load_fixture, strategy):
    f = load_fixture(f"strategy_{strategy}.mat")

    drug_p = _scalarize(_mat_struct_to_dict(f["drug_params"]))
    ctrl_p = _scalarize(_mat_struct_to_dict(f["ctrl_params"]))
    drug_p["pd_mode"] = str(drug_p["pd_mode"])
    u_draws = {"poisson": np.atleast_1d(f["u_draws"].poisson).astype(float)}

    py = strategies.run_strategy(
        strategy=str(f["strategy"]),
        lambda_base=np.atleast_1d(f["lambda_base"]).astype(float),
        drug_params=drug_p,
        ctrl_params=ctrl_p,
        T=np.atleast_1d(f["T"]).astype(float),
        dt=float(f["dt"]),
        use_ekf=bool(f["use_ekf"]),
        u_draws=u_draws,
    )

    ref = f["results"]

    # Scalar metrics — exact integer / float match
    assert py["burden"] == pytest.approx(float(ref.burden), abs=1e-9)
    assert py["total_dose"] == pytest.approx(float(ref.total_dose), abs=1e-9)
    assert py["n_doses"] == int(ref.n_doses)

    # Trajectory arrays — bit-identical to within float epsilon
    for field in ["lambda_obs", "lambda_hat", "x", "doses_given",
                  "infusion_rate", "seizure_counts"]:
        np.testing.assert_allclose(
            py[field],
            np.atleast_1d(getattr(ref, field)).astype(float),
            rtol=0, atol=1e-10,
            err_msg=f"{strategy}/{field}",
        )


# ---------------------------------------------------------------------------
# Adaptive variants used by sensitivity/misspec scripts
# ---------------------------------------------------------------------------
@pytest.mark.needs_matlab_fixture
def test_adaptive_with_ekf_matches_matlab(load_fixture):
    f = load_fixture("adaptive_with_ekf.mat")
    patient = _scalarize(_mat_struct_to_dict(f["patient"]))
    patient["lambda_base"] = np.atleast_1d(patient["lambda_base"]).astype(float)

    dp = _scalarize(_mat_struct_to_dict(f["drug_params"]))
    cp = _scalarize(_mat_struct_to_dict(f["ctrl_params"]))
    ekf_ov = _scalarize(_mat_struct_to_dict(f["ekf_ov"]))

    u_draws = {"poisson": np.atleast_1d(f["u_draws"].poisson).astype(float)}

    py = strategies.run_adaptive_with_ekf(
        patient=patient, dp=dp, ctrl_params=cp,
        T=np.atleast_1d(f["T"]).astype(float),
        dt=float(f["dt"]),
        ekf_ov=ekf_ov,
        u_draws=u_draws,
    )

    ref = f["r"]
    for field in ["lambda_obs", "x", "doses_given",
                  "seizure_counts", "lambda_hat"]:
        np.testing.assert_allclose(
            py[field],
            np.atleast_1d(getattr(ref, field)).astype(float),
            rtol=0, atol=1e-10,
            err_msg=field,
        )


@pytest.mark.needs_matlab_fixture
def test_adaptive_custom_matches_matlab(load_fixture):
    f = load_fixture("adaptive_custom.mat")
    patient = _scalarize(_mat_struct_to_dict(f["patient"]))
    patient["lambda_base"] = np.atleast_1d(patient["lambda_base"]).astype(float)

    dp = _scalarize(_mat_struct_to_dict(f["drug_params"]))
    cp = _scalarize(_mat_struct_to_dict(f["ctrl_params"]))

    u_draws = {
        "poisson":   np.atleast_1d(f["u_draws"].poisson).astype(float),
        "mortality": np.atleast_2d(f["u_draws"].mortality).astype(float),
    }

    py_surv = strategies.run_adaptive_custom(
        patient=patient, dp=dp, cp=cp,
        T=np.atleast_1d(f["T"]).astype(float),
        dt=float(f["dt"]),
        theta_mort=np.atleast_1d(f["theta_mort"]).astype(float),
        Nruns_mort=int(f["Nruns_mort"]),
        q_val=float(f["q_val"]),
        tau_x_val=float(f["tau_x_val"]),
        P_pop_val=float(f["P_pop_val"]),
        floor_val=float(f["floor_val"]),
        omega_val=float(f["omega_val"]),
        u_draws=u_draws,
    )

    assert py_surv == pytest.approx(float(f["surv"]), abs=1e-12)
