"""Tests for ``cicadas2.patient`` — stochastic patient generator.

All bit-identical tests rely on RNG injection: the MATLAB harness
draws a uniform tape, computes the patient via inverse-CDF
sampling, and saves both the tape and the resulting patient struct.
The Python port consumes the same tape via ``scipy.stats.*.ppf``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cicadas2 import patient as patient_mod
from cicadas2.params import default_params


def _mat_struct_to_dict(ms):
    """Flatten a scipy.io mat_struct (or dict) into a plain dict."""
    if hasattr(ms, "_fieldnames"):
        return {k: getattr(ms, k) for k in ms._fieldnames}
    return dict(ms)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------
def test_age_override_skips_age_draw():
    """If params['age'] is set, the patient's age comes straight from it."""
    time, _, _, _ = default_params()
    np.random.seed(0)
    p = patient_mod.generate(time["T"], {"age": 72.5})
    assert p["age"] == 72.5


def test_sofa_override_skips_sofa_draw():
    time, _, _, _ = default_params()
    np.random.seed(0)
    p = patient_mod.generate(time["T"], {"sofa": 4})
    assert p["sofa"] == 4


def test_trajectory_starts_at_zero_and_is_finite():
    time, _, _, _ = default_params()
    np.random.seed(0)
    p = patient_mod.generate(time["T"], {})
    assert p["lambda_base"][0] == 0
    assert np.all(np.isfinite(p["lambda_base"]))
    assert p["lambda_base"].size == time["Nt"]


def test_no_drug_params_defaults_alpha():
    time, _, _, _ = default_params()
    np.random.seed(0)
    p = patient_mod.generate(time["T"], {})
    assert p["alpha"] == 0.02
    assert "C50" not in p   # no Hill params unless requested


# ---------------------------------------------------------------------------
# Bit-identical fixture tests
# ---------------------------------------------------------------------------
@pytest.mark.needs_matlab_fixture
@pytest.mark.parametrize("fixture_name", [
    "patient_basic.mat",
    "patient_with_linear_pd.mat",
    "patient_with_hill_pd.mat",
])
def test_generate_matches_matlab(load_fixture, fixture_name):
    f = load_fixture(fixture_name)

    T = np.atleast_1d(f["T"]).astype(float)
    params = _mat_struct_to_dict(f["params"])
    # Unwrap nested drug_params if present
    if "drug_params" in params:
        dp = _mat_struct_to_dict(params["drug_params"])
        # Normalise scalar fields
        dp = {k: (float(v) if np.ndim(v) == 0 and not isinstance(v, str) else v)
              for k, v in dp.items()}
        if "pd_mode" in dp and not isinstance(dp["pd_mode"], str):
            dp["pd_mode"] = str(dp["pd_mode"])
        params["drug_params"] = dp
    # Normalise scalar params
    params = {k: (float(v) if np.ndim(v) == 0 and not isinstance(v, (str, dict)) else v)
              for k, v in params.items()}

    u_draws = _mat_struct_to_dict(f["u_draws"])
    u_draws = {k: float(v) for k, v in u_draws.items()}

    py = patient_mod.generate(T, params, u_draws=u_draws)
    ref = _mat_struct_to_dict(f["patient"])

    # Scalar fields — exact (modulo CDF numerics ~1e-12)
    for field in ["age", "sofa", "A_patient", "r", "kappa", "peak_rate", "alpha"]:
        assert field in py
        assert py[field] == pytest.approx(float(ref.__dict__[field] if hasattr(ref, "__dict__") else ref[field]), abs=1e-10), \
            f"{field}: py={py[field]} matlab={getattr(ref, field, None) or ref[field]}"

    # Optional Hill fields
    for field in ["C50", "gamma_hill"]:
        if hasattr(ref, field):
            assert py[field] == pytest.approx(float(getattr(ref, field)), abs=1e-10)

    # Trajectory — bit-identical to within float epsilon
    np.testing.assert_allclose(
        py["lambda_base"], np.atleast_1d(ref.lambda_base if hasattr(ref, "lambda_base") else ref["lambda_base"]),
        rtol=0, atol=1e-12,
    )


# ---------------------------------------------------------------------------
# CICADAS SDE generator
# ---------------------------------------------------------------------------
@pytest.mark.needs_matlab_fixture
def test_generate_cicadas_matches_matlab(load_fixture):
    """End-to-end equality for the SDE generator.

    The SDE step accumulates ~2016 normal draws per patient, so any
    off-by-one or PPF discrepancy compounds visibly in the trajectory.
    Bit-identical match is therefore a strong joint test of:
        - inverse-CDF parity (Beta, Gamma, LogN, Normal)
        - 1-based-to-0-based loop indexing
        - boundary handling on the t_switch transition and the L >= 0 floor.
    """
    f = load_fixture("patient_cicadas.mat")
    T = np.atleast_1d(f["T"]).astype(float)
    params = _mat_struct_to_dict(f["params"])
    params = {k: (float(v) if np.ndim(v) == 0 else v) for k, v in params.items()}

    u_draws = _mat_struct_to_dict(f["u_draws"])
    u_draws = {
        k: (float(v) if np.ndim(v) == 0 else np.atleast_1d(v).astype(float))
        for k, v in u_draws.items()
    }

    py = patient_mod.generate_cicadas(T, params, u_draws=u_draws)
    ref = f["patient"]

    for field in ["age", "sofa", "A_patient", "peak_rate", "alpha"]:
        assert py[field] == pytest.approx(float(getattr(ref, field)), abs=1e-10)

    # Trajectory: tight tolerance even with 2016 cumulative SDE steps.
    np.testing.assert_allclose(
        py["lambda_base"],
        np.atleast_1d(ref.lambda_base),
        rtol=1e-10, atol=1e-10,
    )
