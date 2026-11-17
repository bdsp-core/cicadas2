"""Tests for ``cicadas2.pd`` — observed seizure rate after drug suppression."""

from __future__ import annotations

import numpy as np
import pytest

from cicadas2 import pd


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------
def test_linear_zero_drug_no_suppression():
    lam_base = np.array([0.5, 1.0, 1.5, 2.0])
    out = pd.observed_rate(lam_base, np.zeros_like(lam_base),
                           {"mode": "linear", "alpha": 0.02})
    np.testing.assert_allclose(out, lam_base)


def test_linear_floor_at_zero():
    """Linear mode never lets the observed rate go negative."""
    lam_base = np.array([1.0])
    out = pd.observed_rate(lam_base, np.array([1000.0]),   # huge drug, alpha=0.02
                           {"mode": "linear", "alpha": 0.02})
    assert out[0] == 0.0


def test_scalar_pd_params_is_legacy_linear_alpha():
    """Backward compat: bare numeric pd_params is treated as alpha (linear)."""
    lam_base = np.array([2.0])
    out_legacy = pd.observed_rate(lam_base, np.array([10.0]), 0.05)
    out_dict   = pd.observed_rate(lam_base, np.array([10.0]),
                                  {"mode": "linear", "alpha": 0.05})
    np.testing.assert_allclose(out_legacy, out_dict)


def test_hill_at_C50_is_half_suppression():
    """At x == C50, the Hill suppression factor s_x is 0.5."""
    lam_base = np.array([1.0])
    x = np.array([500.0])
    out = pd.observed_rate(lam_base, x,
                           {"mode": "hill", "C50": 500.0, "gamma": 2.0})
    assert out[0] == pytest.approx(0.5)


def test_hill_at_zero_drug_is_no_suppression():
    """At x == 0, the Hill factor is 1 (no suppression)."""
    lam_base = np.array([1.0, 2.0])
    x = np.array([0.0, 0.0])
    out = pd.observed_rate(lam_base, x,
                           {"mode": "hill", "C50": 500.0, "gamma": 2.0})
    np.testing.assert_allclose(out, lam_base)


def test_hill_high_drug_approaches_full_suppression():
    """x >> C50 should drive s_x toward 0."""
    lam_base = np.array([1.0])
    out = pd.observed_rate(lam_base, np.array([1e8]),
                           {"mode": "hill", "C50": 500.0, "gamma": 2.0})
    assert out[0] < 1e-6


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown PD mode"):
        pd.observed_rate(np.array([1.0]), np.array([1.0]),
                         {"mode": "weibull"})


# ---------------------------------------------------------------------------
# Bit-identical fixture tests
# ---------------------------------------------------------------------------
def _mat_struct_to_dict(ms):
    """Flatten a scipy.io mat_struct (or dict) into a plain dict."""
    if hasattr(ms, "_fieldnames"):
        return {k: getattr(ms, k) for k in ms._fieldnames}
    return dict(ms)


@pytest.mark.needs_matlab_fixture
@pytest.mark.parametrize("fixture_name", ["pd_linear.mat", "pd_hill.mat"])
def test_observed_rate_matches_matlab(load_fixture, fixture_name):
    f = load_fixture(fixture_name)
    pd_params = _mat_struct_to_dict(f["pd_params"])
    # Normalise scalars
    pd_params = {k: (float(v) if np.isscalar(v) or np.ndim(v) == 0 else v)
                 for k, v in pd_params.items() if k != "mode"}
    pd_params["mode"] = str(f["pd_params"].mode)
    py = pd.observed_rate(
        lambda_base=np.atleast_1d(f["lambda_base"]).astype(float),
        x=np.atleast_1d(f["x"]).astype(float),
        pd_params=pd_params,
    )
    np.testing.assert_allclose(py, np.atleast_1d(f["lambda_obs"]),
                               rtol=0, atol=1e-12)
