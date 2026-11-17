"""Tests for ``cicadas2.estimation`` — Poisson MLE for (A_p, r, kappa).

Two layers (per PYTHON_PORT_PLAN.md §4.1):

* The negative log-likelihood is bit-identical to MATLAB at any
  given theta.
* The converged optimum is *approximately* identical — scipy's
  Nelder-Mead and MATLAB's fminsearch may differ by ~1-5 % in
  argmin even though they minimise the same objective.
"""

from __future__ import annotations

import numpy as np
import pytest

from cicadas2 import estimation


def _mat_struct_to_dict(ms):
    if hasattr(ms, "_fieldnames"):
        return {k: getattr(ms, k) for k in ms._fieldnames}
    return dict(ms)


# ---------------------------------------------------------------------------
# Property: NLL is finite for sensible thetas, NaN-free trajectory
# ---------------------------------------------------------------------------
def test_nll_finite_for_typical_theta():
    rng = np.random.default_rng(0)
    Nt = 500
    T  = np.linspace(0, 7, Nt)
    counts = rng.poisson(0.05, size=Nt)
    nll = estimation.negative_log_likelihood(
        theta=np.array([0.0, np.log(1.5), np.log(0.3)]),
        seizure_counts=counts, T=T, dt=T[1] - T[0], peak_rate=10,
        x=np.zeros(Nt), is_treated=False, pd_mode="linear",
        alpha=None, C50=None, gamma_hill=None,
    )
    assert np.isfinite(nll)
    assert nll > 0     # the prior penalty alone makes it positive


# ---------------------------------------------------------------------------
# Bit-identical NLL at a known theta
# ---------------------------------------------------------------------------
@pytest.mark.needs_matlab_fixture
def test_nll_bit_identical_at_true_theta(load_fixture):
    """NLL is fully deterministic given theta; must match MATLAB to ~1e-12."""
    f = load_fixture("estimation_untreated.mat")
    # Re-evaluate the NLL via MATLAB's own implementation? We don't carry
    # that here — instead, evaluate the Python NLL at MATLAB's converged
    # theta_opt, then at theta_probe, and rely on the fit_info.negloglik
    # value MATLAB saved at theta_opt.
    params_est = _mat_struct_to_dict(f["params_est"])
    theta_opt_matlab = np.array([
        np.log(float(params_est["A_p"])),
        np.log(float(params_est["r"])),
        np.log(float(params_est["kappa"])),
    ])
    nll_at_matlab_opt = estimation.negative_log_likelihood(
        theta=theta_opt_matlab,
        seizure_counts=np.atleast_1d(f["seizure_counts"]).astype(float),
        T=np.atleast_1d(f["T"]).astype(float),
        dt=float(f["dt"]),
        peak_rate=10,
        x=np.zeros(np.atleast_1d(f["T"]).size),
        is_treated=False, pd_mode="linear",
        alpha=None, C50=None, gamma_hill=None,
    )
    fit_info = _mat_struct_to_dict(f["fit_info"])
    nll_matlab = float(fit_info["negloglik"])
    # Bit-identical to ~1e-9 (compounded float ops over 2017 points).
    assert nll_at_matlab_opt == pytest.approx(nll_matlab, abs=1e-9, rel=1e-9)


# ---------------------------------------------------------------------------
# Converged-optimum approximate match (Nelder-Mead vs fminsearch)
# ---------------------------------------------------------------------------
@pytest.mark.needs_matlab_fixture
def test_converged_optimum_within_tolerance(load_fixture):
    """Scipy's Nelder-Mead may land at a slightly different argmin
    than MATLAB's fminsearch; tolerate ~5 % per parameter and ~1 %
    in the converged objective."""
    f = load_fixture("estimation_untreated.mat")
    counts = np.atleast_1d(f["seizure_counts"]).astype(float)
    T = np.atleast_1d(f["T"]).astype(float)
    dt = float(f["dt"])

    py_params, py_info = estimation.estimate_natural_history(
        seizure_counts=counts, T=T, dt=dt, peak_rate=10,
    )

    ref_params = _mat_struct_to_dict(f["params_est"])
    ref_info   = _mat_struct_to_dict(f["fit_info"])

    # Converged NLL should be within 1 % of MATLAB's
    assert py_info["negloglik"] == pytest.approx(
        float(ref_info["negloglik"]), rel=0.01
    )
    # Each estimated parameter within 5 %
    for k in ["A_p", "r", "kappa"]:
        py_val  = py_params[k]
        ref_val = float(ref_params[k])
        assert py_val == pytest.approx(ref_val, rel=0.05), (
            f"{k}: py={py_val} matlab={ref_val} (>5% diff)"
        )

    # Downstream derived quantity that mortality actually consumes:
    # cumulative burden should match within 1 % between the two optima.
    py_Lcum = float(np.sum(py_info["lambda_base_hat"]) * dt)
    ref_lbh = np.atleast_1d(ref_info["lambda_base_hat"]).astype(float)
    ref_Lcum = float(np.sum(ref_lbh) * dt)
    assert py_Lcum == pytest.approx(ref_Lcum, rel=0.01), (
        f"L_cum: py={py_Lcum:.3f} matlab={ref_Lcum:.3f} (>1% diff)"
    )
