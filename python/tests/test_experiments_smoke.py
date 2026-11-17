"""Smoke tests for every Phase 6 experiment script.

Each test invokes the experiment's ``run(...)`` with very small N and
asserts that the returned dict has the expected keys with the right
shapes. These are *not* equivalence tests — they only verify that
the orchestration runs to completion end-to-end without exceptions,
that the .mat schema is preserved, and that the output dimensions
scale correctly with the input sizes.

Total runtime: ~75 s on a modern laptop (most of it is s16, which
runs three nested g-formula experiments even at N=5).
"""

from __future__ import annotations

import numpy as np
import pytest

from cicadas2.experiments import (
    s0_validate_distributions,
    s3_population_sim,
    s4_threshold_sweep,
    s5_quadrant_analysis,
    s6_parameter_estimation,
    s7_estimation_robustness,
    s8_closed_loop_validation,
    s9_gformula_causal,
    s10_smart_adaptive,
    s11_ke_sensitivity,
    s12_personalized_validation,
    s13_closed_loop_personalized,
    s14_c50_sensitivity,
    s15_ekf_sensitivity,
    s16_misspecification,
)


# ---------------------------------------------------------------------------
# Small-N smokes
# ---------------------------------------------------------------------------
def test_s0_validate_distributions():
    out = s0_validate_distributions.run(N=50)
    assert {"ages", "sofas", "age_mean", "sofa_mean"} <= out.keys()
    assert out["ages"].size == 50


def test_s3_population_sim():
    out = s3_population_sim.run(Npat=5, Nruns=3)
    res = out["results"]
    assert {"surv_none", "surv_fixed", "surv_prn", "surv_adaptive"} <= res.keys()
    assert res["surv_adaptive"].size == 5
    assert int(out["Npat"]) == 5


def test_s4_threshold_sweep():
    out = s4_threshold_sweep.run(Npat=5, Nruns=3, prn_thresholds=[3, 5],
                                 omega_values=[2, 5])
    assert out["results_prn"]["surv_rate"].shape == (2,)
    assert out["results_omega"]["surv_rate"].shape == (2,)


def test_s5_quadrant_analysis():
    out = s5_quadrant_analysis.run(Npat=5, Nruns=3)
    assert out["surv_rates"].shape == (4, 4)
    assert out["burden_means"].shape == (4, 4)


def test_s6_parameter_estimation():
    out = s6_parameter_estimation.run(Npat=5)
    assert out["est_Ap"].shape == (5,)
    assert out["est_kappa"].shape == (5,)


def test_s7_estimation_robustness():
    out = s7_estimation_robustness.run(Npat=5, treatment_fracs=[0.0, 0.5, 1.0])
    assert out["results"]["corr_Lcum"].shape == (3,)


def test_s8_closed_loop_validation():
    out = s8_closed_loop_validation.run(Npat=5, Nruns=3)
    assert out["surv_true"].shape == (4, 4)
    assert out["surv_est"].shape == (4, 4)
    assert "corr_Lcum" in out


def test_s9_gformula_causal():
    out = s9_gformula_causal.run(Npat=10, Nruns_mort=3)
    for key in ("rct_ate", "naive_ate", "gform_ate"):
        assert key in out
        assert np.isfinite(out[key])


def test_s10_smart_adaptive():
    out = s10_smart_adaptive.run(Npat=5, Nruns=3)
    n_omega = out["omega_vals"].size
    assert out["surv_grid"].shape == (5, n_omega, 4)
    assert out["omega_star"].shape == (5, 4)


def test_s11_ke_sensitivity():
    out = s11_ke_sensitivity.run(Npat=5, Nruns=3, hl_multipliers=[0.8, 1.0, 1.2])
    assert out["corr_Lcum"].shape == (3,)
    assert out["surv_est_all"].shape == (4, 4, 3)


def test_s12_personalized_validation():
    out = s12_personalized_validation.run(Npat=5, Nruns=3)
    n_omega = out["omega_vals"].size
    assert out["surv_true_grid"].shape == (5, n_omega, 4)


def test_s13_closed_loop_personalized():
    out = s13_closed_loop_personalized.run(Npat=5, Nruns=3)
    assert out["surv_results"].shape == (4, 3)


def test_s14_c50_sensitivity():
    out = s14_c50_sensitivity.run(Npat=5, Nruns=3, c50_multipliers=[0.8, 1.0, 1.2])
    assert out["corr_Lcum"].shape == (3,)


def test_s15_ekf_sensitivity():
    out = s15_ekf_sensitivity.run(Npat=3, Nruns_mort=3)
    assert len(out["results"]) == 5    # 5 EKF parameters swept


def test_s16_misspecification():
    out = s16_misspecification.run(Npat=5, Nruns_mort=3)
    for key in ("ref_gform_bias", "layerA_gform_bias",
                "layerB_gform_bias", "layerC_gform_bias",
                "mort_gform_bias"):
        assert key in out
    assert out["graded_bias"].size == 11
