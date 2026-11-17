"""End-to-end equivalence test for ``s4_threshold_sweep``.

Distributional verification (per PYTHON_PORT_PLAN.md §4.4):
the Python port and MATLAB reference each generate N=200 patients
with N=50 mortality MC runs per data point. Survival statistics
have MC standard error roughly sqrt(0.5*0.5/50)/sqrt(200) ≈ 0.5%,
so a 2-percentage-point tolerance is loose enough to absorb
RNG-stream differences while still catching real bugs.

This test is marked ``slow`` and takes ~30 s to run; it is excluded
from the default pytest invocation (see pyproject.toml addopts).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cicadas2._matlab_compat import mat_load
from cicadas2.experiments import s4_threshold_sweep


REPO_ROOT = Path(__file__).resolve().parents[2]
MATLAB_REF = REPO_ROOT / "results" / "threshold_sweep_results.mat"


@pytest.mark.slow
@pytest.mark.skipif(not MATLAB_REF.exists(),
                    reason="MATLAB reference results/threshold_sweep_results.mat missing")
def test_threshold_sweep_within_mc_tolerance(tmp_path):
    """Python and MATLAB outputs should agree within MC noise."""
    out = s4_threshold_sweep.run(out_path=tmp_path / "py.mat")

    ref = mat_load(MATLAB_REF)

    # --- Baseline -------------------------------------------------------
    py_base  = out["results_baseline"]["surv_rate"]
    ref_base = float(ref["results_baseline"].surv_rate)
    assert py_base == pytest.approx(ref_base, abs=2.0), (
        f"baseline survival: py={py_base:.1f} ref={ref_base:.1f}"
    )

    # --- PRN sweep ------------------------------------------------------
    py_prn  = out["results_prn"]["surv_rate"]
    ref_prn = np.atleast_1d(ref["results_prn"].surv_rate).astype(float)
    np.testing.assert_allclose(py_prn, ref_prn, atol=2.0,
        err_msg="PRN survival sweep differs > 2pp from MATLAB")

    # --- Omega sweep ----------------------------------------------------
    py_om  = out["results_omega"]["surv_rate"]
    ref_om = np.atleast_1d(ref["results_omega"].surv_rate).astype(float)
    np.testing.assert_allclose(py_om, ref_om, atol=2.0,
        err_msg="omega survival sweep differs > 2pp from MATLAB")

    # --- Argmaxes (best thresholds) should match exactly ---------------
    assert int(np.argmax(py_prn)) == int(np.argmax(ref_prn))
    assert int(np.argmax(py_om))  == int(np.argmax(ref_om))
