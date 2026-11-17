"""Tests for ``cicadas2.params.default_params``.

These checks are deterministic and need no MATLAB fixture — the
expected values are copied from the MATLAB source. If the MATLAB
defaults ever change, both this test and ``default_params.m`` must
be updated together.
"""

from __future__ import annotations

import numpy as np
import pytest

from cicadas2.params import default_params


@pytest.fixture(scope="module")
def params():
    return default_params()


def test_time_grid(params):
    time, _, _, _ = params
    assert time["Tmax"] == 7
    assert time["dt"] == pytest.approx(5 / 60 / 24)
    assert time["Nt"] == 2017                    # 7 days * 288 steps/day + endpoint
    assert time["T"][0] == 0
    assert time["T"][-1] == pytest.approx(7.0)


def test_drug_params(params):
    _, drug, _, _ = params
    assert drug["half_life"] == 8
    # Per-step elimination factor: k_e = exp(-log(2)*dt_h/half_life)
    dt_h = 5 / 60
    expected_ke = float(np.exp(-np.log(2) * dt_h / 8))
    assert drug["k_e"] == pytest.approx(expected_ke)
    assert drug["pd_mode"] == "hill"
    assert drug["C50"] == 500
    assert drug["gamma_hill"] == 2
    assert drug["dosing_interval"] == 8
    assert drug["dose_standard"] == 500


def test_ctrl_params(params):
    _, _, ctrl, _ = params
    assert ctrl["omega"] == 2
    # Five-tier dose menu, matching MATLAB
    assert (ctrl["dose_none"], ctrl["dose_low"], ctrl["dose_intermediate"],
            ctrl["dose_medium"], ctrl["dose_high"]) == (0, 250, 375, 500, 750)


def test_mort_params(params):
    _, _, _, mort = params
    np.testing.assert_array_equal(
        mort["theta"], np.array([-4, 0.5, 1.5, 1.0, 0.15, 0.07])
    )
    assert mort["L_norm"] == 35
    assert mort["X_norm"] == 5000
    assert mort["Tmax"] == 7
