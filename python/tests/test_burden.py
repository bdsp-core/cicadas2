"""Tests for ``cicadas2.burden`` — total seizure count."""

from __future__ import annotations

import numpy as np
import pytest

from cicadas2 import burden


def test_compute_burden_zero():
    assert burden.compute_burden(np.zeros(10)) == 0.0


def test_compute_burden_sum():
    assert burden.compute_burden(np.array([1, 2, 3, 4])) == 10.0


@pytest.mark.needs_matlab_fixture
def test_compute_burden_matches_matlab(load_fixture):
    f = load_fixture("burden.mat")
    py = burden.compute_burden(np.atleast_1d(f["seizure_counts"]))
    assert py == float(f["burden"])
