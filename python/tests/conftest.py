"""Pytest fixtures shared across the cicadas2 test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from cicadas2._matlab_compat import mat_load


REFERENCE_DIR = Path(__file__).parent / "reference_data"


def _fixture_path(name: str) -> Path:
    """Resolve a fixture file path within ``tests/reference_data/``."""
    return REFERENCE_DIR / name


@pytest.fixture
def load_fixture():
    """Return a callable that loads a reference .mat fixture by name.

    Tests that depend on a fixture should be marked with
    ``@pytest.mark.needs_matlab_fixture`` so they get skipped (rather
    than fail) when the fixture is missing — which is the expected
    state until ``tools/matlab_harness.m`` has been run once on a
    machine with MATLAB.
    """
    def _load(name: str) -> dict:
        path = _fixture_path(name)
        if not path.exists():
            pytest.skip(
                f"missing fixture {name}; run tools/matlab_harness.m to "
                f"generate it"
            )
        return mat_load(path)
    return _load
