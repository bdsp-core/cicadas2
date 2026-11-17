"""Port of ``functions/fcn_compute_burden.m``."""

from __future__ import annotations

import numpy as np


def compute_burden(seizure_counts) -> float:
    """Sum of Poisson seizure counts; trivial but kept for parity.

    Not currently called by any MATLAB script (burden is computed
    inline in ``fcn_run_strategy``), but ported anyway for 1:1
    parity with ``functions/``.
    """
    return float(np.sum(seizure_counts))
