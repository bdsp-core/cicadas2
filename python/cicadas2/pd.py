"""Port of ``functions/fcn_observed_rate.m``.

Pharmacodynamic suppression of the baseline seizure rate, in either
a linear or Hill mode.
"""

from __future__ import annotations

import numpy as np


def observed_rate(lambda_base, x, pd_params) -> np.ndarray:
    """Observed seizure rate after drug suppression.

    Two modes are supported:

    * ``linear``: ``lambda_obs = max(0, lambda_base - alpha * x)``
    * ``hill``:   ``lambda_obs = lambda_base * s_x``,
      ``s_x = 1 - 1 / ((C50/x)^gamma + 1)``, with explicit
      ``s_x = 1`` where ``x == 0`` to avoid the divide-by-zero.

    For backward compatibility with the MATLAB call sites that pass a
    bare scalar ``alpha`` (treated as linear mode), ``pd_params`` may
    be either a dict with ``mode`` / ``alpha`` / ``C50`` / ``gamma``
    fields, or a scalar.
    """
    lambda_base = np.asarray(lambda_base, dtype=float)
    x = np.asarray(x, dtype=float)

    # Scalar input = alpha (legacy linear mode)
    if not isinstance(pd_params, dict):
        alpha = float(pd_params)
        return np.maximum(0.0, lambda_base - alpha * x)

    mode = pd_params.get("mode", "linear").lower()

    if mode == "linear":
        alpha = float(pd_params["alpha"])
        return np.maximum(0.0, lambda_base - alpha * x)

    if mode == "hill":
        C50 = float(pd_params["C50"])
        gamma = float(pd_params["gamma"])

        # s_x = 1 where x == 0, otherwise 1 - 1/((C50/x)^gamma + 1).
        s_x = np.ones_like(x)
        mask = x > 0
        if np.any(mask):
            s_x[mask] = 1.0 - 1.0 / ((C50 / x[mask]) ** gamma + 1.0)
        return lambda_base * s_x

    raise ValueError(f"Unknown PD mode: {mode!r}. Use 'linear' or 'hill'.")
