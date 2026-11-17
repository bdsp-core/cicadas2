"""Port of ``functions/fcn_ekf_lograte_ou.m``.

Extended Kalman filter for a latent log-rate state observed through
Poisson counts. Deterministic given the input count vector and the
population prior — no RNG involved, so the bit-identical fixture
test is straightforward.
"""

from __future__ import annotations

import numpy as np


def run(
    y:     np.ndarray,
    mu:    np.ndarray,
    dt:    float,
    tau_x: float,
    q:     float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EKF on a log-deviation state ``x = log(lambda / mu)``.

    State (Ornstein-Uhlenbeck on x)::

        x_{k+1} = phi * x_k + w_k,   w_k ~ N(0, q)
        phi     = exp(-dt / tau_x)

    Observation (Poisson with state-dependent rate)::

        y_k ~ Poisson( mu_k * exp(x_k) * dt )

    Returns
    -------
    lambda_hat : array
        Posterior estimate of the rate at every step.
    xh : array
        Posterior log-deviation history.
    P_hist : array
        Posterior variance history.
    """
    y  = np.asarray(y,  dtype=float)
    mu = np.asarray(mu, dtype=float)
    Nt = mu.size

    phi = float(np.exp(-dt / tau_x))     # AR(1) coefficient

    x = 0.0
    P = 1.0

    lambda_hat = np.zeros(Nt)
    xh         = np.zeros(Nt)
    P_hist     = np.zeros(Nt)

    for k in range(Nt):                  # MATLAB: for k = 1:Nt
        # Predict
        x = phi * x
        P = phi**2 * P + q

        # Update
        lam  = mu[k] * np.exp(x)
        yhat = lam * dt

        H = yhat                          # Jacobian d(yhat)/dx
        R = max(yhat, 1e-3)               # Poisson variance with floor
        K = (P * H) / (H * P * H + R)
        x = x + K * (y[k] - yhat)

        # Joseph form (numerically robust when R is state-dependent)
        P = (1 - K * H) * P * (1 - K * H) + K * R * K

        xh[k]         = x
        P_hist[k]     = P
        lambda_hat[k] = mu[k] * np.exp(x)

    return lambda_hat, xh, P_hist
