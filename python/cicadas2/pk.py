"""Port of ``functions/fcn_drug_pk.m`` and ``fcn_drug_pk_pill.m``.

One-compartment pharmacokinetics with first-order elimination, in two
flavours:

* ``run_continuous`` — driven by an infusion rate vector ``A``.
* ``run_pill`` — discrete bolus dosing at fixed-hour intervals.

Both are deterministic; bit-identical tests use plain input/output
fixtures (no RNG injection needed).
"""

from __future__ import annotations

import numpy as np


def run_continuous(A: np.ndarray, k_e: float, Nt: int) -> np.ndarray:
    """One-compartment PK driven by an infusion rate vector ``A``.

    Recurrence::

        x[k+1] = k_e * x[k] + A[k]

    Equivalent to MATLAB::

        x = fcn_drug_pk(A, k_e, Nt)
    """
    A = np.asarray(A)
    x = np.zeros(Nt)
    for k in range(Nt - 1):                   # MATLAB: for k = 1:(Nt-1)
        x[k + 1] = x[k] * k_e + A[k]
    return x


def run_pill(dose: float,
             dosing_interval: float,
             k_e: float,
             T: np.ndarray,
             dt: float) -> np.ndarray:
    """One-compartment PK with discrete bolus dosing.

    Doses are administered at ``t = 0, dosing_interval, 2 * dosing_interval, …``,
    with ``dosing_interval`` given in **hours** and ``T`` / ``dt`` in **days**.
    Between doses the concentration decays by ``k_e`` per step.

    Edge cases match MATLAB exactly: the first time step is set to
    ``dose`` (no decay, no double-counting), and subsequent dose times
    add ``dose`` to the post-decay concentration.
    """
    T = np.asarray(T, dtype=float)
    Nt = T.size
    x = np.zeros(Nt)

    dosing_interval_days = dosing_interval / 24

    # MATLAB: dose_times = 0:dosing_interval_days:T(end)
    n_doses = int(np.floor(T[-1] / dosing_interval_days)) + 1
    dose_times = np.arange(n_doses) * dosing_interval_days

    for k in range(Nt):
        current_time = T[k]
        is_dose_time = bool(np.any(np.abs(current_time - dose_times) < dt / 2))

        if k == 0:
            x[k] = dose                         # first time step: first dose
        else:
            x[k] = x[k - 1] * k_e               # decay from previous step
            if is_dose_time:
                x[k] = x[k] + dose

    return x
