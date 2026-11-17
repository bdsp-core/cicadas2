"""Port of ``scripts/s0_validate_distributions.m`` (diagnostic).

Just generates N patients and reports the empirical demographic
distributions. No .mat output in the MATLAB version; this Python
port returns the stats dict so the user can plot or assert on it.
"""

from __future__ import annotations

import numpy as np

from .. import patient as patient_mod
from ..params import default_params


def run(N: int = 10000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    time, _, _, _ = default_params()
    T = np.arange(0, 4 + time["dt"]/2, time["dt"])     # 4-day window

    ages  = np.zeros(N)
    sofas = np.zeros(N)
    for i in range(N):
        u = {k: rng.uniform() for k in
             ("age", "sofa", "A_base", "r", "kappa")}
        pat = patient_mod.generate(T, {"peak_rate": 10}, u_draws=u)
        ages[i]  = pat["age"]
        sofas[i] = pat["sofa"]

    return {
        "ages":      ages,
        "sofas":     sofas,
        "age_mean":  float(np.mean(ages)),
        "age_median": float(np.median(ages)),
        "age_std":   float(np.std(ages)),
        "sofa_mean": float(np.mean(sofas)),
        "sofa_median": float(np.median(sofas)),
        "sofa_std":  float(np.std(sofas)),
        "frac_sofa_le_6":   float(np.mean(sofas <= 6)),
        "frac_sofa_7_to_9": float(np.mean((sofas >= 7) & (sofas <= 9))),
        "frac_sofa_10_12":  float(np.mean((sofas >= 10) & (sofas <= 12))),
        "frac_sofa_ge_13":  float(np.mean(sofas >= 13)),
    }
