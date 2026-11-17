"""Port of ``functions/default_params.m``.

Returns four parameter dicts (time, drug, ctrl, mort) whose keys mirror
the field names of the corresponding MATLAB structs so the port reads
1:1 against the MATLAB reference.
"""

from __future__ import annotations

import numpy as np


def default_params() -> tuple[dict, dict, dict, dict]:
    """Return ``(time_params, drug_params, ctrl_params, mort_params)``.

    Equivalent to the MATLAB ``default_params()`` function. Field
    names match the MATLAB struct fields.
    """
    # --- Time --------------------------------------------------------
    Tmax = 7              # days
    dt = 5 / 60 / 24      # 5 min in days
    n_steps = int(round(Tmax / dt))   # 2016 intervals
    T = np.linspace(0, Tmax, n_steps + 1)
    time = {
        "Tmax": Tmax,
        "dt":   dt,
        "T":    T,
        "Nt":   T.size,
    }

    # --- Drug PK -----------------------------------------------------
    dt_hours = dt * 24
    drug = {
        "half_life":  8,                                # hours
        "k_e":        float(np.exp(-np.log(2) * dt_hours / 8)),
        # PD (linear)
        "alpha_base": 0.02,
        "alpha_sofa": -0.3,
        "alpha_age":  0.2,
        # PD (Hill) — default mode
        "pd_mode":     "hill",
        "C50":         500,
        "gamma_hill":  2,
        # Dosing
        "dosing_interval": 8,    # hours
        "dose_standard":   500,
        "infusion_rate":   5,
    }
    drug["alpha"] = drug["alpha_base"]   # population mean

    # --- Control / strategy ------------------------------------------
    ctrl = {
        "omega":             2,     # target rate (sz/day)
        "dose_none":         0,
        "dose_low":          250,
        "dose_intermediate": 375,
        "dose_medium":       500,
        "dose_high":         750,
        "dose_standard":     500,
        "prn_threshold":     5,
        "prn_legacy":        True,   # R&R: True = submitted 4-h rate rule
        "prn_window_hours":  8,      # count rule window (one dosing interval)
        "prn_min_count":     1,      # count rule threshold
        "k_p":               0.5,
        "k_i":               2,
        "A_max":             8,
    }

    # --- Mortality ----------------------------------------------------
    # z = theta0 + theta1*(d/Tmax)^2
    #   + (theta2 + theta3*SOFA/10) * L_cum/L_norm
    #   + (theta4 + theta5*Age/65) * X_cum/X_norm
    # h_d = 1 / (1 + exp(-z)), evaluated daily.
    mort = {
        "theta":  np.array([-4, 0.5, 1.5, 1.0, 0.15, 0.07]),
        "L_norm": 35,           # seizure burden normalization (sz*day)
        "X_norm": 5000,         # drug exposure normalization (AUC units)
        "Tmax":   Tmax,         # survival horizon (days)
    }

    return time, drug, ctrl, mort
