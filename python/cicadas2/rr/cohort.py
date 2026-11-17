"""Reproducible synthetic cohorts (R&R task 4-5, G6).

Patient i draws all of its random inputs from np.random.default_rng((seed, i)).
Consequences: (1) prefix-stable: the first 200 patients of a 1,000-patient
cohort equal a 200-patient cohort with the same seed; (2) nothing an
experiment does with other random numbers can change who the patients are.
The submitted scripts drew patients from one shared stream, so each script
had its own cohort (why Tables 9 and 10 disagree on the same setting).
"""
from __future__ import annotations

import numpy as np

from .. import patient as patient_mod
from ..params import default_params

_KEYS = ("age", "sofa", "A_base", "r", "kappa", "alpha", "C50", "gamma_hill")


def generate(n: int, seed: int, generator: str = "rise_decay", peak_rate: float = 10.0) -> dict:
    time, drug, _, _ = default_params()
    T, Nt = time["T"], time["Nt"]
    out = {k: np.zeros(n) for k in ("age", "sofa", "A_patient", "r", "kappa", "alpha", "C50", "gamma_hill")}
    LB = np.zeros((n, Nt))
    for i in range(n):
        g = np.random.default_rng((seed, i))
        u = {k: g.uniform() for k in _KEYS}
        if generator == "rise_decay":
            p = patient_mod.generate(T, {"peak_rate": peak_rate, "drug_params": drug}, u_draws=u)
        elif generator == "sde":
            u["t_switch"] = g.uniform()
            u["sde"] = g.uniform(size=Nt - 1)
            p = patient_mod.generate_cicadas(T, {"peak_rate": peak_rate, "drug_params": drug}, u_draws=u)
        else:
            raise ValueError(generator)
        for k in out:
            out[k][i] = p.get(k, np.nan)
        LB[i] = p["lambda_base"]
    out.update(n=n, seed=seed, generator=generator, lambda_base=LB, peak_rate=peak_rate)
    return out


def subset(c: dict, idx) -> dict:
    idx = np.asarray(idx)
    s = dict(c)
    for k, v in c.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (c["n"],):
            s[k] = v[idx]
    s["n"] = len(idx)
    return s


def uniform_tape(n: int, Nt: int, seed: int, stream: int = 0, offset: int = 0) -> np.ndarray:
    """Per-patient uniform tape U[i] = default_rng((seed, stream, offset + i)).uniform(size=Nt).

    Using the same tape for every strategy gives common random numbers:
    bin k of patient i uses the same uniform under every strategy, so the
    Poisson inverse CDF can only give fewer seizures where the rate is lower.
    """
    U = np.empty((n, Nt))
    for i in range(n):
        U[i] = np.random.default_rng((seed, stream, offset + i)).uniform(size=Nt)
    return U
