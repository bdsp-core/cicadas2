"""MATLAB compatibility helpers.

Utilities that bridge MATLAB and Python conventions:

* ``mat_load`` / ``mat_save`` wrap ``scipy.io.loadmat`` /
  ``scipy.io.savemat`` with sensible defaults (``squeeze_me=True``,
  ``struct_as_record=False`` on load; v5 ``.mat`` on save).
* ``inverse_cdf_*`` helpers convert a pre-drawn uniform tape into
  draws from named distributions, so the same uniform stream
  produces bit-identical samples on the MATLAB and Python sides
  (see PYTHON_PORT_PLAN.md §4.2 for why).
* ``matlab_to_python_index`` clarifies the 1-based → 0-based
  conversion that bites every MATLAB port.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.stats as st


# ---------------------------------------------------------------------------
# .mat I/O
# ---------------------------------------------------------------------------
def mat_load(path: str | Path) -> dict:
    """Load a .mat file, returning a dict of numpy arrays / scalars.

    Drops scipy's automatic ``__header__`` / ``__version__`` / ``__globals__``
    entries so the returned dict only contains user-saved variables.
    """
    raw = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def mat_save(path: str | Path, **arrays) -> None:
    """Write a v5 .mat file (the format MATLAB loads natively)."""
    sio.savemat(str(path), arrays, do_compression=True, oned_as="column")


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def matlab_to_python_index(idx: int) -> int:
    """Convert a 1-based MATLAB index to 0-based Python.

    Trivial, but useful as a self-documenting marker at every site
    where MATLAB code computed an index that we now subscript into a
    NumPy array.
    """
    return idx - 1


# ---------------------------------------------------------------------------
# Inverse-CDF sampling (RNG bridge)
# ---------------------------------------------------------------------------
#
# MATLAB and NumPy use different algorithms for sampling from
# non-uniform distributions, so identical seeds produce different
# samples even when both initialise the same MT19937 state. To get
# bit-identical samples on both sides we use a shared uniform tape
# and inverse-CDF sampling. The functions below take a uniform draw
# u ∈ (0,1) and return a sample from the requested distribution.
#
# A MATLAB harness calling the same inverse-CDF routine (e.g.
# `betainv(u, a, b)`) on the same `u` tape produces identical
# samples — which is exactly what bit-identical port tests need.
# ---------------------------------------------------------------------------
def beta_from_uniform(u: np.ndarray, a: float, b: float) -> np.ndarray:
    return st.beta.ppf(u, a, b)


def gamma_from_uniform(u: np.ndarray, shape: float, scale: float) -> np.ndarray:
    return st.gamma.ppf(u, shape, scale=scale)


def lognormal_from_uniform(u: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return st.lognorm.ppf(u, s=sigma, scale=np.exp(mu))


def normal_from_uniform(u: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    return st.norm.ppf(u, loc=mu, scale=sigma)


def poisson_from_uniform(u: np.ndarray, lam: np.ndarray | float) -> np.ndarray:
    """Inverse-CDF Poisson; ``u`` and ``lam`` must broadcast."""
    return st.poisson.ppf(u, mu=lam).astype(int)


def poisson_ppf_scalar(u: float, lam: float) -> int:
    """Scalar Poisson inverse CDF.

    Equivalent to ``int(scipy.stats.poisson.ppf(u, lam))`` but ~10-50x
    faster on scalar arguments because it avoids scipy's argsreduce
    / broadcast dispatch machinery. Iterative CDF accumulator
    starting from k=0; for small ``lam`` (the common case in this
    simulator, since ``lam = lambda_obs * dt`` with ``dt = 5 min``)
    it terminates in 1-3 iterations.

    Used in the hot inner loop of ``strategies.run_strategy`` where
    profiling shows this single call accounted for 74% of runtime
    before optimization.
    """
    import math
    if lam <= 0.0:
        return 0
    if u >= 1.0:
        # PPF is +inf at u=1; clamp to a safe large integer
        return 10_000
    p = math.exp(-lam)
    cum = p
    k = 0
    while cum < u:
        k += 1
        p *= lam / k
        cum += p
        if k > 10_000:           # numerical safety net
            break
    return k
