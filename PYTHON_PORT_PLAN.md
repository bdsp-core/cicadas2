# Python port plan for CICADAS-II

A staged plan to port the entire MATLAB codebase to Python, with
explicit per-component verification against the existing MATLAB
reference.

## 1. Scope and goals

### Goals
1. A self-contained Python implementation that reproduces every
   numerical result currently produced by the MATLAB code.
2. **Bit-identical** equivalence for deterministic code paths (PK
   integration, mortality hazard, EKF update equations,
   log-likelihood evaluation, etc.).
3. **Statistical** equivalence for stochastic code paths (patient
   generation, Monte Carlo survival), to a tolerance commensurate
   with Monte Carlo noise (≤0.5 pp on survival metrics at the sample
   sizes used in the paper).
4. A test suite that runs in CI and fails loudly if the Python
   implementation drifts from the MATLAB reference.

### Non-goals
1. Rewriting the manuscript figure script (`make_paper_figures.py`
   already exists and works on either MATLAB- or Python-generated
   `.mat` files).
2. Optimizing for performance beyond making the port usable
   (Python will be slower than MATLAB for tight numerical loops; that
   is acceptable for v1).
3. Refactoring the algorithmic structure. The port should mirror the
   MATLAB structure 1:1 so divergence is easy to spot in code review.

## 2. Codebase inventory

| Layer        | Files | LOC  | Notes |
|--------------|-------|------|-------|
| `functions/` | 14    | 1,282 | Pure math; deterministic except for patient generation |
| `scripts/`   | 17    | 5,458 | Experiment drivers; orchestrate functions and save `.mat` |
| **Total**    | 31    | 6,740 | Roughly 4,000–5,500 LOC Python after porting |

Largest dependencies (by LOC):
- `fcn_run_strategy.m` (271) — central simulator: PK loop, dosing decisions, EKF coupling
- `fcn_estimate_natural_history.m` (169) — Poisson MLE for $(A_p, r, \kappa)$
- `fcn_generate_patient_cicadas.m` (159) — SDE-based alternative generator
- Script extremes: `s16_misspecification.m` (620), `s9_gformula_causal.m` (535), `s6_parameter_estimation.m` (398)

## 3. Target package layout

```
python/                            (new directory)
├── cicadas2/                      (the importable package)
│   ├── __init__.py
│   ├── params.py                  (port of default_params.m)
│   ├── pk.py                      (fcn_drug_pk, fcn_drug_pk_pill)
│   ├── pd.py                      (fcn_observed_rate)
│   ├── patient.py                 (fcn_generate_patient[_cicadas])
│   ├── mortality.py               (fcn_mortality, fcn_mortality_squared)
│   ├── ekf.py                     (fcn_ekf_lograte_ou)
│   ├── estimation.py              (fcn_estimate_natural_history)
│   ├── strategies.py              (fcn_run_strategy and adaptive variants)
│   ├── burden.py                  (fcn_compute_burden)
│   ├── _matlab_compat.py          (RNG bridge, .mat I/O helpers)
│   └── experiments/
│       ├── s0_validate.py
│       ├── s4_threshold_sweep.py
│       ├── s5_quadrant.py
│       ├── …
│       └── s16_misspecification.py
├── tests/
│   ├── conftest.py                (fixtures: load reference .mat files)
│   ├── reference_data/            (small .mat fixtures from MATLAB harness)
│   ├── test_pk.py
│   ├── test_mortality.py
│   ├── test_ekf.py
│   ├── test_estimation.py
│   ├── test_strategies.py
│   └── test_experiments.py        (slow; skipped by default)
├── tools/
│   └── matlab_harness.m           (regenerates reference .mat fixtures)
├── pyproject.toml
└── README.md
```

Notes:
- Python package lives in `python/` so it sits alongside the MATLAB
  code without polluting the existing `functions/`, `scripts/`,
  `Figures_py/` layout.
- `make_paper_figures.py` stays where it is; eventually it can import
  from `cicadas2` instead of relying on cached `.mat` files.

## 4. Verification strategy — the hard part

### 4.1 What "matches MATLAB" means

| Class                          | Verification standard | Tolerance |
|--------------------------------|----------------------|-----------|
| Deterministic functions        | Bit-identical to MATLAB for the same inputs | $|\Delta| < 10^{-10}$ relative |
| Optimization (`fminsearch`)    | Same minimum *value*, possibly different argmin coordinates | objective within $10^{-6}$; argmin within $10^{-3}$ relative |
| Stochastic generators          | Distributional moments match | mean / std / quantiles within MC-noise CI |
| End-to-end experiments         | Reported manuscript numbers match | survival within $0.5$ pp; ATEs within $1.0$ pp |

### 4.2 The MATLAB→Python RNG bridge

This is the single biggest porting risk. MATLAB's `randn`, `rand`,
`betarnd`, `gamrnd`, `lognrnd`, `poissrnd` do **not** produce
identical streams to NumPy even with the same seed because:
- Different PRNG state representations (MATLAB's `twister` ≈ NumPy's
  `MT19937` but with different initial seeding).
- Different distribution-sampling algorithms for non-uniform draws
  (e.g., MATLAB's `gamrnd` uses Marsaglia-Tsang; NumPy uses a
  different rejection scheme depending on shape).

**Pragmatic resolution:** for each stochastic function we have two
options.

**Option A — RNG injection.** Refactor the function to accept a
pre-drawn uniform stream (or a seeded NumPy `Generator`) as an
argument. The matching MATLAB version is also refactored to read its
random numbers from a saved `.mat` of uniform draws. Both sides
consume the same random tape and produce bit-identical output. Best
for unit-testable internals.

**Option B — Distributional equivalence.** Don't try to match the
exact draws; instead assert that 10,000 independent draws have the
same mean / std / quantile distribution within MC-CI. Best for
end-to-end experiments where exact-equality is impossible and
unnecessary.

Plan: **use Option A for unit tests of the leaf stochastic functions
(patient.py); use Option B for the orchestrating experiments
(experiments/*.py).**

### 4.3 The MATLAB harness

A single MATLAB script `tools/matlab_harness.m` regenerates every
reference `.mat` file consumed by the tests. Each fixture pairs an
input snapshot with the MATLAB output. The python test loads both,
runs the python port on the input, and compares.

Example structure per function:

```matlab
% In tools/matlab_harness.m
rng(0, 'twister');
input = struct();
input.lambda_obs = ...;
input.x          = ...;
input.dt         = 5/60/24;
input.theta      = [-4, 0.5, 1.5, 1.0, 0.15, 0.07];
input.age        = 65;
input.sofa       = 4;
input.u_draws    = rand(1, 1000);    % pre-drawn uniforms for RNG injection
output = struct();
[output.alive, output.death_time] = fcn_mortality_seeded(input, input.u_draws);
save('python/tests/reference_data/mortality_seed0.mat', '-struct', 'input', 'output');
```

```python
# In tests/test_mortality.py
def test_fcn_mortality_matches_matlab(fixture):
    inp, out = load_fixture('mortality_seed0.mat')
    py_alive, py_death = cicadas2.mortality.run(
        inp['lambda_obs'], inp['x'], inp['dt'], inp['theta'],
        inp['age'], inp['sofa'],
        u_draws=inp['u_draws'],   # injected uniforms
    )
    assert py_alive == out['alive']
    assert np.isclose(py_death, out['death_time'], equal_nan=True)
```

This pattern scales: one fixture per function per "interesting input
class," ~5-10 fixtures per function. Total reference fixtures: ~150,
total disk usage <50 MB.

### 4.4 End-to-end equivalence

For each numbered experiment (`s4_threshold_sweep`, etc.), provide a
"reproduction harness" that runs the experiment in both languages,
loads both `.mat` outputs, and compares all numeric fields with a
~0.5 pp tolerance on survival statistics and ~1.0 pp on ATEs. This is
a slow integration test (~minutes per script) — gated behind a
pytest mark `@pytest.mark.slow` and run in nightly CI rather than
per-commit.

## 5. Phased execution

| Phase | Scope | Deliverable | Approx effort (engineer-days) |
|------:|-------|-------------|------------------------------:|
| 0 | Scaffolding | `python/` package skeleton, pyproject.toml, pytest setup, .mat I/O helpers, RNG-bridge utilities, CI workflow | 2 |
| 1 | Leaf deterministic functions: `default_params`, `fcn_drug_pk`, `fcn_drug_pk_pill`, `fcn_observed_rate`, `fcn_compute_burden`, `fcn_mortality`, `fcn_mortality_squared` | Python module + unit tests with bit-identical fixtures | 3 |
| 2 | Stochastic generators: `fcn_generate_patient`, `fcn_generate_patient_cicadas` | Python module + unit tests with RNG-injected fixtures + distributional sanity tests | 4 |
| 3 | EKF: `fcn_ekf_lograte_ou` | Python module + per-step fixtures (predict step, update step, full trajectory) | 2 |
| 4 | Estimation: `fcn_estimate_natural_history` | Python module + fixtures; address `fminsearch` vs `scipy.optimize.minimize` minor disagreement | 3 |
| 5 | Strategy runners: `fcn_run_strategy`, `fcn_run_adaptive_custom`, `fcn_run_adaptive_with_ekf` | Python module + fixtures per strategy (none, fixed, PRN, adaptive); compare full PK/dosing trace | 5 |
| 6 | Experiment scripts: s0, s3–s16 (skip s1, s2 — visual demos only) | Python modules under `experiments/`; slow integration tests; reference comparisons against the cached `results/*.mat` files; document expected MC tolerances | 8 |
| 7 | Wire `make_paper_figures.py` to read from the new package directly (optional; can keep `.mat`-based loading) | Optional simplification | 1 |
| 8 | Documentation: how to run, how to add a test, how to extend | README.md, CONTRIBUTING.md | 1 |
| **Total** | | | **≈ 29 engineer-days** (~6 calendar weeks at half-time, ~3 weeks intensive) |

## 6. Per-component validation summary

For each ported component, the merge-gate is the same:

1. The MATLAB harness (re)produces a fixture or set of fixtures.
2. The Python port has at least one unit test per fixture, asserting
   the documented tolerance.
3. The unit tests pass under `pytest python/tests`.
4. End-to-end tests for any *experiment* touched are run manually
   before merging.

A short table tracks per-component status in
`python/PORT_STATUS.md` (to be created during Phase 0).

## 7. Risk register

| Risk | Likelihood | Mitigation |
|------|-----------:|-----------|
| `scipy.optimize.minimize(method='Nelder-Mead')` finds a slightly different argmin than `fminsearch` for some patients | High | Verify on the objective value rather than the argmin; document MAPE on per-patient estimates as the acceptable tolerance |
| Vectorization differences (e.g., `cumsum`, `repmat`) introduce off-by-one errors | Medium | Bit-identical per-step fixtures catch these immediately |
| Old `.mat` files contain MATLAB struct arrays that scipy.io can't round-trip | Medium | Snapshot inputs as simple double arrays in the harness; document any struct fields that need manual unpacking |
| `Beta(3,5)` and `Gamma(2,1.5)` sampling diverge between MATLAB and NumPy even with same seed | Certain | Inject uniform draws and apply inverse-CDF (`scipy.stats.beta.ppf`, `gamma.ppf`) in both languages, identical algorithm |
| The Hill PD function's `1/((C50/x)^gamma + 1)` form has a discontinuity at $x=0$ handled implicitly in MATLAB; NumPy may emit RuntimeWarning | Low | Wrap with explicit `np.where` guard; assert no warnings raised in tests |
| Stochastic SDE generator (`fcn_generate_patient_cicadas`) is hard to verify exactly | High | Use distributional equivalence only; document this explicitly |
| MATLAB `for d = 1:n_days` is 1-indexed; Python is 0-indexed — easy off-by-one | High | Add tests that exercise day-1 and last-day boundaries specifically |
| `fminsearch` `'Display','off','MaxFunEvals',5000,'MaxIter',2000` settings need to be matched in scipy | Medium | Pass `options={'xatol': ..., 'fatol': ..., 'maxiter': ..., 'maxfev': ...}` explicitly |

## 8. First concrete deliverables (week 1)

If we proceed, here's what week 1 should look like:

**Day 1–2 (Phase 0):**
1. Create `python/` directory with `pyproject.toml` declaring deps
   (`numpy>=1.26`, `scipy>=1.11`, `pytest>=7`, `pytest-cov`).
2. Set up `python/tests/conftest.py` with `load_fixture(name)` helper.
3. Add a GitHub Actions workflow that runs `pytest python/tests` on
   pushes touching `python/`.
4. Write `tools/matlab_harness.m` skeleton with one function (
   `fcn_mortality`) wired end-to-end.

**Day 3–5 (Phase 1, half):**
5. Port `default_params.m` → `params.py`.
6. Port `fcn_drug_pk.m`, `fcn_observed_rate.m`,
   `fcn_compute_burden.m` → `pk.py`, `pd.py`, `burden.py`.
7. Write unit tests for each, with fixtures generated by the
   harness.
8. Establish the per-component status table.

End of week 1: ~30 % of the leaf functions ported, the verification
pattern proven, the rest of the work mechanical.

## 9. Open questions for the author

1. **Performance target.** Are there scripts you re-run frequently
   where Python being 2–10× slower than MATLAB would be a real
   pain? If yes, we may want Numba/Cython for the hot loops in
   `fcn_run_strategy`.
2. **Backward compatibility on `.mat` files.** Should the Python
   port write the same `.mat` schema that MATLAB writes, so the
   existing `make_paper_figures.py` works unchanged? (Recommended:
   yes.)
3. **Continuing MATLAB code path.** Do you want to keep the MATLAB
   code as the canonical reference indefinitely, or eventually
   delete it once Python parity is established? (Recommended: keep
   both for ≥6 months after parity, then phase out MATLAB.)
4. **Optimizer fidelity.** If `scipy.optimize.minimize(Nelder-Mead)`
   gives per-patient $(A_p, r, \kappa)$ estimates that differ from
   MATLAB by, say, 5 % MAPE while preserving downstream g-formula
   ATEs, is that acceptable, or do we need exact argmin match?
5. **Test data freshness.** Is it OK if the reference `.mat`
   fixtures live in the repo (~50 MB), or do you want them stored
   externally (e.g., Git LFS, a release artifact)?
