# Reproducibility — CICADAS-II

This document describes how to regenerate every numerical result in
the CICADAS-II manuscript from the MATLAB code in this repository.

## Quick start

From the repo root, in MATLAB:

```matlab
run_all_simulations    % default seed = 42
```

That single command runs every numbered simulation script in
`scripts/` in order. Each script writes its cached output to
`results/*.mat`; those `.mat` files are what the manuscript's
quantitative claims are computed from. After the run completes,
regenerate the figures with

```bash
python make_paper_figures.py
```

in the repo root (requires Python 3.10+, matplotlib, scipy, seaborn).

Total wall time on a modern laptop: approximately 1–3 hours. The
longest scripts are `s12_personalized_validation`,
`s15_ekf_sensitivity`, and `s16_misspecification`.

## Where each manuscript claim comes from

| Manuscript claim                          | Source script                  | Cached output                                  |
|-------------------------------------------|--------------------------------|------------------------------------------------|
| Synthetic cohort age/SOFA breakdown       | `s0_validate_distributions`    | (prints diagnostics; no .mat)                  |
| Parameter recovery (ρ_Lcum = 0.986, …)    | `s6_parameter_estimation`      | `parameter_estimation_results.mat`             |
| G-formula bias reduction (95.4%)          | `s9_gformula_causal`           | `gformula_results.mat`                         |
| RCT / naive / g-formula ATEs (+18.5, −2.1, +19.4) | `s9_gformula_causal`   | `gformula_results.mat`                         |
| Quadrant survival rates                   | `s5_quadrant_analysis`         | `quadrant_results.mat`                         |
| Threshold sweep (28 pp spread)            | `s4_threshold_sweep`           | `threshold_sweep_results.mat`                  |
| Personalized threshold gains (3–11 pp)    | `s12_personalized_validation`  | `personalized_validation_results.mat`          |
| Closed-loop ranking match (≤3 pp)         | `s8_closed_loop_validation`    | `closed_loop_results.mat`                      |
| Uniform vs fixed-vs-dynamic ω             | `s13_closed_loop_personalized` | `closed_loop_personalized_results.mat`         |
| EKF sensitivity (≤2 pp for most params)   | `s15_ekf_sensitivity`          | `ekf_sensitivity_results.mat`                  |
| Misspecification bias (14.5 pp mortality) | `s16_misspecification`         | `misspecification_results.mat`                 |

## Single source of truth for parameters

All default parameters live in
[`functions/default_params.m`](functions/default_params.m). This includes:

- Time grid (`time_params.Tmax`, `time_params.dt`)
- PK constants (`drug_params.half_life`, `drug_params.k_e`, …)
- PD constants (`drug_params.C50`, `drug_params.gamma_hill`)
- Dosing menu (`ctrl_params.dose_low/intermediate/medium/high`)
- Default adaptive threshold (`ctrl_params.omega = 2`)
- Mortality coefficients (`mort_params.theta`)
- **Mortality normalization** (`mort_params.L_norm = 35`,
  `mort_params.X_norm = 5000`, `mort_params.Tmax = 7`)

Any script that needs these calls `default_params()` rather than
hardcoding values. `functions/fcn_mortality.m` accepts an optional
`mort_params` argument and falls back to `default_params()` if
omitted, so the normalization constants exist in exactly one place.

## Notation matches the manuscript

The disease decay rate is named `kappa` in both the code and the
paper (previously `d` in code, `\kappa` in paper). The struct field
`patient.kappa`, the estimator output `params_est.kappa`, and the
.mat keys `true_kappa` / `est_kappa` all use the new name.

Cached `.mat` files generated before this rename still carry the old
`true_d` / `est_d` keys. The Python figure script
([make_paper_figures.py](make_paper_figures.py)) aliases them to the
new keys on load, so existing figures continue to render without
re-running the simulations.  Re-running `run_all_simulations.m` will
produce `.mat` files with the new keys natively.

## Seeds and reproducibility caveats

`run_all_simulations.m` sets a master RNG seed (`rng(42, 'twister')`)
before running anything. Individual scripts may also `rng` internally
for sub-experiments; consult each script's first ~20 lines to confirm.

Reported numerical values are subject to Monte Carlo variability at
~0.5 pp scale even with a fixed master seed if scripts are run out of
order or on a different MATLAB version. The qualitative findings
(sign of ATEs, ordering of strategies across regimes, identifiability
ridge in `(r, κ)`) are not affected by this variability.

## Revision (JNE major revision, Oct 2026): `cicadas2.rr`

All revision analyses run in Python (MATLAB was unavailable during the revision):

```bash
cd python
pip install -e ".[dev]"
python -m cicadas2.rr.run                 # every revision experiment (~15 min on a laptop)
python -m cicadas2.rr.run gformula shapes # or selected experiments
python -m pytest tests/test_rr_engine.py tests/test_rr_survival.py
```

Outputs (new files only; no submitted `results/*.mat` is overwritten):

| Output | Content |
|---|---|
| `results/rr_<experiment>.json` / `.npz` | Full results of each experiment |
| `results/rr_<experiment>_macros.json` | Numbers for the paper |
| `results/rr_ledger.jsonl` | One line per run: time, revision item IDs addressed, seed, n, code fingerprint, outputs, key numbers |
| `CICADAS_II_Paper_Draft/results_macros.tex` | Every revision number as a LaTeX macro (loaded by the manuscript) |
| `CICADAS_II_Paper_Draft/rr_tables/*.tex`, `theta_table.tex` | Generated tables |

Conventions: one cohort (seed 2026, n = 1,000, patient *i* from `default_rng((2026, i))`); common random
numbers across strategies; survival computed exactly with 7 daily hazards (an earlier version of the code used 8); 95% intervals by
patient bootstrap.

| Revision claim | Experiment (`python -m cicadas2.rr.run <name>`) |
|---|---|
| Table 7 with CIs; tuned comparators; tiers; Table 10 | `strategies` |
| G-formula with fitted mortality model, oracle truth | `gformula`, `gformula_scaling` |
| Unmeasured confounding, E-value | `confounding` |
| Balance and overlap | `overlap` |
| Mortality shapes, time trend | `shapes` |
| Personalized omega* with intervals | `personalization` |
| omega* vs Lcum level set | `levelset` |
| Lcum recovery under graded misspecification | `graded_lcum` |
| EKF accuracy | `ekf` |
| Observation window and onset offset | `window` |
| Annotation and dose-record errors | `obs_errors` |
| Two-compartment PK, tolerance | `dgp` |
| Phenotype misassignment | `misclass` |
| Negative control | `refutation` |
