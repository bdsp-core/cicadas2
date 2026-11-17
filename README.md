# CICADAS-II

**C**ausal **I**nference for **C**ritical-Care **A**nti-Seizure Treatment with
**D**isease **D**ynamics, **A**utomated PKPD/Trial Simulation, and **S**urvival
Optimization — *Modified to accommodate infrequent events and bolus dosing.*

CICADAS-II is a simulation framework for developing and validating causal
inference and treatment-optimization methods for non-convulsive seizures in
the ICU. It models the full pipeline from a synthetic patient through latent
disease dynamics, pharmacokinetics/pharmacodynamics (PKPD), sparse Poisson
seizure observations, real-time state estimation, treatment strategies, and a
daily-hazard mortality model — then validates a parametric **g-formula** for
recovering causal treatment effects from confounded observational data.

This repository accompanies the CICADAS-II manuscript (in review). It contains **two
complete implementations** — the original **MATLAB** and a verified **Python**
port — plus the figure-generation pipeline and a full test suite.

> **Scope.** This is a *methodological* study on simulated data. It does not
> analyze real clinical data or provide clinical recommendations. The goal is
> to establish that the inferential pipeline is sound enough to deploy on real
> data in future work.

---

## Repository layout

```
cicadas2/
├── functions/                 MATLAB core functions (the reference implementation)
│   ├── default_params.m         single source of truth for all parameters
│   ├── fcn_generate_patient*.m  synthetic patient generators (rise-decay + CICADAS SDE)
│   ├── fcn_drug_pk*.m           one-compartment PK (infusion + bolus)
│   ├── fcn_observed_rate.m      Hill / linear PD suppression
│   ├── fcn_ekf_lograte_ou.m    Extended Kalman Filter on the log-rate state
│   ├── fcn_estimate_natural_history.m   Poisson MLE for (A_p, r, kappa)
│   ├── fcn_run_strategy.m       central per-patient simulator (5 strategies)
│   ├── fcn_run_adaptive_*.m     adaptive-strategy variants for sensitivity work
│   └── fcn_mortality*.m         daily-hazard mortality (linear + squared burden)
├── scripts/                   MATLAB experiment drivers (s0–s16)
├── results/                   cached *.mat outputs of the s*-scripts
│
├── python/                    Python port (see python/README.md)
│   ├── cicadas2/                importable package, 1:1 with functions/
│   │   └── experiments/         ports of scripts s0–s16
│   ├── tests/                   pytest suite + MATLAB-generated fixtures
│   └── pyproject.toml
├── tools/matlab_harness.m     regenerates the Python test fixtures from MATLAB
│
├── make_paper_figures.py      generates every manuscript figure (Python/matplotlib)
├── make_paper_figures.m       MATLAB equivalent of the figure pipeline
├── Figures_py/                generated figure PDFs
├── run_all_simulations.m      master MATLAB script: regenerates all results/*.mat
│
├── CICADAS_II_Paper_Draft/    generated LaTeX macros, tables and figures used by the manuscript
├── PYTHON_PORT_PLAN.md        the staged plan used to build the Python port
├── REPRODUCIBILITY.md         how to regenerate every reported number
└── PYTHON_PORT_PLAN.md / python/PORT_STATUS.md   per-component port status
```

---

## Requirements and tested versions

### MATLAB (reference implementation)
- **MATLAB R2025b** (tested). R2016b+ should work (uses local functions in scripts).
- **Statistics and Machine Learning Toolbox** — required, for
  `betarnd`/`gamrnd`/`lognrnd`/`poissrnd` (sampling),
  `betainv`/`gaminv`/`logninv`/`norminv`/`poissinv` (inverse-CDF, used by the
  reproducibility harness), and `corr`/`prctile`/`ksdensity` (analysis/plots).
- No other toolboxes are required (`fminsearch`, `polyfit` are base MATLAB).

### Python (port)
- **Python ≥ 3.10** (tested on 3.12; CI runs 3.10, 3.11, 3.12).
- **NumPy ≥ 1.26** and **SciPy ≥ 1.11** (tested with NumPy 2.4, SciPy 1.17).
- Dev/test extras: **pytest ≥ 7**, **pytest-cov**.
- Figure generation additionally needs **matplotlib** and **seaborn**.

```bash
python3.12 -m venv python/.venv
source python/.venv/bin/activate
pip install -e "python[dev]"      # installs the cicadas2 package + test deps
pip install matplotlib seaborn    # only needed for make_paper_figures.py
```

---

## Quick start

### Run the whole MATLAB pipeline
```matlab
% From the repo root, in MATLAB:
run_all_simulations          % regenerates every results/*.mat (1–3 h)
```
Then regenerate the figures (either language works on the same `results/*.mat`):
```bash
python make_paper_figures.py            # all figures -> Figures_py/
python make_paper_figures.py --figs 6,9 # only specific figures
```

### Run an experiment in Python
```python
from cicadas2.experiments import s9_gformula_causal
out = s9_gformula_causal.run(out_path="results/gformula_results.mat")
print(out["rct_ate"], out["naive_ate"], out["gform_ate"])
```
Every experiment module exposes a `run(...)` whose defaults mirror the MATLAB
script and writes a `.mat` with the **same schema**, so `make_paper_figures.py`
consumes either implementation's output unchanged.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the script → result → claim
mapping.

---

## The simulation pipeline (one patient)

1. **Patient generation** — covariates (age ~ Beta, SOFA ~ Gamma) and a
   latent rise-decay disease trajectory `λ_base(t) = A_p·𝒫·S(t)/max S(t)`,
   `S(t)=e^{−κt}(1−e^{−rt})`. (`fcn_generate_patient.m`; a CICADAS-style SDE
   alternative lives in `fcn_generate_patient_cicadas.m`.)
2. **PKPD** — one-compartment PK with bolus dosing every 8 h
   (`fcn_drug_pk_pill.m`) and Hill-type PD suppression
   (`fcn_observed_rate.m`), giving the observed rate `λ_obs(t)`.
3. **Observation** — sparse Poisson counts `N_k ~ Poisson(λ_obs(t_k)·Δt)`.
4. **State estimation** — an EKF on the log-deviation state recovers `λ̂(t)`
   from the counts in real time (`fcn_ekf_lograte_ou.m`).
5. **Treatment** — `none`, `fixed`, `prn`, or `adaptive` (a 5-tier dose menu
   thresholded on `λ̂` vs. `ω`) (`fcn_run_strategy.m`).
6. **Mortality** — discrete-time daily logistic hazard driven by cumulative
   seizure burden and drug exposure (`fcn_mortality.m`).
7. **Causal inference** — the parametric g-formula estimates each patient's
   natural-history parameters from observations, then simulates counterfactual
   survival under each strategy (`scripts/s9_gformula_causal.m`).

---

## Figures

All figures are produced by [`make_paper_figures.py`](make_paper_figures.py)
(one function per figure) and saved to [`Figures_py/`](Figures_py/). Every
figure reads a cached `results/*.mat` produced by one of the `s*` experiment
scripts (except Fig. 1, which is generated inline). Global styling — Okabe-Ito
colorblind-safe palette, Helvetica, a fixed font-size hierarchy, and a uniform
6.5″ figure width — is applied once at the top of the script.

| Fig | Generator function | Source data (`results/…`) | Producing script | What it shows |
|-----|--------------------|---------------------------|------------------|---------------|
| 1   | `fig01_demographics`        | *(generated inline)*               | `s0_validate_distributions` | Synthetic-cohort age and SOFA distributions; confirms the population matches published neuro-ICU cohorts. |
| 2   | `fig02_param_recovery`      | `parameter_estimation_results.mat` | `s6_parameter_estimation`   | True-vs-estimated scatter for the disease parameters `(A_p, r, κ)`, untreated (top) and treated (bottom). Severity `A_p` recovers well; rise/decay rates are structurally non-identifiable. Square panels with a 45° identity line. |
| 3   | `fig03_derived_recovery`    | `parameter_estimation_results.mat` | `s6_parameter_estimation`   | True-vs-estimated for the *derived* quantities that actually feed mortality: cumulative burden `L_cum` (ρ≈0.99), peak height, and peak time. Shows the composites are recoverable even when `(r, κ)` are not. |
| 4   | `fig04_confounding`         | `gformula_results.mat`             | `s9_gformula_causal`        | Peak-rate histograms by treatment group under RCT (balanced) vs. observational (treated patients sicker) assignment — the confounding-by-indication that flips the naive ATE's sign. |
| 5   | `fig04b_method_comparison`  | `gformula_results.mat`             | `s9_gformula_causal`        | Grouped survival bars (treated vs. untreated) across the three analyses — RCT (truth), naive observational, and g-formula — with ATE annotations (+18.5 / −2.1 / +19.4 pp). |
| 6   | `fig05_te_by_severity`      | `gformula_results.mat`             | `s9_gformula_causal`        | Per-patient g-formula treatment effect vs. peak seizure rate, colored by SOFA, with the population ATE and its 95% bootstrap CI — reveals heterogeneity masked by the ATE. |
| 7   | `fig06_quadrant`            | `quadrant_results.mat`             | `s5_quadrant_analysis`      | 2×2 grid of survival-by-strategy bar charts, one per harm phenotype; the optimal strategy changes across quadrants (Adaptive wins 3/4; *no treatment* wins Low-Sz/High-Drug). |
| 8   | `fig07_threshold_sweep`     | `threshold_sweep_results.mat`      | `s4_threshold_sweep`        | Survival vs. threshold for PRN (left) and adaptive `ω` (right), with the no-treatment baseline and the swept optimum marked. |
| 9   | `fig08_personalized`        | `personalized_validation_results.mat` | `s12_personalized_validation` | Per-regime survival under uniform `ω`, personalized-from-true-`λ`, and personalized-from-estimated-`λ` — estimated-parameter personalization recovers 94–105% of the oracle benefit. |
| 10  | `fig09_uniform_vs_dynamic`  | `closed_loop_personalized_results.mat` | `s13_closed_loop_personalized` | Per-regime survival under uniform `ω=2`, fixed regime-optimal `ω*`, and the dynamic `ω(t)` controller — admission-time selection dominates within the 7-day window. |
| 11  | `fig10_closed_loop`         | `closed_loop_results.mat`          | `s8_closed_loop_validation` | 2×4 grid: survival by strategy using true (top) vs. estimated (bottom) trajectories, per regime. The winning strategy matches in all four regimes (≤3 pp difference). |
| 12  | `fig11_ekf_sensitivity`     | `ekf_sensitivity_results.mat`      | `s15_ekf_sensitivity`       | Adaptive-strategy survival as each EKF tuning parameter (`q`, `τ_x`, `𝒫_pop`, floor, `ω`) is swept individually; robust (≤2 pp) except prior peak and threshold. |
| 13  | `fig12_graded_misspec`      | `misspecification_results.mat`     | `s16_misspecification`      | G-formula residual bias vs. the fraction of SDE-generated (misspecified-trajectory) patients — the robustness curve under graded model misspecification. |

> Figure numbers above follow the manuscript. The Python generator names keep
> the historical `figNN` scheme (with `fig04b` inserted), so e.g. manuscript
> Fig. 5 is produced by `fig04b_method_comparison`.

To regenerate one figure after re-running its experiment:
```bash
python make_paper_figures.py --figs 5      # uses internal slot numbers; --list to inspect
```

---

## Testing and verification

The Python port is verified against the MATLAB reference at three levels
(see [`PYTHON_PORT_PLAN.md`](PYTHON_PORT_PLAN.md) §4):

1. **Bit-identical** for deterministic code (PK, PD, EKF, mortality, the MLE
   objective) and, via a shared uniform-tape + inverse-CDF "RNG bridge", for
   the stochastic generators too — to ~1e-10.
2. **Within-tolerance** for the Nelder-Mead optimizer (objective matches to
   1e-9; argmin within ~5%).
3. **Distributional** for whole experiments — Python reproduces MATLAB's
   reported quantities within Monte Carlo noise. Verified at full sample size
   for the threshold sweep (s4), quadrant analysis (s5), and g-formula (s9);
   e.g. s9 at N=1000 gives naive ATE −4.1 pp (MATLAB −2.1) and g-formula ATE
   +18.3 pp (MATLAB +19.4), both within ~2 pp.

```bash
cd python
pytest -m "not slow"     # 52 unit tests (Phases 0-5) + 15 experiment smokes
```

The MATLAB-generated reference fixtures live in
`python/tests/reference_data/` and are regenerated by
[`tools/matlab_harness.m`](tools/matlab_harness.m). GitHub Actions runs the
unit and smoke suites on every push touching `python/`
(`.github/workflows/python-tests.yml`).

`python/PORT_STATUS.md` tracks per-component status. The decay-rate symbol is
named `kappa` consistently in both the code and the manuscript.

---

## Manuscript outputs

[`CICADAS_II_Paper_Draft/`](CICADAS_II_Paper_Draft/) holds what the code generates for the manuscript:
`results_macros.tex` (every reported number as a LaTeX macro), `rr_tables/` and `theta_table.tex` (tables), and
`figures_rr/` (figures). Text, figures and tables in the paper all come from these files, so they cannot disagree.
The manuscript text itself is not part of this repository.

### Reproducing the revised manuscript

All revision analyses run in Python (`python/cicadas2/rr/`), on one shared cohort of 1,000 synthetic patients with
per-patient seeds and exact 7-day survival.

```bash
cd python
python -m cicadas2.rr.run                 # every experiment (about 40 min); writes results/rr_*.json/.npz,
                                          # results_macros.tex and the tables in CICADAS_II_Paper_Draft/rr_tables/
python -m cicadas2.rr.paper_figures       # every figure -> CICADAS_II_Paper_Draft/figures_rr/
python -m cicadas2.rr.run strategies harm # or a subset of experiments by name
```

`results/rr_ledger.jsonl` records every run with its seed, cohort size and a fingerprint of the code that produced
it. The real-cohort numbers in Section 6 of the manuscript are aggregate counts produced in a separate, access-restricted
analysis environment; no patient-level data are stored here.

## Citation

If you use this framework, please cite the CICADAS-II manuscript (in
preparation) and the original CICADAS paper (McCauley et al., 2025).
