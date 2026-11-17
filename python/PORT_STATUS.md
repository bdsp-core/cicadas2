# Port status tracker

Tracks the Python port of each MATLAB function and experiment script
against the plan in [../PYTHON_PORT_PLAN.md](../PYTHON_PORT_PLAN.md).

Legend:
- ⬜  not started
- 🟡  port written, fixture pending
- 🔵  port + fixture, tests passing
- ✅  port + fixture + downstream callers verified

## Phase 1 — leaf deterministic functions

| MATLAB                          | Python module        | Status |
|---------------------------------|----------------------|--------|
| `default_params.m`              | `cicadas2.params`    | 🔵 (4 tests passing) |
| `fcn_drug_pk.m`                 | `cicadas2.pk` (`run_continuous`) | 🔵 (bit-identical against `pk_continuous.mat`; 3 tests passing) |
| `fcn_drug_pk_pill.m`            | `cicadas2.pk` (`run_pill`) | 🔵 (bit-identical against `pk_pill.mat`; 3 tests passing) |
| `fcn_observed_rate.m`           | `cicadas2.pd`        | 🔵 (bit-identical against `pd_linear.mat` + `pd_hill.mat`; 9 tests passing) |
| `fcn_compute_burden.m`          | `cicadas2.burden`    | 🔵 (bit-identical against `burden.mat`; 3 tests passing) |
| `fcn_mortality.m`               | `cicadas2.mortality` (`run`) | 🔵 (bit-identical against `mortality_seed0.mat` + `mortality_high_burden.mat`; 6 tests passing) |
| `fcn_mortality_squared.m`       | `cicadas2.mortality` (`run_squared`) | 🔵 (bit-identical against `mortality_squared.mat`; 1 test passing) |

## Phase 2 — stochastic generators

| MATLAB                          | Python module          | Status |
|---------------------------------|------------------------|--------|
| `fcn_generate_patient.m`        | `cicadas2.patient` (`generate`) | 🔵 (bit-identical via RNG injection against patient_basic / patient_with_linear_pd / patient_with_hill_pd; 7 tests passing) |
| `fcn_generate_patient_cicadas.m`| `cicadas2.patient` (`generate_cicadas`) | 🔵 (bit-identical 2016-step SDE trajectory; tolerance 1e-10; 1 test passing) |

## Phase 3 — state estimation

| MATLAB                          | Python module          | Status |
|---------------------------------|------------------------|--------|
| `fcn_ekf_lograte_ou.m`          | `cicadas2.ekf`         | 🔵 (bit-identical against ekf_lograte_ou.mat; 4 tests passing) |

## Phase 4 — parameter estimation

| MATLAB                          | Python module          | Status |
|---------------------------------|------------------------|--------|
| `fcn_estimate_natural_history.m`| `cicadas2.estimation`  | 🔵 (NLL bit-identical at fixed theta; converged optimum within 5%/1% of MATLAB's fminsearch; 3 tests passing) |

## Phase 5 — strategy runners

| MATLAB                          | Python module          | Status |
|---------------------------------|------------------------|--------|
| `fcn_run_strategy.m`            | `cicadas2.strategies` (`run_strategy`) | 🔵 (bit-identical for none/fixed/prn/adaptive; 4 fixture tests passing) |
| `fcn_run_adaptive_custom.m`     | `cicadas2.strategies` (`run_adaptive_custom`) | 🔵 (bit-identical incl. 10-run mortality MC; 1 test passing) |
| `fcn_run_adaptive_with_ekf.m`   | `cicadas2.strategies` (`run_adaptive_with_ekf`) | 🔵 (bit-identical full trajectory; 1 test passing) |

## Phase 6 — experiment scripts

| MATLAB                          | Python module                         | Status |
|---------------------------------|---------------------------------------|--------|
| `s0_validate_distributions.m`   | `cicadas2.experiments.s0_validate_distributions` | 🟡 (port runs end-to-end; diagnostic only, no .mat) |
| `s3_population_sim.m`           | `cicadas2.experiments.s3_population_sim`         | 🟡 (port runs end-to-end; smoke test passes) |
| `s4_threshold_sweep.m`          | `cicadas2.experiments.s4_threshold_sweep`        | 🟡 (port runs end-to-end; full N=200 distributional-equivalence test pending) |
| `s5_quadrant_analysis.m`        | `cicadas2.experiments.s5_quadrant_analysis`      | 🟡 (port runs end-to-end; smoke test passes) |
| `s6_parameter_estimation.m`     | `cicadas2.experiments.s6_parameter_estimation`   | 🟡 |
| `s7_estimation_robustness.m`    | `cicadas2.experiments.s7_estimation_robustness`  | 🟡 |
| `s8_closed_loop_validation.m`   | `cicadas2.experiments.s8_closed_loop_validation` | 🟡 |
| `s9_gformula_causal.m`          | `cicadas2.experiments.s9_gformula_causal`        | 🟡 (smoke test reproduces qualitative pattern: naive ATE -8 pp, g-formula +18 pp) |
| `s10_smart_adaptive.m`          | `cicadas2.experiments.s10_smart_adaptive`        | 🟡 |
| `s11_ke_sensitivity.m`          | `cicadas2.experiments.s11_ke_sensitivity`        | 🟡 |
| `s12_personalized_validation.m` | `cicadas2.experiments.s12_personalized_validation` | 🟡 |
| `s13_closed_loop_personalized.m`| `cicadas2.experiments.s13_closed_loop_personalized` | 🟡 |
| `s14_c50_sensitivity.m`         | `cicadas2.experiments.s14_c50_sensitivity`       | 🟡 |
| `s15_ekf_sensitivity.m`         | `cicadas2.experiments.s15_ekf_sensitivity`       | 🟡 |
| `s16_misspecification.m`        | `cicadas2.experiments.s16_misspecification`      | 🟡 |
| (skipped: s1, s2 — visual demos) | —                                               | n/a    |

**Phase 6 status**: all 15 experiment scripts have working Python ports
that run end-to-end on smoke-size inputs. Each module exposes a
``run(...)`` function with MATLAB-matching defaults that writes a
``.mat`` file with the same schema as the MATLAB counterpart, so
``make_paper_figures.py`` works on either output unchanged.

Full N=full distributional-equivalence integration tests (per
PYTHON_PORT_PLAN.md §4.4) are deferred — each takes 10-60 minutes of
wall time and requires running the MATLAB reference at matching N.

## Discrepancies between code and paper (surfaced by porting)

These are MATLAB-code behaviors that mismatch the manuscript and may
warrant a small fix to either side after the port stabilises:

- **`fcn_mortality` evaluates 8 daily hazards, not 7.** The loop is
  `for d = 1:ceil(Nt*dt)` and `Nt*dt = 7.003`, so `n_days = 8`. The
  8th iteration reuses the last index of `L_cum / X_cum` and adds an
  extra hazard at `(d/Tmax)^2 = (8/7)^2 ≈ 1.31`. The paper describes
  a daily hazard over `d ∈ {1, ..., Tmax = 7}`. Impact is small (most
  patients won't die on day 8 given typical hazards) but should be
  reconciled.

## Performance notes

(Populated as we go; per open-question #1 from the plan.)

- TBD: which experiments are routinely re-run? Add Numba JIT only
  for those.

## R&R additions (Oct 2026)

| Item | MATLAB | Python | Parity |
|---|---|---|---|
| PRN count rule (`prn_legacy`, `prn_window_hours`, `prn_min_count`) | `fcn_run_strategy.m`, `default_params.m` | `strategies.py`, `params.py` | Identical doses on shared tapes, both rules (checked in Octave) |
| Exact survival | `fcn_survival_exact.m` | `rr/survival.py` | Agree to 4e-19 (linear); note MATLAB `'squared'` squares both terms like `fcn_mortality_squared`, Python uses `shape="squared", drug_shape="squared"` for the same model |
| Paired bootstrap, summaries, macro writer, theta registry | `fcn_paired_bootstrap.m`, `fcn_summarize_survival.m`, `fcn_write_macros.m`, `theta_registry.m` | `rr/stats.py`, `rr/macros.py`, `rr/theta.py` | Same definitions; not cross-tested |
| Vectorised engine, cohorts, g-formula with fitted theta, all revision experiments | none (Python only, MATLAB unavailable during the revision) | `rr/engine.py`, `rr/cohort.py`, `rr/gformula.py`, `rr/experiments.py` | Engine reproduces `run_strategy` (identical counts and doses) |
