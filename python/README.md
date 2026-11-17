# cicadas2 — Python port

Python port of the MATLAB CICADAS-II framework. See
[`../PYTHON_PORT_PLAN.md`](../PYTHON_PORT_PLAN.md) for the staged plan
and [`PORT_STATUS.md`](PORT_STATUS.md) for live progress.

## Quick start

From the repo root:

```bash
python3.12 -m venv python/.venv
source python/.venv/bin/activate
pip install -e python[dev]
pytest python/tests
```

Tests that require a MATLAB-generated reference fixture are skipped
when the `.mat` is missing (look for `SKIPPED` lines like
`missing fixture mortality_seed0.mat; run tools/matlab_harness.m`).

## Generating the reference fixtures

Once, on a machine with MATLAB:

```matlab
addpath(genpath('.'));
run('tools/matlab_harness.m');
```

This populates `python/tests/reference_data/*.mat`. Commit them so CI
and other developers don't need MATLAB.

## Package layout

```
cicadas2/
├── __init__.py
├── _matlab_compat.py   # .mat I/O + RNG-bridge utilities
├── params.py           # port of default_params.m
├── mortality.py        # port of fcn_mortality.m
├── …                   # one module per MATLAB function (mirrors structure)
└── experiments/        # one module per s*.m script (Phase 6)
```

The 1:1 mapping is intentional — divergence between MATLAB and
Python is easy to spot in code review.

## Conventions

- **MATLAB struct → Python dict.** Field names preserved verbatim.
- **MATLAB 1-based index → Python 0-based.** Annotated with the
  `matlab_to_python_index()` helper at every conversion site so
  off-by-one bugs are easy to grep for.
- **Stochastic functions.** Always accept an optional `u_draws=`
  argument (pre-drawn uniforms) so deterministic verification is
  possible. Default is to draw fresh uniforms via numpy.
- **.mat schema.** When experiments save their results, they emit
  the same field names as the MATLAB version so the existing
  `make_paper_figures.py` works on either Python- or MATLAB-generated
  outputs.

## Verification levels

1. **Property tests** — no MATLAB fixture needed. Catch coding bugs
   via invariants (range checks, monotonicity, etc.).
2. **Bit-identical fixture tests** — load MATLAB-saved inputs and
   outputs; assert the Python port reproduces the outputs exactly.
   Marked `@pytest.mark.needs_matlab_fixture`.
3. **End-to-end experiments** — run a whole `s*` experiment in both
   languages, compare summary statistics within MC tolerance.
   Marked `@pytest.mark.slow`; gated to nightly CI.
