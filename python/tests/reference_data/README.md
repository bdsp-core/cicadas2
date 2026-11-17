# Reference data fixtures

`.mat` files in this directory are *reference* outputs of the MATLAB
code, used to verify the Python port produces identical results.

Each fixture is produced by `tools/matlab_harness.m`. Run that
script in MATLAB once (from the repo root):

```matlab
addpath(genpath('.'));
run('tools/matlab_harness.m');
```

The harness writes one `.mat` per function-under-test into this
directory. The Python tests that load these files are decorated with
`@pytest.mark.needs_matlab_fixture` and are *skipped* (not failed)
when the file is missing.

Total disk usage is small (<50 MB once all fixtures exist); committing
them to the repo is the chosen tradeoff per
[../../../PYTHON_PORT_PLAN.md](../../../PYTHON_PORT_PLAN.md) §9.
