"""CLI: python -m cicadas2.rr.run [experiment ...] [--results DIR]

Runs the R&R experiments (all by default), then rebuilds results_macros.tex.
"""
import argparse
import json
import time
from pathlib import Path

from . import experiments, macros, tables

ROOT = Path(__file__).resolve().parents[3]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(experiments.ALL))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--tex", default=str(ROOT / "CICADAS_II_Paper_Draft" / "results_macros.tex"))
    a = ap.parse_args(argv)
    for nm in a.names:
        t0 = time.time()
        out = experiments.ALL[nm](a.results)
        print(f"== {nm} ({time.time() - t0:.0f} s)")
        print(json.dumps(out.get("headline", {}), indent=1, default=float)[:3000])
    n = macros.write_all(a.results, a.tex)
    print(f"wrote {n} macros to {a.tex}")
    try:
        tables.main()
    except FileNotFoundError as e:   # tables need every experiment; skip when only some were run
        print("tables skipped:", e)


if __name__ == "__main__":
    main()
