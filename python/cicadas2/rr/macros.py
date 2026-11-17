"""LaTeX macro writer (R&R task 6). Every number in the paper comes from one file.

Each experiment returns a list of (name, value, fmt). write_all() merges every
results/rr_*_macros.json into CICADAS_II_Paper_Draft/results_macros.tex.
Names must be letters only (a LaTeX rule); duplicates are refused.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path


def save(macros, path):
    rows = [[n, v, f] for (n, v, f) in macros]
    Path(path).write_text(json.dumps(rows, indent=0))


def _fmt(v, f):
    if isinstance(v, str):
        return v
    s = (f or "{:.1f}").format(v)
    return s.replace("-", r"\ensuremath{-}")


def write_all(results_dir, out_tex):
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob("rr_*_macros.json"))
    seen = {}
    lines = [f"% AUTO-GENERATED {_dt.datetime.now():%Y-%m-%d %H:%M} by cicadas2.rr.macros.write_all. Do not edit by hand.",
             "% Sources: " + ", ".join(f.name for f in files)]
    for f in files:
        for name, v, fmt in json.loads(f.read_text()):
            if not re.fullmatch(r"[A-Za-z]+", name):
                raise ValueError(f"macro {name!r} must be letters only")
            if name in seen:
                raise ValueError(f"duplicate macro {name!r} in {f.name} and {seen[name]}")
            seen[name] = f.name
            lines.append(f"\\providecommand{{\\{name}}}{{}}\\renewcommand{{\\{name}}}{{{_fmt(v, fmt)}}}")
    Path(out_tex).write_text("\n".join(lines) + "\n")
    return len(seen)
