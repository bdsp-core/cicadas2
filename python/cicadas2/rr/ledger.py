"""Append-only run ledger: every experiment run records what it addressed and what it found.

results/rr_ledger.jsonl gets one JSON line per run: timestamp, experiment,
revision item IDs addressed, seed, n, code fingerprint, output files, and the key
numbers, so every reported number traces to a recorded run.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path

_PKG = Path(__file__).resolve().parent


def code_fingerprint():
    h = hashlib.sha256()
    # computational modules only (tables/macros/ledger/run do not change results)
    for name in ("cohort.py", "engine.py", "survival.py", "stats.py", "gformula.py", "experiments.py", "theta.py"):
        h.update((_PKG / name).read_bytes())
    for name in ("strategies.py", "estimation.py", "patient.py", "params.py"):
        h.update((_PKG.parent / name).read_bytes())
    return h.hexdigest()[:12]


def record(results_dir, experiment, ids, seed, n, outputs, key_numbers, notes=""):
    entry = dict(time=_dt.datetime.now().isoformat(timespec="seconds"), experiment=experiment,
                 addresses=list(ids), seed=seed, n=n, code=code_fingerprint(),
                 outputs=list(outputs), key_numbers=key_numbers, notes=notes)
    p = Path(results_dir) / "rr_ledger.jsonl"
    with p.open("a") as fh:
        fh.write(json.dumps(entry, default=float) + "\n")
    return entry
