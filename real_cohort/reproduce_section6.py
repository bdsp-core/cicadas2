#!/usr/bin/env python3
"""Reproduce every number in Section 6 ("Feasibility in a real cohort") of the CICADAS-II manuscript from this
package alone, and compare each one with the published aggregate file reference/rr_real_cohort.json.

  python code/reproduce_section6.py                 # from the package root
  python code/reproduce_section6.py --root /path/to/cicadas2 --expected reference/rr_real_cohort.json

Needs Python >= 3.10 with numpy, pandas and pyarrow. Runs in about 10 s and needs ~4.5 GB of memory.
Prints one PASS/FAIL line per quantity (JSON precision), then one line per number as the manuscript prints it
(LaTeX macro precision and the Table rows), then a summary. Exit code 0 only if everything passes.
Only aggregate values are printed.

What is computed here, and from which table
  cohort, demographics                 data/patients.parquet
  EEG hours, EA burden (stage 19)      recomputed from data/eeg_minute (and checked against patients.parquet)
  seizure sparsity                     data/eeg_minute
  drug exposure, levetiracetam         data/medications.parquet (and checked against patients.parquet flags)
  cumulative propofol (stage 20)       data/propofol_charting.parquet (and checked against patients.parquet)
  four groups, OR, sensitivity         stage-20 rules applied to the above
The definitions follow the source pipeline (cicadas-real-data pipeline/19_patient_ea_burden.py and
20_quadrants.py; cicadas3 pipeline/04_summarize_cohort_meds.py; the descriptive stage that wrote the JSON).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------------ constants
WINDOW_S = 120 * 3600                 # first 5 days of EEG, from FirstEEG (eeg_window_start_h on the package clock)
N_HOURS = 120
MIN_REC_24H = 6.0                     # a 24-h window enters the peak only with >= 6 h recorded (stage 19)
LAB = ["iiic", "sz_pp", "lpd", "gpd"]
CUT = 0.10
GROUPS = ["Low EA / no propofol", "Low EA / propofol", "High EA / no propofol", "High EA / propofol"]
SZ_THRESHOLDS = [1, 10, 30]           # seconds of post-processed seizure needed for a minute to count as seizure
DAY_MIN_RECORDED_MIN = 12 * 60        # a patient-day counts if >= 12 h (720 recorded minutes) of EEG
LEV_GAP_MIN_H, LEV_GAP_MAX_H = 2.0, 36.0   # intervals between consecutive levetiracetam doses: 2 < gap <= 36 h
RESCUE_BENZOS = ["lorazepam", "diazepam", "clonazepam", "midazolam"]
KEY = {"levetiracetam": ["levetiracetam"], "lacosamide": ["lacosamide"], "valproate": ["valproate"],
       "phenytoin_fosphenytoin": ["phenytoin", "fosphenytoin"], "phenobarbital": ["phenobarbital"],
       "propofol": ["propofol"], "dexmedetomidine": ["dexmedetomidine"]}


def q3(x, scale=1.0):
    x = pd.Series(x, dtype=float).dropna() * scale
    return [float(v) for v in x.quantile([0.25, 0.5, 0.75])]


def pct(mask):
    mask = np.asarray(mask, bool)
    return 100.0 * mask.sum() / len(mask)


# ------------------------------------------------------------------------------------------------ loading
def load(root: Path):
    p = pd.read_parquet(root / "data" / "patients.parquet")
    em = pd.read_parquet(root / "data" / "eeg_minute")
    med = pd.read_parquet(root / "data" / "medications.parquet")
    pc = pd.read_parquet(root / "data" / "propofol_charting.parquet")
    for t in (p, em, med, pc):
        t["BDSPPatientID"] = t["BDSPPatientID"].astype(str)
    return p, em, med, pc


def minute_arrays(p: pd.DataFrame, em: pd.DataFrame):
    """Integer-second clock: t_s = minute start, w0_s = FirstEEG, both on the package clock."""
    em = em.sort_values(["BDSPPatientID", "t_h"], kind="stable").reset_index(drop=True)
    idx = pd.Index(p["BDSPPatientID"])
    code = idx.get_indexer(em["BDSPPatientID"])
    assert (code >= 0).all(), "eeg_minute has a patient that is not in patients.parquet"
    t_s = np.round(em["t_h"].to_numpy() * 3600).astype(np.int64)                    # minute starts: whole seconds
    w0_s = np.round(p["eeg_window_start_h"].to_numpy() * 3600, 6)                    # FirstEEG: microseconds
    return em, code, t_s, w0_s


# ------------------------------------------------------------------------------------------------ stage 19
def ea_burden(p, em, code, t_s, w0_s):
    """Stage 19 on the minute table: a minute counts toward the window and hour containing its centre.
    Stage 19 truncates FirstEEG to whole seconds (np.datetime64(..., "s")), so the window starts there."""
    c = t_s + 30 - np.floor(w0_s)[code]
    m = (c >= 0) & (c < WINDOW_S)
    hr = np.clip(np.floor(c[m] / 3600).astype(int), 0, N_HOURS - 1)
    n = len(p)
    flat = code[m] * N_HOURS + hr

    def hourly(col):
        return np.bincount(flat, weights=em[col].to_numpy()[m].astype(float), minlength=n * N_HOURS).reshape(n, N_HOURS)

    rec = hourly("rec_s")
    out = pd.DataFrame({"BDSPPatientID": p["BDSPPatientID"].to_numpy()})
    out["rec_h_5d"] = rec.sum(1) / 3600
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(rec, 1)], 1)
    cr = cs[:, 24:] - cs[:, :-24]                                        # 97 sliding 24-h windows
    ok = cr >= MIN_REC_24H * 3600
    for k in LAB:
        x = hourly(f"{k}_s")
        tot = rec.sum(1)
        out[f"burden_{k}"] = np.where(tot > 0, x.sum(1) / np.where(tot > 0, tot, 1), np.nan)
        cx = np.concatenate([np.zeros((n, 1)), np.cumsum(x, 1)], 1)
        cx = cx[:, 24:] - cx[:, :-24]
        r = np.where(ok, cx / np.where(ok, cr, 1), -np.inf)
        pk = r.max(1)
        out[f"peak24_{k}"] = np.where(ok.any(1), pk, np.nan)
    out["n_24h_windows_ok"] = ok.sum(1)
    return out


# ------------------------------------------------------------------------------------------------ seizures
def seizure_sparsity(p, em, code, t_s, w0_s):
    """Seizure onsets in the first 5 days (minute start in [FirstEEG, FirstEEG + 120 h)).

    seizure minute  sz_pp_s >= threshold
    onset           a seizure minute whose immediately preceding clock minute is not a recorded seizure minute
    recorded day    number of recorded minutes / 1440
    5-min bins      the wall-clock 5-minute grid (grid5_phase_min aligns the package clock to it); share of a
                    patient's recorded bins that contain an onset
    patient-days    24-h days from FirstEEG with >= 720 recorded minutes
    """
    hs = t_s - w0_s[code]
    m = (hs >= 0) & (hs < WINDOW_S)
    cd, ts, hs = code[m], t_s[m], hs[m]
    sz_s = em["sz_pp_s"].to_numpy()[m]
    n = len(p)
    same = np.r_[False, cd[1:] == cd[:-1]]
    contig = same & np.r_[False, np.diff(ts) == 60]
    tmin = ts // 60
    phase = p["grid5_phase_min"].to_numpy().astype(np.int64)[cd]
    b5 = (tmin + phase) // 5
    day = np.floor(hs / 86400).astype(int)
    rec_min = np.bincount(cd, minlength=n)
    has = rec_min > 0
    res = {}
    for k in SZ_THRESHOLDS:
        sz = sz_s >= k
        on = sz & ~(np.r_[False, sz[:-1]] & contig)
        n_on = np.bincount(cd, weights=on, minlength=n)
        rate = n_on[has] / (rec_min[has] / 1440)
        anyp = n_on[has] > 0
        bins = pd.DataFrame({"p": cd, "b": b5, "on": on}).groupby(["p", "b"])["on"].sum()
        nb = bins.groupby(level=0).size()
        nb_on = (bins > 0).groupby(level=0).sum()
        szp = nb_on[nb_on > 0].index
        share = 100 * nb_on[szp] / nb[szp]
        days = pd.DataFrame({"p": cd, "d": day, "on": on}).groupby(["p", "d"]).agg(n=("on", "size"), o=("on", "sum"))
        days = days[days["n"] >= DAY_MIN_RECORDED_MIN]["o"]
        res[f"min{k}s"] = {
            "min_seizure_seconds_per_minute": k,
            "patients_with_any_pct": 100 * (n_on > 0).sum() / n,
            "onsets_per_recorded_day_all": q3(rate),
            "onsets_per_recorded_day_among_sz_patients": q3(rate[anyp]),
            "share_5min_bins_with_onset_among_sz_patients_pct": q3(share),
            "share_5min_bins_with_2plus_onsets_given_any_pct": 100 * (bins >= 2).sum() / (bins > 0).sum(),
            "patient_days_12h_recorded": int(len(days)),
            "patient_days_zero_onsets_pct": pct(days == 0),
            "patient_days_1to4_pct": pct(days.between(1, 4)),
            "patient_days_5to24_pct": pct(days.between(5, 24)),
            "patient_days_25plus_pct": pct(days >= 25),
        }
    return res


# ------------------------------------------------------------------------------------------------ medications
def med_hours(p, t: pd.DataFrame):
    """Hours from FirstEEG (cicadas3 / stage-22 'hours_from_first_eeg') on the package clock."""
    w0 = p.set_index("BDSPPatientID")["eeg_window_start_h"]
    return t["t_h"] - t["BDSPPatientID"].map(w0)


def drug_exposure(p, med):
    """cicadas3 stage-04 flags: any given administration with 0 <= hours_from_first_eeg <= 120."""
    h = med_hours(p, med)
    g = med[med["given"] & (h >= -1e-9) & (h <= 120 + 1e-9)]
    sched = g[(g["drug_class"] == "ASM") & g["subclass"].isin(["standard", "barbiturate"])]
    flags = pd.DataFrame(index=p["BDSPPatientID"])
    flags["asm_in_eeg5d"] = flags.index.isin(sched["BDSPPatientID"])
    for k, ds in KEY.items():
        flags[f"eeg5d_{k}"] = flags.index.isin(g.loc[g["drug"].isin(ds), "BDSPPatientID"])
    flags["eeg5d_rescue_benzo_bolus"] = flags.index.isin(g.loc[g["drug"].isin(RESCUE_BENZOS) & ~g["infusion"], "BDSPPatientID"])
    flags["eeg5d_sedative_hypnotic_infusion"] = flags.index.isin(
        g.loc[(g["subclass"] == "IV sedative-hypnotic") & g["infusion"], "BDSPPatientID"])
    return flags.astype(int)


def asm_doses(p, med):
    """One row per scheduled-ASM bolus (patient, drug, time): BI sums split components, MGH keeps the largest."""
    a = med[(med["drug_class"] == "ASM") & med["subclass"].isin(["standard", "barbiturate"]) & med["given"] & ~med["infusion"]]
    out = (a.groupby(["BDSPPatientID", "site", "drug", "t_h"])
             .agg(dose_sum=("dose_mg", lambda x: x.sum(min_count=1)), dose_max=("dose_mg", "max"))
             .reset_index())
    out["dose_mg"] = np.where(out["site"] == "BI", out["dose_sum"], out["dose_max"])
    out["hours_from_first_eeg"] = med_hours(p, out)
    return out


def levetiracetam(p, med):
    d = asm_doses(p, med)
    w = d[(d["drug"] == "levetiracetam") & (d["hours_from_first_eeg"] >= -1e-9) & (d["hours_from_first_eeg"] <= 120 + 1e-9)]
    w = w.sort_values(["BDSPPatientID", "t_h"], kind="stable")
    gap = (w.groupby("BDSPPatientID")["t_h"].diff() * 3600).round() / 3600          # whole seconds
    gap = gap.dropna()
    gap = gap[(gap > LEV_GAP_MIN_H) & (gap <= LEV_GAP_MAX_H)]
    q8, q12 = pct(gap.between(6, 10)), pct(gap.between(10, 14))
    per_day = w.groupby("BDSPPatientID").size() / 5.0
    return {"patients": int(w["BDSPPatientID"].nunique()), "doses": int(len(w)), "dose_mg": q3(w["dose_mg"]),
            "interval_h": q3(gap), "interval_share_pct": {"q8h (6-10 h)": q8, "q12h (10-14 h)": q12, "other": 100 - q8 - q12},
            "doses_per_patient_day": q3(per_day)}


def propofol_mg(p, pc):
    """Stage 20 propofol_mg(): cumulative propofol (mg) in hours [0, 120) from FirstEEG."""
    e = pc.copy()
    e["hours_from_first_eeg"] = med_hours(p, e)
    h = e["hours_from_first_eeg"]
    bol = e[(~e["infusion"].fillna(False).astype(bool)) & e["given"].astype(bool) & (h >= 0) & (h < 120)]
    bol_mg = bol.groupby("BDSPPatientID")["dose_mg"].sum()
    inf = e[e["infusion"].fillna(False).astype(bool) & h.notna()].sort_values(["BDSPPatientID", "hours_from_first_eeg"])
    rate = pd.to_numeric(inf["infusion_rate"], errors="coerce").fillna(0.0)
    keep = ~inf["action"].eq("Canceled Entry") & ~(rate > 150)
    inf, rate = inf[keep.to_numpy()], rate[keep.to_numpy()]
    rate = np.where(inf["infusion_rate_unit"].eq("mL/hr"), rate * 10.0, 0.0)
    rate = np.where(inf["action"].isin(["Stopped", "Paused", "Missed", "Held"]), 0.0, rate)
    inf = inf.assign(mg_h=rate)
    inf["t_next"] = inf.groupby("BDSPPatientID")["hours_from_first_eeg"].shift(-1)
    inf["t_next"] = inf["t_next"].fillna(inf["hours_from_first_eeg"] + 1.0)
    inf["t_next"] = np.minimum(inf["t_next"], inf["hours_from_first_eeg"] + 4.0)
    lo = inf["hours_from_first_eeg"].clip(0, 120)
    hi = inf["t_next"].clip(0, 120)
    inf["mg"] = inf["mg_h"] * (hi - lo).clip(lower=0)
    return inf.groupby("BDSPPatientID")["mg"].sum().add(bol_mg, fill_value=0.0)


# ------------------------------------------------------------------------------------------------ stage 20
def cells(d, ea_col="ea_peak", cut=CUT, prop_col="prop", label=""):
    he = (d[ea_col] >= cut).astype(int)
    t = pd.crosstab(he, d[prop_col]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    n00, n01, n10, n11 = (int(t.loc[0, 0]), int(t.loc[0, 1]), int(t.loc[1, 0]), int(t.loc[1, 1]))
    or_ = (n11 + .5) * (n00 + .5) / ((n10 + .5) * (n01 + .5))
    se = math.sqrt(sum(1 / (x + .5) for x in (n00, n01, n10, n11)))
    phi = (n11 * n00 - n10 * n01) / math.sqrt((n10 + n11) * (n00 + n01) * (n01 + n11) * (n00 + n10))
    return {"definition": label, "n": len(d), "low_ea_no_prop": n00, "low_ea_prop": n01, "high_ea_no_prop": n10,
            "high_ea_prop": n11, "smallest_cell_pct": 100 * min(n00, n01, n10, n11) / len(d),
            "odds_ratio": or_, "or_lo": math.exp(math.log(or_) - 1.96 * se), "or_hi": math.exp(math.log(or_) + 1.96 * se),
            "phi": phi}


def quadrants(d):
    out = {}
    m = d[d["site"] == "MGH"].copy()
    for name, x in [("all", d), ("MGH", m)]:
        x = x[x["ea_peak"].notna()]
        out[f"groups_{name}"] = {g: int((x["group"] == g).sum()) for g in GROUPS}
        out[f"quadrants_{name}"] = cells(x, label="primary: peak 24-h EA >= 10%, any propofol")
    m = m[m["ea_peak"].notna()].copy()
    m["prop_hi_mg"] = (m["propofol_mg_5d"] > m.loc[m.propofol_mg_5d > 0, "propofol_mg_5d"].median()).astype(int)
    sens = [cells(m, cut=x, label=f"peak 24-h EA >= {x:.0%}") for x in (0.01, 0.10, 0.25)]
    sens += [cells(m, ea_col="burden_iiic", label="overall 5-day EA burden >= 10%"),
             cells(m, ea_col="peak24_sz_pp", cut=0.10, label="seizure-only peak 24-h >= 10%"),
             cells(m, ea_col="peak24_sz_pp", cut=0.01, label="seizure-only peak 24-h >= 1%"),
             cells(m, prop_col="prop_hi_mg", label="propofol above median cumulative mg (exposed) vs rest"),
             cells(m[m.site == "MGH"], label="MGH only"),
             cells(m[m.rec_h_5d >= 24], label="EEG recorded >= 24 h in window")]
    out["quadrants_MGH_sensitivity"] = sens
    return out


# ------------------------------------------------------------------------------------------------ comparison
def decimals(v) -> int:
    s = str(v)
    return len(s.split(".")[1]) if "." in s else 0


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(x, list) and x and isinstance(x[0], dict):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{v.get('definition', v.get('target', i))}]")
    else:
        yield path, x


def get(d, path):
    cur = d
    for part in path.replace("[", ".[").split("."):
        if part.startswith("["):
            key = part[1:-1]
            cur = next(r for r in cur if r.get("definition", r.get("target")) == key)
        else:
            cur = cur[part]
    return cur


def same(exp, got):
    if isinstance(exp, list):
        if not isinstance(got, list) or len(exp) != len(got):
            return False
        dec = max(decimals(e) for e in exp)
        return all(round(float(g) + 1e-12, dec) == round(float(e), dec) for e, g in zip(exp, got))
    try:
        e = float(exp)
    except (TypeError, ValueError):
        return str(exp) == str(got)
    if isinstance(got, str):
        try:
            got = float(got)
        except ValueError:
            return False
    return round(float(got) + 1e-12, decimals(exp)) == round(e, decimals(exp))


def show(exp, got):
    if isinstance(exp, list):
        dec = max(decimals(e) for e in exp)
        return [round(float(g), dec + 2) for g in got]
    if isinstance(got, float):
        return round(got, decimals(exp) + 2)
    return got


def paper_macros(d):
    """The numbers the manuscript prints: the macros of cicadas2.rr.real.build(), same formats."""
    s10, s30, qm, sens = d["seizures"]["min10s"], d["seizures"]["min30s"], d["quadrants_MGH"], d["quadrants_MGH_sensitivity"]
    lev, dr = d["levetiracetam"], d["drugs_in_window_pct"]
    i = float
    return [("RealN", d["n"], "{:,}"), ("RealNMGH", d["site"]["MGH"], "{:,}"), ("RealNBI", d["site"]["BI"], "{:,}"),
            ("RealAgeMed", d["age"][1], "{:.0f}"), ("RealAgeLo", d["age"][0], "{:.0f}"), ("RealAgeHi", d["age"][2], "{:.0f}"),
            ("RealMalePct", d["male_pct"], "{:.0f}"),
            ("RealEEGhMed", d["eeg_h_5d"][1], "{:.0f}"), ("RealEEGhLo", d["eeg_h_5d"][0], "{:.0f}"), ("RealEEGhHi", d["eeg_h_5d"][2], "{:.0f}"),
            ("RealSzAnyPct", s10["patients_with_any_pct"], "{:.0f}"),
            ("RealSzDayMed", s10["onsets_per_recorded_day_among_sz_patients"][1], "{:.0f}"),
            ("RealSzDayLo", s10["onsets_per_recorded_day_among_sz_patients"][0], "{:.0f}"),
            ("RealSzDayHi", s10["onsets_per_recorded_day_among_sz_patients"][2], "{:.0f}"),
            ("RealBinsWithMed", s10["share_5min_bins_with_onset_among_sz_patients_pct"][1], "{:.0f}"),
            ("RealBinsWithLo", s10["share_5min_bins_with_onset_among_sz_patients_pct"][0], "{:.0f}"),
            ("RealBinsWithHi", s10["share_5min_bins_with_onset_among_sz_patients_pct"][2], "{:.0f}"),
            ("RealBinsMultiPct", s10["share_5min_bins_with_2plus_onsets_given_any_pct"], "{:.0f}"),
            ("RealPdays", s10["patient_days_12h_recorded"], "{:,}"),
            ("RealPdaysZeroPct", s10["patient_days_zero_onsets_pct"], "{:.0f}"),
            ("RealPdaysFewPct", s10["patient_days_1to4_pct"], "{:.0f}"),
            ("RealPdaysMidPct", s10["patient_days_5to24_pct"], "{:.0f}"),
            ("RealPdaysManyPct", s10["patient_days_25plus_pct"], "{:.0f}"),
            ("RealPdaysZeroPctThirty", s30["patient_days_zero_onsets_pct"], "{:.0f}"),
            ("RealPdaysManyPctThirty", s30["patient_days_25plus_pct"], "{:.0f}"),
            ("RealEAPeakMed", d["ea"]["peak24_iiic"][1], "{:.0f}"), ("RealEAHighPct", d["ea"]["peak24_ge10_pct"], "{:.0f}"),
            ("RealASMPct", dr["asm_in_eeg5d"], "{:.0f}"), ("RealLevPct", dr["levetiracetam"], "{:.0f}"),
            ("RealPropPct", dr["propofol"], "{:.0f}"), ("RealBenzoPct", dr["rescue_benzo_bolus"], "{:.0f}"),
            ("RealLevDoseMed", lev["dose_mg"][1], "{:,.0f}"), ("RealLevDoseLo", lev["dose_mg"][0], "{:,.0f}"),
            ("RealLevIntMed", lev["interval_h"][1], "{:.0f}"),
            ("RealLevQtwelvePct", lev["interval_share_pct"]["q12h (10-14 h)"], "{:.0f}"),
            ("RealLevQeightPct", lev["interval_share_pct"]["q8h (6-10 h)"], "{:.0f}"),
            ("RealQuadLL", i(qm["low_ea_no_prop"]), "{:,.0f}"), ("RealQuadLP", i(qm["low_ea_prop"]), "{:,.0f}"),
            ("RealQuadHL", i(qm["high_ea_no_prop"]), "{:,.0f}"), ("RealQuadHP", i(qm["high_ea_prop"]), "{:,.0f}"),
            ("RealQuadOR", i(qm["odds_ratio"]), "{:.2f}"), ("RealQuadORLo", i(qm["or_lo"]), "{:.2f}"), ("RealQuadORHi", i(qm["or_hi"]), "{:.2f}"),
            ("RealQuadMinCellPct", min(i(r["smallest_cell_pct"]) for r in sens), "{:.1f}"),
            ("RealQuadORMin", min(i(r["odds_ratio"]) for r in sens), "{:.2f}"), ("RealQuadORMax", max(i(r["odds_ratio"]) for r in sens), "{:.2f}"),
            ("RealBIPropN", i(d["quadrants_all"]["low_ea_prop"]) + i(d["quadrants_all"]["high_ea_prop"])
             - i(qm["low_ea_prop"]) - i(qm["high_ea_prop"]), "{:,.0f}")]


TABLE_ROWS = ["peak 24-h EA >= 1%", "peak 24-h EA >= 10%", "peak 24-h EA >= 25%", "overall 5-day EA burden >= 10%",
              "seizure-only peak 24-h >= 10%", "propofol above median cumulative mg (exposed) vs rest",
              "EEG recorded >= 24 h in window"]


def table_row(r):
    return (f"{int(float(r['low_ea_no_prop'])):,} & {int(float(r['low_ea_prop'])):,} & {int(float(r['high_ea_no_prop'])):,} & "
            f"{int(float(r['high_ea_prop'])):,} & {float(r['odds_ratio']):.2f} ({float(r['or_lo']):.2f}, {float(r['or_hi']):.2f})")


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=here)
    ap.add_argument("--expected", type=Path, default=None)
    args = ap.parse_args()
    exp_path = args.expected or args.root / "reference" / "rr_real_cohort.json"
    exp = json.loads(exp_path.read_text())

    p, em, med, pc = load(args.root)
    em, code, t_s, w0_s = minute_arrays(p, em)
    checks = []                                             # internal consistency of the package

    # --- stage 19 from the minute table, checked against the stored summaries
    b = ea_burden(p, em, code, t_s, w0_s)
    for c in ["rec_h_5d", "burden_iiic", "peak24_iiic", "burden_sz_pp", "peak24_sz_pp", "burden_lpd", "peak24_lpd",
              "burden_gpd", "peak24_gpd"]:
        a, z = b[c].to_numpy(float), p[c].to_numpy(float)
        ok = np.array_equal(np.isnan(a), np.isnan(z)) and np.nanmax(np.abs(a - z)) < 1e-9
        checks.append((f"eeg_minute recompute == patients.{c}", ok, f"max |diff| {np.nanmax(np.abs(a - z)):.2e}"))
    d = p.drop(columns=[c for c in b.columns if c != "BDSPPatientID"]).merge(b, on="BDSPPatientID")

    # --- drugs, checked against the stored flags
    fl = drug_exposure(p, med)
    for c in fl.columns:
        if c in p.columns:
            agree = int((fl[c].to_numpy() == p.set_index("BDSPPatientID").loc[fl.index, c].to_numpy()).sum())
            checks.append((f"medications recompute == patients.{c}", agree == len(fl), f"{agree:,}/{len(fl):,} patients agree"))
    mg = propofol_mg(p, pc).reindex(p["BDSPPatientID"]).fillna(0.0).to_numpy()
    dmg = np.abs(mg - p["propofol_mg_5d"].to_numpy())
    checks.append(("propofol_charting recompute == patients.propofol_mg_5d", dmg.max() < 1e-6, f"max |diff| {dmg.max():.2e} mg"))

    d = d.drop(columns=[c for c in fl.columns if c in d.columns]).merge(fl, left_on="BDSPPatientID", right_index=True)
    d["propofol_mg_5d"] = mg
    d["ea_peak"] = d["peak24_iiic"].fillna(d["burden_iiic"])
    d["high_ea"] = (d["ea_peak"] >= CUT).astype(int)
    d["prop"] = d["eeg5d_propofol"].astype(int)
    d["group"] = [GROUPS[2 * e + q] for e, q in zip(d["high_ea"], d["prop"])]

    got = {
        "n": len(d),
        "site": {"MGH": int((d.site == "MGH").sum()), "BI": int((d.site == "BI").sum())},
        "age": q3(d["age"]),
        "male_pct": pct(d["sex"] == "Male"),
        "sofa": q3(d["SOFA_score"]),
        "primary_dx_top": {k: 100 * v for k, v in d["primary_dx"].value_counts(normalize=True).head(6).items()},
        "eeg_h_5d": q3(d["rec_h_5d"]),
        "seizures": seizure_sparsity(p, em, code, t_s, w0_s),
        "ea": {"peak24_iiic": q3(d["peak24_iiic"], 100), "overall_iiic": q3(d["burden_iiic"], 100),
               "peak24_ge10_pct": pct(d["ea_peak"] >= CUT)},
        "drugs_in_window_pct": {"asm_in_eeg5d": pct(d.asm_in_eeg5d == 1), "levetiracetam": pct(d.eeg5d_levetiracetam == 1),
                                "lacosamide": pct(d.eeg5d_lacosamide == 1), "valproate": pct(d.eeg5d_valproate == 1),
                                "phenytoin_fosphenytoin": pct(d.eeg5d_phenytoin_fosphenytoin == 1),
                                "rescue_benzo_bolus": pct(d.eeg5d_rescue_benzo_bolus == 1), "propofol": pct(d.eeg5d_propofol == 1),
                                "sedative_hypnotic_infusion": pct(d.eeg5d_sedative_hypnotic_infusion == 1)},
        "levetiracetam": levetiracetam(p, med),
    }
    got.update(quadrants(d))

    print(f"package: {args.root}\nexpected: {exp_path}\n")
    print("== Internal consistency of the package")
    for name, ok, det in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({det})")

    print("\n== rr_real_cohort.json, every quantity (compared at the precision stored in the JSON)")
    n_pass = n_fail = n_skip = 0
    for path, e in leaves(exp):
        if path == "source":
            continue
        if path.startswith("quadrants_MGH_separability"):
            n_skip += 1
            continue
        try:
            g = get(got, path)
        except (KeyError, StopIteration):
            print(f"FAIL  {path}: expected {e}, not computed")
            n_fail += 1
            continue
        ok = same(e, g)
        n_pass += ok
        n_fail += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {path}: expected {e} | computed {show(e, g)}")
    if n_skip:
        print(f"SKIP  quadrants_MGH_separability ({n_skip} values): cross-validated AUCs from baseline covariates "
              "(CCI, days from admission to EEG, admitting unit) that are not in this package; Section 6 does not print them")

    print("\n== As printed in the manuscript (Section 6 macros and Table 'real cohort')")
    m_pass = m_fail = 0
    for (name, ev_, fmt), (_, gv, _) in zip(paper_macros(exp), paper_macros(got)):
        a, z = fmt.format(ev_), fmt.format(gv)
        ok = a == z
        m_pass += ok
        m_fail += not ok
        print(f"{'PASS' if ok else 'FAIL'}  \\{name}: printed {a} | computed {z}")
    for lab in TABLE_ROWS:
        er = next(r for r in exp["quadrants_MGH_sensitivity"] if r["definition"] == lab)
        gr = next(r for r in got["quadrants_MGH_sensitivity"] if r["definition"] == lab)
        a, z = table_row(er), table_row(gr)
        ok = a == z
        m_pass += ok
        m_fail += not ok
        print(f"{'PASS' if ok else 'FAIL'}  Table row '{lab}': printed {a} | computed {z}")

    c_fail = sum(not ok for _, ok, _ in checks)
    print(f"\nSUMMARY  consistency {len(checks) - c_fail}/{len(checks)} | JSON quantities {n_pass}/{n_pass + n_fail} PASS"
          f" ({n_skip} skipped) | printed numbers {m_pass}/{m_pass + m_fail} PASS")
    return 0 if (n_fail == 0 and m_fail == 0 and c_fail == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
