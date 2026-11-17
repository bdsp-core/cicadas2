"""Real-cohort descriptive numbers (editor point E1) -> LaTeX macros and a table.

Input: results/rr_real_cohort.json, the aggregate summary written by the cicadas-real-data repo
(branch real-cohort: pipeline/20_quadrants.py --site MGH and pipeline/26_real_cohort_descriptive.py, 2026-10-27).
Only aggregate counts, shares and quartiles cross into this repo; patient-level data stay on the analysis machine.
No outcome variable was read to make them.

  python -m cicadas2.rr.real      # writes results/rr_real_cohort_macros.json and rr_tables/table_real_cohort.tex
"""
from __future__ import annotations

import json
from pathlib import Path

from . import ledger, macros

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results"
TAB = ROOT / "CICADAS_II_Paper_Draft" / "rr_tables"


def build():
    d = json.loads((RES / "rr_real_cohort.json").read_text())
    s10 = d["seizures"]["min10s"]
    s30 = d["seizures"]["min30s"]
    qm = d["quadrants_MGH"]
    sens = d["quadrants_MGH_sensitivity"]
    lev = d["levetiracetam"]
    dr = d["drugs_in_window_pct"]
    i = lambda x: float(x)
    mac = [("RealN", d["n"], "{:,}"), ("RealNMGH", d["site"]["MGH"], "{:,}"), ("RealNBI", d["site"]["BI"], "{:,}"),
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
           ("RealBIPropN", float(d["quadrants_all"]["low_ea_prop"]) + float(d["quadrants_all"]["high_ea_prop"]) - float(qm["low_ea_prop"]) - float(qm["high_ea_prop"]), "{:,.0f}")]
    macros.save(mac, RES / "rr_real_cohort_macros.json")
    ledger.record(RES, "real_cohort", ["E1", "E2", "R1 M1b"], None, d["n"], ["rr_real_cohort.json", "rr_real_cohort_macros.json"],
                  {"groups_MGH": [qm["low_ea_no_prop"], qm["low_ea_prop"], qm["high_ea_no_prop"], qm["high_ea_prop"]],
                   "or": qm["odds_ratio"]}, d.get("source", ""))

    rows = [r"\begin{table}[ht]\centering\small",
            r"\caption{Four groups defined by epileptiform activity (EA) and propofol in the first 5 days of EEG, "
            r"MGH patients of the real cohort ($n = \RealNMGH$). High EA: peak 24-h EA burden at or above the cutoff; "
            r"EA is seizure, lateralized or generalized periodic discharges (LPD, GPD) as labelled by an automated "
            r"detector. Propofol: any administration in the window. OR: odds ratio between high EA and propofol, "
            r"with 95\% confidence interval; a value near 1 means the two axes vary independently. No outcome was used. "
            r"BIDMC patients are excluded because propofol is rarely charted in that site's medication record.}",
            r"\label{tab:realquad}", r"\begin{tabular}{@{}p{4.3cm}rrrrl@{}}", r"\toprule",
            r" & \multicolumn{2}{c}{Low EA} & \multicolumn{2}{c}{High EA} & \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"Definition & No propofol & Propofol & No propofol & Propofol & OR (95\% CI) \\",
            r"\midrule"]
    lab = {"peak 24-h EA >= 1%": r"Peak 24-h EA $\geq$ 1\%", "peak 24-h EA >= 10%": r"Peak 24-h EA $\geq$ 10\% (primary)",
           "peak 24-h EA >= 25%": r"Peak 24-h EA $\geq$ 25\%", "overall 5-day EA burden >= 10%": r"Overall 5-day EA $\geq$ 10\%",
           "seizure-only peak 24-h >= 10%": r"Seizures only, peak $\geq$ 10\%",
           "propofol above median cumulative mg (exposed) vs rest": r"Propofol above median dose vs rest",
           "EEG recorded >= 24 h in window": r"Patients with $\geq$ 24 h of EEG"}
    for r in sens:
        if r["definition"] not in lab:
            continue
        rows.append(f"{lab[r['definition']]} & {int(r['low_ea_no_prop']):,} & {int(r['low_ea_prop']):,} & {int(r['high_ea_no_prop']):,} & "
                    f"{int(r['high_ea_prop']):,} & {float(r['odds_ratio']):.2f} ({float(r['or_lo']):.2f}, {float(r['or_hi']):.2f}) \\\\")
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "table_real_cohort.tex").write_text("% AUTO-GENERATED by cicadas2.rr.real from results/rr_real_cohort.json\n" + "\n".join(rows) + "\n")
    return len(mac)


if __name__ == "__main__":
    print(build(), "macros")
