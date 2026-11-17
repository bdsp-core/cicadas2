"""LaTeX tables for the revised paper, generated from results/rr_*.json (python -m cicadas2.rr.tables).

Writes CICADAS_II_Paper_Draft/rr_tables/*.tex. Tables are rebuilt from the same
files as the macros, so tables, text and figures share one source (G1-G4).
"""
from __future__ import annotations

import json
from pathlib import Path

from .theta import QUADS, REGISTRY, write_table

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results"
OUT = ROOT / "CICADAS_II_Paper_Draft" / "rr_tables"
LAB = {q: REGISTRY[q]["label"] for q in QUADS}


def _j(name):
    return json.loads((RES / f"rr_{name}.json").read_text())


def _two(a, b):
    """Two stacked lines in one table cell."""
    return r"\begin{tabular}[t]{@{}l@{}}" + a + r"\\{}" + b + r"\end{tabular}"


def _n(x, k=1):
    """Number with a typographic minus."""
    t = f"{x:.{k}f}"
    if t.startswith("-") and float(t) == 0:
        t = t[1:]
    return t.replace("-", "$-$")


def _ci(d, k=1):
    return f"{_n(d['est'], k)} ({_n(d['lo'], k)}, {_n(d['hi'], k)})"


def table7():
    import numpy as np
    from . import stats
    s = _j("strategies")
    z = np.load(RES / "rr_harm.npz", allow_pickle=True)
    hl = list(z["labels"])
    rows = [r"\begin{table}[ht]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
            r"\caption{Seven-day survival (\%) by strategy in the four harm phenotypes, with 95\% confidence intervals "
            r"($n = 1{,}000$; survival computed exactly from the daily hazard over 7 days). Bold: highest mean in the row. "
            r"Fixed: 500~mg every 8~h. PRN (\emph{pro re nata}): a dose if at least one seizure was counted in the preceding 8~h. "
            r"Adaptive: tiered controller driven by the extended Kalman filter, $\omega = 2$. Harm-balanced: the controller of "
            r"Section~\ref{sec:harm}. Same numbers as Figure~\ref{fig:quadrant}. Sz: seizure harm; Drug: drug harm.}",
            r"\label{tab:quadrant}", r"\begin{tabular}{@{}llllll@{}}", r"\toprule",
            r"Phenotype & None & Fixed & PRN & Adaptive & Harm-balanced \\", r"\midrule"]
    for q in QUADS:
        t = dict(s["regimes"][q]["table7"])
        vals = [t["none"], t["fixed"], t["prn8_k1"], t["adapt_2"], stats.mean_ci(z[f"S_{q}"][:, hl.index("harm_est")])]
        top = max(v["est"] for v in vals)
        cells = [(r"\textbf{" + _ci(v) + "}") if abs(v["est"] - top) < 0.05 else _ci(v) for v in vals]
        rows.append(f"{LAB[q]} & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


def table_tuned():
    s = _j("strategies")
    fams = [("prn8", ("PRN,", "8-h count")), ("prn24", ("PRN,", "24-h count")), ("prnekf", ("PRN,", "EKF rate")),
            ("fixedfam", ("Fixed", "dose")), ("adaptma", ("Tiered, 24-h", "average"))]

    def setting(f, ch):
        if "_" not in ch:
            return "500 mg"
        v = ch.split("_", 1)[1]
        if f in ("prn8", "prn24"):
            k = int(v.lstrip("k"))
            return rf"$\geq${k} seizure" + ("s" if k > 1 else "")
        if f == "prnekf":
            return f"{v} sz/day"
        if f == "fixedfam":
            return f"{v} mg"
        return rf"$\omega = {v}$"

    rows = [r"\begin{table}[ht]\centering\footnotesize\setlength{\tabcolsep}{3pt}",
            r"\caption{The tiered adaptive controller against comparators whose threshold or dose was tuned on held-out "
            r"patients (all settings, including the controller's own threshold $\omega$, chosen on the even-numbered "
            r"patients and evaluated on the odd-numbered ones, $n = 500$ each). Entries: survival of the tiered controller "
            r"minus the comparator, in percentage points, with 95\% paired-bootstrap confidence interval; the tuned setting "
            r"is given below each entry. Positive values favor the tiered controller. PRN (\emph{pro re nata}): a dose when "
            r"at least the given number of seizures was counted in the preceding 8 or 24~h, or when the extended Kalman "
            r"filter (EKF) rate exceeds the threshold. Tiered, 24-h average: the same dose tiers driven by a 24-hour "
            r"moving average of counts instead of the EKF. Sz: seizure harm; Drug: drug harm. A tuned fixed dose beats the tiered controller in High Sz / Low Drug and Low Sz / Low Drug; when the drug is harmful, rules that rarely dose (PRN, no treatment) match or beat it.}",
            r"\label{tab:tuned}", r"\begin{tabular}{@{}l" + "l" * len(fams) + "@{}}", r"\toprule",
            " & ".join([_two("Phenotype", r"(tiered $\omega$)")] + [_two(a_, b_) for _, (a_, b_) in fams]) + r" \\", r"\midrule"]
    for q in QUADS:
        R = s["regimes"][q]
        cells = []
        for f, _ in fams:
            c = R["tuned_contrasts"][f"adapt-{f}"]
            cells.append(_two(_ci(c), setting(f, R["tuned_test"][f]["choice"])))
        w = R["tuned_test"]["adapt"]["choice"].split("_", 1)[1]
        a_, b_ = LAB[q].split(" / ")
        rows.append(_two(a_ + " / " + b_, rf"($\omega = {w}$)") + " & " + " & ".join(cells) + r" \\[2pt]")
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


def table_smd():
    o = _j("overlap")
    rows = [r"\begin{table}[ht]\centering\small",
            r"\caption{Covariate balance between treated and untreated patients. SMD: standardized mean difference "
            r"(difference in means divided by the pooled standard deviation); $|\text{SMD}| > 0.1$ is the usual flag for imbalance. "
            r"Observational: confounded assignment (Section~\ref{sec:overlap}); randomized: coin-flip assignment of the same patients. "
            r"SOFA: Sequential Organ Failure Assessment score. $n = 1{,}000$ (874 treated, 126 untreated). The observational cohort is strongly imbalanced on first-day seizure activity; randomizing the same patients removes the imbalance.}",
            r"\label{tab:smd}", r"\begin{tabular}{lrr}", r"\toprule", r"First-day feature & Observational SMD & Randomized SMD \\", r"\midrule"]
    for f, a, b in zip(o["features"], o["smd_observational"], o["smd_rct"]):
        rows.append(f"{f} & ${a:.2f}$ & ${b:.2f}$ \\\\")
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


def table_robustness():
    o = _j("obs_errors")["gformula"]
    sh = _j("shapes")["gformula"]
    dg = _j("dgp")["variants"]
    lab_o = {"baseline": "Baseline", "sens075": "25\\% of seizures missed", "fp3": "3 false detections per day",
             "sens075_fp3": "Both", "jitter1h": "Dose times shifted, SD 1~h", "jitter2h": "Dose times shifted, SD 2~h",
             "drop10": "10\\% of doses missing from record", "drop25": "25\\% of doses missing from record"}
    lab_s = {"squared": "Squared burden", "threshold": "Threshold burden", "saturating": "Saturating burden",
             "piecewise": "Piecewise-linear burden", "time_linear": "Linear time trend"}
    lab_d = {"two_compartment": "Two-compartment PK", "tolerance_010": "Tolerance, $C_{50}$ +10\\%/day",
             "tolerance_025": "Tolerance, $C_{50}$ +25\\%/day"}
    rows = [r"\begin{table}[ht]\centering\small",
            r"\caption{G-formula error under misspecification and data errors. True ATE: oracle effect of adaptive dosing versus no "
            r"treatment (percentage points); bias: g-formula with fitted mortality model minus truth, with 95\% bootstrap interval "
            r"(mortality model refitted on each of 200 resamples); relative bias: bias as a percentage of the true ATE. "
            r"EKF: extended Kalman filter; SDE: stochastic differential equation; PK: pharmacokinetics; $C_{50}$: half-maximal concentration. $n = 1{,}000$ in every scenario. The bias interval excludes zero only for false detections, missing dose records, saturating harm and strong drug tolerance; unmeasured confounding is analysed separately (Figure~\ref{fig:hiddenconf}).}",
            r"\label{tab:robust}", r"\begin{tabular}{lrrr}", r"\toprule", r"Scenario & True ATE & Bias (95\% CI) & Relative bias \\", r"\midrule"]
    def row(lab, v):
        return (f"{lab} & {_n(v['oracle_ate'])} & {_n(v['bias'])} ({_n(v['ci']['bias'][0])}, {_n(v['ci']['bias'][1])}) & "
                f"{_n(100 * v['bias'] / v['oracle_ate'], 0)}\\% \\\\")
    for k, l in lab_o.items():
        rows.append(row(l, o[k]))
    rows.append(r"\midrule")
    for k, l in lab_s.items():
        rows.append(row(l, sh[k]))
    rows.append(r"\midrule")
    for k, l in lab_d.items():
        rows.append(row(l, dg[k]["gformula"]))
    if (RES / "rr_misspec.json").exists():
        ly = _j("misspec")["layers"]
        rows.append(r"\midrule")
        for k, l in [("ekf", "Miscalibrated EKF in the deployed strategy"), ("traj", "CICADAS SDE natural history"),
                     ("joint", "Both")]:
            rows.append(row(l, ly[k]))
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


# ------------------------------------------------------------------------------------------
# Supplementary tables (R2 6): numbers behind figures that moved, and duplicates of figures
# ------------------------------------------------------------------------------------------
def _tab(caption, label, spec, header, body, size=r"\small"):
    return "\n".join([r"\begin{table}[ht]\centering" + size, r"\caption{" + caption + "}", r"\label{" + label + "}",
                      r"\begin{tabular}{" + spec + "}", r"\toprule", header, r"\midrule", *body, r"\bottomrule",
                      r"\end{tabular}", r"\end{table}"])


def table_personalized():
    p = _j("personalization")["regimes"]
    body = []
    for q in QUADS:
        r = p[q]
        mt, me = r["median_omega_true"], r["median_omega_est"]
        fm = lambda w: r"$\infty$" if str(w) in ("inf", "Infinity") or (isinstance(w, float) and w > 1e6) else f"{float(w):g}"
        body.append(f"{LAB[q]} & {fm(mt)} & {fm(me)} & {_ci(r['uniform'])} & {_ci(r['personal_true'])} & {_ci(r['personal_est'])} & "
                    f"{_n(100 * r['recovered_frac'], 0)}\\% \\\\")
    cap = (r"Personalized dose threshold. For every patient the tiered controller's threshold $\omega$ was chosen from "
           r"$\{0.125, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, \infty\}$ ($\infty$: no treatment) to maximize that patient's 7-day survival, "
           r"using either the true natural history or the one estimated from observed counts; survival is then evaluated on the "
           r"true trajectory. Entries: mean survival (\%) with 95\% confidence interval, $n = 1{,}000$. Recovered: gain of the "
           r"estimated over the uniform threshold as a share of the gain with the true natural history. Sz: seizure harm; Drug: drug harm. Thresholds chosen from estimated trajectories recover 96--99\% of the gain except when both harms are high (63\%).")
    return _tab(cap, "tab:personalized", "lllllll",
                r"Phenotype & \multicolumn{2}{c}{Median $\omega^*$} & Uniform $\omega = 2$ & Personal (true) & Personal (est.) & Recovered \\"
                "\n" + r" & True & Est. & & & & \\", body, r"\footnotesize")


def table_dynamic():
    s = _j("strategies")["regimes"]
    body = []
    for q in QUADS:
        t = s[q]["table10_test"]
        body.append(f"{LAB[q]} & {_ci(t['uniform'])} & {_ci(t['fixed_best'])} [{(t['fixed_best']['choice'].split('_')[1] if '_' in t['fixed_best']['choice'] else '$\\infty$')}] & {_ci(t['dynamic'])} \\\\")
    cap = (r"Three levels of personalization of the tiered controller, held-out half ($n = 500$): a uniform threshold "
           r"$\omega = 2$; the best fixed threshold for the phenotype, chosen on the other half (in brackets); and the dynamic "
           r"$\omega(t)$ rule of Section~\ref{sec:dynomega}. Entries: 7-day survival (\%) with 95\% confidence interval. Sz: seizure harm; Drug: drug harm. A threshold fixed per phenotype captures most of the gain over the uniform threshold, and the dynamic rule does no better (paired differences in Section~\ref{sec:personalized}).")
    return _tab(cap, "tab:dynamic_omega", "llll", r"Phenotype & Uniform & Fixed $\omega^*$ & Dynamic $\omega(t)$ \\", body)


def table_metrics():
    t8 = _j("strategies")["regimes"]["quad_HSLD"]["table8"]
    nm = [("none", "None"), ("fixed", "Fixed 500~mg"), ("prn8_k1", "PRN"), ("adapt_2", r"Adaptive ($\omega = 2$)")]
    body = [f"{l} & {t8[k]['Lcum']:.1f} & {t8[k]['Xcum']:,.0f} & {t8[k]['dose']:,.0f} \\\\".replace(",", "{,}") for k, l in nm]
    cap = (r"Mean 7-day cumulative seizure burden $L_{\mathrm{cum}}$ (seizure-days), cumulative drug exposure $X_{\mathrm{cum}}$ "
           r"(concentration-hours) and total dose (mg) by strategy, $n = 1{,}000$. These do not depend on the harm phenotype, "
           r"because the phenotype changes only the mortality model.")
    return _tab(cap, "tab:strategy_metrics", "lrrr", r"Strategy & $L_{\mathrm{cum}}$ & $X_{\mathrm{cum}}$ & Total dose (mg) \\", body)


def table_ekf_sens():
    e = _j("ekf_sensitivity")
    nm = {"q": r"$q$ (process noise)", "tau_x": r"$\tau_x$ (mean reversion, days)", "P_pop": r"$\mathcal{P}_{\mathrm{pop}}$ (prior peak, sz/day)",
          "floor": "Floor (prior minimum, sz/day)", "omega": r"$\omega$ (dose threshold, sz/day)"}
    body = []
    for k, l in nm.items():
        d = e["sweeps"][k]; v = [r_["value"] for r_ in d["rows"]]
        rng = r"$10^{-6}$--$10^{-2}$" if k == "q" else f"{min(v):g}--{max(v):g}"
        body.append(f"{l} & {rng} & {_n(d['min'])}--{_n(d['max'])} & {_n(d['spread'])} & {'Yes' if d['all_above_fixed'] else 'No'} \\\\")
    cap = (r"Sensitivity of the tiered controller to its filter settings (default mortality model, $n = 1{,}000$). One setting "
           r"is varied at a time, the others held at $q = 10^{-4}$, $\tau_x = 0.5$~days, $\mathcal{P}_{\mathrm{pop}} = 5$, floor 0.1, "
           r"$\omega = 2$. Spread: maximum minus minimum survival (percentage points). Above fixed: the 95\% confidence interval "
           f"of the difference from fixed 500~mg dosing ({_n(e['baselines']['fixed']['est'])}\\%) excludes zero at every value tested. "
           f"No treatment gives {_n(e['baselines']['none']['est'])}\\%.")
    return _tab(cap, "tab:ekf_sensitivity", "lllrl", r"Setting & Range & Survival (\%) & Spread & Above fixed \\", body)


def table_graded():
    g = _j("misspec")["graded"]
    body = [f"{100 * r['sde_fraction']:.0f}\\% & {_ci({'est': r['oracle_ate'], 'lo': r['ci']['oracle'][0], 'hi': r['ci']['oracle'][1]})} & "
            f"{_ci({'est': r['gform_ate'], 'lo': r['ci']['gform'][0], 'hi': r['ci']['gform'][1]})} & "
            f"{_ci({'est': r['bias'], 'lo': r['ci']['bias'][0], 'hi': r['ci']['bias'][1]})} \\\\" for r in g]
    cap = (r"Graded trajectory misspecification. A random, nested fraction of the 1{,}000 patients follows the CICADAS "
           r"stochastic differential equation (SDE) natural history; the estimator always assumes the rise--decay form. ATE: "
           r"average treatment effect of adaptive dosing versus none (percentage points); truth is the oracle effect; bias is "
           r"g-formula minus truth. 95\% bootstrap intervals, mortality model refitted on each of 200 resamples.")
    return _tab(cap, "tab:graded", "rlll", r"SDE fraction & True ATE & G-formula ATE & Bias \\", body)


def table_recovery():
    r = _j("recovery")
    p = r["params"]
    nm = [("A_p", r"$A_p$ (severity)"), ("r", r"$r$ (rise rate)"), ("kappa", r"$\kappa$ (decay rate)")]
    body = [f"{l} & {p[k]['untreated']['corr']:.2f} & {p[k]['untreated']['mdape']:.0f}\\% & {p[k]['treated']['corr']:.2f} & "
            f"{p[k]['treated']['mdape']:.0f}\\% \\\\" for k, l in nm]
    cap = (r"Recovery of the natural-history parameters by Poisson maximum likelihood ($n = 1{,}000$: odd-numbered patients "
           r"untreated, even-numbered patients on fixed 500~mg with the drug effect inverted using the patient's own "
           r"pharmacodynamics). $\rho$: Pearson correlation between true and estimated values; error: median absolute "
           r"percentage error. "
           f"{r['converged_share'] * 100:.0f}\\% of fits converged.")
    t1 = _tab(cap, "tab:param_recovery", "lcccc",
              r" & \multicolumn{2}{c}{Untreated ($n = 500$)} & \multicolumn{2}{c}{Treated ($n = 500$)} \\" "\n"
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n" r"Parameter & $\rho$ & Error & $\rho$ & Error \\", body)
    d = r["derived"]
    body = [rf"$L_{{\mathrm{{cum}}}}$ (7-day burden) & {d['Lcum']['overall']:.2f} & {d['Lcum']['untreated']:.2f} & {d['Lcum']['treated']:.2f} & {d['Lcum']['mdape']:.0f}\% \\",
            rf"Peak seizure rate & {d['peak']['overall']:.2f} & {d['peak']['untreated']:.2f} & {d['peak']['treated']:.2f} & {d['peak']['mdape']:.0f}\% \\",
            rf"Time of peak & {d['tpeak']['overall']:.2f} & {d['tpeak']['untreated']:.2f} & {d['tpeak']['treated']:.2f} & MAE {d['tpeak']['mae_days']:.2f}~d \\"]
    cap = (r"Recovery of the quantities derived from the estimated trajectory, same patients as Table~\ref{tab:param_recovery}. "
           r"$\rho$: Pearson correlation; error: median absolute percentage error; MAE: mean absolute error. Only "
           r"$L_{\mathrm{cum}}$ enters the mortality model.")
    t2 = _tab(cap, "tab:derived", "lcccc", r"Quantity & Overall $\rho$ & Untreated $\rho$ & Treated $\rho$ & Error \\", body)
    po = r["population"]
    body = [f"{l} & {_n(po[k]['true'][0], 2)} & {_n(po[k]['true'][1], 2)} & {_n(po[k]['est'][0], 2)} & {_n(po[k]['est'][1], 2)} \\\\" for k, l in nm]
    cap = (r"Recovery of the population distribution of each parameter: mean and standard deviation of the logarithm of the "
           r"true and the estimated values, $n = 1{,}000$.")
    t3 = _tab(cap, "tab:pop_recovery", "lcccc",
              r" & \multicolumn{2}{c}{True} & \multicolumn{2}{c}{Estimated} \\" "\n" r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n"
              r"Parameter & Log-mean & Log-SD & Log-mean & Log-SD \\", body)
    return t1, t2, t3


def table_demographics():
    d = _j("figdata")["sofa_categories"]
    body = [rf"\quad {k} & {v:.1f}\% \\" for k, v in d.items()]
    cap = (r"SOFA severity categories in the shared synthetic cohort ($n = 1{,}000$). SOFA: Sequential Organ Failure "
           r"Assessment score.")
    return _tab(cap, "tab:demographics", "lr", r"SOFA category & Patients \\", body)


def table_harm():
    h = _j("harm")
    f = h["filter"]["chosen"]
    rows = [r"\begin{table}[ht]\centering\small",
            r"\caption{Harm-balanced controller against comparators tuned on held-out patients (tuned on even-numbered, "
            r"scored on odd-numbered patients, $n = 500$ each). The controller is not tuned on outcomes: its doses follow "
            r"from the phenotype's hazard coefficients, the patient's SOFA score and age, and a forecast of the drug-free "
            r"seizure rate. Estimated: the forecast comes from an extended Kalman filter (EKF) whose measurement model "
            r"includes the drug; its prior and persistence were chosen by tracking error on the tuning half "
            f"(prior peak {f['P_pop']:g} seizures/day, persistence {f['tau_x']:g} days). Oracle: the true drug-free rate and "
            r"the patient's own dose-response parameters. Entries: survival difference in percentage points with 95\% "
            r"paired-bootstrap interval; the tuned setting is in brackets. Positive values favor the harm-balanced controller. Sz: seizure harm; Drug: drug harm. SOFA: Sequential Organ Failure Assessment. PRN (\emph{pro re nata}): a dose when seizures were counted in the preceding hours. The controller ties the best tuned fixed dose when the drug is not harmful (where the maximum dose is optimal) and beats every tuned comparator when it is.}",
            r"\label{tab:harm}", r"\begin{tabular}{@{}lllll@{}}", r"\toprule",
            " & ".join([_two("Phenotype", ""), _two("vs tuned", "fixed dose"), _two("vs tuned", "tiered adaptive"),
                        _two("vs no", "treatment"), _two("Oracle vs", "tuned fixed")]) + r" \\", r"\midrule"]
    for q in QUADS:
        R = h["regimes"][q]
        c = R["contrasts"]
        fx = R["tuned"]["fixed"].split("_")[1] + " mg"
        om = r"$\omega=" + R["tuned"]["adapt"].split("_")[1] + "$"
        a, b = LAB[q].split(" / ")
        rows.append(f"{_two(a + ' /', b)} & {_two(_ci(c['est-fixed']), '[' + fx + ']')} & {_two(_ci(c['est-adapt']), '[' + om + ']')} & "
                    f"{_ci(c['est-none'])} & {_ci(c['oracle-fixed'])} \\\\[2pt]")
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


def table_closed_loop():
    c = _j("closed_loop")
    names = [("none", "None"), ("fixed", "Fixed 500~mg"), ("fixed_750", "Fixed 750~mg"), ("prn8_k1", "PRN"),
             ("adapt_2", r"Tiered ($\omega = 2$)"), ("harm", "Harm-balanced")]
    rows = [r"\begin{table}[ht]\centering\small",
            r"\caption{Closed-loop validation: 7-day survival (\%) under each strategy simulated on the true disease "
            r"trajectory with each patient's own pharmacology (True) and on the trajectory the analyst estimates from "
            r"observed counts with population pharmacology and an independent random-number sequence (Est.). "
            r"$n = 1{,}000$; even-numbered patients were observed on fixed 500~mg dosing and odd-numbered patients "
            r"untreated before estimation. Bold: best strategy within each column. The ranking of all six strategies "
            r"is the same under true and estimated trajectories in every phenotype. PRN: dose when a seizure was "
            r"counted in the preceding 8~h. Sz: seizure harm; Drug: drug harm. Fixed 750~mg: the highest fixed dose. Harm-balanced: the controller of Section~\ref{sec:harm}. Entries are point estimates; their 95\% confidence intervals are shown in Figure~\ref{fig:closed_loop}.}",
            r"\label{tab:closedloop}", r"\begin{tabular}{@{}l" + "rr" * 4 + "@{}}", r"\toprule",
            " & " + " & ".join(r"\multicolumn{2}{c}{" + LAB[q].split(" / ")[0] + " /}" for q in QUADS) + r" \\",
            " & " + " & ".join(r"\multicolumn{2}{c}{" + LAB[q].split(" / ")[1] + "}" for q in QUADS) + r" \\",
            "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(4)),
            " & " + " & ".join(["True & Est."] * 4) + r" \\", r"\midrule"]
    best = {q: {src: max(c["regimes"][q][src], key=lambda k: c["regimes"][q][src][k]["est"]) for src in ("true", "est")} for q in QUADS}
    for k, nm in names:
        cells = []
        for q in QUADS:
            for src in ("true", "est"):
                v = c["regimes"][q][src][k]["est"]
                top = max(c["regimes"][q][src][kk]["est"] for kk, _ in names)
                cells.append((r"\textbf{%.1f}" % v) if abs(v - top) < 0.05 else "%.1f" % v)
        rows.append(nm + " & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hdr = "% AUTO-GENERATED by cicadas2.rr.tables from results/rr_*.json\n"
    for name, fn in [("table_quadrant", table7), ("table_tuned", table_tuned), ("table_smd", table_smd),
                     ("table_robustness", table_robustness), ("table_harm", table_harm), ("table_closed_loop", table_closed_loop),
                     ("table_personalized", table_personalized), ("table_dynamic", table_dynamic), ("table_metrics", table_metrics)]:
        (OUT / f"{name}.tex").write_text(hdr + fn() + "\n")
    optional = [("table_ekf_sensitivity", "ekf_sensitivity", table_ekf_sens), ("table_graded", "misspec", table_graded),
                ("table_demographics", "figdata", table_demographics)]
    for name, src, fn in optional:
        if (RES / f"rr_{src}.json").exists():
            (OUT / f"{name}.tex").write_text(hdr + fn() + "\n")
    if (RES / "rr_recovery.json").exists():
        for name, t in zip(("table_param_recovery", "table_derived", "table_pop_recovery"), table_recovery()):
            (OUT / f"{name}.tex").write_text(hdr + t + "\n")
    write_table(ROOT / "CICADAS_II_Paper_Draft" / "theta_table.tex")
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
