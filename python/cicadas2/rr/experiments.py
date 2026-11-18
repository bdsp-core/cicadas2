"""R&R experiments (tasks 9-18 of the roadmap). Run: python -m cicadas2.rr.run [names...]

Every experiment:
  * uses the seeded cohort (cohort.generate(N, SEED)) so all experiments share patients,
  * computes survival exactly with 7 daily hazards (G10) unless marked "legacy",
  * reports 95% CIs (paired bootstrap for contrasts),
  * writes results/rr_<name>.npz (arrays), results/rr_<name>.json (summary tables),
    results/rr_<name>_macros.json (LaTeX macros) and one line in results/rr_ledger.jsonl.
Nothing here overwrites a submitted results/*.mat file.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from ..params import default_params
from . import cohort as cohort_mod, engine, gformula, ledger, macros, stats, survival
from .theta import QUADS, REGISTRY, theta

SEED = 2026
N = 1000
TIME, DRUG, CTRL, MORT = default_params()
T, dt, Nt = TIME["T"], TIME["dt"], TIME["Nt"]
# One bootstrap for every g-formula analysis: the same 500 patient resamples (seed 21) in every experiment, so
# the baseline interval is identical wherever it is reported and scenario intervals are paired with it.
GF_B, GF_SEED = 500, 21
# Replicate outcome draws (2026-11-18): every experiment that redraws the simulated deaths uses the same DEATH_DRAWS
# sets, death uniforms from default_rng((SEED, DEATH_SEED0 + j)), with patients, trajectories and treatment
# assignment held fixed. The spread over these draws is the error from fitting the mortality model alone.
# (Previously 50, 40 and 20 draws with different seeds were used in three places, which gave SDs of 4.1, 3.7 and
# 2.9 pp and mean biases of -0.8, -2.1 and -1.5 pp for the same quantity.)
DEATH_DRAWS, DEATH_SEED0 = 200, 900


def _death_draws(P, th, n):
    """gform_ate and its bias over the DEATH_DRAWS replicate sets of observed deaths (same patients and assignment)."""
    ate, bias = [], []
    for j in range(DEATH_DRAWS):
        Pd = dict(P, u_death=np.random.default_rng((SEED, DEATH_SEED0 + j)).uniform(size=(n, 8)))
        r_ = gformula.evaluate(Pd, th)
        ate.append(float(r_["gform_ate"])); bias.append(float(r_["gform_ate"] - r_["oracle_ate"]))
    ate, bias = np.array(ate), np.array(bias)
    return {"mean": float(ate.mean()), "sd": float(ate.std(ddof=1)), "p2_5": float(np.percentile(ate, 2.5)),
            "p97_5": float(np.percentile(ate, 97.5)), "mean_bias": float(bias.mean()),
            "mc_se": float(ate.std(ddof=1) / np.sqrt(DEATH_DRAWS)), "draws": DEATH_DRAWS}

QLAB = {"quad_HSLD": "HSLD", "quad_HSHD": "HSHD", "quad_LSLD": "LSLD", "quad_LSHD": "LSHD",
        "default": "Default", "gformula": "Gform"}
REGIMES = QUADS + ["default", "gformula"]
OMEGAS = [0.125, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8]
_COHORT = {}


def get_cohort(n=N, seed=SEED, generator="rise_decay"):
    key = (n, seed, generator)
    if key not in _COHORT:
        _COHORT[key] = cohort_mod.generate(n, seed, generator)
    return _COHORT[key]


def _save(results_dir, name, arrays, summary, mac, ids, notes=""):
    rd = Path(results_dir)
    rd.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rd / f"rr_{name}.npz", **arrays)
    (rd / f"rr_{name}.json").write_text(json.dumps(summary, indent=1, default=float))
    macros.save(mac, rd / f"rr_{name}_macros.json")
    key = {k: v for k, v in summary.items() if k in ("headline", "checks")}
    ledger.record(rd, name, ids, SEED, summary.get("n", N),
                  [f"rr_{name}.npz", f"rr_{name}.json", f"rr_{name}_macros.json"], key, notes)


def _r(x, k=2):
    return float(np.round(x, k))


# ----------------------------------------------------------------------------------------
# E1  Strategy grid: PRN fairness (R1 M3), tiers (R1 m3), omega grid (G7), dynamic omega,
#     Table 7 / 8 / 10 analogues with CIs (R1 M7, R1 m1), legacy replication (G9, G10)
# ----------------------------------------------------------------------------------------
def strategy_specs(c):
    S = [("none", {"strategy": "none"}), ("fixed", {"strategy": "fixed"})]
    # fixed-dose family on the same menu as adaptive (max 750 mg), so the
    # adaptive-vs-fixed comparison is not a comparison of dose intensity
    S += [(f"fixed_{d}", {"strategy": "fixed", "dose_standard": d}) for d in (250, 375, 750)]
    S += [(f"prn8_k{k}", {"strategy": "prn", "prn_legacy": False, "prn_window_hours": 8, "prn_min_count": k}) for k in (1, 2, 3, 4, 5)]
    S += [(f"prnlegacy_{t}", {"strategy": "prn", "prn_legacy": True, "prn_threshold": t}) for t in (2, 3, 5, 7, 10, 15)]
    S += [(f"prn24_k{k}", {"strategy": "prn", "prn_legacy": False, "prn_window_hours": 24, "prn_min_count": k}) for k in (1, 2, 3, 4, 6, 8)]
    S += [(f"prnekf_{t}", {"strategy": "prn_ekf", "prn_threshold": t}) for t in (0.5, 1, 2, 3, 5)]
    S += [(f"adapt_{w}", {"strategy": "adaptive", "omega": w}) for w in OMEGAS]
    S += [(f"adaptma_{w}", {"strategy": "adaptive", "omega": w, "rate_source": "ma", "ma_hours": 24}) for w in OMEGAS]
    dose5 = np.array([0, CTRL["dose_low"], CTRL["dose_intermediate"], CTRL["dose_medium"], CTRL["dose_high"]], float)
    S += [("adapt2_dose075", {"strategy": "adaptive", "omega": 2, "tier_doses": dose5 * 0.75}),
          ("adapt2_dose125", {"strategy": "adaptive", "omega": 2, "tier_doses": dose5 * 1.25}),
          ("adapt2_3tier", {"strategy": "adaptive", "omega": 2, "tier_bounds": [0.75, 1.5], "tier_doses": [0, 375, 750]}),
          ("adapt2_7tier", {"strategy": "adaptive", "omega": 2, "tier_bounds": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                            "tier_doses": [0, 125, 250, 375, 500, 625, 750]})]
    for q in QUADS:
        S += [(f"dyn_{QLAB[q]}", {"strategy": "adaptive", "dynamic": {"theta": theta(q), "age": c["age"], "sofa": c["sofa"]}})]
    return S


def run_strategy_grid(c, seed=SEED, extra=None):
    U = cohort_mod.uniform_tape(c["n"], Nt, seed, stream=1)
    specs = strategy_specs(c)
    labels, Ld, Xd, dose, nd = [], [], [], [], []
    for lab, sp in specs:
        sp = dict(sp, **(extra or {}))
        o = engine.simulate(sp, c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt)
        labels.append(lab); Ld.append(o["Lday"]); Xd.append(o["Xday"]); dose.append(o["total_dose"]); nd.append(o["n_doses"])
    return labels, np.stack(Ld, 1), np.stack(Xd, 1), np.stack(dose, 1), np.stack(nd, 1)


def surv_matrix(c, Ld, Xd, key, n_days=7, shape="linear", time_power=2):
    return survival.exact(Ld, Xd, c["age"], c["sofa"], theta(key), shape, n_days=n_days, time_power=time_power)


def exp_strategies(results_dir):
    t0 = time.time()
    c = get_cohort()
    labels, Ld, Xd, dose, nd = run_strategy_grid(c)
    L = {l: i for i, l in enumerate(labels)}
    tune = np.arange(N) % 2 == 0
    test = ~tune
    out = {"n": N, "labels": labels, "regimes": {}, "checks": {}}
    mac = []
    arrays = dict(Lday=Ld, Xday=Xd, total_dose=dose, n_doses=nd, labels=np.array(labels))
    fams = {"prn8": [l for l in labels if l.startswith("prn8_")], "prn24": [l for l in labels if l.startswith("prn24_")],
            "prnekf": [l for l in labels if l.startswith("prnekf_")], "prnlegacy": [l for l in labels if l.startswith("prnlegacy_")],
            "adapt": [l for l in labels if l.startswith("adapt_")],
            "adaptma": [l for l in labels if l.startswith("adaptma_")],
            "fixedfam": ["fixed_250", "fixed_375", "fixed", "fixed_750"]}
    for rk in REGIMES:
        S = surv_matrix(c, Ld, Xd, rk)
        S8 = surv_matrix(c, Ld, Xd, rk, n_days=8)
        arrays[f"S_{rk}"] = S
        R = {"surv": {l: stats.mean_ci(S[:, L[l]]) for l in labels}}
        # Table 7 analogue (main four) with p(best)
        main = ["none", "fixed", "prn8_k1", "adapt_2"]
        pb = stats.p_best(S[:, [L[m] for m in main]], seed=3)
        R["table7"] = {m: dict(R["surv"][m], p_best=float(pb[j])) for j, m in enumerate(main)}
        R["table7_legacy"] = {m: 100 * S8[:, L[m]].mean() for m in ["none", "fixed", "prnlegacy_5", "adapt_2"]}
        R["contrasts"] = {f"{a}-{b}": stats.paired(S[:, L[a]], S[:, L[b]], seed=5)
                          for a, b in [("adapt_2", "fixed"), ("adapt_2", "prn8_k1"), ("adapt_2", "none"),
                                       ("fixed", "none"), ("prn8_k1", "prnlegacy_5"), ("prn8_k1", "none")]}
        # R1 M3: each family tuned on the even half, scored on the odd half
        tuned = {}
        for f, members in fams.items():
            best = max(members, key=lambda l: S[tune, L[l]].mean())
            tuned[f] = dict(choice=best, **stats.mean_ci(S[test, L[best]]))
        R["tuned_test"] = tuned
        R["tuned_contrasts"] = {f"adapt-{f}": stats.paired(S[test, L[tuned["adapt"]["choice"]]], S[test, L[tuned[f]["choice"]]], seed=7)
                                for f in ("prn8", "prn24", "prnekf", "prnlegacy", "fixedfam", "adaptma")}
        R["tuned_contrasts"]["adapt-none"] = stats.paired(S[test, L[tuned["adapt"]["choice"]]], S[test, L["none"]], seed=8)
        R["tuned_contrasts"]["adapt-fixed"] = stats.paired(S[test, L[tuned["adapt"]["choice"]]], S[test, L["fixed"]], seed=9)
        # Tiers (R1 minor 3)
        R["tiers"] = {t: dict(stats.mean_ci(S[:, L[t]]), vs_std=stats.paired(S[:, L[t]], S[:, L["adapt_2"]], seed=11))
                      for t in ["adapt2_dose075", "adapt2_dose125", "adapt2_3tier", "adapt2_7tier"]}
        R["tiers_rank_ok"] = {t: bool((S[:, L[t]].mean() > S[:, L["none"]].mean()) == (S[:, L["adapt_2"]].mean() > S[:, L["none"]].mean()))
                              for t in R["tiers"]}
        # Table 10 analogue (quadrants only): uniform 2, regime-best fixed omega (tuned), dynamic
        if rk in QUADS:
            cand = fams["adapt"] + ["none"]
            wbest = max(cand, key=lambda l: S[tune, L[l]].mean())
            dyn = f"dyn_{QLAB[rk]}"
            R["table10_test"] = {"uniform": stats.mean_ci(S[test, L["adapt_2"]]), "fixed_best": dict(choice=wbest, **stats.mean_ci(S[test, L[wbest]])),
                                 "dynamic": stats.mean_ci(S[test, L[dyn]]),
                                 "fixed-uniform": stats.paired(S[test, L[wbest]], S[test, L["adapt_2"]], seed=12),
                                 "dynamic-uniform": stats.paired(S[test, L[dyn]], S[test, L["adapt_2"]], seed=13),
                                 "dynamic-fixed": stats.paired(S[test, L[dyn]], S[test, L[wbest]], seed=14)}
        # Table 8 analogue
        R["table8"] = {m: dict(Lcum=float(Ld[:, L[m], 6].mean()), Xcum=float(Xd[:, L[m], 6].mean()), dose=float(dose[:, L[m]].mean()))
                       for m in ["none", "fixed", "prn8_k1", "prnlegacy_5", "adapt_2"]}
        out["regimes"][rk] = R
        q = QLAB[rk]
        cnm = {"prn8": "PRN", "prn24": "PRNday", "prnekf": "PRNekf", "prnlegacy": "PRNold", "fixedfam": "FixedTuned",
               "adaptma": "AdaptMA", "none": "None", "fixed": "FixedFive"}
        for key, c_ in R["tuned_contrasts"].items():
            tag = cnm[key.split("-")[1]]
            mac += [(f"Tuned{q}AdaptVs{tag}", c_["est"], "{:.1f}"), (f"Tuned{q}AdaptVs{tag}Lo", c_["lo"], "{:.1f}"),
                    (f"Tuned{q}AdaptVs{tag}Hi", c_["hi"], "{:.1f}")]
        mac += [(f"Tuned{q}AdaptOmega", float(tuned["adapt"]["choice"].split("_")[1]), "{:g}"),
                (f"Tuned{q}FixedDose", float(tuned["fixedfam"]["choice"].split("_")[1]) if "_" in tuned["fixedfam"]["choice"] else 500.0, "{:.0f}")]
        if rk in QUADS:
            t10 = R["table10_test"]
            for key, tag in [("fixed-uniform", "FixedVsUniform"), ("dynamic-uniform", "DynVsUniform"), ("dynamic-fixed", "DynVsFixed")]:
                mac += [(f"Tten{q}{tag}", t10[key]["est"], "{:.1f}"), (f"Tten{q}{tag}Lo", t10[key]["lo"], "{:.1f}"),
                        (f"Tten{q}{tag}Hi", t10[key]["hi"], "{:.1f}")]
        for m, nm in [("none", "None"), ("fixed", "Fixed"), ("prn8_k1", "PRN"), ("adapt_2", "Adapt")]:
            mac += [(f"Strat{q}{nm}", R["surv"][m]["est"], "{:.1f}"), (f"Strat{q}{nm}Lo", R["surv"][m]["lo"], "{:.1f}"),
                    (f"Strat{q}{nm}Hi", R["surv"][m]["hi"], "{:.1f}")]
    # Supplementary Table S4 quantities (phenotype-free) quoted in the Methods: Lcum in seizures (expected count),
    # Xcum in mg*day (x is the amount in mg, integrated over days), doses of fixed 500 mg q8h over the 7 days
    t8 = out["regimes"]["default"]["table8"]
    mac += [("MetFixedXcum", t8["fixed"]["Xcum"], "{:,.0f}"), ("MetFixedLcum", t8["fixed"]["Lcum"], "{:.1f}"),
            ("MetNoneLcum", t8["none"]["Lcum"], "{:.1f}"), ("MetFixedDose", t8["fixed"]["dose"], "{:,.0f}"),
            ("MetFixedNDoses", float(nd[:, L["fixed"]].mean()), "{:.0f}"),
            ("MetXnormRatio", MORT["X_norm"] / t8["fixed"]["Xcum"], "{:.2f}")]
    # PRN legacy plateau check (Fig 8): thresholds 2,3,5 are the same rule in counts
    Sd = arrays["S_default"]
    out["checks"]["legacy_prn_235_identical"] = bool(np.allclose(Sd[:, L["prnlegacy_2"]], Sd[:, L["prnlegacy_5"]]) and
                                                     np.allclose(Sd[:, L["prnlegacy_3"]], Sd[:, L["prnlegacy_5"]]))
    out["checks"]["legacy_prn_7_10_identical"] = bool(np.allclose(Sd[:, L["prnlegacy_7"]], Sd[:, L["prnlegacy_10"]]))
    out["runtime_s"] = time.time() - t0
    hl = {q: {k: _r(v["est"], 1) for k, v in out["regimes"][q]["table7"].items()} for q in QUADS}
    out["headline"] = hl
    _save(results_dir, "strategies", arrays, out, mac, ["R1 M3", "R1 M7", "R1 m1", "R1 m3", "G6", "G7", "G9", "G10", "G12"],
          "Strategy grid on the shared 1,000-patient cohort; tuned families scored on held-out half.")
    return out


# ----------------------------------------------------------------------------------------
# E2  G-formula baseline (G5, G11, R2 11, R1 M7) and legacy replication
# ----------------------------------------------------------------------------------------
_PREP = {}


def get_prep(pkpd="oracle", dgp=None, jitter=0.0, drop=0.0, n=N, seed=SEED):
    key = (pkpd, json.dumps(dgp, sort_keys=True, default=str), jitter, drop, n, seed)
    if key not in _PREP:
        _PREP[key] = gformula.prepare(get_cohort(n, seed), DRUG, CTRL, T, dt, seed, pkpd=pkpd, dgp=dgp,
                                      dose_jitter_h=jitter, dose_drop=drop)
    return _PREP[key]


def _summ(r, b=None):
    s = {k: float(r[k]) for k in ["oracle_ate", "gform_ate", "gform_known_ate", "naive_ate", "naive_deaths_ate", "rct_ate", "frac_treated"]}
    s["bias"] = s["gform_ate"] - s["oracle_ate"]
    s["bias_known"] = s["gform_known_ate"] - s["oracle_ate"]
    s["theta_hat"] = [float(v) for v in r["theta_hat"]]
    if b is not None:
        s["ci"] = {cname: [float(b["lo"][j]), float(b["hi"][j])] for j, cname in enumerate(b["cols"])}
        s["se"] = {cname: float(b["se"][j]) for j, cname in enumerate(b["cols"])}
    return s


def exp_gformula(results_dir, B=500):
    t0 = time.time()
    th = theta("gformula")
    out = {"n": N, "checks": {}}
    mac = []
    arrays = {}
    for pk in ("oracle", "pop"):
        P = get_prep(pk)
        r = gformula.evaluate(P, th)
        b = gformula.bootstrap(P, th, B=GF_B, seed=GF_SEED)
        s = _summ(r, b)
        # spread of a single coin-flip RCT over re-randomisations (G5)
        g = np.random.default_rng(31)
        rcts = np.array([100 * (r["S_t"][a].mean() - r["S_n"][~a].mean())
                         for a in (g.uniform(size=N) > 0.5 for _ in range(2000))])
        s["rct_sd_over_randomisations"] = float(rcts.std(ddof=1))
        # 95% interval of the one simulated RCT actually reported (Fig 3): patient bootstrap with the same GF_B
        # resamples as gformula.bootstrap, each patient keeping the arm it was randomized to
        rct = P["u_rct"] > 0.5
        gb = np.random.default_rng(GF_SEED)
        rb = []
        for _ in range(GF_B):
            ii = gb.integers(0, N, size=N)
            m = rct[ii]
            rb.append(100 * (r["S_t"][ii][m].mean() - r["S_n"][ii][~m].mean()))
        s["rct_ci"] = [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))]
        s["rct_95_range"] = [float(np.percentile(rcts, 2.5)), float(np.percentile(rcts, 97.5))]
        s["oracle_se_patient_sampling"] = float(100 * (r["S_t"] - r["S_n"]).std(ddof=1) / np.sqrt(N))
        s["theta_hat_se"] = [float(v) for v in np.sqrt(np.diag(r["theta_cov"]))]
        s["theta_true"] = [float(v) for v in th]
        # spread of the fitted g-formula over DEATH_DRAWS independent sets of observed deaths (same patients/assignment)
        s["fitted_over_death_draws"] = _death_draws(P, th, N)
        rl = gformula.evaluate(P, th, n_days=8)
        s["legacy_8day_knowntheta"] = {"oracle": float(rl["oracle_ate"]), "gform_known": float(rl["gform_known_ate"]),
                                       "naive": float(rl["naive_ate"]), "rct": float(rl["rct_ate"])}
        s["Lcum_corr"] = {"treated": float(np.corrcoef(P["Lcum_hat_tt"], P["Lcum_true_t"])[0, 1]),
                          "untreated": float(np.corrcoef(P["Lcum_hat_nn"], P["Lcum_true_n"])[0, 1]),
                          "Lbase_from_treated": float(np.corrcoef(P["Lbase_hat_t"], P["Lbase_true"])[0, 1]),
                          "Lbase_from_untreated": float(np.corrcoef(P["Lbase_hat_n"], P["Lbase_true"])[0, 1])}
        out[pk] = s
        arrays[f"{pk}_ite_oracle"] = r["S_t"] - r["S_n"]
        arrays[f"{pk}_ite_gform"] = r["G_t"] - r["G_n"]
        arrays[f"{pk}_ps_true"] = r["ps_true"]
        arrays[f"{pk}_A"] = r["A"]
        arrays[f"{pk}_rct_draws"] = rcts
    o = out["oracle"]
    mac += [("GFOracleATE", o["oracle_ate"], "{:.1f}"), ("GFOracleLo", o["ci"]["oracle"][0], "{:.1f}"), ("GFOracleHi", o["ci"]["oracle"][1], "{:.1f}"),
            ("GFFitATE", o["gform_ate"], "{:.1f}"), ("GFFitLo", o["ci"]["gform"][0], "{:.1f}"), ("GFFitHi", o["ci"]["gform"][1], "{:.1f}"),
            ("GFBias", o["bias"], "{:.1f}"), ("GFBiasLo", o["ci"]["bias"][0], "{:.1f}"), ("GFBiasHi", o["ci"]["bias"][1], "{:.1f}"),
            ("GFKnownATE", o["gform_known_ate"], "{:.1f}"), ("GFNaiveATE", o["naive_ate"], "{:.1f}"),
            ("GFNaiveLo", o["ci"]["naive"][0], "{:.1f}"), ("GFNaiveHi", o["ci"]["naive"][1], "{:.1f}"),
            ("GFRCTsd", o["rct_sd_over_randomisations"], "{:.1f}"), ("GFRCTATE", o["rct_ate"], "{:.1f}"),
            ("GFRCTLo", o["rct_ci"][0], "{:.1f}"), ("GFRCTHi", o["rct_ci"][1], "{:.1f}"), ("GFFracTreated", 100 * o["frac_treated"], "{:.0f}"),
            ("GFPopFitATE", out["pop"]["gform_ate"], "{:.1f}"), ("GFPopBias", out["pop"]["bias"], "{:.1f}"),
            ("GFDrawMean", o["fitted_over_death_draws"]["mean"], "{:.1f}"), ("GFDrawSD", o["fitted_over_death_draws"]["sd"], "{:.1f}"),
            ("GFDrawBias", o["fitted_over_death_draws"]["mean_bias"], "{:.1f}"), ("GFDrawSE", o["fitted_over_death_draws"]["mc_se"], "{:.1f}"),
            ("GFDrawN", DEATH_DRAWS, "{:d}")]
    out["headline"] = {"oracle": _r(o["oracle_ate"]), "gform_fitted": _r(o["gform_ate"]), "bias": _r(o["bias"]),
                       "bias_ci": [_r(v) for v in o["ci"]["bias"]], "naive": _r(o["naive_ate"]),
                       "known_theta": _r(o["gform_known_ate"]), "pop_pkpd_bias": _r(out["pop"]["bias"])}
    out["checks"]["legacy_vs_submitted"] = {"submitted": {"rct": 18.5, "naive": -2.1, "gform": 19.4},
                                            "replicated_8day_knowntheta": o["legacy_8day_knowntheta"]}
    out["runtime_s"] = time.time() - t0
    _save(results_dir, "gformula", arrays, out, mac, ["G5", "G10", "G11", "R2 11", "R1 M7", "R1 m1"],
          "Oracle ATE truth; theta fitted by pooled logistic; bootstrap refits theta.")
    return out


# ----------------------------------------------------------------------------------------
# E3  Hidden confounder sweep and E-value (R1 M6b)
# ----------------------------------------------------------------------------------------
def exp_confounding(results_dir, kappas=(0, 0.25, 0.5, 1, 2, 4), a_t=1.0, a_m=0.5, B=200):
    t0 = time.time()
    P = get_prep("oracle")
    th = theta("gformula")
    rows = []
    for k in kappas:
        r = gformula.evaluate(P, th, gamma_u=k * a_t, delta_u=k * a_m)
        b = gformula.bootstrap(P, th, B=GF_B, seed=GF_SEED, gamma_u=k * a_t, delta_u=k * a_m)
        s = _summ(r, b)
        s["kappa"] = k
        s["gamma_u"], s["delta_u"] = k * a_t, k * a_m
        s["covers_oracle"] = bool(b["lo"][1] <= r["oracle_ate"] <= b["hi"][1])
        s["bias_ci_covers_0"] = bool(b["lo"][2] <= 0 <= b["hi"][2])
        rows.append(s)
    grid = []
    for gu in (0, 0.5, 1, 2):
        for du in (0, 0.25, 0.5, 1):
            r = gformula.evaluate(P, th, gamma_u=gu, delta_u=du)
            grid.append(dict(gamma_u=gu, delta_u=du, oracle=float(r["oracle_ate"]), gform=float(r["gform_ate"]),
                             bias=float(r["gform_ate"] - r["oracle_ate"]), naive=float(r["naive_ate"])))
    # E-value for the baseline g-formula estimate (death-risk ratio, treated vs untreated)
    r0 = gformula.evaluate(P, th)
    risk_t, risk_n = 1 - r0["G_t"].mean(), 1 - r0["G_n"].mean()
    b0 = gformula.bootstrap(P, th, B=GF_B, seed=GF_SEED)
    # CI limit closest to the null on the risk-ratio scale, from bootstrap of G means
    g = np.random.default_rng(42)
    rrs = []
    for _ in range(B):
        idx = g.integers(0, N, size=N)
        rb = gformula.evaluate(P, th, idx=idx)
        rrs.append((1 - rb["G_t"].mean()) / (1 - rb["G_n"].mean()))
    rr = risk_t / risk_n
    rr_hi = float(np.percentile(rrs, 97.5))
    ev = {"rr_death": float(rr), "rr_ci_upper": rr_hi, "evalue_point": float(stats.evalue_rr(rr)),
          "evalue_ci": float(stats.evalue_rr(rr_hi)) if rr_hi < 1 else 1.0}
    first_fail = next((s["kappa"] for s in rows if not s["covers_oracle"]), None)
    sign_flip = next((s["kappa"] for s in rows if s["gform_ate"] <= 0), None)
    out = {"n": N, "a_t": a_t, "a_m": a_m, "rows": rows, "grid": grid, "evalue": ev,
           "headline": {"kappa_first_ci_miss": first_fail, "kappa_sign_flip": sign_flip,
                        "bias_by_kappa": {str(s["kappa"]): _r(s["bias"]) for s in rows}, "evalue": _r(ev["evalue_point"]),
                        "evalue_ci": _r(ev["evalue_ci"])},
           "checks": {"kappa0_matches_baseline": _r(rows[0]["gform_ate"])}, "runtime_s": time.time() - t0}
    mac = [("ConfEvalue", ev["evalue_point"], "{:.2f}"), ("ConfEvalueCI", ev["evalue_ci"], "{:.2f}"),
           ("ConfRRDeath", ev["rr_death"], "{:.2f}")]
    names = {0: "Zero", 0.25: "Quarter", 0.5: "Half", 1: "One", 2: "Two", 4: "Four"}
    for s in rows:
        mac += [(f"ConfBias{names[s['kappa']]}", s["bias"], "{:.1f}"), (f"ConfOracle{names[s['kappa']]}", s["oracle_ate"], "{:.1f}"),
                (f"ConfGform{names[s['kappa']]}", s["gform_ate"], "{:.1f}")]
    _save(results_dir, "confounding", {"kappas": np.array(kappas)}, out, mac, ["R1 M6b"],
          f"Hidden U ~ N(0,1): +{a_t}*kappa on treatment logit, +{a_m}*kappa on daily death log-odds; never seen by estimation.")
    return out


# ----------------------------------------------------------------------------------------
# E4  Balance and overlap (R1 M6c)
# ----------------------------------------------------------------------------------------
def exp_overlap(results_dir):
    P = get_prep("oracle")
    th = theta("gformula")
    r = gformula.evaluate(P, th)
    A = r["A"]
    F = P["F"]
    smd_obs = gformula.smd(F, A)
    smd_rct = gformula.smd(F, P["u_rct"] > 0.5)
    ps_hat = gformula.propensity_fit(F, A.astype(float))
    trim = (ps_hat >= 0.05) & (ps_hat <= 0.95)
    rt = gformula.evaluate(P, th, idx=np.nonzero(trim)[0])
    hi_region = ps_hat > 0.95
    out = {"n": N, "features": gformula.FEATURE_NAMES, "smd_observational": smd_obs.tolist(), "smd_rct": smd_rct.tolist(),
           "n_treated": int(A.sum()), "n_untreated": int((~A).sum()),
           "ps_true_quantiles_treated": np.percentile(r["ps_true"][A], [5, 25, 50, 75, 95]).tolist(),
           "ps_true_quantiles_untreated": np.percentile(r["ps_true"][~A], [5, 25, 50, 75, 95]).tolist(),
           "frac_ps_hat_gt_095": float(np.mean(ps_hat > 0.95)), "frac_ps_hat_lt_005": float(np.mean(ps_hat < 0.05)),
           "untreated_in_ps_gt_095": int((~A & hi_region).sum()), "n_in_ps_gt_095": int(hi_region.sum()),
           "trimmed": {"n": int(trim.sum()), "oracle": float(rt["oracle_ate"]), "gform": float(rt["gform_ate"]),
                       "bias": float(rt["gform_ate"] - rt["oracle_ate"])},
           "full": {"oracle": float(r["oracle_ate"]), "gform": float(r["gform_ate"]), "bias": float(r["gform_ate"] - r["oracle_ate"])},
           "high_ps_region_profile": {"mean_peak_day1": float(F[hi_region, 1].mean()) if hi_region.any() else None,
                                      "mean_sofa": float(F[hi_region, 5].mean()) if hi_region.any() else None}}
    out["headline"] = {"max_abs_smd": _r(np.max(np.abs(smd_obs))), "n_untreated": out["n_untreated"],
                       "frac_ps_gt_095": _r(out["frac_ps_hat_gt_095"], 3), "trimmed_bias": _r(out["trimmed"]["bias"])}
    mac = [("OvlMaxSMD", float(np.max(np.abs(smd_obs))), "{:.2f}"), ("OvlNUntreated", out["n_untreated"], "{:d}"),
           ("OvlFracHighPS", 100 * out["frac_ps_hat_gt_095"], "{:.1f}"), ("OvlTrimN", out["trimmed"]["n"], "{:d}"),
           ("OvlTrimBias", out["trimmed"]["bias"], "{:.1f}")]
    _save(results_dir, "overlap", {"ps_hat": ps_hat, "ps_true": r["ps_true"], "A": A, "F": F}, out, mac, ["R1 M6c"],
          "SMD = difference in means / pooled SD; propensity re-estimated by logistic regression on the 7 recorded features.")
    return out


# ----------------------------------------------------------------------------------------
# E5  Mortality-shape misspecification (R1 M2a, R1 M2b, G11) and ranking robustness
# ----------------------------------------------------------------------------------------
def exp_shapes(results_dir, B=200):
    t0 = time.time()
    P = get_prep("oracle")
    th = theta("gformula")
    rows = {}
    for sh in ["linear", "squared", "threshold", "saturating", "piecewise"]:
        r = gformula.evaluate(P, th, shape=sh)
        b = gformula.bootstrap(P, th, B=GF_B, seed=GF_SEED, shape=sh)
        rows[sh] = _summ(r, b)
    # time trend (R1 M2a): truth uses a linear time term, the fitted model keeps the quadratic one
    r = gformula.evaluate(P, th, time_power=1)
    b = gformula.bootstrap(P, th, B=GF_B, seed=GF_SEED, time_power=1)
    rows["time_linear"] = _summ(r, b)
    # replication of the submitted squared-burden test (both burden terms squared, as fcn_mortality_squared):
    # shape AND coefficients differ from what the g-formula assumes, theta not fitted
    rep = {}
    for nd in (7, 8):
        r2 = gformula.evaluate(P, theta("squared_gen"), shape="squared", drug_shape="squared", n_days=nd, theta_known=th)
        rep[f"{nd}day"] = {"oracle": float(r2["oracle_ate"]), "known_linear_theta": float(r2["gform_known_ate"]),
                           "bias_known": float(r2["gform_known_ate"] - r2["oracle_ate"]),
                           "fitted": float(r2["gform_ate"]), "bias_fitted": float(r2["gform_ate"] - r2["oracle_ate"])}
    # rankings under each shape and time trend, from the strategy grid (true model, no estimation)
    c = get_cohort()
    labels, Ld, Xd, _, _ = run_strategy_grid(c)
    L = {l: i for i, l in enumerate(labels)}
    main = ["none", "fixed", "prn8_k1", "adapt_2"]
    rank = {}
    for q in QUADS:
        rank[q] = {}
        for sh in ["linear", "squared", "threshold", "saturating", "piecewise"]:
            S = survival.exact(Ld[:, [L[m] for m in main]], Xd[:, [L[m] for m in main]], c["age"], c["sofa"], theta(q), sh)
            rank[q][sh] = {"surv": {m: float(100 * S[:, j].mean()) for j, m in enumerate(main)},
                           "winner": main[int(np.argmax(S.mean(0)))], "p_best": stats.p_best(S, seed=51).tolist()}
        S = survival.exact(Ld[:, [L[m] for m in main]], Xd[:, [L[m] for m in main]], c["age"], c["sofa"], theta(q), time_power=1)
        rank[q]["time_linear"] = {"surv": {m: float(100 * S[:, j].mean()) for j, m in enumerate(main)},
                                  "winner": main[int(np.argmax(S.mean(0)))]}
    out = {"n": N, "gformula": rows, "submitted_squared_replication": rep, "rankings": rank, "runtime_s": time.time() - t0}
    out["headline"] = {"bias_fitted": {k: _r(v["bias"]) for k, v in rows.items()},
                       "bias_known_theta": {k: _r(v["bias_known"]) for k, v in rows.items()},
                       "submitted_squared_8day_known": _r(rep["8day"]["bias_known"]),
                       "winners": {QLAB[q]: {s: rank[q][s]["winner"] for s in rank[q]} for q in QUADS}}
    mac = []
    nm = {"linear": "Linear", "squared": "Squared", "threshold": "Threshold", "saturating": "Saturating", "piecewise": "Piecewise", "time_linear": "TimeLinear"}
    for k, v in rows.items():
        mac += [(f"Shape{nm[k]}Bias", v["bias"], "{:.1f}"), (f"Shape{nm[k]}BiasLo", v["ci"]["bias"][0], "{:.1f}"),
                (f"Shape{nm[k]}BiasHi", v["ci"]["bias"][1], "{:.1f}"), (f"Shape{nm[k]}Oracle", v["oracle_ate"], "{:.1f}")]
    mac += [("ShapeSubmittedRepBias", rep["8day"]["bias_known"], "{:.1f}")]
    _save(results_dir, "shapes", {}, out, mac, ["R1 M2a", "R1 M2b", "G11"],
          "Shapes anchored at g(1)=1; theta refitted (linear) on data from each shape.")
    return out


# ----------------------------------------------------------------------------------------
# E6  Personalised thresholds with intervals (R1 M7, G7, Secs 5.3.2 and 5.3.4)
# ----------------------------------------------------------------------------------------
def _omega_grid_surv(LB, C50, gam, U, keys, c):
    S = {k: np.zeros((LB.shape[0], len(OMEGAS) + 1)) for k in keys}
    cum = []
    for j, w in enumerate(OMEGAS + [np.inf]):
        sp = {"strategy": "none"} if np.isinf(w) else {"strategy": "adaptive", "omega": w}
        o = engine.simulate(sp, LB, C50, gam, U, DRUG, CTRL, T, dt)
        cum.append((o["Lday"], o["Xday"]))
        for k in keys:
            S[k][:, j] = survival.exact(o["Lday"], o["Xday"], c["age"], c["sofa"], theta(k))
    return S, cum


def exp_personalization(results_dir, B=1000):
    t0 = time.time()
    c = get_cohort()
    U = cohort_mod.uniform_tape(N, Nt, SEED, stream=1)
    U_cf = cohort_mod.uniform_tape(N, Nt, SEED, stream=2)
    treated = np.arange(N) % 2 == 0          # half observed on fixed dosing, half untreated (Sec 5.3.4 design)
    fx = engine.simulate({"strategy": "fixed"}, c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt, store=("counts", "doses"))
    nn = engine.simulate({"strategy": "none"}, c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt, store=("counts",))
    popC = np.full(N, float(DRUG["C50"])); popG = np.full(N, float(DRUG["gamma_hill"]))
    LBh = np.zeros_like(c["lambda_base"])
    it = np.nonzero(treated)[0]; iu = np.nonzero(~treated)[0]
    LBh[it], _ = gformula.estimate_trajectories(fx["counts"][it], fx["doses"][it], popC[it], popG[it], DRUG, T, dt, True)
    LBh[iu], _ = gformula.estimate_trajectories(nn["counts"][iu], None, popC[iu], popG[iu], DRUG, T, dt, False)
    S_true, _ = _omega_grid_surv(c["lambda_base"], c["C50"], c["gamma_hill"], U, QUADS, c)
    S_est, _ = _omega_grid_surv(LBh, popC, popG, U_cf, QUADS, c)       # analyst's model: estimated trajectory, population PKPD
    wlab = OMEGAS + [np.inf]
    j2 = OMEGAS.index(2)
    out = {"n": N, "omegas": [str(w) for w in wlab], "regimes": {}}
    mac = []
    arrays = {}
    for q in QUADS:
        St, Se = S_true[q], S_est[q]
        jt = St.argmax(1); je = Se.argmax(1)
        uni = St[:, j2]; pt = St[np.arange(N), jt]; pe = St[np.arange(N), je]
        g_true = stats.paired(pt, uni, seed=61); g_est = stats.paired(pe, uni, seed=62)
        rec = stats.bootstrap_stat(lambda idx: (pe[idx] - uni[idx]).mean() / max((pt[idx] - uni[idx]).mean(), 1e-12), N, B=B, seed=63)
        wt = np.array(wlab)[jt]; we = np.array(wlab)[je]
        med_t = stats.bootstrap_stat(lambda idx: np.median(wt[idx]), N, B=B, seed=64)
        med_e = stats.bootstrap_stat(lambda idx: np.median(we[idx]), N, B=B, seed=65)
        within1 = float(np.mean(np.abs(jt - je) <= 1))
        exact_match = float(np.mean(jt == je))
        R = {"uniform": stats.mean_ci(uni), "personal_true": stats.mean_ci(pt), "personal_est": stats.mean_ci(pe),
             "gain_true": g_true, "gain_est": g_est,
             "recovered_frac": float((pe - uni).mean() / max((pt - uni).mean(), 1e-12)), "recovered_ci": [rec["lo"], rec["hi"]],
             "median_omega_true": float(np.median(wt)), "median_omega_true_ci": [med_t["lo"], med_t["hi"]],
             "median_omega_est": float(np.median(we)), "median_omega_est_ci": [med_e["lo"], med_e["hi"]],
             "omega_true_dist": {str(w): float(np.mean(wt == w)) for w in wlab},
             "omega_est_dist": {str(w): float(np.mean(we == w)) for w in wlab},
             "exact_match": exact_match, "within_one_step": within1,
             "share_at_lowest_grid": float(np.mean(jt == 0))}
        out["regimes"][q] = R
        arrays[f"S_true_{q}"] = St; arrays[f"S_est_{q}"] = Se
        Q = QLAB[q]
        mac += [(f"Pers{Q}GainTrue", g_true["est"], "{:.1f}"), (f"Pers{Q}GainTrueLo", g_true["lo"], "{:.1f}"), (f"Pers{Q}GainTrueHi", g_true["hi"], "{:.1f}"),
                (f"Pers{Q}GainEst", g_est["est"], "{:.1f}"), (f"Pers{Q}GainEstLo", g_est["lo"], "{:.1f}"), (f"Pers{Q}GainEstHi", g_est["hi"], "{:.1f}"),
                (f"Pers{Q}Recovered", 100 * R["recovered_frac"], "{:.0f}"), (f"Pers{Q}RecoveredLo", 100 * rec["lo"], "{:.0f}"),
                (f"Pers{Q}RecoveredHi", 100 * rec["hi"], "{:.0f}"), (f"Pers{Q}MedOmega", R["median_omega_true"], "{:g}")]
    out["headline"] = {QLAB[q]: {"gain_true": _r(out["regimes"][q]["gain_true"]["est"]), "gain_true_ci": [_r(out["regimes"][q]["gain_true"]["lo"]), _r(out["regimes"][q]["gain_true"]["hi"])],
                                 "gain_est": _r(out["regimes"][q]["gain_est"]["est"]), "recovered": _r(out["regimes"][q]["recovered_frac"], 2),
                                 "median_omega": out["regimes"][q]["median_omega_true"], "at_lowest_grid": _r(out["regimes"][q]["share_at_lowest_grid"], 2)}
                       for q in QUADS}
    out["runtime_s"] = time.time() - t0
    _save(results_dir, "personalization", arrays, out, mac, ["R1 M7", "G7", "R1 m1"],
          "omega grid 0.125-8 plus none; estimates use population PKPD; true theta used for scoring (as submitted).")
    return out


# ----------------------------------------------------------------------------------------
# E7  Lcum level-set test (R1 M4a)
# ----------------------------------------------------------------------------------------
def exp_levelset(results_dir, r_mult=(0.5, 1.0, 2.0, 4.0)):
    t0 = time.time()
    c = get_cohort()
    U = cohort_mod.uniform_tape(N, Nt, SEED, stream=1)
    base_int = gformula.day_cumsum(c["lambda_base"], dt)[:, 6]
    variants = []
    for m in r_mult:
        r = c["r"] * m
        sh = np.exp(-c["kappa"][:, None] * T[None, :]) * (1 - np.exp(-r[:, None] * T[None, :]))
        sh = sh / sh.max(1, keepdims=True)
        integ = gformula.day_cumsum(sh, dt)[:, 6]
        LB = sh * (base_int / integ)[:, None]          # rescale so the 7-day integral is unchanged
        variants.append(LB)
    peak_h = np.stack([v.max(1) for v in variants], 1)
    peak_t = np.stack([T[v.argmax(1)] for v in variants], 1)
    res = {}
    for q in QUADS:
        Sv = []
        for LB in variants:
            S, _ = _omega_grid_surv(LB, c["C50"], c["gamma_hill"], U, [q], c)
            Sv.append(S[q])
        Sv = np.stack(Sv, 0)                             # (V, n, W)
        jstar = Sv.argmax(2)                             # (V, n)
        ref = list(r_mult).index(1.0)
        same_all = float(np.mean(np.all(jstar == jstar[ref], axis=0)))
        loss = np.stack([Sv[v][np.arange(N), jstar[v]] - Sv[v][np.arange(N), jstar[ref]] for v in range(len(variants))], 0)
        res[q] = {"share_same_omega_all_variants": same_all, "mean_loss_pp_using_ref_omega": float(100 * loss.mean()),
                  "p95_loss_pp": float(100 * np.percentile(loss.max(0), 95)), "max_loss_pp": float(100 * loss.max())}
    out = {"n": N, "r_mult": list(r_mult), "peak_height_ratio_max_min": float(np.median(peak_h.max(1) / peak_h.min(1))),
           "peak_time_range_days": float(np.median(peak_t.max(1) - peak_t.min(1))),
           "lcum_check_max_rel_diff": float(np.max(np.abs(np.stack([gformula.day_cumsum(v, dt)[:, 6] for v in variants], 1) / base_int[:, None] - 1))),
           "regimes": res, "runtime_s": time.time() - t0}
    out["headline"] = {QLAB[q]: {"same_omega": _r(res[q]["share_same_omega_all_variants"], 2), "mean_loss_pp": _r(res[q]["mean_loss_pp_using_ref_omega"], 2),
                                 "p95_loss_pp": _r(res[q]["p95_loss_pp"], 2)} for q in QUADS}
    mac = []
    for q in QUADS:
        mac += [(f"Lvl{QLAB[q]}Same", 100 * res[q]["share_same_omega_all_variants"], "{:.0f}"),
                (f"Lvl{QLAB[q]}Loss", res[q]["mean_loss_pp_using_ref_omega"], "{:.2f}"),
                (f"Lvl{QLAB[q]}LossPNinetyFive", res[q]["p95_loss_pp"], "{:.1f}")]
    mac += [("LvlPeakRatio", out["peak_height_ratio_max_min"], "{:.1f}"), ("LvlPeakTimeRange", out["peak_time_range_days"], "{:.1f}")]
    _save(results_dir, "levelset", {}, out, mac, ["R1 M4a"],
          "Rise rate r scaled x0.5-x4 with amplitude rescaled so the 7-day natural-history integral is unchanged.")
    return out


# ----------------------------------------------------------------------------------------
# E8  EKF tracking accuracy (R2 9, R2 8)
# ----------------------------------------------------------------------------------------
def exp_ekf(results_dir):
    c = get_cohort()
    U = cohort_mod.uniform_tape(N, Nt, SEED, stream=1)
    out = {"n": N, "strategies": {}}
    mac = []
    burn = int(round(0.25 / dt))                         # ignore the first 6 h for steady-state metrics
    for strat in ("none", "adaptive"):
        o = engine.simulate({"strategy": strat}, c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt,
                            store=("lambda_hat", "ekf_P", "counts", "lambda_obs"))
        lam, lh, P, cnt = o["lambda_obs"], o["lambda_hat"], o["ekf_P"], o["counts"]
        prior = np.maximum(0.1, 5 * (np.exp(-0.3 * T) * (1 - np.exp(-1.5 * T))) / np.max(np.exp(-0.3 * T) * (1 - np.exp(-1.5 * T))))
        lo = prior[None, :] * np.exp(np.log(lh / prior[None, :]) - 1.96 * np.sqrt(P))
        hi = prior[None, :] * np.exp(np.log(lh / prior[None, :]) + 1.96 * np.sqrt(P))
        sl = slice(burn, None)
        mask = lam[:, sl] >= 0.5                          # rates where log error is meaningful
        est = {"EKF": lh}
        for hrs in (4, 24):
            w = int(round(hrs / 24 / dt))
            cs = np.cumsum(cnt, axis=1)
            trail = cs - np.concatenate([np.zeros((N, w)), cs[:, :-w]], axis=1)
            nwin = np.minimum(np.arange(1, Nt + 1), w)
            est[f"MA{hrs}h"] = trail / (nwin * dt)
        R = {}
        for k, e in est.items():
            err = e[:, sl] - lam[:, sl]
            rel = np.abs(err[mask]) / lam[:, sl][mask]
            R[k] = {"mae_sz_per_day": float(np.mean(np.abs(err))), "median_rel_err": float(np.median(rel)),
                    "rmse_sz_per_day": float(np.sqrt(np.mean(err ** 2)))}
        R["EKF"]["coverage_95"] = float(np.mean((lam[:, sl] >= lo[:, sl]) & (lam[:, sl] <= hi[:, sl])))
        R["EKF"]["coverage_95_when_rate_ge_05"] = float(np.mean(((lam[:, sl] >= lo[:, sl]) & (lam[:, sl] <= hi[:, sl]))[mask]))
        out["strategies"][strat] = R
        nm = "Untr" if strat == "none" else "Adapt"
        mac += [(f"Ekf{nm}MAE", R["EKF"]["mae_sz_per_day"], "{:.2f}"), (f"Ekf{nm}MAEfour", R["MA4h"]["mae_sz_per_day"], "{:.2f}"),
                (f"Ekf{nm}MAEday", R["MA24h"]["mae_sz_per_day"], "{:.2f}"), (f"Ekf{nm}Cov", 100 * R["EKF"]["coverage_95"], "{:.0f}"),
                (f"Ekf{nm}RelErr", 100 * R["EKF"]["median_rel_err"], "{:.0f}")]
    out["headline"] = {s: {k: _r(v["mae_sz_per_day"]) for k, v in out["strategies"][s].items()} for s in out["strategies"]}
    out["headline"]["coverage"] = {s: _r(out["strategies"][s]["EKF"]["coverage_95"], 3) for s in out["strategies"]}
    _save(results_dir, "ekf", {}, out, mac, ["R2 9", "R2 8"], "Errors after the first 6 h; trailing moving averages of counts as alternatives.")
    return out


# ----------------------------------------------------------------------------------------
# E9  Phenotype misassignment (R1 M1b, E2). Rewritten 2026-11-18: the submitted version compared simple
#     strategies only, ignored the harm-balanced controller and had no intervals.
# ----------------------------------------------------------------------------------------
# drug-harm multipliers for the "one theta for everyone" controller family: (theta4, theta5) of High Sz / High
# Drug scaled by m, seizure terms kept; the drug-to-seizure harm ratio then spans those of all four phenotypes
MIS_DRUG_SCALE = [0.04, 0.08, 0.16, 0.24, 0.33, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0, 6.0]
MIS_PS = [round(0.05 * i, 2) for i in range(21)]


def _mis_wrong(q, scheme):
    """Phenotypes a patient of true phenotype q can be wrongly assigned to, with equal probability."""
    others = [o for o in QUADS if o != q]
    if scheme == "uniform":
        return others
    sz, dr = (lambda o: o[5]), (lambda o: o[7])  # "quad_HSLD": o[5] = seizure-harm level, o[7] = drug-harm level
    keep = {"seizure": lambda o: sz(o) != sz(q) and dr(o) == dr(q), "drug": lambda o: sz(o) == sz(q) and dr(o) != dr(q)}[scheme]
    return [o for o in others if keep(o)]


def exp_misclass(results_dir, ps=MIS_PS, B=2000):
    """How much survival is lost when phenotype-dependent policies are given the wrong phenotype?

    Population: every held-out patient (odd-numbered, n = 500) is scored under each of the four true phenotypes
    (an equal mix), on the common random-number tape of exp_strategies. Policies:
      matched harm-balanced: the controller of Sec 4.5 (estimated rate, filter of exp_harm) given the theta of the
        ASSIGNED phenotype; matched simple: the best strategy for the assigned phenotype among no treatment, the
        fixed-dose family (250-750 mg), the tiered controller (omega grid) and the count-based PRN family, chosen
        on the even half; best single policy: one policy for everyone, from the same simple strategies and from
        harm-balanced controllers with one fixed theta (the four phenotype vectors and MIS_DRUG_SCALE), chosen on
        the even half to maximise survival averaged over the four phenotypes.
    Misassignment: with probability p the assigned phenotype is replaced by a wrong one, uniformly among the other
    three (primary), or only along one axis (secondary); survival is always scored under the true phenotype.
    Survival is the exact expectation over the assignment error, so it is linear in p:
      V(p) = (1 - p) * matched + p * wrong.  Crossover p* = (matched - single) / (matched - wrong).
    Intervals: patient bootstrap of the held-out half (B resamples, paired across policies).
    """
    t0 = time.time()
    c = get_cohort()
    U = cohort_mod.uniform_tape(N, Nt, SEED, stream=1)
    tune = np.arange(N) % 2 == 0
    test = ~tune
    hj = Path(results_dir) / "rr_harm.json"
    filt = json.loads(hj.read_text())["filter"]["chosen"] if hj.exists() else dict(P_pop=10, tau_x=5, q=1e-4)
    labels, Ld, Xd, dose, _ = run_strategy_grid(c)
    L = {l: i for i, l in enumerate(labels)}
    simple = (["none", "fixed_250", "fixed_375", "fixed", "fixed_750"] + [f"adapt_{w:g}" for w in OMEGAS]
              + [f"prn8_k{k}" for k in (1, 2, 3, 4, 5)])
    # harm-balanced controllers: one per theta (phenotype vectors and the one-parameter family)
    TH = {f"harm_{QLAB[q]}": theta(q) for q in QUADS}
    for m in MIS_DRUG_SCALE:
        t_ = theta("quad_HSHD").copy(); t_[4:] *= m
        TH[f"harmscale_{m:g}"] = t_
    H = {}
    for k, th in TH.items():
        sp = {"strategy": "harm", "harm": dict(theta=th, age=c["age"], sofa=c["sofa"]), "ekf": dict(filt, drug_aware=True)}
        H[k] = engine.simulate(sp, c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt)
    pol = simple + list(TH)
    # S[q][:, j]: survival of every patient under true phenotype q with policy pol[j]
    S = {}
    for q in QUADS:
        cols = [survival.exact(Ld[:, L[l]], Xd[:, L[l]], c["age"], c["sofa"], theta(q)) for l in simple]
        cols += [survival.exact(H[k]["Lday"], H[k]["Xday"], c["age"], c["sofa"], theta(q)) for k in TH]
        S[q] = np.stack(cols, 1)
    P = {l: j for j, l in enumerate(pol)}
    doses = {l: float(dose[:, L[l]].mean()) for l in simple}
    doses.update({k: float(H[k]["total_dose"].mean()) for k in TH})
    avg_tune = {l: np.mean([S[q][tune, P[l]].mean() for q in QUADS]) for l in pol}
    simple_by_q = {q: max(simple, key=lambda l: S[q][tune, P[l]].mean()) for q in QUADS}
    harm_by_q = {q: f"harm_{QLAB[q]}" for q in QUADS}
    single = max(pol, key=avg_tune.get)
    single_simple = max(simple, key=avg_tune.get)
    single_harm = max(TH, key=avg_tune.get)

    def per_patient(by_q, scheme):
        """Per-patient survival averaged over the 4 true phenotypes: correctly assigned and wrongly assigned."""
        right = np.mean([S[q][:, P[by_q[q]]] for q in QUADS], 0)
        wrong = np.mean([np.mean([S[q][:, P[by_q[o]]] for o in _mis_wrong(q, scheme)], 0) for q in QUADS], 0)
        return right[test], wrong[test]
    one = lambda l: np.mean([S[q][:, P[l]] for q in QUADS], 0)[test]
    v_single, v_single_simple, v_single_harm = one(single), one(single_simple), one(single_harm)
    n_t = int(test.sum())
    idx = np.random.default_rng((SEED, 909)).integers(0, n_t, size=(B, n_t))

    def pstar(M, W, Sg):
        m, w, s_ = M.mean(), W.mean(), Sg.mean()
        if m <= s_:
            return 0.0
        return float((m - s_) / (m - w)) if w < s_ else np.inf
    res = {}
    for name, by_q in (("harm", harm_by_q), ("simple", simple_by_q)):
        for scheme in ("uniform", "seizure", "drug"):
            M, W = per_patient(by_q, scheme)
            ps_b = np.array([pstar(M[i], W[i], v_single[i]) for i in idx])
            curve = []
            for p in ps:
                V = (1 - p) * M + p * W
                curve.append({"p": p, **stats.mean_ci(V), "vs_single": stats.paired(V, v_single, seed=95)})
            res[f"{name}_{scheme}"] = {
                "matched": stats.mean_ci(M), "all_wrong": stats.mean_ci(W),
                "matched_vs_single": stats.paired(M, v_single, seed=93), "wrong_vs_single": stats.paired(W, v_single, seed=94),
                "loss_per_10pct": float(10 * (M.mean() - W.mean())),
                "p_star": pstar(M, W, v_single),
                "p_star_ci": [float(np.percentile(np.minimum(ps_b, 9.99), 2.5)), float(np.percentile(np.minimum(ps_b, 9.99), 97.5))],
                "p_star_share_never": float(np.mean(np.isinf(ps_b))), "curve": curve}
    out = {"n": n_t, "n_cohort": N, "filter": filt, "policies": pol, "drug_scale_family": MIS_DRUG_SCALE,
           "matched_simple_by_phenotype": {QLAB[q]: simple_by_q[q] for q in QUADS},
           "single_policy": single, "single_simple": single_simple, "single_harm": single_harm,
           "single": stats.mean_ci(v_single), "single_simple_ci": stats.mean_ci(v_single_simple),
           "single_harm_ci": stats.mean_ci(v_single_harm),
           "mean_dose": doses, "avg_survival_tune": {k: float(100 * v) for k, v in avg_tune.items()},
           "per_phenotype_test": {QLAB[q]: {l: float(100 * S[q][test, P[l]].mean()) for l in pol} for q in QUADS},
           "results": res, "runtime_s": time.time() - t0}
    h = res["harm_uniform"]; sm = res["simple_uniform"]
    out["headline"] = {"matched_harm": _r(h["matched"]["est"]), "matched_simple": _r(sm["matched"]["est"]),
                       "single": [single, _r(out["single"]["est"])], "harm_minus_single": [_r(h["matched_vs_single"][k]) for k in ("est", "lo", "hi")],
                       "simple_minus_single": [_r(sm["matched_vs_single"][k]) for k in ("est", "lo", "hi")],
                       "p_star_harm": h["p_star"], "p_star_harm_ci": h["p_star_ci"], "p_star_simple": sm["p_star"]}
    mac = []
    pct = lambda v: 100 * v if np.isfinite(v) else 999.0
    for name, tag in (("harm", "Harm"), ("simple", "Simple")):
        r_ = res[f"{name}_uniform"]
        mac += [(f"Mis{tag}Matched", r_["matched"]["est"], "{:.1f}"), (f"Mis{tag}MatchedLo", r_["matched"]["lo"], "{:.1f}"),
                (f"Mis{tag}MatchedHi", r_["matched"]["hi"], "{:.1f}"),
                (f"Mis{tag}Wrong", r_["all_wrong"]["est"], "{:.1f}"),
                (f"Mis{tag}VsSingle", r_["matched_vs_single"]["est"], "{:.1f}"), (f"Mis{tag}VsSingleLo", r_["matched_vs_single"]["lo"], "{:.1f}"),
                (f"Mis{tag}VsSingleHi", r_["matched_vs_single"]["hi"], "{:.1f}"),
                (f"Mis{tag}WrongVsSingle", r_["wrong_vs_single"]["est"], "{:.1f}"),
                (f"Mis{tag}WrongVsSingleLo", r_["wrong_vs_single"]["lo"], "{:.1f}"), (f"Mis{tag}WrongVsSingleHi", r_["wrong_vs_single"]["hi"], "{:.1f}"),
                (f"Mis{tag}LossPerTen", r_["loss_per_10pct"], "{:.1f}"),
                (f"Mis{tag}Pstar", pct(r_["p_star"]), "{:.0f}"), (f"Mis{tag}PstarLo", pct(r_["p_star_ci"][0]), "{:.0f}"),
                (f"Mis{tag}PstarHi", min(100.0, pct(r_["p_star_ci"][1])), "{:.0f}")]
        for scheme, stag in (("seizure", "SzAxis"), ("drug", "DrugAxis")):
            r2 = res[f"{name}_{scheme}"]
            mac += [(f"Mis{tag}{stag}Wrong", r2["all_wrong"]["est"], "{:.1f}"), (f"Mis{tag}{stag}Pstar", pct(r2["p_star"]), "{:.0f}")]
    mac += [("MisSingle", out["single"]["est"], "{:.1f}"), ("MisSingleLo", out["single"]["lo"], "{:.1f}"), ("MisSingleHi", out["single"]["hi"], "{:.1f}"),
            ("MisSingleSimple", out["single_simple_ci"]["est"], "{:.1f}"), ("MisSingleHarm", out["single_harm_ci"]["est"], "{:.1f}"),
            ("MisNTest", n_t, "{:d}")]
    # plain-language description of the chosen single policies, for the text
    def describe(l):
        if l.startswith("harmscale_"):
            return r"harm-balanced controller with the High~Sz~/~High~Drug drug-harm coefficients scaled by " + l.split("_")[1]
        if l.startswith("harm_"):
            return "harm-balanced controller with the " + REGISTRY[[q for q in QUADS if QLAB[q] == l.split("_")[1]][0]]["label"] + r" coefficients"
        if l == "none":
            return "no treatment"
        if l.startswith("fixed"):
            return ("fixed " + (l.split("_")[1] if "_" in l else "500") + r"~mg every 8~h")
        if l.startswith("adapt_"):
            return r"tiered controller with $\omega = " + l.split("_")[1] + "$"
        return r"PRN dosing ($\geq$" + l.split("_k")[1] + " seizures in 8~h)"
    mac += [("MisSingleDesc", describe(single), None), ("MisSingleSimpleDesc", describe(single_simple), None),
            ("MisSingleHarmDesc", describe(single_harm), None)]
    arrays = {f"S_{q}": S[q] for q in QUADS}
    arrays["policies"] = np.array(pol)
    _save(results_dir, "misclass", arrays, out, mac, ["R1 M1b", "E2"],
          "Held-out half scored under all four true phenotypes; wrong phenotype uniform among the other three "
          "(also one-axis errors); exact expectation over assignment errors; patient bootstrap.")
    return out


# ----------------------------------------------------------------------------------------
# E10 Observation errors: annotation errors, medication-record errors (R1 M1a)
# ----------------------------------------------------------------------------------------
def exp_obs_errors(results_dir):
    t0 = time.time()
    c = get_cohort()
    U = cohort_mod.uniform_tape(N, Nt, SEED, stream=1)
    Ut = cohort_mod.uniform_tape(N, Nt, SEED, stream=3)
    Uf = cohort_mod.uniform_tape(N, Nt, SEED, stream=4)
    grid = []
    base = {}
    for sens in (1.0, 0.9, 0.75, 0.5):
        for fp in (0.0, 1.0, 3.0, 6.0):
            row = {"sens": sens, "fp_per_day": fp}
            for st_lab, sp in (("adaptive", {"strategy": "adaptive"}), ("prn", {"strategy": "prn", "prn_legacy": False})):
                o = engine.simulate(dict(sp, detect_sens=sens, detect_fp_per_day=fp), c["lambda_base"], c["C50"], c["gamma_hill"],
                                    U, DRUG, CTRL, T, dt, U_thin=Ut, U_fp=Uf)
                for q in QUADS:
                    row[f"{st_lab}_{QLAB[q]}"] = float(100 * survival.exact(o["Lday"], o["Xday"], c["age"], c["sofa"], theta(q)).mean())
                row[f"{st_lab}_dose"] = float(o["total_dose"].mean())
            grid.append(row)
    th = theta("gformula")
    scen = {"baseline": dict(), "sens075": dict(dgp={"detect_sens": 0.75}), "fp3": dict(dgp={"detect_fp_per_day": 3.0}),
            "sens075_fp3": dict(dgp={"detect_sens": 0.75, "detect_fp_per_day": 3.0}),
            "jitter1h": dict(jitter=1.0), "jitter2h": dict(jitter=2.0), "drop10": dict(drop=0.10), "drop25": dict(drop=0.25)}
    gf = {}
    for k, kw in scen.items():
        P = get_prep("oracle", **kw)
        r = gformula.evaluate(P, th)
        b = gformula.bootstrap(P, th, B=GF_B, seed=GF_SEED)
        s = _summ(r, b)
        s["Lcum_corr_treated"] = float(np.corrcoef(P["Lbase_hat_t"], P["Lbase_true"])[0, 1])
        s["Lcum_corr_untreated"] = float(np.corrcoef(P["Lbase_hat_n"], P["Lbase_true"])[0, 1])
        gf[k] = s
    out = {"n": N, "strategy_grid": grid, "gformula": gf, "runtime_s": time.time() - t0}
    out["headline"] = {k: {"bias": _r(v["bias"]), "ci": [_r(x) for x in v["ci"]["bias"]], "Lcum_corr_t": _r(v["Lcum_corr_treated"], 3)}
                       for k, v in gf.items()}
    nm = {"baseline": "Base", "sens075": "SensLow", "fp3": "FalsePos", "sens075_fp3": "Both", "jitter1h": "JitterOne",
          "jitter2h": "JitterTwo", "drop10": "DropTen", "drop25": "DropTwentyFive"}
    mac = []
    for k, v in gf.items():
        mac += [(f"Obs{nm[k]}Bias", v["bias"], "{:.1f}"), (f"Obs{nm[k]}BiasLo", v["ci"]["bias"][0], "{:.1f}"),
                (f"Obs{nm[k]}BiasHi", v["ci"]["bias"][1], "{:.1f}")]
    _save(results_dir, "obs_errors", {}, out, mac, ["R1 M1a"],
          "Detection errors act on the counts the EKF/PRN and the estimator see; mortality uses the true rate.")
    return out


# ----------------------------------------------------------------------------------------
# E11 Data-generating misspecification: two-compartment PK (R1 M5), drug tolerance (R2 15)
# ----------------------------------------------------------------------------------------
def exp_dgp(results_dir):
    t0 = time.time()
    c = get_cohort()
    U = cohort_mod.uniform_tape(N, Nt, SEED, stream=1)
    th = theta("gformula")
    variants = {"one_compartment": {}, "two_compartment": {"pk2": {"k12_per_h": 0.5, "k21_per_h": 0.1}},
                "tolerance_010": {"c50_drift_per_day": 0.10}, "tolerance_025": {"c50_drift_per_day": 0.25}}
    out = {"n": N, "variants": {}}
    mac = []
    main = ["none", "fixed", "prn8", "adaptive"]
    for vk, dgp in variants.items():
        Ls, Xs = [], []
        for sp in ({"strategy": "none"}, {"strategy": "fixed"}, {"strategy": "prn", "prn_legacy": False}, {"strategy": "adaptive"}):
            o = engine.simulate(dict(sp, **dgp), c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt)
            Ls.append(o["Lday"]); Xs.append(o["Xday"])
        Ld, Xd = np.stack(Ls, 1), np.stack(Xs, 1)
        rank = {}
        for q in QUADS:
            S = survival.exact(Ld, Xd, c["age"], c["sofa"], theta(q))
            rank[QLAB[q]] = {"surv": {m: float(100 * S[:, j].mean()) for j, m in enumerate(main)}, "winner": main[int(np.argmax(S.mean(0)))],
                             "p_best": stats.p_best(S, seed=80).tolist()}
        P = get_prep("oracle", dgp=dgp or None)
        r = gformula.evaluate(P, th)
        b = gformula.bootstrap(P, th, B=GF_B, seed=GF_SEED)
        out["variants"][vk] = {"rankings": rank, "gformula": _summ(r, b)}
    out["headline"] = {vk: {"winners": {q: v["rankings"][q]["winner"] for q in v["rankings"]}, "gform_bias": _r(v["gformula"]["bias"]),
                            "gform_sign_correct": bool(np.sign(v["gformula"]["gform_ate"]) == np.sign(v["gformula"]["oracle_ate"]))}
                       for vk, v in out["variants"].items()}
    nm = {"one_compartment": "OneComp", "two_compartment": "TwoComp", "tolerance_010": "TolTen", "tolerance_025": "TolTwentyFive"}
    for vk, v in out["variants"].items():
        mac += [(f"Dgp{nm[vk]}Bias", v["gformula"]["bias"], "{:.1f}"), (f"Dgp{nm[vk]}Oracle", v["gformula"]["oracle_ate"], "{:.1f}"),
                (f"Dgp{nm[vk]}Gform", v["gformula"]["gform_ate"], "{:.1f}")]
    out["runtime_s"] = time.time() - t0
    _save(results_dir, "dgp", {}, out, mac, ["R1 M5", "R2 15"],
          "Two-compartment: k12=0.5/h, k21=0.1/h (illustrative); tolerance: C50 rises 10% or 25% per day. Estimator keeps the one-compartment, fixed-C50 model.")
    return out


# ----------------------------------------------------------------------------------------
# E12 Refutation and negative controls (handoff Task C; supports R1 M6)
# ----------------------------------------------------------------------------------------
def exp_refutation(results_dir):
    P = get_prep("oracle")
    th = theta("gformula")
    out = {"n": N}
    null_th = th.copy(); null_th[2:] = 0.0                       # seizures and drug do not affect death
    r = gformula.evaluate(P, null_th)
    b = gformula.bootstrap(P, null_th, B=GF_B, seed=GF_SEED)
    out["null_outcome"] = _summ(r, b)
    out["null_outcome"]["ci_covers_0"] = bool(b["lo"][1] <= 0 <= b["hi"][1])
    # repeat over the DEATH_DRAWS sets of observed deaths: does the pipeline manufacture an effect on average?
    for lab, tt in (("null", null_th), ("base", th)):
        d_ = _death_draws(P, tt, N)
        out[f"{lab}_bias_over_draws"] = {"mean": d_["mean_bias"], "sd": d_["sd"], "se": d_["mc_se"], "draws": d_["draws"]}
    g = np.random.default_rng(91)
    subs = []
    for _ in range(20):
        idx = np.sort(g.choice(N, N // 2, replace=False))
        rs = gformula.evaluate(P, th, idx=idx)
        subs.append(float(rs["gform_ate"] - rs["oracle_ate"]))
    out["subset_bias_sd"] = float(np.std(subs, ddof=1))
    out["subset_bias_mean"] = float(np.mean(subs))
    out["headline"] = {"null_gform": _r(out["null_outcome"]["gform_ate"]), "null_ci": [_r(x) for x in out["null_outcome"]["ci"]["gform"]],
                       "null_covers_0": out["null_outcome"]["ci_covers_0"], "subset_bias_sd": _r(out["subset_bias_sd"])}
    out["headline"]["null_mean_bias_over_draws"] = _r(out["null_bias_over_draws"]["mean"])
    out["headline"]["base_mean_bias_over_draws"] = _r(out["base_bias_over_draws"]["mean"])
    mac = [("RefNullDrawMean", out["null_bias_over_draws"]["mean"], "{:.1f}"), ("RefNullDrawSD", out["null_bias_over_draws"]["sd"], "{:.1f}"),
           ("RefNullDrawSE", out["null_bias_over_draws"]["se"], "{:.2f}"),
           ("RefBaseDrawMean", out["base_bias_over_draws"]["mean"], "{:.1f}"), ("RefBaseDrawSE", out["base_bias_over_draws"]["se"], "{:.1f}"),
           ("RefNullATE", out["null_outcome"]["gform_ate"], "{:.1f}"), ("RefNullLo", out["null_outcome"]["ci"]["gform"][0], "{:.1f}"),
           ("RefNullHi", out["null_outcome"]["ci"]["gform"][1], "{:.1f}"), ("RefSubsetSD", out["subset_bias_sd"], "{:.1f}")]
    _save(results_dir, "refutation", {}, out, mac, ["R1 M6a", "R1 M6b"], "Null outcome: theta2..theta5 = 0 in truth; 20 random half-samples.")
    return out


# ----------------------------------------------------------------------------------------
# E13 Estimation vs observation window and onset offset (R2 10)
# ----------------------------------------------------------------------------------------
def _masked_fit(counts, mask, peak_rate=10.0):
    Tm = T[mask]; cm = counts[mask]

    def nll(p):
        A, r, k = np.exp(p)
        sh = np.exp(-k * T) * (1 - np.exp(-r * T))
        mx = sh.max()
        if mx < 1e-10:
            return 1e10
        lam = A * peak_rate * (np.exp(-k * Tm) * (1 - np.exp(-r * Tm))) / mx
        mu = np.maximum(lam * dt, 1e-10)
        return -np.sum(cm * np.log(mu) - mu) + 0.01 * np.sum(p ** 2)
    best = None
    for p0 in ([0.0, np.log(1.5), np.log(0.3)], [np.log(2.0), np.log(1.0), np.log(0.5)]):
        res = minimize(nll, p0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-6, maxiter=4000, maxfev=8000))
        if best is None or res.fun < best.fun:
            best = res
    A, r, k = np.exp(best.x)
    sh = np.exp(-k * T) * (1 - np.exp(-r * T))
    return A * peak_rate * sh / sh.max()


def exp_window(results_dir, n=300):
    t0 = time.time()
    c = cohort_mod.subset(get_cohort(), np.arange(n))
    U = cohort_mod.uniform_tape(n, Nt, SEED, stream=1)
    o = engine.simulate({"strategy": "none"}, c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt, store=("counts",))
    true_L = gformula.day_cumsum(c["lambda_base"], dt)[:, 6]
    true_pk = c["lambda_base"].max(1)
    rows = []
    for kind, vals in (("window_days", (1, 2, 3, 5, 7)), ("offset_days", (0, 0.5, 1, 2))):
        for v in vals:
            mask = (T <= v) if kind == "window_days" else (T >= v)
            est = np.stack([_masked_fit(o["counts"][i], mask) for i in range(n)])
            eL = gformula.day_cumsum(est, dt)[:, 6]
            rows.append({kind: v, "Lcum_corr": float(np.corrcoef(eL, true_L)[0, 1]),
                         "Lcum_mape": float(np.median(np.abs(eL / true_L - 1))), "peak_corr": float(np.corrcoef(est.max(1), true_pk)[0, 1])})
    out = {"n": n, "rows": rows, "runtime_s": time.time() - t0,
           "headline": {(f"W{r_['window_days']}" if "window_days" in r_ else f"O{r_['offset_days']}"): _r(r_["Lcum_corr"], 3) for r_ in rows}}
    mac = []
    words = {1: "One", 2: "Two", 3: "Three", 5: "Five", 7: "Seven", 0: "Zero", 0.5: "Half"}
    for r_ in rows:
        if "window_days" in r_:
            mac.append((f"WinLcumCorr{words[r_['window_days']]}", r_["Lcum_corr"], "{:.2f}"))
        else:
            mac.append((f"OffLcumCorr{words[r_['offset_days']]}", r_["Lcum_corr"], "{:.2f}"))
    _save(results_dir, "window", {}, out, mac, ["R2 10"], "Untreated patients; likelihood over observed bins only; Lcum = 7-day natural-history integral.")
    return out


def exp_gformula_scaling(results_dir, sizes=(1000, 2000, 4000)):
    """Is the fitted-theta g-formula consistent? Mean and spread over outcome draws as n grows (G11).

    For each size the cohort is the first n patients (prefix-stable), the treatment assignment and trajectories are
    held fixed, and only the simulated deaths are redrawn (DEATH_DRAWS sets, the same seeds as exp_gformula), so the
    SD is the error of the fitted mortality model and should fall about as 1/sqrt(n); the bias splits into the part
    left with the true coefficients (known_theta - oracle: error of the estimated trajectories) and the rest.
    """
    t0 = time.time()
    th = theta("gformula")
    rows = []
    for n in sizes:
        P = get_prep("oracle", n=n)
        r0 = gformula.evaluate(P, th)
        d_ = _death_draws(P, th, n)
        rows.append({"n": n, "n_untreated": int((~r0["A"]).sum()), "oracle": float(r0["oracle_ate"]), "known_theta": float(r0["gform_known_ate"]),
                     "known_bias": float(r0["gform_known_ate"] - r0["oracle_ate"]),
                     "fitted_mean": d_["mean"], "fitted_sd": d_["sd"], "mean_bias": d_["mean_bias"], "mc_se": d_["mc_se"],
                     "draws": d_["draws"]})
    for r_ in rows:
        r_["sd_ratio_vs_first"] = r_["fitted_sd"] / rows[0]["fitted_sd"]
        r_["sqrt_n_prediction"] = float(np.sqrt(rows[0]["n"] / r_["n"]))
    out = {"n": max(sizes), "rows": rows, "runtime_s": time.time() - t0,
           "headline": {str(r_["n"]): {"mean_bias": _r(r_["mean_bias"]), "sd": _r(r_["fitted_sd"]), "known_bias": _r(r_["known_bias"])} for r_ in rows}}
    words = {1000: "OneK", 2000: "TwoK", 4000: "FourK"}
    mac = []
    for r_ in rows:
        w = words[r_["n"]]
        mac += [(f"GFScale{w}Bias", r_["mean_bias"], "{:.1f}"), (f"GFScale{w}SD", r_["fitted_sd"], "{:.1f}"),
                (f"GFScale{w}SE", r_["mc_se"], "{:.1f}"), (f"GFScale{w}KnownBias", r_["known_bias"], "{:.1f}"),
                (f"GFScale{w}NUntr", r_["n_untreated"], "{:d}")]
    _save(results_dir, "gformula_scaling", {}, out, mac, ["G11", "R1 M7"],
          f"{DEATH_DRAWS} outcome draws per size (seeds {DEATH_SEED0}+); same cohort prefix; assignment fixed.")
    return out


def exp_graded_lcum(results_dir, fracs=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    """Lcum recovery when a growing fraction of patients follows the CICADAS SDE (R1 M4b)."""
    t0 = time.time()
    out = {"n": N, "groups": {}}
    est = {}
    for gen in ("rise_decay", "sde"):
        c = get_cohort(N, SEED, gen)
        U = cohort_mod.uniform_tape(N, Nt, SEED, stream=1)
        un = engine.simulate({"strategy": "none"}, c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt, store=("counts",))
        tr = engine.simulate({"strategy": "adaptive"}, c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt, store=("counts", "doses"))
        LBn, _ = gformula.estimate_trajectories(un["counts"], None, c["C50"], c["gamma_hill"], DRUG, T, dt, False)
        LBt, _ = gformula.estimate_trajectories(tr["counts"], tr["doses"], c["C50"], c["gamma_hill"], DRUG, T, dt, True)
        true_L = gformula.day_cumsum(c["lambda_base"], dt)[:, 6]
        est[gen] = dict(true=true_L, un=gformula.day_cumsum(LBn, dt)[:, 6], tr=gformula.day_cumsum(LBt, dt)[:, 6])
        out["groups"][gen] = {"corr_untreated": float(np.corrcoef(est[gen]["un"], true_L)[0, 1]),
                              "corr_treated": float(np.corrcoef(est[gen]["tr"], true_L)[0, 1]),
                              "median_abs_rel_err_untreated": float(np.median(np.abs(est[gen]["un"] / true_L - 1))),
                              "median_abs_rel_err_treated": float(np.median(np.abs(est[gen]["tr"] / true_L - 1)))}
    rows = []
    for f in fracs:
        k = int(round(f * N))
        tr_ = np.concatenate([est["sde"]["true"][:k], est["rise_decay"]["true"][k:]])
        e_u = np.concatenate([est["sde"]["un"][:k], est["rise_decay"]["un"][k:]])
        e_t = np.concatenate([est["sde"]["tr"][:k], est["rise_decay"]["tr"][k:]])
        half = np.arange(N) % 2 == 0          # half treated, half untreated, as in the graded experiment
        e_mix = np.where(half, e_t, e_u)
        rows.append({"sde_fraction": f, "corr": float(np.corrcoef(e_mix, tr_)[0, 1]),
                     "median_abs_rel_err": float(np.median(np.abs(e_mix / tr_ - 1)))})
    out["rows"] = rows
    out["runtime_s"] = time.time() - t0
    out["headline"] = {"corr_by_fraction": {str(r_["sde_fraction"]): _r(r_["corr"], 3) for r_ in rows},
                       "groups": {g: {k: _r(v, 3) for k, v in d.items()} for g, d in out["groups"].items()}}
    mac = [("GradedCorrZero", rows[0]["corr"], "{:.2f}"), ("GradedCorrHalf", rows[5]["corr"], "{:.2f}"), ("GradedCorrAll", rows[-1]["corr"], "{:.2f}"),
           ("GradedSDECorrTreated", out["groups"]["sde"]["corr_treated"], "{:.2f}"), ("GradedSDECorrUntreated", out["groups"]["sde"]["corr_untreated"], "{:.2f}")]
    _save(results_dir, "graded_lcum", {}, out, mac, ["R1 M4b"], "Natural-history Lcum over 7 days; SDE patients from fcn_generate_patient_cicadas port.")
    return out


# ----------------------------------------------------------------------------------------
# E16  Harm-balanced adaptive controller (R1 M3 follow-up; the tuned fixed dose beat the tiered rule)
# ----------------------------------------------------------------------------------------
HARM_FILTERS = [dict(P_pop=p, tau_x=t, q=qq) for p in (5, 10, 15) for t in (0.5, 5, 20) for qq in (1e-4, 1e-3)]


def _harm_spec(c, key, filt=None, **kw):
    sp = {"strategy": "harm", "harm": dict(theta=theta(key), age=c["age"], sofa=c["sofa"], **kw)}
    if filt is not None:
        sp["ekf"] = dict(filt, drug_aware=True)
    return sp


def exp_harm(results_dir):
    """Harm-balanced controller against tuned comparators.

    The controller picks each dose to minimise the forecast integral of f = a*lambda_base*s(x) + b*x over 24 h,
    with a, b from the phenotype's theta and the patient's SOFA and age. It needs a forecast of the drug-free rate:
      oracle "full": true lambda_base and the patient's own C50 and gamma (ceiling)
      oracle "rate": true lambda_base, population PD
      estimated:     drug-aware EKF; its prior peak, persistence and process noise are chosen on the even half by
                     tracking error of lambda_base only (no outcomes), then frozen
    Comparators (fixed family, tiered adaptive, PRN) are tuned on the even half by survival, as in exp_strategies.
    All contrasts are scored on the odd half. Cap sensitivity: 1,500 mg for both harm and fixed.
    """
    t0 = time.time()
    c = get_cohort()
    U = cohort_mod.uniform_tape(N, Nt, SEED, stream=1)
    tune = np.arange(N) % 2 == 0
    test = ~tune
    sim = lambda sp: engine.simulate(sp, c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt)

    # 1. filter chosen on the even half by tracking error (default-phenotype controller drives the doses)
    track = []
    for f in HARM_FILTERS:
        o = engine.simulate(_harm_spec(c, "default", f), c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt,
                            store=("lambda_hat",))
        err = np.abs(o["lambda_hat"] - c["lambda_base"])
        track.append(dict(f, mae_tune=float(err[tune].mean()), mae_test=float(err[test].mean())))
    best_f = min(range(len(track)), key=lambda i: track[i]["mae_tune"])
    filt = {k: track[best_f][k] for k in ("P_pop", "tau_x", "q")}
    sub = lambda f: next(r for r in track if all(r[k] == f[k] for k in f))
    default_f = dict(P_pop=5, tau_x=0.5, q=1e-4)

    # 2. comparators (phenotype-free, simulated once)
    comp = {"none": {"strategy": "none"}}
    comp.update({f"fixed_{d}": {"strategy": "fixed", "dose_standard": d} for d in (250, 375, 500, 750, 1000, 1500)})
    comp.update({f"fixedload_{d}": {"strategy": "fixed", "dose_standard": d, "loading_dose": 2 * d} for d in (500, 750)})
    comp.update({f"adapt_{w}": {"strategy": "adaptive", "omega": w} for w in OMEGAS})
    comp.update({f"prn8_k{k}": {"strategy": "prn", "prn_legacy": False, "prn_window_hours": 8, "prn_min_count": k} for k in (1, 2, 3, 4, 5)})
    co = {k: sim(v) for k, v in comp.items()}
    fams = {"fixed": [f"fixed_{d}" for d in (250, 375, 500, 750)], "fixed1500": [f"fixed_{d}" for d in (250, 375, 500, 750, 1000, 1500)],
            "adapt": [f"adapt_{w}" for w in OMEGAS], "prn": [f"prn8_k{k}" for k in (1, 2, 3, 4, 5)]}

    out = {"n": N, "filter": dict(chosen=filt, grid=track, default=sub(default_f)), "regimes": {}}
    mac = [("HarmFilterPpop", filt["P_pop"], "{:g}"), ("HarmFilterTau", filt["tau_x"], "{:g}"), ("HarmFilterQ", filt["q"], "{:g}"),
           ("HarmTrackMAEDefault", sub(default_f)["mae_test"], "{:.2f}"), ("HarmTrackMAEChosen", track[best_f]["mae_test"], "{:.2f}")]
    arrays = {}
    for rk in QUADS + ["default"]:
        q = QLAB[rk]
        th = theta(rk)
        surv = lambda o: survival.exact(o["Lday"], o["Xday"], c["age"], c["sofa"], th)
        Sc = {k: surv(o) for k, o in co.items()}
        hv = {"harm_oracle_full": _harm_spec(c, rk, oracle="full"), "harm_oracle_rate": _harm_spec(c, rk, oracle="rate"),
              "harm_est": _harm_spec(c, rk, filt), "harm_est_oldfilter": _harm_spec(c, rk, default_f),
              "harm_est_load": _harm_spec(c, rk, filt, d_max_load=1500),
              "harm_est_1500": _harm_spec(c, rk, filt, d_max=1500), "harm_oracle_full_1500": _harm_spec(c, rk, oracle="full", d_max=1500)}
        ho = {k: sim(v) for k, v in hv.items()}
        Sh = {k: surv(o) for k, o in ho.items()}
        S = {**Sc, **Sh}
        R = {"surv": {k: stats.mean_ci(v[test]) for k, v in S.items()},
             "dose": {k: float(o["total_dose"].mean()) for k, o in {**co, **ho}.items()}}
        tuned = {f: max(m, key=lambda l: S[l][tune].mean()) for f, m in fams.items()}
        R["tuned"] = tuned
        pr = lambda a, b, sd: stats.paired(S[a][test], S[b][test], seed=sd)
        R["contrasts"] = {
            "est-fixed": pr("harm_est", tuned["fixed"], 21), "est-adapt": pr("harm_est", tuned["adapt"], 22),
            "est-prn": pr("harm_est", tuned["prn"], 23), "est-none": pr("harm_est", "none", 24),
            "oracle-fixed": pr("harm_oracle_full", tuned["fixed"], 25), "oraclerate-fixed": pr("harm_oracle_rate", tuned["fixed"], 26),
            "est-oracle": pr("harm_est", "harm_oracle_full", 27), "est-oldfilter": pr("harm_est", "harm_est_oldfilter", 28),
            "estload-fixedload750": pr("harm_est_load", "fixedload_750", 29),
            "est1500-fixed1500": pr("harm_est_1500", tuned["fixed1500"], 30),
            "oracle1500-fixed1500": pr("harm_oracle_full_1500", tuned["fixed1500"], 31)}
        # share of the oracle's gain over the tuned fixed dose that the estimated controller keeps
        g_or = S["harm_oracle_full"][test].mean() - S[tuned["fixed"]][test].mean()
        g_es = S["harm_est"][test].mean() - S[tuned["fixed"]][test].mean()
        R["recovered_share"] = float(g_es / g_or) if g_or > 0.005 else None
        out["regimes"][rk] = R
        arrays[f"S_{rk}"] = np.stack([S[k] for k in sorted(S)], 1)
        for key, tag in [("est-fixed", "EstVsFixed"), ("est-adapt", "EstVsAdapt"), ("est-prn", "EstVsPRN"), ("est-none", "EstVsNone"),
                         ("oracle-fixed", "OracleVsFixed"), ("est-oracle", "EstVsOracle"), ("est-oldfilter", "EstVsOldFilter"),
                         ("estload-fixedload750", "LoadVsFixedLoad"), ("est1500-fixed1500", "CapEstVsFixed"),
                         ("oracle1500-fixed1500", "CapOracleVsFixed")]:
            c_ = R["contrasts"][key]
            mac += [(f"Harm{q}{tag}", c_["est"], "{:.1f}"), (f"Harm{q}{tag}Lo", c_["lo"], "{:.1f}"), (f"Harm{q}{tag}Hi", c_["hi"], "{:.1f}")]
        for k, nm in [("harm_est", "Est"), ("harm_oracle_full", "Oracle"), (tuned["fixed"], "TunedFixed"), (tuned["adapt"], "TunedAdapt")]:
            mac += [(f"Harm{q}Surv{nm}", R["surv"][k]["est"], "{:.1f}")]
        mac += [(f"Harm{q}DoseEst", R["dose"]["harm_est"], "{:.0f}"), (f"Harm{q}DoseOracle", R["dose"]["harm_oracle_full"], "{:.0f}")]
    arrays["labels"] = np.array(sorted(S))
    out["runtime_s"] = time.time() - t0
    out["headline"] = {QLAB[r]: {k: _r(out["regimes"][r]["contrasts"][k]["est"], 1) for k in ("est-fixed", "est-adapt", "oracle-fixed", "est-none")}
                       for r in QUADS}
    _save(results_dir, "harm", arrays, out, mac, ["R1 M3", "R1 M4c"],
          "Harm-balanced controller; filter chosen on even half by tracking error; contrasts on odd half.")
    return out


# ----------------------------------------------------------------------------------------
# E17  Closed-loop validation (Sec 5.3.4; R2 13, R2 14) on the shared cohort, with CIs
# ----------------------------------------------------------------------------------------
def exp_closed_loop(results_dir):
    """Does the analyst's counterfactual simulator rank strategies as the truth does?

    Design (as submitted, now on the shared 1,000 patients): even-numbered patients are observed on fixed
    500 mg dosing, odd-numbered patients untreated. Each patient's natural history is estimated from those
    observations by Poisson MLE, inverting the drug with POPULATION PKPD. Every strategy is then simulated twice:
    on the true trajectory with the patient's own PKPD (truth) and on the estimated trajectory with population
    PKPD and an independent random-number tape (the analyst's counterfactual simulator). "Fixed" in the design is
    the treatment the treated half received in the observed data; it is also one of the strategies compared.
    """
    t0 = time.time()
    c = get_cohort()
    U = cohort_mod.uniform_tape(N, Nt, SEED, stream=1)
    U_cf = cohort_mod.uniform_tape(N, Nt, SEED, stream=2)
    treated = np.arange(N) % 2 == 0
    fx = engine.simulate({"strategy": "fixed"}, c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt, store=("counts", "doses"))
    nn = engine.simulate({"strategy": "none"}, c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt, store=("counts",))
    popC = np.full(N, float(DRUG["C50"])); popG = np.full(N, float(DRUG["gamma_hill"]))
    LBh = np.zeros_like(c["lambda_base"])
    it = np.nonzero(treated)[0]; iu = np.nonzero(~treated)[0]
    LBh[it], _ = gformula.estimate_trajectories(fx["counts"][it], fx["doses"][it], popC[it], popG[it], DRUG, T, dt, True)
    LBh[iu], _ = gformula.estimate_trajectories(nn["counts"][iu], None, popC[iu], popG[iu], DRUG, T, dt, False)
    Lt = c["lambda_base"].sum(1) * dt; Le = LBh.sum(1) * dt
    base = [("none", {"strategy": "none"}), ("fixed", {"strategy": "fixed"}), ("fixed_750", {"strategy": "fixed", "dose_standard": 750}),
            ("prn8_k1", {"strategy": "prn", "prn_legacy": False, "prn_window_hours": 8, "prn_min_count": 1}),
            ("adapt_2", {"strategy": "adaptive", "omega": 2})]
    filt = json.loads((Path(results_dir) / "rr_harm.json").read_text())["filter"]["chosen"] if (Path(results_dir) / "rr_harm.json").exists() \
        else dict(P_pop=10, tau_x=5, q=1e-4)
    out = {"n": N, "lcum_corr": float(np.corrcoef(Lt, Le)[0, 1]), "lcum_corr_treated": float(np.corrcoef(Lt[it], Le[it])[0, 1]),
           "lcum_corr_untreated": float(np.corrcoef(Lt[iu], Le[iu])[0, 1]), "regimes": {}}
    sim_t = lambda sp: engine.simulate(sp, c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt)
    sim_e = lambda sp: engine.simulate(sp, LBh, popC, popG, U_cf, DRUG, CTRL, T, dt)
    ot = {k: sim_t(v) for k, v in base}
    oe = {k: sim_e(v) for k, v in base}
    mac = [("CLLcumCorr", out["lcum_corr"], "{:.2f}")]
    arrays = {}
    for rk in QUADS:
        q = QLAB[rk]
        th = theta(rk)
        hs = _harm_spec(c, rk, filt)
        ot_q = dict(ot, harm=sim_t(hs)); oe_q = dict(oe, harm=sim_e(hs))
        St = {k: survival.exact(o["Lday"], o["Xday"], c["age"], c["sofa"], th) for k, o in ot_q.items()}
        Se = {k: survival.exact(o["Lday"], o["Xday"], c["age"], c["sofa"], th) for k, o in oe_q.items()}
        R = {"true": {k: stats.mean_ci(v) for k, v in St.items()}, "est": {k: stats.mean_ci(v) for k, v in Se.items()},
             "est_minus_true": {k: stats.paired(Se[k], St[k], seed=71) for k in St}}
        R["winner_true"] = max(St, key=lambda k: St[k].mean()); R["winner_est"] = max(Se, key=lambda k: Se[k].mean())
        R["max_abs_diff"] = float(max(abs(Se[k].mean() - St[k].mean()) * 100 for k in St))
        # ranking agreement over all strategies
        rt = sorted(St, key=lambda k: -St[k].mean()); re_ = sorted(Se, key=lambda k: -Se[k].mean())
        R["rank_true"], R["rank_est"] = rt, re_
        # value of each strategy relative to no treatment: true vs analyst's estimate (the decision-relevant quantity)
        R["effect_vs_none"] = {k: {"true": stats.paired(St[k], St["none"], seed=72), "est": stats.paired(Se[k], Se["none"], seed=73)}
                               for k in St if k != "none"}
        out["regimes"][rk] = R
        arrays[f"S_true_{rk}"] = np.stack([St[k] for k in sorted(St)], 1)
        arrays[f"S_est_{rk}"] = np.stack([Se[k] for k in sorted(Se)], 1)
        mac += [(f"CL{q}MaxDiff", R["max_abs_diff"], "{:.1f}"), (f"CL{q}WinnerMatch", "yes" if R["winner_true"] == R["winner_est"] else "no", None)]
        for k, nm in [("none", "None"), ("fixed", "Fixed"), ("prn8_k1", "PRN"), ("adapt_2", "Adapt"), ("harm", "Harm")]:
            mac += [(f"CL{q}{nm}True", R["true"][k]["est"], "{:.1f}"), (f"CL{q}{nm}Est", R["est"][k]["est"], "{:.1f}")]
    arrays["labels"] = np.array(sorted(St))
    out["runtime_s"] = time.time() - t0
    out["headline"] = {QLAB[r]: {"winner_true": out["regimes"][r]["winner_true"], "winner_est": out["regimes"][r]["winner_est"],
                                 "max_abs_diff": _r(out["regimes"][r]["max_abs_diff"], 1)} for r in QUADS}
    _save(results_dir, "closed_loop", arrays, out, mac, ["R2 13", "R2 14", "R1 M7"],
          "Closed-loop validation on the shared cohort; estimated trajectories with population PKPD; independent tape.")
    return out


# ----------------------------------------------------------------------------------------
# E18  Figure data for the redrawn submitted figures (Figs 1, 4, 5, 6 of the submission)
# ----------------------------------------------------------------------------------------
def exp_figdata(results_dir):
    """Per-patient arrays the redrawn figures need, from the same g-formula preparation as exp_gformula."""
    t0 = time.time()
    c = get_cohort()
    P = get_prep("oracle")
    th = theta("gformula")
    r = gformula.evaluate(P, th)
    rct = P["u_rct"] > 0.5
    S_obs = np.where(r["A"], r["S_t"], r["S_n"])
    peak = c["lambda_base"].max(1)
    arms = {"oracle": (r["S_t"], r["S_n"]), "rct": (r["S_t"][rct], r["S_n"][~rct]),
            "naive": (S_obs[r["A"]], S_obs[~r["A"]]), "gformula": (r["G_t"], r["G_n"])}
    out = {"n": N, "arms": {k: {"treated": stats.mean_ci(a), "untreated": stats.mean_ci(b)} for k, (a, b) in arms.items()},
           "n_treated_obs": int(r["A"].sum()), "n_treated_rct": int(rct.sum()),
           "peak_mean": {"rct_treated": float(peak[rct].mean()), "rct_untreated": float(peak[~rct].mean()),
                         "obs_treated": float(peak[r["A"]].mean()), "obs_untreated": float(peak[~r["A"]].mean())},
           "ite_range": [float(np.percentile(100 * (r["G_t"] - r["G_n"]), q)) for q in (2.5, 50, 97.5)],
           "ite_share_positive": float(np.mean(r["G_t"] > r["G_n"])),
           "sofa_categories": {lab: float(100 * np.mean((np.round(c["sofa"]) >= lo) & (np.round(c["sofa"]) <= hi)))
                               for lab, lo, hi in (("0--6 (low)", 0, 6), ("7--9 (moderate)", 7, 9), ("10--12 (high)", 10, 12),
                                                   ("13 or more (very high)", 13, 99))},
           "demographics": {"age": [float(np.mean(c["age"])), float(np.median(c["age"])), float(np.std(c["age"], ddof=1))],
                            "sofa": [float(np.mean(c["sofa"])), float(np.median(c["sofa"])), float(np.std(c["sofa"], ddof=1))]}}
    # Figures 2 and 4 plot peaks up to the 99th percentile; the captions give what lies beyond
    p99 = float(np.percentile(peak, 99))
    out["peak_range"] = {"p50": float(np.median(peak)), "p99": p99, "max": float(peak.max()), "n_above_p99": int((peak > p99).sum())}
    # untreated simulated counts summarised exactly as the real cohort (Section 6): onsets per day among patients with
    # any seizure, share of 5-min bins with an onset, and patient-days by number of onsets (7 full days per patient)
    U1 = cohort_mod.uniform_tape(N, Nt, SEED, stream=1)
    un = engine.simulate({"strategy": "none"}, c["lambda_base"], c["C50"], c["gamma_hill"], U1, DRUG, CTRL, T, dt, store=("counts",))
    cnt = un["counts"][:, :int(round(TIME["Tmax"] / dt))]           # bins of [0, 7 d)
    per_day = cnt.reshape(N, 7, -1).sum(2)
    anysz = cnt.sum(1) > 0
    rate = cnt.sum(1)[anysz] / 7.0
    bins_with = 100 * (cnt[anysz] > 0).mean(1)
    pdays = per_day.ravel()
    out["sim_counts"] = {"any_pct": float(100 * anysz.mean()), "rate_med": float(np.median(rate)),
                         "rate_iqr": [float(v) for v in np.percentile(rate, [25, 75])],
                         "bins_with_med": float(np.median(bins_with)), "bins_with_iqr": [float(v) for v in np.percentile(bins_with, [25, 75])],
                         "multi_pct": float(100 * (cnt > 1).sum() / max((cnt > 0).sum(), 1)),
                         "pdays_zero": float(100 * np.mean(pdays == 0)), "pdays_few": float(100 * np.mean((pdays >= 1) & (pdays <= 4))),
                         "pdays_mid": float(100 * np.mean((pdays >= 5) & (pdays <= 24))), "pdays_many": float(100 * np.mean(pdays >= 25)),
                         "pdays_max": float(pdays.max())}
    arrays = dict(peak=peak, sofa=c["sofa"], age=c["age"], A=r["A"], rct=rct, S_t=r["S_t"], S_n=r["S_n"], G_t=r["G_t"], G_n=r["G_n"])
    ite = 100 * (r["G_t"] - r["G_n"])
    mac = [("FigITELo", out["ite_range"][0], "{:.0f}"), ("FigITEMed", out["ite_range"][1], "{:.0f}"), ("FigITEHi", out["ite_range"][2], "{:.0f}"),
           ("FigITEPosPct", 100 * out["ite_share_positive"], "{:.0f}"),
           ("FigPeakObsT", out["peak_mean"]["obs_treated"], "{:.1f}"), ("FigPeakObsU", out["peak_mean"]["obs_untreated"], "{:.1f}"),
           ("FigPeakRctT", out["peak_mean"]["rct_treated"], "{:.1f}"), ("FigPeakRctU", out["peak_mean"]["rct_untreated"], "{:.1f}"),
           ("FigNTreatedObs", out["n_treated_obs"], "{:,}"), ("FigNUntreatedObs", N - out["n_treated_obs"], "{:,}"),
           ("FigNTreatedRct", out["n_treated_rct"], "{:,}"), ("FigNUntreatedRct", N - out["n_treated_rct"], "{:,}"),
           ("DemAgeMean", out["demographics"]["age"][0], "{:.1f}"), ("DemAgeMedian", out["demographics"]["age"][1], "{:.1f}"),
           ("DemAgeSD", out["demographics"]["age"][2], "{:.1f}"), ("DemSofaMean", out["demographics"]["sofa"][0], "{:.1f}"),
           ("DemSofaMedian", out["demographics"]["sofa"][1], "{:.0f}"), ("DemSofaSD", out["demographics"]["sofa"][2], "{:.1f}"),
           ("DemSofaLowPct", out["sofa_categories"]["0--6 (low)"], "{:.1f}"),
           ("FigPeakMedian", out["peak_range"]["p50"], "{:.1f}"), ("FigPeakPNinetyNine", out["peak_range"]["p99"], "{:.0f}"),
           ("FigPeakMax", out["peak_range"]["max"], "{:.0f}"), ("FigPeakNAbove", out["peak_range"]["n_above_p99"], "{:d}")]
    sc_ = out["sim_counts"]
    mac += [("SimSzAnyPct", sc_["any_pct"], "{:.0f}"), ("SimSzDayMed", sc_["rate_med"], "{:.1f}"),
            ("SimSzDayLo", sc_["rate_iqr"][0], "{:.1f}"), ("SimSzDayHi", sc_["rate_iqr"][1], "{:.1f}"),
            ("SimBinsWithMed", sc_["bins_with_med"], "{:.1f}"), ("SimBinsWithLo", sc_["bins_with_iqr"][0], "{:.1f}"),
            ("SimBinsWithHi", sc_["bins_with_iqr"][1], "{:.1f}"), ("SimBinsMultiPct", sc_["multi_pct"], "{:.1f}"),
            ("SimPdaysZeroPct", sc_["pdays_zero"], "{:.0f}"), ("SimPdaysFewPct", sc_["pdays_few"], "{:.0f}"),
            ("SimPdaysMidPct", sc_["pdays_mid"], "{:.0f}"), ("SimPdaysManyPct", sc_["pdays_many"], "{:.0f}")]
    for k, nm in [("oracle", "Oracle"), ("rct", "RCT"), ("naive", "Naive"), ("gformula", "GF")]:
        mac += [(f"Fig{nm}SurvT", out["arms"][k]["treated"]["est"], "{:.1f}"), (f"Fig{nm}SurvU", out["arms"][k]["untreated"]["est"], "{:.1f}")]
    # threshold sweep under the default mortality model (Fig 8 of the submission), from exp_strategies
    sd = json.loads((Path(results_dir) / "rr_strategies.json").read_text())["regimes"]["default"]["surv"]
    prn = {k: sd[f"prn8_k{k}"]["est"] for k in (1, 2, 3, 4, 5)}
    ada = {w: sd[f"adapt_{w:g}"]["est"] for w in OMEGAS}
    wb = max(ada, key=ada.get); kb = max(prn, key=prn.get)
    out["sweep_default"] = {"prn": prn, "adapt": {str(k): v for k, v in ada.items()}, "none": sd["none"]["est"]}
    mac += [("SweepNone", sd["none"]["est"], "{:.1f}"), ("SweepPRNBest", prn[kb], "{:.1f}"), ("SweepPRNBestK", kb, "{:d}"),
            ("SweepPRNWorst", min(prn.values()), "{:.1f}"), ("SweepAdaptBest", ada[wb], "{:.1f}"), ("SweepAdaptBestOmega", wb, "{:g}"),
            ("SweepAdaptWorst", min(ada.values()), "{:.1f}"), ("SweepAdaptSpread", max(ada.values()) - min(ada.values()), "{:.1f}"),
            ("SweepPRNSpread", max(prn.values()) - min(prn.values()), "{:.1f}"), ("SweepAdaptTwo", ada[2], "{:.1f}")]
    out["headline"] = {"ite_range": [_r(v, 1) for v in out["ite_range"]], "peak_mean": {k: _r(v, 1) for k, v in out["peak_mean"].items()}}
    out["runtime_s"] = time.time() - t0
    _save(results_dir, "figdata", arrays, out, mac, ["R2 6", "E3"], "Arrays for redrawn Figs 1, 4, 5, 6; same preparation as exp_gformula.")
    return out


# ----------------------------------------------------------------------------------------
# E19  Parameter recovery (submitted Sec 4.2.1, Tables 3-5, Figs 2-3) on the shared cohort
# ----------------------------------------------------------------------------------------
def exp_recovery(results_dir):
    """Even-numbered patients observed on fixed 500 mg (drug inverted with their own PKPD), odd-numbered untreated."""
    from .. import estimation
    t0 = time.time()
    c = get_cohort()
    U = cohort_mod.uniform_tape(N, Nt, SEED, stream=1)
    fx = engine.simulate({"strategy": "fixed"}, c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt, store=("counts", "doses"))
    nn = engine.simulate({"strategy": "none"}, c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt, store=("counts",))
    treated = np.arange(N) % 2 == 0
    est = {k: np.zeros(N) for k in ("A_p", "r", "kappa")}
    LBh = np.zeros_like(c["lambda_base"])
    conv = np.zeros(N, bool)
    for i in range(N):
        if treated[i]:
            pe, fi = estimation.estimate_natural_history(fx["counts"][i], T, dt, peak_rate=c["peak_rate"], k_e=DRUG["k_e"],
                                                         doses_given=fx["doses"][i], pd_mode="hill",
                                                         C50=float(c["C50"][i]), gamma_hill=float(c["gamma_hill"][i]))
        else:
            pe, fi = estimation.estimate_natural_history(nn["counts"][i], T, dt, peak_rate=c["peak_rate"])
        for k in est:
            est[k][i] = pe[k]
        LBh[i] = fi["lambda_base_hat"]
        conv[i] = bool(fi["exitflag"])
    true = {"A_p": c["A_patient"], "r": c["r"], "kappa": c["kappa"]}
    der_t = {"Lcum": c["lambda_base"].sum(1) * dt, "peak": c["lambda_base"].max(1), "tpeak": T[np.argmax(c["lambda_base"], 1)]}
    der_e = {"Lcum": LBh.sum(1) * dt, "peak": LBh.max(1), "tpeak": T[np.argmax(LBh, 1)]}
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    mape = lambda a, b: float(100 * np.median(np.abs(b / a - 1)))
    out = {"n": N, "n_treated": int(treated.sum()), "converged_share": float(conv.mean()), "params": {}, "derived": {}, "population": {}}
    for k in est:
        out["params"][k] = {g: {"corr": corr(true[k][m], est[k][m]), "mdape": mape(true[k][m], est[k][m])}
                            for g, m in (("untreated", ~treated), ("treated", treated))}
        out["population"][k] = {"true": [float(np.mean(np.log(true[k]))), float(np.std(np.log(true[k]), ddof=1))],
                                "est": [float(np.mean(np.log(est[k]))), float(np.std(np.log(est[k]), ddof=1))]}
    for k in der_t:
        d = {g: corr(der_t[k][m], der_e[k][m]) for g, m in (("overall", np.ones(N, bool)), ("untreated", ~treated), ("treated", treated))}
        d["mdape"] = mape(der_t[k], der_e[k]) if k != "tpeak" else None
        d["mae_days"] = float(np.mean(np.abs(der_t[k] - der_e[k]))) if k == "tpeak" else None
        out["derived"][k] = d
    arrays = dict(treated=treated, **{f"true_{k}": v for k, v in true.items()}, **{f"est_{k}": v for k, v in est.items()},
                  **{f"true_{k}": v for k, v in der_t.items()}, **{f"est_{k}": v for k, v in der_e.items()})
    mac = [("RecConvPct", 100 * out["converged_share"], "{:.0f}"), ("RecNTreated", out["n_treated"], "{:,}"), ("RecNUntreated", N - out["n_treated"], "{:,}")]
    nm = {"A_p": "Ap", "r": "R", "kappa": "Kappa"}
    for k in est:
        for g, G in (("untreated", "Untr"), ("treated", "Tr")):
            mac += [(f"Rec{nm[k]}Corr{G}", out["params"][k][g]["corr"], "{:.2f}"), (f"Rec{nm[k]}Err{G}", out["params"][k][g]["mdape"], "{:.0f}")]
        mac += [(f"Rec{nm[k]}LogMeanTrue", out["population"][k]["true"][0], "{:.2f}"), (f"Rec{nm[k]}LogSDTrue", out["population"][k]["true"][1], "{:.2f}"),
                (f"Rec{nm[k]}LogMeanEst", out["population"][k]["est"][0], "{:.2f}"), (f"Rec{nm[k]}LogSDEst", out["population"][k]["est"][1], "{:.2f}")]
    dn = {"Lcum": "Lcum", "peak": "Peak", "tpeak": "Tpeak"}
    for k in der_t:
        for g, G in (("overall", "All"), ("untreated", "Untr"), ("treated", "Tr")):
            mac += [(f"Rec{dn[k]}Corr{G}", out["derived"][k][g], "{:.2f}")]
    mac += [("RecLcumErr", out["derived"]["Lcum"]["mdape"], "{:.0f}"), ("RecPeakErr", out["derived"]["peak"]["mdape"], "{:.0f}"),
            ("RecTpeakMAE", out["derived"]["tpeak"]["mae_days"], "{:.2f}")]
    out["headline"] = {"Ap_corr": {g: _r(out["params"]["A_p"][g]["corr"]) for g in ("untreated", "treated")},
                       "Lcum_corr": _r(out["derived"]["Lcum"]["overall"], 3)}
    out["runtime_s"] = time.time() - t0
    _save(results_dir, "recovery", arrays, out, mac, ["G2", "G3", "R1 M4c"], "Per-patient MLE on the shared cohort; error = median absolute percentage error.")
    return out


# ----------------------------------------------------------------------------------------
# E20  EKF tuning sensitivity of the tiered controller (submitted Sec 5.4, Fig 12, Table 12)
# ----------------------------------------------------------------------------------------
EKF_SWEEPS = {"q": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2], "tau_x": [0.1, 0.25, 0.5, 1.0, 2.0],
              "P_pop": [1, 2, 3, 5, 7, 10, 15, 20], "floor": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
              "omega": [0.5, 1, 1.5, 2, 3, 5, 7]}
EKF_DEFAULTS = {"q": 1e-4, "tau_x": 0.5, "P_pop": 5, "floor": 0.1, "omega": 2}


def exp_ekf_sensitivity(results_dir):
    t0 = time.time()
    c = get_cohort()
    U = cohort_mod.uniform_tape(N, Nt, SEED, stream=1)
    th = theta("default")
    sim = lambda sp: engine.simulate(sp, c["lambda_base"], c["C50"], c["gamma_hill"], U, DRUG, CTRL, T, dt)
    S = lambda o: survival.exact(o["Lday"], o["Xday"], c["age"], c["sofa"], th)
    base = {k: stats.mean_ci(S(sim({"strategy": k}))) for k in ("none", "fixed")}
    S_fixed = S(sim({"strategy": "fixed"}))
    out = {"n": N, "baselines": base, "defaults": EKF_DEFAULTS, "sweeps": {}}
    arrays = {}
    for par, vals in EKF_SWEEPS.items():
        rows = []
        for v in vals:
            ek = {k: EKF_DEFAULTS[k] for k in ("q", "tau_x", "P_pop", "floor")}
            om = EKF_DEFAULTS["omega"]
            if par == "omega":
                om = v
            else:
                ek[par] = v
            Sv = S(sim({"strategy": "adaptive", "omega": om, "ekf": ek}))
            rows.append({"value": v, "surv": stats.mean_ci(Sv), "vs_fixed": stats.paired(Sv, S_fixed, seed=91)})
        ests = [r_["surv"]["est"] for r_ in rows]
        out["sweeps"][par] = {"rows": rows, "spread": float(max(ests) - min(ests)), "min": float(min(ests)), "max": float(max(ests)),
                              "all_above_fixed": bool(all(r_["vs_fixed"]["lo"] > 0 for r_ in rows))}
    mac = [("EkfSensNone", base["none"]["est"], "{:.1f}"), ("EkfSensFixed", base["fixed"]["est"], "{:.1f}")]
    pn = {"q": "Q", "tau_x": "Tau", "P_pop": "Ppop", "floor": "Floor", "omega": "Omega"}
    for par, d in out["sweeps"].items():
        mac += [(f"EkfSens{pn[par]}Spread", d["spread"], "{:.1f}"), (f"EkfSens{pn[par]}Min", d["min"], "{:.1f}"),
                (f"EkfSens{pn[par]}Max", d["max"], "{:.1f}"),
                (f"EkfSens{pn[par]}AllAbove", "Yes" if d["all_above_fixed"] else "No", None)]
        dflt = [r_ for r_ in d["rows"] if r_["value"] == EKF_DEFAULTS[par]][0]
        mac += [(f"EkfSensDefault", dflt["surv"]["est"], "{:.1f}")] if par == "q" else []
    out["headline"] = {p: _r(d["spread"], 1) for p, d in out["sweeps"].items()}
    out["runtime_s"] = time.time() - t0
    _save(results_dir, "ekf_sensitivity", arrays, out, mac, ["G8", "R2 6"], "Tiered controller, default theta; one EKF setting varied at a time.")
    return out


# ----------------------------------------------------------------------------------------
# E21  Misspecification layers and graded SDE fraction for the g-formula (submitted Sec 5.5.1-5.5.2,
#      Tables 13-14, Fig 13), scored against the oracle effect
# ----------------------------------------------------------------------------------------
BAD_EKF = {"ekf": {"P_pop": 20, "tau_x": 2.0, "q": 1e-2, "floor": 1.0}}


def _merge_prep(Pa, Pb, take_b):
    """Patient-wise mix of two preparations of the same seed: patient i comes from Pb where take_b[i]."""
    m = np.asarray(take_b, bool)
    def mix(a, b):
        if isinstance(a, tuple):
            return tuple(mix(x, y) for x, y in zip(a, b))
        if isinstance(a, np.ndarray) and a.shape[:1] == (Pa["n"],):
            return np.where(m.reshape((-1,) + (1,) * (a.ndim - 1)), b, a)
        return a
    P = {k: mix(v, Pb[k]) for k, v in Pa.items() if k != "cf"}
    P["cf"] = {k: mix(Pa["cf"][k], Pb["cf"][k]) for k in Pa["cf"]}
    return P


def exp_misspec(results_dir, fracs=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), B=200):
    t0 = time.time()
    th = theta("gformula")
    c_rd = get_cohort(N, SEED, "rise_decay"); c_sde = get_cohort(N, SEED, "sde")
    prep = lambda c, ts=None: gformula.prepare(c, DRUG, CTRL, T, dt, SEED, treat_spec=ts)
    P = {"well": get_prep("oracle"), "ekf": prep(c_rd, BAD_EKF), "traj": prep(c_sde), "joint": prep(c_sde, BAD_EKF)}
    out = {"n": N, "bad_ekf": BAD_EKF["ekf"], "layers": {}, "graded": []}
    for k, Pk in P.items():
        out["layers"][k] = _summ(gformula.evaluate(Pk, th), gformula.bootstrap(Pk, th, B=GF_B, seed=GF_SEED))
    rng = np.random.default_rng((SEED, 404))
    order = rng.permutation(N)                     # which patients switch to the SDE first (random, nested)
    for f in fracs:
        take = np.zeros(N, bool); take[order[:int(round(f * N))]] = True
        Pm = _merge_prep(P["well"], P["traj"], take)
        s = _summ(gformula.evaluate(Pm, th), gformula.bootstrap(Pm, th, B=GF_B, seed=GF_SEED))
        s["sde_fraction"] = f
        out["graded"].append(s)
    lay = out["layers"]
    mac = []
    for k, nm in [("well", "Well"), ("ekf", "EKF"), ("traj", "Traj"), ("joint", "Joint")]:
        d = lay[k]
        mac += [(f"Lay{nm}Oracle", d["oracle_ate"], "{:.1f}"), (f"Lay{nm}GF", d["gform_ate"], "{:.1f}"),
                (f"Lay{nm}Bias", d["bias"], "{:.1f}"), (f"Lay{nm}BiasLo", d["ci"]["bias"][0], "{:.1f}"),
                (f"Lay{nm}BiasHi", d["ci"]["bias"][1], "{:.1f}"), (f"Lay{nm}RelBias", 100 * d["bias"] / d["oracle_ate"], "{:.0f}")]
    inter = lay["joint"]["bias"] - lay["ekf"]["bias"] - lay["traj"]["bias"] + lay["well"]["bias"]
    mac += [("LayInteraction", inter, "{:.1f}")]
    g0, g1 = out["graded"][0], out["graded"][-1]
    absb = [abs(g["bias"]) for g in out["graded"]]
    mac += [("GrOracleZero", g0["oracle_ate"], "{:.1f}"), ("GrOracleAll", g1["oracle_ate"], "{:.1f}"),
            ("GrGFZero", g0["gform_ate"], "{:.1f}"), ("GrGFAll", g1["gform_ate"], "{:.1f}"),
            ("GrBiasZero", g0["bias"], "{:.1f}"), ("GrBiasAll", g1["bias"], "{:.1f}"),
            ("GrMaxAbsBias", max(absb), "{:.1f}"), ("GrMaxAbsBiasFrac", 100 * out["graded"][int(np.argmax(absb))]["sde_fraction"], "{:.0f}"),
            ("GrCICoverCount", sum(g["ci"]["bias"][0] <= 0 <= g["ci"]["bias"][1] for g in out["graded"]), "{:d}")]
    out["interaction"] = inter
    out["headline"] = {"layers_bias": {k: _r(v["bias"], 1) for k, v in lay.items()},
                       "graded_bias": {str(g["sde_fraction"]): _r(g["bias"], 1) for g in out["graded"]}}
    out["runtime_s"] = time.time() - t0
    _save(results_dir, "misspec", {}, out, mac, ["G1", "R2 14", "R1 M2"],
          "Oracle ATE truth; layer A = miscalibrated EKF in the deployed strategy; B = CICADAS SDE natural history; graded = random nested SDE subsets.")
    return out


ALL = {"strategies": exp_strategies, "gformula": exp_gformula, "confounding": exp_confounding, "overlap": exp_overlap,
       "shapes": exp_shapes, "personalization": exp_personalization, "levelset": exp_levelset, "ekf": exp_ekf,
       "misclass": exp_misclass, "obs_errors": exp_obs_errors, "dgp": exp_dgp, "refutation": exp_refutation, "window": exp_window,
       "gformula_scaling": exp_gformula_scaling, "graded_lcum": exp_graded_lcum,
       "harm": exp_harm, "closed_loop": exp_closed_loop, "figdata": exp_figdata, "recovery": exp_recovery,
       "ekf_sensitivity": exp_ekf_sensitivity, "misspec": exp_misspec}
