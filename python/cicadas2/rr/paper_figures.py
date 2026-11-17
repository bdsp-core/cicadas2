"""The submitted paper figures, redrawn from the revision's results files (R2 6, E3, G1-G4, G8).

  python -m cicadas2.rr.paper_figures            # all figures
  python -m cicadas2.rr.paper_figures 2 3 13     # only some

Style is copied from make_paper_figures.py (the submitted figures): seaborn "ticks", Helvetica, Okabe-Ito
strategy colours, bold value labels. Data now come from results/rr_*.json / .npz, the same files that
produce the LaTeX macros and tables, so text, tables and figures cannot disagree. Every survival figure
shows 95% confidence intervals instead of the submitted standard errors.

Output: CICADAS_II_Paper_Draft/figures_rr/paper_figNN_<name>.pdf (the submitted figures/ folder is left
untouched for the tracked-changes comparison).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .theta import QUADS, REGISTRY  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results"
OUT = ROOT / "CICADAS_II_Paper_Draft" / "figures_rr"

# Helvetica on the Mac; Liberation Sans (metric-compatible TrueType) where Helvetica is not installed
sns.set_theme(context="paper", style="ticks", font_scale=1.0, rc={
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Liberation Sans", "Arial", "DejaVu Sans"],
    "figure.titlesize": 13, "figure.titleweight": "bold", "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "legend.title_fontsize": 9, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.15,
    "grid.linewidth": 0.4, "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "dejavusans"})

FS_VALUE_LABEL = 8.5
FS_INLINE = 8.5
FS_PANEL_HDR = 11
STRAT_COLORS = {"None": "#7F7F7F", "Fixed": "#0072B2", "PRN": "#D55E00", "Adaptive": "#009E73",
                "Harm-bal.": "#CC79A7", "Fixed 750": "#56B4E9"}
COHORT_COLORS = {"untreated": "#0072B2", "treated": "#D55E00"}
LAB = {q: REGISTRY[q]["label"] for q in QUADS}


def _j(name):
    return json.loads((RES / f"rr_{name}.json").read_text())


def _z(name):
    return np.load(RES / f"rr_{name}.npz", allow_pickle=True)


def savepdf(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("  saved", OUT / name)


def annotate_value(ax, x, y, value, *, dy=2, fontsize=FS_VALUE_LABEL, weight="bold"):
    ax.text(x, y + dy, value, ha="center", va="bottom", fontsize=fontsize, fontweight=weight)


def identity_line(ax, lim, **kw):
    kw.setdefault("color", "0.3"); kw.setdefault("linestyle", "--"); kw.setdefault("linewidth", 1); kw.setdefault("zorder", 0)
    ax.plot(lim, lim, **kw)


def _err(d):
    """Asymmetric 95% CI half-widths from a {est, lo, hi} dict."""
    return [d["est"] - d["lo"], d["hi"] - d["est"]]


# ---------------------------------------------------------------------------------------------
# Figure 1: demographics of the shared cohort
# ---------------------------------------------------------------------------------------------
def fig01_demographics():
    z = _z("figdata")
    ages, sofas = z["age"], np.round(z["sofa"]).astype(int)
    n = len(ages)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4))
    sns.kdeplot(ages, ax=axes[0], color=COHORT_COLORS["untreated"], linewidth=2.2, clip=(18, 100), fill=True, alpha=0.18)
    axes[0].set(xlabel="Age (years)", ylabel="Density", title=f"Age distribution (N = {n:,})", xlim=(18, 100))
    axes[0].text(0.97, 0.95, f"mean {ages.mean():.1f}\nmedian {np.median(ages):.1f}\nSD {ages.std(ddof=1):.1f}",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=9,
                 bbox=dict(facecolor="white", edgecolor="0.8", boxstyle="round,pad=0.4"))
    pct = np.bincount(sofas, minlength=25)[:25] / n * 100
    axes[1].bar(np.arange(25), pct, color=COHORT_COLORS["treated"], alpha=0.78, edgecolor="white", linewidth=0.6)
    axes[1].set(xlabel="SOFA score", ylabel="Patients (%)", title=f"SOFA distribution (N = {n:,})", xlim=(-0.5, 14.5))
    axes[1].text(0.97, 0.95, f"mean {sofas.mean():.1f}\nmedian {int(np.median(sofas))}\nSD {sofas.std(ddof=1):.1f}",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=9,
                 bbox=dict(facecolor="white", edgecolor="0.8", boxstyle="round,pad=0.4"))
    savepdf(fig, "paper_fig01_demographics.pdf")


# ---------------------------------------------------------------------------------------------
# Figures 2-3: parameter and derived-quantity recovery
# ---------------------------------------------------------------------------------------------
def fig02_param_recovery():
    z = _z("recovery")
    tx = z["treated"].astype(bool)
    params = [("A_p", r"$A_p$"), ("r", r"$r$"), ("kappa", r"$\kappa$")]
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 5.0), sharex="col", sharey="col")
    for col, (k, label) in enumerate(params):
        t, e = z[f"true_{k}"], z[f"est_{k}"]
        # axis set by the true values; estimates beyond it are drawn at the edge (stated in the caption)
        lim = (0, (np.concatenate([t, e]).max() * 1.05) if col == 0 else t.max() * 1.6)
        for row, (idx, color, name) in enumerate([(~tx, COHORT_COLORS["untreated"], "untreated"),
                                                  (tx, COHORT_COLORS["treated"], "treated")]):
            ax = axes[row, col]
            r = np.corrcoef(t[idx], e[idx])[0, 1]
            ax.scatter(t[idx], np.clip(e[idx], None, lim[1]), s=10, alpha=0.45, color=color, edgecolor="white", linewidth=0.3)
            identity_line(ax, lim)
            ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{label} ({name})  r = {r:.2f}")
            if col == 0:
                ax.set_ylabel(f"Estimated {label}")
            if row == 1:
                ax.set_xlabel(f"True {label}")
    fig.tight_layout()
    savepdf(fig, "paper_fig02_param_recovery.pdf")


def fig03_derived_recovery():
    z = _z("recovery")
    tx = z["treated"].astype(bool)
    trip = [("Lcum", r"$L_{\mathrm{cum}}$", r"$L_{\mathrm{cum}}$ (sz$\cdot$day)", None, 3),
            ("peak", "Peak height", "Peak height (sz/day)", None, 3),
            ("tpeak", "Peak time", "Peak time (days)", (0, 3.5), 2)]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.6))
    for ax, (k, name, axlabel, clip, dig) in zip(axes, trip):
        t, e = z[f"true_{k}"], z[f"est_{k}"]
        r = np.corrcoef(t, e)[0, 1]
        lim = clip if clip is not None else (0, np.concatenate([t, e]).max() * 1.05)
        ax.scatter(t[~tx], e[~tx], s=11, alpha=0.45, color=COHORT_COLORS["untreated"], edgecolor="white", linewidth=0.3, label="Untreated")
        ax.scatter(t[tx], e[tx], s=11, alpha=0.45, color=COHORT_COLORS["treated"], edgecolor="white", linewidth=0.3, label="Treated")
        identity_line(ax, lim)
        ax.set(xlim=lim, ylim=lim, xlabel=f"True {axlabel}", ylabel=f"Estimated {axlabel}", title=f"{name}  r = {r:.{dig}f}")
        ax.set_aspect("equal", adjustable="box")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    savepdf(fig, "paper_fig03_derived_recovery.pdf")


# ---------------------------------------------------------------------------------------------
# Figure 4: confounding by indication; Figure 5: method comparison; Figure 6: effect by severity
# ---------------------------------------------------------------------------------------------
def fig04_confounding():
    z = _z("figdata")
    peak, rct, obs = z["peak"], z["rct"].astype(bool), z["A"].astype(bool)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.4))
    bins = np.linspace(0, np.percentile(peak, 99), 30)

    def panel(ax, m, title):
        ax.hist(peak[~m], bins=bins, color=STRAT_COLORS["None"], alpha=0.55, edgecolor="white", linewidth=0.4,
                label=f"Untreated (n={(~m).sum()})")
        ax.hist(peak[m], bins=bins, color=STRAT_COLORS["Adaptive"], alpha=0.65, edgecolor="white", linewidth=0.4,
                label=f"Treated (n={m.sum()})")
        ax.axvline(peak[m].mean(), color=STRAT_COLORS["Adaptive"], linestyle=":", linewidth=1.5)
        ax.axvline(peak[~m].mean(), color=STRAT_COLORS["None"], linestyle=":", linewidth=1.5)
        ax.set(xlabel="Peak seizure rate (sz/day)", ylabel="Patients", title=title)
        ax.legend(loc="upper right")
    panel(axes[0], rct, "Randomized assignment")
    panel(axes[1], obs, "Observational assignment")
    fig.tight_layout()
    savepdf(fig, "paper_fig04_confounding.pdf")


def fig04b_method_comparison():
    f = _j("figdata")["arms"]
    g = _j("gformula")["oracle"]
    methods = [("oracle", "Oracle (truth)", g["oracle_ate"], g["ci"]["oracle"]),
               ("rct", "Simulated RCT", g["rct_ate"], None),
               ("naive", "Naive obs.", g["naive_ate"], g["ci"]["naive"]),
               ("gformula", "G-formula\n(fitted mortality model)", g["gform_ate"], g["ci"]["gform"])]
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    x = np.arange(len(methods)); w = 0.36
    U = [f[k]["untreated"] for k, *_ in methods]; Tr = [f[k]["treated"] for k, *_ in methods]
    ax.bar(x - w / 2, [d["est"] for d in U], w, yerr=np.array([_err(d) for d in U]).T, capsize=3,
           error_kw=dict(ecolor="0.25", lw=1), color=STRAT_COLORS["None"], edgecolor="white", linewidth=0.8, label="Untreated")
    ax.bar(x + w / 2, [d["est"] for d in Tr], w, yerr=np.array([_err(d) for d in Tr]).T, capsize=3,
           error_kw=dict(ecolor="0.25", lw=1), color=STRAT_COLORS["Adaptive"], edgecolor="white", linewidth=0.8, label="Treated")
    for xi, (u, t) in enumerate(zip(U, Tr)):
        annotate_value(ax, xi - w / 2, u["hi"], f"{u['est']:.1f}", dy=1.0)
        annotate_value(ax, xi + w / 2, t["hi"], f"{t['est']:.1f}", dy=1.0)
    ymax = max(d["hi"] for d in U + Tr)
    for xi, (_, _, ate, ci) in enumerate(methods):
        txt = f"ATE = {ate:+.1f} pp" + (f"\n[{ci[0]:+.1f}, {ci[1]:+.1f}]" if ci else "")
        ax.text(xi, ymax + 10, txt, ha="center", va="bottom", fontsize=FS_INLINE, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([m[1] for m in methods])
    ax.set_ylabel("7-day survival (%)"); ax.set_ylim(0, ymax + 28)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    savepdf(fig, "paper_fig04b_method_comparison.pdf")


def fig05_te_by_severity():
    z = _z("figdata")
    g = _j("gformula")["oracle"]
    peak, sofa = z["peak"], z["sofa"]
    ite = 100 * (z["G_t"] - z["G_n"])
    ate, (lo, hi) = g["gform_ate"], g["ci"]["gform"]
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    sc = ax.scatter(peak, ite, c=sofa, s=11, alpha=0.55, cmap="viridis", edgecolor="none")
    ax.axhline(0, color="0.3", linestyle="--", linewidth=1)
    ax.axhspan(lo, hi, color=STRAT_COLORS["Adaptive"], alpha=0.15)
    ax.axhline(ate, color=STRAT_COLORS["Adaptive"], linewidth=2, label=f"G-formula ATE = {ate:+.1f} pp  (95% CI [{lo:+.1f}, {hi:+.1f}])")
    ax.axhline(g["oracle_ate"], color="0.15", linewidth=1.2, linestyle="-.", label=f"Oracle ATE = {g['oracle_ate']:+.1f} pp")
    ax.set_xlim(0, np.percentile(peak, 99))
    ax.set(xlabel="Peak seizure rate (sz/day)", ylabel="Individual treatment effect, adaptive vs none (pp)")
    cbar = fig.colorbar(sc, ax=ax, pad=0.015); cbar.set_label("SOFA score")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="0.8")
    fig.tight_layout()
    savepdf(fig, "paper_fig05_te_by_severity.pdf")


# ---------------------------------------------------------------------------------------------
# Figure 7: phenotype-dependent strategy preference (now with 95% CIs and the harm-balanced controller)
# ---------------------------------------------------------------------------------------------
def _bars(ax, vals, names, colors, *, best_marker=True, ylim=(0, 100), label_fmt="{:.0f}", dy=1.5):
    est = np.array([v["est"] for v in vals])
    yerr = np.array([_err(v) for v in vals]).T
    bars = ax.bar(names, est, color=colors, edgecolor="white", linewidth=0.8, width=0.7,
                  yerr=yerr, capsize=3, error_kw=dict(ecolor="0.25", lw=1))
    ax.set_ylim(*ylim); ax.grid(axis="x", visible=False)
    if best_marker:
        b = bars[int(np.argmax(est))]; b.set_edgecolor("#222"); b.set_linewidth(1.8)
    for i, v in enumerate(vals):
        annotate_value(ax, i, v["hi"], label_fmt.format(v["est"]), dy=dy)
    return bars


def _harm_ci(q):
    from . import stats
    z = _z("harm")
    lab = list(z["labels"])
    return stats.mean_ci(z[f"S_{q}"][:, lab.index("harm_est")])


def fig06_quadrant():
    s = _j("strategies")["regimes"]
    keys = [("none", "None"), ("fixed", "Fixed"), ("prn8_k1", "PRN"), ("adapt_2", "Adaptive")]
    names = [k[1] for k in keys] + ["Harm-bal."]
    colors = [STRAT_COLORS[n] for n in names]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.9))
    for q, ax in zip(QUADS, axes.flat):
        vals = [s[q]["surv"][k] for k, _ in keys] + [_harm_ci(q)]
        _bars(ax, vals, names, colors)
        ib = int(np.argmax([v["est"] for v in vals]))
        ax.set_title(f"{LAB[q]}\nbest: {names[ib]} ({vals[ib]['est']:.0f}%)")
        ax.set_ylabel("7-day survival (%)")
        ax.tick_params(axis="x", labelsize=8.5)
        for tl in ax.get_xticklabels():
            tl.set_rotation(25); tl.set_horizontalalignment("right")
    axes[0, 0].text(0.02, 0.97, f"n = {_j('strategies')['n']:,} patients\nError bars: 95% CI",
                    transform=axes[0, 0].transAxes, fontsize=8.5, color="0.3", va="top", ha="left")
    fig.tight_layout()
    savepdf(fig, "paper_fig06_quadrant.pdf")


# ---------------------------------------------------------------------------------------------
# Figure 8: threshold sweep under the default mortality model
# ---------------------------------------------------------------------------------------------
def fig07_threshold_sweep():
    s = _j("strategies")["regimes"]["default"]["surv"]
    prn = [(k, s[f"prn8_k{k}"]) for k in (1, 2, 3, 4, 5)]
    om = [(w, s[f"adapt_{w:g}"]) for w in (0.125, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8)]
    base = s["none"]["est"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.4), sharey=True)
    panels = [(prn, STRAT_COLORS["PRN"], "PRN: optimal count = {0:g}", "PRN trigger (seizures counted in preceding 8 h)", False),
              (om, STRAT_COLORS["Adaptive"], r"Adaptive: optimal $\omega$ = {0:g}", r"Adaptive dose threshold $\omega$ (sz/day)", True)]
    for ax, (pts, color, title, xlabel, logx) in zip(axes, panels):
        xs = np.array([p[0] for p in pts], float); d = [p[1] for p in pts]
        ys = np.array([v["est"] for v in d]); err = np.array([_err(v) for v in d]).T
        ax.errorbar(xs, ys, yerr=err, fmt="o-", color=color, markerfacecolor=color, markeredgecolor="white",
                    markersize=7, linewidth=1.8, capsize=3, elinewidth=1)
        if logx:
            ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels([f"{v:g}" for v in xs])
            ax.minorticks_off()
        ax.axhline(base, color="0.5", linestyle="--", linewidth=1)
        ax.text(xs[-1], base + 1, "no treatment", va="bottom", ha="right", fontsize=FS_INLINE, color="0.5")
        i = int(np.argmax(ys))
        ax.axvline(xs[i], color="0.4", linestyle=":", linewidth=1)
        ax.annotate("optimum", xy=(xs[i], ys[i]), xytext=(8, 6), textcoords="offset points", fontsize=8.5, color="0.3", fontweight="bold")
        ax.set(xlabel=xlabel, title=title.format(xs[i]))
    axes[0].set_ylabel("7-day survival (%)")
    axes[0].text(0.97, 0.97, "Error bars: 95% CI", transform=axes[0].transAxes, fontsize=FS_INLINE, color="0.4", ha="right", va="top")
    fig.tight_layout()
    savepdf(fig, "paper_fig07_threshold_sweep.pdf")


# ---------------------------------------------------------------------------------------------
# Figures 9-10: personalization; uniform vs fixed-optimal vs dynamic omega
# ---------------------------------------------------------------------------------------------
def _grouped(ax, rows, series, colors, *, gain_from=0, min_gain=0.0):
    n = len(rows); x = np.arange(n); w = 0.26
    for i, (lab, col) in enumerate(zip(series, colors)):
        est = np.array([r[i]["est"] for r in rows]); err = np.array([_err(r[i]) for r in rows]).T
        bars = ax.bar(x + (i - 1) * w, est, w, yerr=err, capsize=2.5, error_kw=dict(ecolor="0.25", lw=0.9),
                      label=lab, color=col, edgecolor="white", linewidth=0.7)
        if i != gain_from:
            for j, b in enumerate(bars):
                gain = rows[j][i]["est"] - rows[j][gain_from]["est"]
                if abs(gain) >= min_gain:
                    annotate_value(ax, b.get_x() + b.get_width() / 2, rows[j][i]["hi"], f"{gain:+.1f}", dy=1, fontsize=FS_INLINE)
    return x


def fig08_personalized():
    p = _j("personalization")["regimes"]
    rows = [[p[q]["uniform"], p[q]["personal_true"], p[q]["personal_est"]] for q in QUADS]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = _grouped(ax, rows, [r"Uniform ($\omega = 2$)", r"Personalized (true $\lambda$)", r"Personalized (estimated $\lambda$)"],
                 ["#9A9A9A", "#4FAD5B", "#4878CF"])
    ax.set(xticks=x, xticklabels=[LAB[q] for q in QUADS], ylabel="7-day survival (%)", ylim=(0, 100))
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    savepdf(fig, "paper_fig08_personalized.pdf")


def fig09_uniform_vs_dynamic():
    s = _j("strategies")["regimes"]
    rows = [[s[q]["table10_test"]["uniform"], s[q]["table10_test"]["fixed_best"], s[q]["table10_test"]["dynamic"]] for q in QUADS]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = _grouped(ax, rows, [r"Uniform ($\omega = 2$)", r"Fixed $\omega^*$ per regime", r"Dynamic $\omega(t)$"],
                 ["#9A9A9A", "#4FAD5B", "#4878CF"], min_gain=0.5)
    ax.set(xticks=x, xticklabels=[LAB[q] for q in QUADS], ylabel="7-day survival (%)", ylim=(0, 100))
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    savepdf(fig, "paper_fig09_uniform_vs_dynamic.pdf")


# ---------------------------------------------------------------------------------------------
# Figure 11: closed-loop validation (true trajectory top, analyst's estimate bottom)
# ---------------------------------------------------------------------------------------------
CL_STRATS = [("none", "None"), ("fixed", "Fixed"), ("fixed_750", "Fixed 750"), ("prn8_k1", "PRN"),
             ("adapt_2", "Adaptive"), ("harm", "Harm-bal.")]


def fig10_closed_loop():
    c = _j("closed_loop")
    names = [n for _, n in CL_STRATS]
    colors = [STRAT_COLORS[n] for n in names]
    fig, axes = plt.subplots(2, 4, figsize=(6.9, 5.8), sharey=True)
    for col, q in enumerate(QUADS):
        for row, src in enumerate(("true", "est")):
            ax = axes[row, col]
            vals = [c["regimes"][q][src][k] for k, _ in CL_STRATS]
            _bars(ax, vals, names, colors, best_marker=True, label_fmt="{:.0f}", dy=1.0)
            for t in list(ax.texts):
                t.remove()                       # bars too narrow for value labels; the table gives the numbers
            ib = int(np.argmax([v["est"] for v in vals]))
            ax.set_title(f"best: {names[ib]} ({vals[ib]['est']:.0f}%)", fontsize=8.5)
            ax.set_ylabel(("True" if row == 0 else "Estimated") + "\n7-day survival (%)" if col == 0 else "")
            ax.tick_params(axis="x", labelsize=7.5)
            for tl in ax.get_xticklabels():
                tl.set_rotation(45); tl.set_horizontalalignment("right")
            ax.set_ylim(0, 112)
        axes[0, col].text(0.5, 1.22, LAB[q].replace(" / ", " /\n"), ha="center", va="bottom", transform=axes[0, col].transAxes,
                          fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    savepdf(fig, "paper_fig10_closed_loop.pdf")


# ---------------------------------------------------------------------------------------------
# Figure 12: EKF tuning sensitivity (legend and caption now match the markers, G8)
# ---------------------------------------------------------------------------------------------
def fig11_ekf_sensitivity():
    e = _j("ekf_sensitivity")
    base_none, base_fix = e["baselines"]["none"]["est"], e["baselines"]["fixed"]["est"]
    order = [("q", "q (process noise)", "q"), ("tau_x", r"$\tau_x$ (mean-reversion, days)", r"$\tau_x$"),
             ("P_pop", r"$\mathcal{P}_{\mathrm{pop}}$ (prior peak, sz/day)", r"$\mathcal{P}_{\mathrm{pop}}$"),
             ("floor", "floor (prior minimum)", "floor"), ("omega", r"$\omega$ (dose threshold, sz/day)", r"$\omega$")]
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 7.4))
    for (par, title, short), ax in zip(order, axes.flat):
        rows = e["sweeps"][par]["rows"]
        xs = np.array([r["value"] for r in rows], float); ys = np.array([r["surv"]["est"] for r in rows])
        err = np.array([_err(r["surv"]) for r in rows]).T
        ax.errorbar(xs, ys, yerr=err, fmt="o-", color=STRAT_COLORS["Adaptive"], markerfacecolor=STRAT_COLORS["Adaptive"],
                    markeredgecolor="white", markersize=7, linewidth=1.8, capsize=2.5, elinewidth=0.9)
        if par in ("q", "floor"):
            ax.set_xscale("log")
        ax.axhline(base_none, color=STRAT_COLORS["PRN"], linestyle="--", linewidth=1.3)
        ax.axhline(base_fix, color="0.3", linestyle="--", linewidth=1.3)
        i_def = int(np.argmin(np.abs(xs - e["defaults"][par])))
        ax.scatter(xs[i_def], ys[i_def], s=110, facecolor="none", edgecolor=STRAT_COLORS["PRN"], linewidth=2, zorder=4)
        ax.set_title(title, fontsize=9.5); ax.set_ylabel("7-day survival (%)"); ax.set_xlabel(short)
    ax = axes.flat[-1]
    spreads = np.array([e["sweeps"][p]["spread"] for p, *_ in order]); idx = np.argsort(spreads)
    ax.barh(np.arange(len(order)), spreads[idx], color=STRAT_COLORS["Adaptive"], edgecolor="white", linewidth=0.6)
    ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([order[i][2] for i in idx])
    ax.set_xlabel("Survival spread (pp)"); ax.set_title("Sensitivity ranking", fontsize=9.5); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, spreads.max() * 1.25)
    for i, v in enumerate(spreads[idx]):
        ax.text(v + 0.4, i, f"{v:.1f}", va="center", fontsize=FS_INLINE)
    handles = [Line2D([0], [0], marker="o", linestyle="-", color=STRAT_COLORS["Adaptive"], markerfacecolor=STRAT_COLORS["Adaptive"],
                      markeredgecolor="white", markersize=7, label="Tiered adaptive (95% CI)"),
               Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=STRAT_COLORS["PRN"],
                      markeredgewidth=2, markersize=10, label="Default value"),
               Line2D([0], [0], color="0.3", linestyle="--", label="Fixed 500 mg"),
               Line2D([0], [0], color=STRAT_COLORS["PRN"], linestyle="--", label="No treatment")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    savepdf(fig, "paper_fig11_ekf_sensitivity.pdf")


# ---------------------------------------------------------------------------------------------
# Figure 13: graded trajectory misspecification (G1: figure and table now come from one run)
# ---------------------------------------------------------------------------------------------
def fig12_graded_misspec():
    g = _j("misspec")["graded"]
    mix = np.array([r["sde_fraction"] for r in g]) * 100
    orc = np.array([r["oracle_ate"] for r in g]); gf = np.array([r["gform_ate"] for r in g])
    olo, ohi = np.array([r["ci"]["oracle"] for r in g]).T; glo, ghi = np.array([r["ci"]["gform"] for r in g]).T
    b = np.array([r["bias"] for r in g]); blo, bhi = np.array([r["ci"]["bias"] for r in g]).T
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 6.4), sharex=True, gridspec_kw={"height_ratios": [1, 0.85], "hspace": 0.10})
    ax1.fill_between(mix, olo, ohi, color="0.2", alpha=0.10, lw=0)
    ax1.plot(mix, orc, "o-", color="0.2", linewidth=2, markersize=6, markeredgecolor="white", label="Oracle ATE (truth)")
    ax1.fill_between(mix, glo, ghi, color=STRAT_COLORS["PRN"], alpha=0.12, lw=0)
    ax1.plot(mix, gf, "s--", color=STRAT_COLORS["PRN"], linewidth=1.8, markersize=6, markeredgecolor="white", label="G-formula ATE")
    ax1.set_ylabel("ATE (percentage points)")
    ax1.legend(loc="upper right")
    ax2.axhline(0, color="0.3", linewidth=1)
    ax2.fill_between(mix, blo, bhi, color=STRAT_COLORS["Fixed"], alpha=0.18, lw=0)
    ax2.plot(mix, b, "o-", color=STRAT_COLORS["Fixed"], linewidth=2, markersize=6, markeredgecolor="white",
             label="G-formula bias (estimate minus truth)")
    ax2.set_xlabel("Fraction of SDE-generated patients (%)")
    ax2.set_ylabel("Bias (pp)")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="0.8")
    fig.tight_layout()
    savepdf(fig, "paper_fig12_graded_misspec.pdf")


# ---------------------------------------------------------------------------------------------
# Revision figures in the same style: hidden confounder, overlap, harm-balanced controller
# ---------------------------------------------------------------------------------------------
def fig_hidden_confounding():
    c = _j("confounding")
    k = np.array([r["kappa"] for r in c["rows"]], float)
    b = np.array([r["bias"] for r in c["rows"]])
    lo = np.array([r["ci"]["bias"][0] for r in c["rows"]]); hi = np.array([r["ci"]["bias"][1] for r in c["rows"]])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.axhline(0, color="0.3", linestyle="--", linewidth=1)
    ax.fill_between(k, lo, hi, color=STRAT_COLORS["Fixed"], alpha=0.18, lw=0)
    ax.plot(k, b, "o-", color=STRAT_COLORS["Fixed"], linewidth=2, markersize=6, markeredgecolor="white", label="G-formula bias (95% CI)")
    ax.set(xlabel=r"Strength of hidden confounder $\kappa_U$", ylabel="Bias, estimate minus truth (pp)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    savepdf(fig, "fig_confounding.pdf")


def fig_overlap():
    z = _z("overlap")
    ps, A = z["ps_hat"], z["A"].astype(bool)
    bins = np.linspace(0, 1, 26)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.hist(ps[~A], bins=bins, density=True, color=STRAT_COLORS["None"], alpha=0.55, edgecolor="white", linewidth=0.4,
            label=f"Untreated (n = {(~A).sum()})")
    ax.hist(ps[A], bins=bins, density=True, color=STRAT_COLORS["Adaptive"], alpha=0.55, edgecolor="white", linewidth=0.4,
            label=f"Treated (n = {A.sum()})")
    ax.axvline(0.95, color="0.3", linestyle="--", linewidth=1)
    ax.set(xlabel="Estimated probability of treatment (propensity score)", ylabel="Density (area 1 per group)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    savepdf(fig, "fig_overlap.pdf")


def fig_harm_forest():
    from . import stats
    h = _j("harm"); z = _z("harm")
    labels = list(z["labels"]); test = np.arange(z["S_default"].shape[0]) % 2 == 1
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 3.0), sharey=True)
    rows = [("No treatment", "none"), ("PRN, tuned", None), ("Tiered adaptive, tuned", None), ("Harm-balanced", "harm_est"),
            ("Harm-balanced, oracle", "harm_oracle_full")]
    for ax, q in zip(axes, QUADS):
        t = h["regimes"][q]["tuned"]; S = z[f"S_{q}"][test]
        col = lambda k: S[:, labels.index(k)]
        ref = col(t["fixed"])
        keys = ["none", t["prn"], t["adapt"], "harm_est", "harm_oracle_full"]
        y = np.arange(len(rows))[::-1]
        ax.axvline(0, color="0.3", linewidth=1)
        for yi, (lab, _), k in zip(y, rows, keys):
            d = stats.paired(col(k), ref, seed=81)
            harm = k.startswith("harm")
            c = STRAT_COLORS["Harm-bal."] if harm else "0.35"
            ax.plot([d["lo"], d["hi"]], [yi, yi], color=c, lw=1.6, solid_capstyle="round")
            ax.plot(d["est"], yi, "o", ms=6 if harm else 5, color=c, mfc="white" if k == "harm_oracle_full" else c, mew=1.4)
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
        ax.set_title(f"{LAB[q].replace(' / ', ' /' + chr(10))}\nref.: fixed {t['fixed'].split('_')[1]} mg", fontsize=9)
        ax.grid(axis="y", visible=False)
    fig.supxlabel("Survival difference from the tuned fixed dose (pp, 95% paired CI)", fontsize=9.5, y=0.0)
    fig.tight_layout()
    savepdf(fig, "fig_harm_controller.pdf")


DISPATCH = {1: fig01_demographics, 2: fig02_param_recovery, 3: fig03_derived_recovery, 4: fig04_confounding,
            13: fig04b_method_comparison, 5: fig05_te_by_severity, 6: fig06_quadrant, 7: fig07_threshold_sweep,
            8: fig08_personalized, 9: fig09_uniform_vs_dynamic, 10: fig10_closed_loop, 11: fig11_ekf_sensitivity,
            12: fig12_graded_misspec, 21: fig_hidden_confounding, 22: fig_overlap, 23: fig_harm_forest}


def main(argv=None):
    want = [int(a) for a in (argv if argv is not None else sys.argv[1:])] or sorted(DISPATCH)
    for k in want:
        if k == 12 and not (RES / "rr_misspec.json").exists():
            print("  skip 12 (rr_misspec.json missing)"); continue
        DISPATCH[k]()


if __name__ == "__main__":
    main()
