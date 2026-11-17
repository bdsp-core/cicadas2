"""Generate all 12 figures in CICADAS_II.pdf from cached MATLAB simulation results.

Inputs : results/*.mat (cached outputs of the s0..s16 MATLAB scripts)
Outputs: Figures_py/paper_figNN_*.pdf  (vector PDFs via matplotlib)

Why a Python port:
  - MATLAB's default aesthetic is dated; matplotlib + seaborn produces a
    cleaner, journal-ready look with no extra effort.
  - Iteration is fast (~seconds) because no simulation is re-run.
  - The same results/*.mat files drive both make_paper_figures.m and
    this script, so the two pipelines stay in sync.

Run:
    python make_paper_figures.py            # all figures
    python make_paper_figures.py --figs 1,6 # only specific figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Global aesthetic — applied once, then every figure inherits it
# ---------------------------------------------------------------------------
sns.set_theme(
    context="paper",
    style="ticks",
    font="Helvetica",
    # font_scale acts as a multiplier on every text size below; keep 1.0 and
    # set each size explicitly so the hierarchy is visible at a glance.
    font_scale=1.0,
    rc={
        # Type hierarchy — set once, used everywhere.
        # Suptitle  > axes title > axis label > tick label > legend / annotation
        "figure.titlesize":   13,
        "figure.titleweight": "bold",
        "axes.titlesize":     11,
        "axes.titleweight":   "bold",
        "axes.labelsize":     10,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "legend.fontsize":      9,
        "legend.title_fontsize": 9,
        "font.size":            9,   # default for ax.text / ax.annotate
        # Layout / aesthetic
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.15,
        "grid.linewidth":    0.4,
        "legend.frameon":    False,
        # Output
        "figure.dpi":  300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,   # TrueType (editable in Illustrator)
        "ps.fonttype":  42,
    },
)

# Standard sizes for in-axes overrides — only use these when rc default is wrong
FS_VALUE_LABEL = 8.5   # numbers above bars
FS_INLINE      = 8.5   # small inline notes (n=, "Error bars: SEM", "optimum")
FS_PANEL_HDR   = 11    # header text above grouped panels (e.g. fig 10)

# Okabe-Ito colorblind-safe palette — distinguishable for deuteranopia and
# protanopia, unlike the more common red/green pairing.
#   None     — neutral gray
#   Fixed    — sky blue           (cool, mechanical)
#   PRN      — vermilion          (reactive)
#   Adaptive — bluish green       (controlled, healthy)
STRAT_COLORS = {
    "None":     "#7F7F7F",
    "Fixed":    "#0072B2",
    "PRN":      "#D55E00",
    "Adaptive": "#009E73",
}
STRAT_LIST = ["None", "Fixed", "PRN", "Adaptive"]

# Untreated / treated cohort colors (Okabe-Ito)
COHORT_COLORS = {"untreated": "#0072B2", "treated": "#D55E00"}

REPO = Path(__file__).resolve().parent
RESULTS_DIR = REPO / "results"
OUT_DIR = REPO / "Figures_py"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_mat(name: str) -> dict:
    S = sio.loadmat(RESULTS_DIR / name, squeeze_me=True, struct_as_record=False)
    # Backward compatibility: the decay-rate variable was renamed from `d`
    # to `kappa` to match the manuscript. Cached .mat files generated before
    # that rename still carry the old names; alias them so callers always
    # see the new keys.
    for old, new in (("true_d", "true_kappa"),
                     ("est_d",  "est_kappa")):
        if old in S and new not in S:
            S[new] = S[old]
    return S


def regime_names(S) -> list[str]:
    return [r.name for r in S["regimes"]]


def savepdf(fig: plt.Figure, name: str) -> None:
    out = OUT_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def annotate_value(ax, x, y, value, *, dy=2, fontsize=FS_VALUE_LABEL,
                   weight="bold"):
    ax.text(x, y + dy, value, ha="center", va="bottom",
            fontsize=fontsize, fontweight=weight)


def identity_line(ax, lim, **kw):
    kw.setdefault("color", "0.3")
    kw.setdefault("linestyle", "--")
    kw.setdefault("linewidth", 1)
    kw.setdefault("zorder", 0)
    ax.plot(lim, lim, **kw)


# ---------------------------------------------------------------------------
# Figure 1: Demographics
# ---------------------------------------------------------------------------
def fig01_demographics():
    """No cached .mat — generate inline (matches s0 logic, fast)."""
    # Replicate fcn_generate_patient sampling for age and SOFA.
    # Age   ~ 40 + 55 * Beta(3, 5)         (Eq 1)
    # SOFA  ~ Gamma(2, 1.5) truncated [0, 24], rounded (Eq 2)
    rng = np.random.default_rng(42)
    N = 10_000
    ages = 40 + 55 * rng.beta(3, 5, N)
    sofa_raw = rng.gamma(2, 1.5, N)
    sofas = np.clip(np.round(sofa_raw), 0, 24).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4))

    # Age — smooth KDE
    sns.kdeplot(ages, ax=axes[0], color=COHORT_COLORS["untreated"], linewidth=2.2,
                clip=(18, 100), fill=True, alpha=0.18)
    axes[0].set(xlabel="Age (years)", ylabel="Density",
                title="Age distribution (N = 10,000)", xlim=(18, 100))
    stats_age = (f"mean {ages.mean():.1f}\nmedian {np.median(ages):.1f}\nSD {ages.std():.1f}")
    axes[0].text(0.97, 0.95, stats_age, transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=9,
                 bbox=dict(facecolor="white", edgecolor="0.8", boxstyle="round,pad=0.4"))

    # SOFA — discrete bar
    sofa_vals = np.arange(25)
    counts = np.bincount(sofas, minlength=25)
    pct = counts / N * 100
    axes[1].bar(sofa_vals, pct, color=COHORT_COLORS["treated"], alpha=0.78,
                edgecolor="white", linewidth=0.6)
    axes[1].set(xlabel="SOFA score", ylabel="Patients (%)",
                title="SOFA distribution (N = 10,000)", xlim=(-0.5, 14.5))
    stats_sofa = (f"mean {sofas.mean():.1f}\nmedian {int(np.median(sofas))}\nSD {sofas.std():.1f}")
    axes[1].text(0.97, 0.95, stats_sofa, transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=9,
                 bbox=dict(facecolor="white", edgecolor="0.8", boxstyle="round,pad=0.4"))

    fig.suptitle("Synthetic patient demographics", y=1.02)
    savepdf(fig, "paper_fig01_demographics.pdf")


# ---------------------------------------------------------------------------
# Figure 2: Individual-level parameter recovery (6 panels: Ap, r, d × ut/tx)
# ---------------------------------------------------------------------------
def fig02_param_recovery():
    S = load_mat("parameter_estimation_results.mat")
    ut = ~S["is_treated"].astype(bool)
    tx = S["is_treated"].astype(bool)

    params = [
        ("Ap",    "true_Ap",    "est_Ap",    r"$A_p$"),
        ("r",     "true_r",     "est_r",     r"$r$"),
        ("kappa", "true_kappa", "est_kappa", r"$\kappa$"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 5.0), sharex="col", sharey="col")

    for col, (_, tk, ek, label) in enumerate(params):
        x_all = np.concatenate([S[tk], S[ek]])
        # Clip outliers for r and d (s6 does this in MATLAB code)
        lim_max = np.percentile(x_all, 99.5) if col > 0 else x_all.max()
        lim = (0, lim_max * 1.05)

        for row, (idx, color, name) in enumerate([
            (ut, COHORT_COLORS["untreated"], "untreated"),
            (tx, COHORT_COLORS["treated"],   "treated"),
        ]):
            ax = axes[row, col]
            r = np.corrcoef(S[tk][idx], S[ek][idx])[0, 1]
            ax.scatter(S[tk][idx], S[ek][idx], s=18, alpha=0.55,
                       color=color, edgecolor="white", linewidth=0.3)
            identity_line(ax, lim)
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(f"{label} ({name})  r = {r:.2f}")
            if col == 0: ax.set_ylabel(f"Estimated {label}")
            if row == 1: ax.set_xlabel(f"True {label}")

    fig.suptitle("Parameter recovery: true vs estimated", y=1.01)
    fig.tight_layout()
    savepdf(fig, "paper_fig02_param_recovery.pdf")


# ---------------------------------------------------------------------------
# Figure 3: Derived quantity recovery (Lcum, peak height, peak timing)
# ---------------------------------------------------------------------------
def fig03_derived_recovery():
    S = load_mat("parameter_estimation_results.mat")
    ut = ~S["is_treated"].astype(bool)
    tx = S["is_treated"].astype(bool)

    # Clip peak-time axes to data range — full 7-day extent wastes space because
    # ~all data sits in [0, 3].
    triplets = [
        ("L_cum",      "true_Lcum",        "est_Lcum",        r"$L_{\mathrm{cum}}$ (sz$\cdot$day)", None),
        ("Peak height","true_peak_height", "est_peak_height", "Peak height (sz/day)",                None),
        ("Peak time",  "true_peak_time",   "est_peak_time",   "Peak time (days)",                    (0, 3.5)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.6))

    for ax, (name, tk, ek, axlabel, clip) in zip(axes, triplets):
        r = np.corrcoef(S[tk], S[ek])[0, 1]
        if clip is not None:
            lim = clip
        else:
            x_all = np.concatenate([S[tk], S[ek]])
            lim = (0, x_all.max() * 1.05)

        ax.scatter(S[tk][ut], S[ek][ut], s=20, alpha=0.55,
                   color=COHORT_COLORS["untreated"],
                   edgecolor="white", linewidth=0.3, label="Untreated")
        ax.scatter(S[tk][tx], S[ek][tx], s=20, alpha=0.55,
                   color=COHORT_COLORS["treated"],
                   edgecolor="white", linewidth=0.3, label="Treated")
        identity_line(ax, lim)
        ax.set(xlim=lim, ylim=lim,
               xlabel=f"True {axlabel}", ylabel=f"Estimated {axlabel}",
               title=f"{name}  r = {r:.3f}")
        ax.set_aspect('equal', adjustable='box')

    # Single shared legend instead of one per panel
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Derived quantity recovery: what matters for mortality",
                 y=1.04)
    fig.tight_layout()
    savepdf(fig, "paper_fig03_derived_recovery.pdf")


# ---------------------------------------------------------------------------
# Helper: per-patient covariates for s9 figures
# ---------------------------------------------------------------------------
def _s9_covariates(S: dict):
    """Return (peak_rates, sofas) for the s9 cohort, in the same patient
    order as the cached gform_surv_treated / obs_assignment / rct_assignment
    vectors. Populated by augment_gformula_results.m (which replays s9's
    patient generation under rng(42) and saves the per-patient covariates)."""
    if "peak_rates" not in S or "sofas" not in S:
        raise RuntimeError(
            "gformula_results.mat is missing peak_rates / sofas.\n"
            "Run `matlab -batch augment_gformula_results` once to add them."
        )
    return np.asarray(S["peak_rates"]).flatten(), np.asarray(S["sofas"]).flatten()


# ---------------------------------------------------------------------------
# Figure 4: Confounding by indication (RCT vs Observational)
# ---------------------------------------------------------------------------
def fig04_confounding():
    S = load_mat("gformula_results.mat")
    peak_rates, _ = _s9_covariates(S)
    rct = np.asarray(S["rct_assignment"]).astype(bool)
    obs = np.asarray(S["obs_assignment"]).astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.4))

    def panel(ax, mask, title):
        treated = peak_rates[mask]
        untreated = peak_rates[~mask]
        bins = np.linspace(0, np.percentile(peak_rates, 99), 30)
        ax.hist(untreated, bins=bins, color=STRAT_COLORS["None"], alpha=0.55,
                edgecolor="white", linewidth=0.4, label=f"Untreated (n={(~mask).sum()})")
        ax.hist(treated,  bins=bins, color=STRAT_COLORS["Adaptive"], alpha=0.65,
                edgecolor="white", linewidth=0.4, label=f"Treated (n={mask.sum()})")
        ax.axvline(treated.mean(), color=STRAT_COLORS["Adaptive"], linestyle=":", linewidth=1.5)
        ax.axvline(untreated.mean(), color=STRAT_COLORS["None"], linestyle=":", linewidth=1.5)
        ax.set(xlabel="Peak seizure rate (sz/day)", ylabel="Patients", title=title)
        ax.legend(loc="upper right")

    panel(axes[0], rct,
          f"RCT — coin-flip assignment\nmean(treated)={peak_rates[rct].mean():.1f},  mean(untreated)={peak_rates[~rct].mean():.1f}")
    panel(axes[1], obs,
          f"Observational — confounding by indication\nmean(treated)={peak_rates[obs].mean():.1f},  mean(untreated)={peak_rates[~obs].mean():.1f}")

    fig.suptitle("Treatment assignment bias", y=1.03)
    fig.tight_layout()
    savepdf(fig, "paper_fig04_confounding.pdf")


# ---------------------------------------------------------------------------
# Figure 4b: Method comparison — RCT truth vs naive observational vs g-formula
# ---------------------------------------------------------------------------
def fig04b_method_comparison():
    S = load_mat("gformula_results.mat")
    methods   = ["RCT (truth)", "Naive obs.", "G-formula"]
    untreated = [float(S["rct_surv_untreated"]),
                 float(S["naive_surv_untreated"]),
                 float(S["gform_surv_untreat_pct"])]
    treated   = [float(S["rct_surv_treated"]),
                 float(S["naive_surv_treated"]),
                 float(S["gform_surv_treat_pct"])]
    ates      = [float(S["rct_ate"]),
                 float(S["naive_ate"]),
                 float(S["gform_ate"])]

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    x = np.arange(len(methods))
    width = 0.36
    ax.bar(x - width/2, untreated, width, color=STRAT_COLORS["None"],
           edgecolor="white", linewidth=0.8, label="Untreated")
    ax.bar(x + width/2, treated, width, color=STRAT_COLORS["Adaptive"],
           edgecolor="white", linewidth=0.8, label="Treated")

    for xi, (u, t) in enumerate(zip(untreated, treated)):
        annotate_value(ax, xi - width/2, u, f"{u:.1f}", dy=1.0, fontsize=8.5)
        annotate_value(ax, xi + width/2, t, f"{t:.1f}", dy=1.0, fontsize=8.5)

    ymax = max(max(untreated), max(treated))
    for xi, a in enumerate(ates):
        ax.text(xi, ymax + 6, f"ATE = {a:+.1f} pp",
                ha="center", va="bottom",
                fontsize=FS_INLINE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("7-day survival (%)")
    ax.set_ylim(0, ymax + 12)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Causal effect recovery: naive bias vs g-formula")
    fig.tight_layout()
    savepdf(fig, "paper_fig04b_method_comparison.pdf")


# ---------------------------------------------------------------------------
# Figure 5: G-formula treatment effect by patient severity
# ---------------------------------------------------------------------------
def fig05_te_by_severity():
    S = load_mat("gformula_results.mat")
    peak_rates, sofa = _s9_covariates(S)

    individual_te = (S["gform_surv_treated"] - S["gform_surv_untreated"]) * 100
    ate = float(S["gform_ate"])

    # Bootstrap CI for the ATE
    rng = np.random.default_rng(0)
    n_boot = 2000
    boot_ates = rng.choice(individual_te, size=(n_boot, len(individual_te))).mean(axis=1)
    ate_lo, ate_hi = np.percentile(boot_ates, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    # Smaller markers + lower alpha to expose density without overplotting
    sc = ax.scatter(peak_rates, individual_te, c=sofa, s=11,
                    alpha=0.55, cmap="viridis", edgecolor="none")
    ax.axhline(0, color="0.3", linestyle="--", linewidth=1)
    ax.axhspan(ate_lo, ate_hi, color=STRAT_COLORS["Adaptive"], alpha=0.15)
    ax.axhline(ate, color=STRAT_COLORS["Adaptive"], linewidth=2,
               label=f"ATE = {ate:+.1f} pp  (95% CI [{ate_lo:+.1f}, {ate_hi:+.1f}])")
    # Clip x-axis to where data density lives (right tail is sparse outliers)
    ax.set_xlim(0, np.percentile(peak_rates, 99))
    ax.set(xlabel="Peak seizure rate (sz/day)",
           ylabel="Individual treatment effect, adaptive vs none (pp)",
           title="G-formula: treatment effect by patient severity")
    cbar = fig.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("SOFA score")
    ax.legend(loc="upper right")
    fig.tight_layout()
    savepdf(fig, "paper_fig05_te_by_severity.pdf")


# ---------------------------------------------------------------------------
# Figure 6: Quadrant analysis — optimal strategy depends on harm profile
# ---------------------------------------------------------------------------
def _quadrant_bars(ax, surv_row, strats, *, ylim=(0, 100),
                   best_marker=True, yerr=None):
    colors = [STRAT_COLORS[s] for s in strats]
    bars = ax.bar(strats, surv_row, color=colors, edgecolor="white",
                  linewidth=0.8, width=0.7,
                  yerr=yerr, capsize=3, error_kw=dict(ecolor="0.25", lw=1))
    ax.set_ylim(*ylim)
    ax.set_ylabel("7-day survival (%)")
    ax.grid(axis="x", visible=False)
    if best_marker:
        i_best = int(np.argmax(surv_row))
        bars[i_best].set_edgecolor("#222")
        bars[i_best].set_linewidth(1.8)
    for x, v in enumerate(surv_row):
        # Lift label above error-bar cap if present
        dy = 1.5 if yerr is None else float(yerr[x]) + 1.5
        annotate_value(ax, x, v, f"{v:.0f}", dy=dy, fontsize=8.5)
    return bars


def fig06_quadrant():
    S = load_mat("quadrant_results.mat")
    surv = S["surv_rates"]
    names = regime_names(S)
    strats = list(S["strat_labels"])
    Npat = int(S["all_results"][0].surv_pat.shape[0])

    # SE from per-patient survival in each regime/strategy
    se = np.zeros_like(surv)
    for qi in range(surv.shape[0]):
        sp = np.asarray(S["all_results"][qi].surv_pat)  # [Npat, n_strats]
        se[qi, :] = sp.std(axis=0) / np.sqrt(sp.shape[0]) * 100

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.6))
    quad_order = [0, 1, 2, 3]
    for qi, ax in zip(quad_order, axes.flat):
        _quadrant_bars(ax, surv[qi, :], strats, yerr=se[qi, :])
        i_best = int(np.argmax(surv[qi, :]))
        ax.set_title(f"{names[qi]}\nbest: {strats[i_best]} ({surv[qi, i_best]:.0f}%)")

    # One n= annotation + error-bar definition in the top-left axes
    axes[0, 0].text(0.02, 0.96,
                    f"n = {Npat} patients/regime\nError bars: SEM",
                    transform=axes[0, 0].transAxes,
                    fontsize=8.5, color="0.3", va="top", ha="left")

    fig.suptitle("Optimal strategy depends on harm profile", y=1.00)
    fig.tight_layout()
    savepdf(fig, "paper_fig06_quadrant.pdf")


# ---------------------------------------------------------------------------
# Figure 7: Threshold sweep (PRN and Adaptive)
# ---------------------------------------------------------------------------
def fig07_threshold_sweep():
    S = load_mat("threshold_sweep_results.mat")
    rp = S["results_prn"]; ro = S["results_omega"]; rb = S["results_baseline"]

    # rp/ro are scipy mat_struct → expose attributes directly
    prn_x = np.asarray(S["prn_thresholds"])
    om_x  = np.asarray(S["omega_values"])
    prn_y = np.asarray(rp.surv_rate); prn_err = np.asarray(rp.surv_std)
    om_y  = np.asarray(ro.surv_rate); om_err  = np.asarray(ro.surv_std)
    base  = float(rb.surv_rate)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.4), sharey=True)

    for ax, (xs, ys, errs, color, title, best_label) in zip(
        axes,
        [(prn_x, prn_y, prn_err, STRAT_COLORS["PRN"],
          "PRN: optimal threshold = {0:.0f}", "PRN threshold (counts in 8h window)"),
         (om_x,  om_y,  om_err,  STRAT_COLORS["Adaptive"],
          r"Adaptive: optimal $\omega$ = {0:.0f}", r"Adaptive target rate $\omega$ (sz/day)")],
    ):
        ax.errorbar(xs, ys, yerr=errs, fmt="o-", color=color,
                    markerfacecolor=color, markeredgecolor="white",
                    markersize=7, linewidth=1.8, capsize=3, elinewidth=1)
        ax.axhline(base, color="0.5", linestyle="--", linewidth=1)
        ax.text(xs[-1], base, " no treatment", va="center", ha="left",
                fontsize=FS_INLINE, color="0.5")
        i = int(np.argmax(ys))
        # Subtle "optimum" indicator instead of oversized yellow marker
        ax.axvline(xs[i], color="0.4", linestyle=":", linewidth=1)
        ax.annotate(f"optimum", xy=(xs[i], ys[i]),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=8.5, color="0.3", fontweight="bold")
        ax.set(xlabel=best_label, title=title.format(xs[i]))
        ax.set_xlim(0, max(xs) + 1)

    axes[0].set_ylabel("7-day survival (%)")
    # Error bars annotation
    axes[0].text(0.02, 0.04, "Error bars: SEM", transform=axes[0].transAxes,
                 fontsize=FS_INLINE, color="0.4", ha="left", va="bottom")
    fig.suptitle(f"Threshold sweep (N = {int(S['Npat'])} patients)", y=1.02)
    fig.tight_layout()
    savepdf(fig, "paper_fig07_threshold_sweep.pdf")


# ---------------------------------------------------------------------------
# Figure 8: Personalized threshold validation
# ---------------------------------------------------------------------------
def fig08_personalized():
    S = load_mat("personalized_validation_results.mat")
    names = regime_names(S)
    omega_vals = np.asarray(S["omega_vals"])
    i_uniform = int(np.argmin(np.abs(omega_vals - 2.0)))

    n = len(names)
    bar_data = np.zeros((n, 3))
    for reg in range(n):
        bar_data[reg, 0] = float(S["surv_true_grid"][:, i_uniform, reg].mean()) * 100
        bar_data[reg, 1] = float(S["surv_at_true_opt"][:, reg].mean()) * 100
        bar_data[reg, 2] = float(S["surv_at_est_opt"][:, reg].mean()) * 100

    series_labels = [r"Uniform ($\omega = 2$)",
                     r"Personalized (true $\lambda$)",
                     r"Personalized (estimated $\lambda$)"]
    colors = ["#9A9A9A", "#4FAD5B", "#4878CF"]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(n); w = 0.26
    for i, (lab, col) in enumerate(zip(series_labels, colors)):
        bars = ax.bar(x + (i - 1) * w, bar_data[:, i], w,
                      label=lab, color=col, edgecolor="white", linewidth=0.7)
        if i > 0:
            for j, b in enumerate(bars):
                gain = bar_data[j, i] - bar_data[j, 0]
                annotate_value(ax, b.get_x() + b.get_width() / 2,
                               b.get_height(), f"{gain:+.1f}", dy=1, fontsize=FS_INLINE)

    ax.set(xticks=x, xticklabels=names, ylabel="7-day survival (%)",
           ylim=(0, 100),
           title="Can estimated parameters recover personalized benefit?")
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, -0.18))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    savepdf(fig, "paper_fig08_personalized.pdf")


# ---------------------------------------------------------------------------
# Figure 9: Uniform vs Fixed-Optimal vs Dynamic omega
# ---------------------------------------------------------------------------
def fig09_uniform_vs_dynamic():
    S = load_mat("closed_loop_personalized_results.mat")
    surv = np.asarray(S["surv_results"])  # [n_regimes, 3]
    names = regime_names(S)

    n = len(names)
    labels = [r"Uniform ($\omega = 2$)", r"Fixed $\omega^*$ per regime",
              r"Dynamic $\omega(t)$"]
    colors = ["#9A9A9A", "#4FAD5B", "#4878CF"]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(n); w = 0.26
    for i, (lab, col) in enumerate(zip(labels, colors)):
        bars = ax.bar(x + (i - 1) * w, surv[:, i], w, label=lab,
                      color=col, edgecolor="white", linewidth=0.7)
        if i > 0:
            for j, b in enumerate(bars):
                gain = surv[j, i] - surv[j, 0]
                if abs(gain) > 0.5:
                    annotate_value(ax, b.get_x() + b.get_width() / 2,
                                   b.get_height(), f"{gain:+.1f}", dy=1, fontsize=FS_INLINE)

    ax.set(xticks=x, xticklabels=names, ylabel="7-day survival (%)",
           ylim=(0, 100), title=r"Uniform vs fixed-optimal vs dynamic $\omega$")
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, -0.18))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    savepdf(fig, "paper_fig09_uniform_vs_dynamic.pdf")


# ---------------------------------------------------------------------------
# Figure 10: Closed-loop pipeline validation (True top, Estimated bottom)
# ---------------------------------------------------------------------------
def fig10_closed_loop():
    S = load_mat("closed_loop_results.mat")
    names = regime_names(S)
    strats = list(S["strat_labels"])
    surv_t = S["surv_true"]
    surv_e = S["surv_est"]
    n = len(names)

    fig, axes = plt.subplots(2, n, figsize=(6.5, 5.6), sharey=True)
    for col in range(n):
        for row, surv in enumerate((surv_t, surv_e)):
            ax = axes[row, col]
            _quadrant_bars(ax, surv[col, :], strats, best_marker=False)
            i_best = int(np.argmax(surv[col, :]))
            ax.set_title(f"{strats[i_best]} ({surv[col, i_best]:.0f}%)")
            if col == 0:
                row_label = "True" if row == 0 else "Estimated"
                ax.set_ylabel(f"{row_label}\n7-day survival (%)")
            else:
                ax.set_ylabel("")
            for tl in ax.get_xticklabels():
                tl.set_rotation(25); tl.set_horizontalalignment("right")
        axes[0, col].text(0.5, 1.28, names[col],
                          ha="center", va="bottom",
                          transform=axes[0, col].transAxes,
                          fontsize=FS_PANEL_HDR, fontweight="bold")

    fig.suptitle(f"Closed-loop validation: true (top) vs estimated (bottom), "
                 f"n = {int(S['Npat'])}", y=1.04)
    fig.tight_layout()
    savepdf(fig, "paper_fig10_closed_loop.pdf")


# ---------------------------------------------------------------------------
# Figure 11: EKF parameter sensitivity (6 panels)
# ---------------------------------------------------------------------------
def fig11_ekf_sensitivity():
    S = load_mat("ekf_sensitivity_results.mat")
    results = S["results"]
    sweeps = S["sweeps"]
    base_none = float(S["baseline_none"])
    base_fix  = float(S["baseline_fixed"])

    n = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 7.4))
    for s, ax in enumerate(axes.flat[:n]):
        r = results[s]
        xs = np.asarray(r.values)
        ys = np.asarray(r.survival)
        log_x = (r.param == "q")
        if log_x:
            ax.semilogx(xs, ys, "o-", color=STRAT_COLORS["Adaptive"],
                        markerfacecolor=STRAT_COLORS["Adaptive"],
                        markeredgecolor="white", markersize=7, linewidth=1.8)
        else:
            ax.plot(xs, ys, "o-", color=STRAT_COLORS["Adaptive"],
                    markerfacecolor=STRAT_COLORS["Adaptive"],
                    markeredgecolor="white", markersize=7, linewidth=1.8)
        ax.axhline(base_none, color=STRAT_COLORS["PRN"],   linestyle="--", linewidth=1)
        ax.axhline(base_fix,  color="0.3",                  linestyle="--", linewidth=1)
        ax.text(xs[-1], base_none, " no tx",   va="center", ha="left",
                fontsize=FS_INLINE, color=STRAT_COLORS["PRN"])
        ax.text(xs[-1], base_fix,  " fixed",   va="center", ha="left",
                fontsize=FS_INLINE, color="0.3")

        default = float(sweeps[s].default)
        i_default = int(np.argmin(np.abs(xs - default)))
        ax.scatter(xs[i_default], ys[i_default], s=110,
                   facecolor="none", edgecolor=STRAT_COLORS["PRN"],
                   linewidth=2, zorder=4)
        ax.set_title(r.name)
        ax.set_ylabel("7-day survival (%)")
        ax.set_xlabel(r.param)

    # Sensitivity ranking (last panel)
    ax = axes.flat[-1]
    spreads = np.array([np.ptp(np.asarray(r.survival)) for r in results])
    order = np.argsort(spreads)
    params = [results[i].param for i in order]
    ax.barh(np.arange(n), spreads[order], color=STRAT_COLORS["Adaptive"],
            edgecolor="white", linewidth=0.6)
    ax.set_yticks(np.arange(n)); ax.set_yticklabels(params)
    ax.set_xlabel("Survival spread (percentage points)")
    ax.set_title("Sensitivity ranking")
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(spreads[order]):
        ax.text(v + 0.4, i, f"{v:.1f}", va="center", fontsize=FS_INLINE)

    # Shared legend explaining the open red circle marker
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="-",
               color=STRAT_COLORS["Adaptive"],
               markerfacecolor=STRAT_COLORS["Adaptive"],
               markeredgecolor="white", markersize=7,
               label="Adaptive strategy"),
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor="none", markeredgecolor=STRAT_COLORS["PRN"],
               markeredgewidth=2, markersize=10,
               label="Default parameter value"),
        Line2D([0], [0], color="0.3", linestyle="--", label="Fixed dosing"),
        Line2D([0], [0], color=STRAT_COLORS["PRN"], linestyle="--",
               label="No treatment"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("EKF parameter sensitivity", y=1.02)
    fig.tight_layout()
    savepdf(fig, "paper_fig11_ekf_sensitivity.pdf")


# ---------------------------------------------------------------------------
# Figure 12: Graded trajectory misspecification
# ---------------------------------------------------------------------------
def fig12_graded_misspec():
    """Two stacked panels rather than dual y-axis (the dual axis was visually
    misleading because the y-scales are unrelated)."""
    S = load_mat("misspecification_results.mat")
    mix = np.asarray(S["mix_fractions"]) * 100
    bias  = np.asarray(S["graded_bias"])
    gform = np.asarray(S["graded_gform"])
    rct   = np.asarray(S["graded_rct"])

    fig, (ax_ate, ax_bias) = plt.subplots(
        2, 1, figsize=(6.5, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1, 0.85], "hspace": 0.10},
    )

    # Top: the two ATEs (estimator vs truth)
    ax_ate.plot(mix, rct,  "o-", color="0.2", linewidth=2,
                markersize=6, markeredgecolor="white",
                label="RCT ATE (truth)")
    ax_ate.plot(mix, gform, "s--", color=STRAT_COLORS["PRN"],
                linewidth=1.8, markersize=6, markeredgecolor="white",
                label="G-formula ATE")
    ax_ate.fill_between(mix, gform, rct, color=STRAT_COLORS["PRN"], alpha=0.10)
    ax_ate.set_ylabel("ATE (percentage points)")
    ax_ate.set_title("Graded trajectory misspecification — robustness curve")
    ax_ate.legend(loc="upper right")

    # Bottom: the absolute bias (|gform - rct|)
    ax_bias.fill_between(mix, 0, bias, color=STRAT_COLORS["Fixed"], alpha=0.18)
    ax_bias.plot(mix, bias, "o-", color=STRAT_COLORS["Fixed"], linewidth=2,
                 markersize=6, markeredgecolor="white",
                 label="G-formula residual bias")
    ax_bias.set_xlabel("Fraction of SDE-generated patients (%)")
    ax_bias.set_ylabel("Residual bias (pp)")
    ax_bias.set_ylim(0, max(bias.max() * 1.25, 1))
    ax_bias.legend(loc="upper left")

    fig.tight_layout()
    savepdf(fig, "paper_fig12_graded_misspec.pdf")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
DISPATCH = {
    1:  fig01_demographics,
    2:  fig02_param_recovery,
    3:  fig03_derived_recovery,
    4:  fig04_confounding,
    13: fig04b_method_comparison,   # sits between fig04 and fig05 in the paper
    5:  fig05_te_by_severity,
    6:  fig06_quadrant,
    7:  fig07_threshold_sweep,
    8:  fig08_personalized,
    9:  fig09_uniform_vs_dynamic,
    10: fig10_closed_loop,
    11: fig11_ekf_sensitivity,
    12: fig12_graded_misspec,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figs", type=str, default="",
                        help="Comma-separated figure numbers (default: all)")
    args = parser.parse_args()
    if args.figs:
        wanted = [int(x) for x in args.figs.split(",")]
    else:
        wanted = sorted(DISPATCH.keys())

    for k in wanted:
        print(f"=== Paper figure {k} ===")
        try:
            DISPATCH[k]()
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()

    print(f"\nDone. PDFs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
