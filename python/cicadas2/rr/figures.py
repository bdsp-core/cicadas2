"""Revision figures drawn from results/rr_*.json / .npz (task 22; no MATLAB needed).

  python -m cicadas2.rr.figures      # writes CICADAS_II_Paper_Draft/figures_rr/*.pdf and .png

Every figure reads the same results files as the macros and tables, so figures, tables and text agree (G1-G4).
Style: thin marks, recessive axes, identity carried by axis labels so no figure depends on colour alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .theta import QUADS, REGISTRY  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results"
OUT = ROOT / "CICADAS_II_Paper_Draft" / "figures_rr"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
LAB = {q: REGISTRY[q]["label"] for q in QUADS}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.5,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


def _j(name):
    return json.loads((RES / f"rr_{name}.json").read_text())


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_harm():
    """Paired survival difference from the tuned fixed dose (held-out half), per phenotype."""
    from . import stats
    h = _j("harm")
    z = np.load(RES / "rr_harm.npz")
    labels = list(z["labels"])
    test = np.arange(z["S_default"].shape[0]) % 2 == 1
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4))
    for ax, q in zip(axes, QUADS):
        R = h["regimes"][q]
        t = R["tuned"]
        S = z[f"S_{q}"][test]
        col = lambda k: S[:, labels.index(k)]
        ref = col(t["fixed"])
        rows = [("No treatment", "none"), ("PRN, tuned", t["prn"]), ("Tiered adaptive, tuned", t["adapt"]),
                ("Harm-balanced", "harm_est"), ("Harm-balanced, oracle", "harm_oracle_full")]
        y = np.arange(len(rows))[::-1]
        ax.axvline(0, color=INK2, lw=0.8)
        for yi, (lab, k) in zip(y, rows):
            d = stats.paired(col(k), ref, seed=81)
            harm = k.startswith("harm")
            c = BLUE if harm else INK2
            ax.plot([d["lo"], d["hi"]], [yi, yi], color=c, lw=1.4, solid_capstyle="round")
            ax.plot(d["est"], yi, "o", ms=4.5 if harm else 3.5, color=c, mfc=c if k != "harm_oracle_full" else "white", mew=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows] if ax is axes[0] else [""] * len(rows))
        ax.set_title(f"{LAB[q]}\n(reference: fixed {t['fixed'].split('_')[1]} mg)", fontsize=7.5)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    fig.supxlabel("Survival difference from the tuned fixed dose (percentage points, 95% paired CI)", fontsize=8, y=0.02)
    fig.tight_layout(w_pad=0.6)
    _save(fig, "fig_harm_controller")


def fig_confounding():
    """G-formula bias as a hidden confounder strengthens."""
    c = _j("confounding")
    k = [r["kappa"] if "kappa" in r else kk for r, kk in zip(c["rows"], [0, 0.25, 0.5, 1, 2, 4])]
    b = np.array([r["bias"] for r in c["rows"]])
    lo = np.array([r["ci"]["bias"][0] for r in c["rows"]]); hi = np.array([r["ci"]["bias"][1] for r in c["rows"]])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.axhline(0, color=INK2, lw=0.6)
    ax.fill_between(k, lo, hi, color=BLUE, alpha=0.15, lw=0)
    ax.plot(k, b, "-o", color=BLUE, ms=4)
    ax.set_xlabel(r"Strength of hidden confounder $\kappa$")
    ax.set_ylabel("G-formula bias (pp)")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    _save(fig, "fig_confounding")


def fig_overlap():
    """Estimated propensity of treatment, treated vs untreated patients."""
    z = np.load(RES / "rr_overlap.npz")
    ps, A = z["ps_hat"], z["A"].astype(bool)
    bins = np.linspace(0, 1, 26)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.hist(ps[A], bins=bins, density=True, histtype="step", color=BLUE, lw=1.6, label=f"Treated (n = {A.sum()})")
    ax.hist(ps[~A], bins=bins, density=True, histtype="step", color=ORANGE, lw=1.6, label=f"Untreated (n = {(~A).sum()})")
    ax.axvline(0.95, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("Estimated propensity of treatment")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "fig_overlap")


def fig_closed_loop():
    """Survival under each strategy from the true trajectory vs the analyst's estimated trajectory."""
    p = RES / "rr_closed_loop.json"
    if not p.exists():
        return
    c = json.loads(p.read_text())
    names = {"none": "None", "fixed": "Fixed 500", "fixed_750": "Fixed 750", "prn8_k1": "PRN", "adapt_2": "Tiered", "harm": "Harm-bal."}
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2))
    for ax, q in zip(axes, QUADS):
        R = c["regimes"][q]
        tv = [R["true"][k]["est"] for k in names]; ev = [R["est"][k]["est"] for k in names]
        lo, hi = min(tv + ev) - 3, max(tv + ev) + 3
        ax.plot([lo, hi], [lo, hi], color=GRID, lw=1, zorder=0)
        for k, nm in names.items():
            col = BLUE if k == "harm" else INK2
            ax.plot(R["true"][k]["est"], R["est"][k]["est"], "o", color=col, ms=4)
        win_t, win_e = R["winner_true"], R["winner_est"]
        ax.set_title(f"{LAB[q]}\nbest: {names[win_t]}" + ("" if win_t == win_e else f" (est. {names[win_e]})"), fontsize=7.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("True trajectory (%)")
        if ax is axes[0]:
            ax.set_ylabel("Estimated trajectory (%)")
    fig.tight_layout(w_pad=0.8)
    _save(fig, "fig_closed_loop")


def main():
    """The paper now uses the restyled versions in paper_figures.py (same data, house style of the submitted
    figures); the functions above are kept for reference."""
    from . import paper_figures
    paper_figures.main([])
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
