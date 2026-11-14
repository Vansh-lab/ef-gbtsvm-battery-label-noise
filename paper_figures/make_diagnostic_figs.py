"""Two diagnostic/comparison figures for the paper, from already-computed Phase B results only.
No new experiment is run and no number is invented; everything plotted is read from
results/phys_phase_b.log (fig1) or results/phase_b_summary_table.csv (fig2).

    python paper_figures/make_diagnostic_figs.py

Writes results/fig1_noise_calibration.{png,pdf} and results/fig2_grid_comparison.{png,pdf}.
Style matches paper_figures/build_ieee_tex.py's charts() (serif, font.size 7.5, no top/right
spines, single-column figsize, grey/blue palette).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False})

GREY, BLUE = "#6b6b6b", "#1f5fbf"

# ---- Figure 1: noise-estimate calibration -------------------------------------------------
cal = pd.read_csv("results/fig_noise_calibration.csv")
cal["injected_pct"] = cal["injected_noise"] * 100
cal["est_pct"] = cal["est_noise"] * 100

fig, ax = plt.subplots(figsize=(3.5, 1.75))
rng = np.random.RandomState(0)
jitter = rng.uniform(-0.6, 0.6, size=len(cal))
ax.plot([0, 20], [0, 20], color=GREY, ls="--", lw=0.8, zorder=1, label="estimate = injected")
ax.scatter(cal["injected_pct"] + jitter, cal["est_pct"], s=16, color=BLUE, alpha=0.85, zorder=3,
           label="per fold (EF-GBTSVM-A)")

lines = ["injected   mean est.   mean $\\tau$"]
for noise_pct, grp in cal.groupby("injected_pct"):
    mean_est = grp["est_pct"].mean()
    mean_tau = grp["purity_used"].mean()
    lines.append(f"{noise_pct:>5.0f}%   {mean_est:>8.1f}%   {mean_tau:>8.3f}")
ax.text(0.98, 0.32, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
        fontsize=6, color="#333333", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", linewidth=0.5))

ax.set_xlim(-2, 24)
ax.set_ylim(-1, 34)
ax.set_xticks([0, 10, 20])
ax.set_xlabel("injected training-label noise (%)")
ax.set_ylabel("est. noise, $\\hat\\eta$ (%)")
ax.grid(axis="y", color="#dddddd", lw=0.5)
ax.legend(loc="upper left", frameon=False, fontsize=5.5, handletextpad=0.3, borderaxespad=0.2)
fig.tight_layout()
fig.savefig("results/fig1_noise_calibration.png", dpi=300)
fig.savefig("results/fig1_noise_calibration.pdf")
plt.close(fig)

# ---- Figure 2: accuracy vs noise, grid/variant comparison ---------------------------------
summ = pd.read_csv("results/phase_b_summary_table.csv")
MODELS = ["EF-GBTSVM", "EF-GBTSVM-L", "EF-GBTSVM-A", "Stump-fair"]
LABELS = {"EF-GBTSVM": "EF-GBTSVM (default grid)", "EF-GBTSVM-L": "EF-GBTSVM-L (wide grid)",
          "EF-GBTSVM-A": "EF-GBTSVM-A (noise-adaptive $\\tau$)", "Stump-fair": "Stump-fair (one-feature rule)"}
COLORS = {"EF-GBTSVM": "#6b6b6b", "EF-GBTSVM-L": "#c98a2a", "EF-GBTSVM-A": "#1f5fbf", "Stump-fair": "#2a9d6b"}
MARKERS = {"EF-GBTSVM": "o", "EF-GBTSVM-L": "s", "EF-GBTSVM-A": "^", "Stump-fair": "D"}

fig, ax = plt.subplots(figsize=(3.5, 1.95))
for m in MODELS:
    r = summ[summ.model == m].sort_values("noise_pct")
    ax.errorbar(r.noise_pct, r.ACC_mean, yerr=r.ACC_std, label=LABELS[m], color=COLORS[m],
                marker=MARKERS[m], markersize=4, lw=1, capsize=2, capthick=0.8, elinewidth=0.8)

ax.set_xticks([0, 10, 20])
ax.set_xlim(-3, 23)
ax.set_ylim(75, 100)
ax.set_xlabel("training-label noise (%)")
ax.set_ylabel("test accuracy (%)")
ax.grid(axis="y", color="#dddddd", lw=0.5)
ax.legend(loc="lower right", frameon=False, fontsize=5.5, handletextpad=0.3, borderaxespad=0.2,
          labelspacing=0.3)
fig.tight_layout()
fig.savefig("results/fig2_grid_comparison.png", dpi=300)
fig.savefig("results/fig2_grid_comparison.pdf")
plt.close(fig)

print("wrote results/fig1_noise_calibration.{png,pdf} and results/fig2_grid_comparison.{png,pdf}")
